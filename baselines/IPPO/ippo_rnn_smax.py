"""
Based on PureJaxRL Implementation of PPO
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from jaxmarl import make

from jaxmarl.wrappers.baselines import SMAXLogWrapper
from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX

import wandb
import functools
import matplotlib.pyplot as plt
from jaxmarl.viz.visualizer import Visualizer, SMAXVisualizer


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, avail_actions = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)

        pi = distrax.Categorical(logits=action_logits)

        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray

class LearnedPolicy(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        actor_mean = nn.Dense(self.action_dim)(x)
        actor_mean = activation(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(1)(x)
        critic = activation(critic)

        return pi, jnp.squeeze(critic, axis=-1)
    
def rollout(env, trained_params, config, max_steps=15, key=jax.random.PRNGKey(2)):
    """Run a rollout using the trained policy."""
    obs, state = env.reset(key)
    state_seq = []
    returns = {a: 0 for a in env.agents}

    # Initialize hidden states for RNN
    hidden_states = {
        agent: ScannedRNN.initialize_carry(1, config["GRU_HIDDEN_DIM"])
        for agent in env.agents
    }

    for _ in range(max_steps):
        key, key_s, key_seq = jax.random.split(key, 3)
        actions = {}

        for agent in env.agents[:env.num_allies]:  # Use policy for allied agents
            agent_obs = jnp.expand_dims(obs[agent], axis=0)  
            agent_avail = jnp.expand_dims(env.get_avail_actions(state)[agent], axis=0)

            ac_in = (jnp.expand_dims(agent_obs, axis=0), jnp.zeros((1, 1), dtype=bool), agent_avail)
            hidden_states[agent], pi, _ = ActorCriticRNN(env.action_space(agent).n, config=config).apply(
                trained_params, hidden_states[agent], ac_in
            )

            actions[agent] = pi.sample(seed=key_seq).squeeze()

        for agent in env.agents[env.num_allies:]:  # Random actions for enemies
            actions[agent] = env.action_space(agent).sample(key_seq)

        state_seq.append((key_s, state, actions))
        obs, state, rewards, dones, infos = env.step(key_s, state, actions)
        returns = {a: returns[a] + rewards[a] for a in env.agents}
        
        if dones["__all__"]:
            break

    print(f"Returns: {returns}")
    return state_seq



def compute_jsd_from_counts(action_sequences):
    """ Compute JSD using normalized high-level action distributions. """
    
    categories = ["move", "shoot", "wait"]
    action_counts = {agent: {cat: 0 for cat in categories} for agent in action_sequences}

    for agent, actions in action_sequences.items():
        for action in actions:
            action_counts[agent][action] += 1

    distributions = []
    for agent, counts in action_counts.items():
        total = sum(counts.values())
        if total > 0:
            distributions.append([counts[cat] / total for cat in categories])
        else:
            distributions.append([0.0, 0.0, 0.0])  
    
    if len(distributions) >= 2:
        return generalized_jsd(distributions)  
    return 0.0

def categorize_high_level_action(action, num_movement_actions):
    if action >= num_movement_actions - 1:  # Attack action
        return "shoot"
    elif action == num_movement_actions - 1:  # Wait action
        return "wait"
    else:
        return "move"

def compute_high_level_distributions(action_sequences, num_movement_actions):
    high_level_counts = {agent: {"move": 0, "shoot": 0, "wait": 0} for agent in action_sequences}

    for agent, actions in action_sequences.items():
        for action in actions:
            high_level_action = categorize_high_level_action(action, num_movement_actions)
            high_level_counts[agent][high_level_action] += 1

    # Convert counts to probability distributions
    high_level_distributions = {}
    for agent, counts in high_level_counts.items():
        total = sum(counts.values())
        high_level_distributions[agent] = {k: v / total for k, v in counts.items()} if total > 0 else counts

    return high_level_distributions


def generalized_jsd(distributions, weights=None):
    """
    Compute the generalized Jensen-Shannon Divergence for N distributions.
    """
    n = len(distributions)
    
    if n == 0:
        return 0.0
    
    if n == 1:
        return 0.0
        
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.array(weights) / np.sum(weights)

    distributions = [np.array(dist).flatten() for dist in distributions]
    
    mixture = np.zeros_like(distributions[0])
    for i in range(n):
        mixture += weights[i] * distributions[i]
    
    divergences = np.zeros(n)
    for i in range(n):
        # KL divergence from distribution i to the mixture
        kl_div = 0.0
        for j in range(len(distributions[i])):
            if distributions[i][j] > 0 and mixture[j] > 0:
                kl_div += distributions[i][j] * np.log(distributions[i][j] / mixture[j])
        divergences[i] = kl_div
    
    return np.sum(weights * divergences)

def compute_trajectory_generalized_jsd(trained_params, config, num_steps=100):
    """
    Compute generalized JSD across all agents over a trajectory using high-level actions and log to wandb.
    """

    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    
    key = jax.random.PRNGKey(0)
    key, key_reset = jax.random.split(key)
    obs, state = env.reset(key_reset)
    init_hstate = ScannedRNN.initialize_carry(1, config["GRU_HIDDEN_DIM"])
    
    generalized_jsd_values = []
    action_entropy_values = {}
    action_sequences = {agent: [] for agent in env.agents}
    
    for agent in env.agents[:env.num_allies]:
        action_entropy_values[agent] = []

    # Define high-level action mapping
    def categorize_high_level_action(action):
        if action >= env.num_movement_actions - 1:  # Attack action
            return "shoot"
        elif action == env.num_movement_actions - 1:  # Wait action
            return "wait"
        else:
            return "move"

    for step in range(num_steps):
        key, key_step = jax.random.split(key)
        avail_actions = env.get_avail_actions(state)
        action_probs = {}
        actions = {}
        
        # For allied agents, use trained policy
        for i, agent in enumerate(env.agents[:env.num_allies]):
            agent_obs = jnp.expand_dims(obs[agent], axis=0)  
            agent_avail = jnp.expand_dims(avail_actions[agent], axis=0)
            
            ac_in = (
                jnp.expand_dims(agent_obs, axis=0),  
                jnp.zeros((1, 1), dtype=bool),  
                agent_avail
            )
            
            # Apply network to get policy
            hstate, pi, _ = ActorCriticRNN(env.action_space(agent).n, config=config).apply(
                trained_params, init_hstate, ac_in
            )
            action_probs[agent] = np.array(pi.probs)
            
            agent_entropy = entropy(action_probs[agent])
            action_entropy_values[agent].append(agent_entropy)
            
            key, key_action = jax.random.split(key)
            action = pi.sample(seed=key_action).squeeze()
            actions[agent] = action
            action_sequences[agent].append(categorize_high_level_action(action))  # Store as high-level action
        
        for i, agent in enumerate(env.agents[env.num_allies:], start=env.num_allies):
            key, key_enemy = jax.random.split(key)
            actions[agent] = env.action_space(agent).sample(key_enemy)
            action_sequences[agent].append(categorize_high_level_action(actions[agent]))  # Store as high-level action
        
        # Compute high-level action distributions
        high_level_counts = {agent: {"move": 0, "shoot": 0, "wait": 0} for agent in env.agents[:env.num_allies]}

        for agent, actions_list in action_sequences.items():
            if agent in high_level_counts:
                for action in actions_list:
                    high_level_counts[agent][action] += 1

        high_level_distributions = {}
        for agent, counts in high_level_counts.items():
            total = sum(counts.values())
            high_level_distributions[agent] = {k: v / total for k, v in counts.items()} if total > 0 else counts

        # Compute JSD over high-level distributions
        distributions = [list(dist.values()) for dist in high_level_distributions.values()]
        if len(distributions) >= 2:
            # gen_jsd = generalized_jsd(distributions)
            gen_jsd = compute_jsd_from_counts(action_sequences)
            generalized_jsd_values.append(gen_jsd)

        obs, state, rewards, done, info = env.step(key_step, state, actions)
        
        if done["__all__"]:
            break

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(generalized_jsd_values)), generalized_jsd_values, label='Generalized JSD')
    
    plt.xlabel('Step')
    plt.ylabel('JSD Value')
    plt.title('Generalized JSD Across All Agents Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    wandb.log({"generalized_jsd_trajectory": wandb.Image(plt)})
    plt.close()
    
    avg_gen_jsd = np.mean(generalized_jsd_values) if generalized_jsd_values else 0
    action_logs = {
        agent: wandb.Table(columns=["Step", "Action"], data=[[step, act] for step, act in enumerate(action_sequences[agent])])
        for agent in env.agents
    }
    
    for agent, table in action_logs.items():
        print(f"Action sequence for {agent}:")
        
        table_data = table.data
        for row in table_data:
            print(row)

    wandb.log({
        "avg_generalized_jsd": avg_gen_jsd,
    })
    
    return {
        'generalized_jsd': generalized_jsd_values,
        'entropy': action_entropy_values,
        'action_sequences': action_sequences
    }


# def compute_trajectory_generalized_jsd(trained_params, config, num_steps=100):
#     """
#     Compute generalized JSD across all agents over a trajectory and log to wandb
#     """

#     scenario = map_name_to_scenario(config["MAP_NAME"])
#     env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    
#     key = jax.random.PRNGKey(0)
#     key, key_reset = jax.random.split(key)
#     obs, state = env.reset(key_reset)
#     init_hstate = ScannedRNN.initialize_carry(1, config["GRU_HIDDEN_DIM"])
    
#     generalized_jsd_values = []
#     action_entropy_values = {}
#     action_sequences = {agent: [] for agent in env.agents}
    
#     for agent in env.agents[:env.num_allies]:
#         action_entropy_values[agent] = []
    
#     for step in range(num_steps):
#         key, key_step = jax.random.split(key)
#         avail_actions = env.get_avail_actions(state)
#         action_probs = {}
#         actions = {}
        
#         # For allied agents, use trained policy
#         for i, agent in enumerate(env.agents[:env.num_allies]):
#             agent_obs = jnp.expand_dims(obs[agent], axis=0)  
#             agent_avail = jnp.expand_dims(avail_actions[agent], axis=0)
            
#             ac_in = (
#                 jnp.expand_dims(agent_obs, axis=0),  
#                 jnp.zeros((1, 1), dtype=bool),  
#                 agent_avail
#             )
            
#             # Apply network to get policy
#             hstate, pi, _ = ActorCriticRNN(env.action_space(agent).n, config=config).apply(
#                 trained_params, init_hstate, ac_in
#             )
#             action_probs[agent] = np.array(pi.probs)
            
#             agent_entropy = entropy(action_probs[agent])
#             action_entropy_values[agent].append(agent_entropy)
            
#             key, key_action = jax.random.split(key)
#             action = pi.sample(seed=key_action).squeeze()
#             actions[agent] = action
#             action_sequences[agent].append(action)
        
#         for i, agent in enumerate(env.agents[env.num_allies:], start=env.num_allies):
#             key, key_enemy = jax.random.split(key)
#             actions[agent] = env.action_space(agent).sample(key_enemy)
#             action_sequences[agent].append(int(action))
        
#         # Compute generalized JSD between all agents
#         distributions = [action_probs[agent] for agent in env.agents[:env.num_allies]]
#         if len(distributions) >= 2:
#             gen_jsd = generalized_jsd(distributions)
#             generalized_jsd_values.append(gen_jsd)
        
#         obs, state, rewards, done, info = env.step(key_step, state, actions)
        
#         if done["__all__"]:
#             break
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(len(generalized_jsd_values)), generalized_jsd_values, label='Generalized JSD')
    
#     plt.xlabel('Step')
#     plt.ylabel('JSD Value')
#     plt.title('Generalized JSD Across All Agents Over Time')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     wandb.log({"generalized_jsd_trajectory": wandb.Image(plt)})
#     plt.close()
    
#     avg_gen_jsd = np.mean(generalized_jsd_values) if generalized_jsd_values else 0
#     action_logs = {
#         agent: wandb.Table(columns=["Step", "Action"], data=[[step, act] for step, act in enumerate(action_sequences[agent])])
#         for agent in env.agents
#     }
    
#     for agent, table in action_logs.items():
#         print(f"Action sequence for {agent}:")
        
#         table_data = table.data
#         for row in table_data:
#             print(row)

#     wandb.log({
#         "avg_generalized_jsd": avg_gen_jsd,
#     })
    
#     return {
#         'generalized_jsd': generalized_jsd_values,
#         'entropy': action_entropy_values,
#         'action_sequences': action_sequences
#     }

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def log_final_visualization(trained_params, config):
    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    key = jax.random.PRNGKey(0)
    key, key_reset = jax.random.split(key)

    obs, state = env.reset(key_reset)
    init_hstate = ScannedRNN.initialize_carry(1, config["GRU_HIDDEN_DIM"])
    
    state_seq = []
    done = {"__all__": False}
    max_steps = 200  # Maximum episode length
    step_count = 0
    
    while not done["__all__"] and step_count < max_steps:
        key, key_step = jax.random.split(key)
        
        avail_actions = env.get_avail_actions(state)
        actions = {}

        for i, agent in enumerate(env.agents[:env.num_allies]):
            agent_obs = jnp.expand_dims(obs[agent], axis=0)  # Add batch dimension
            agent_avail = jnp.expand_dims(avail_actions[agent], axis=0)
            
            ac_in = (
                jnp.expand_dims(agent_obs, axis=0), 
                jnp.zeros((1, 1), dtype=bool), 
                agent_avail
            )
            
            # Apply network to get policy
            hstate, pi, _ = ActorCriticRNN(env.action_space(agent).n, config=config).apply(
                trained_params, init_hstate, ac_in
            )
            
            # Sample action
            key, key_action = jax.random.split(key)
            action = pi.sample(seed=key_action).squeeze()
            actions[agent] = action
        
        for i, agent in enumerate(env.agents[env.num_allies:], start=env.num_allies):
            key, key_enemy = jax.random.split(key)
            actions[agent] = env.action_space(agent).sample(key_enemy)
        
        state_seq.append((key_step, state, actions))
        obs, state, rewards, done, info = env.step(key_step, state, actions)
        step_count += 1
    
    viz = SMAXVisualizer(env, state_seq)
    
    gif_path = "final_episode.gif"
    viz.animate(view=False, save_fname=gif_path)
    wandb.log({"final_episode": wandb.Video(gif_path, fps=4, format="gif")})

    import os
    if os.path.exists(gif_path):
        os.remove(gif_path)


def make_train(config):
    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )

    env = SMAXLogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n)),
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        network_params = network.init(_rng, init_hstate, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])
        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    avail_actions,
                )
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                info = jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    avail_actions,
                )
                runner_state = (train_state, env_state, obsv, done_batch, hstate, rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            avail_actions = jnp.ones(
                (config["NUM_ACTORS"], env.action_space(env.agents[0]).n)
            )
            ac_in = (
                last_obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
                avail_actions,
            )
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done, traj_batch.avail_actions),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean()

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                # adding an additional "fake" dimensionality to perform minibatching correctly
                init_hstate = jnp.reshape(
                    init_hstate, (1, config["NUM_ACTORS"], -1)
                )
                batch = (
                    init_hstate,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate.squeeze(),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            metric = jax.tree_util.tree_map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)
                ),
                traj_batch.info,
            )
            ratio_0 = loss_info[1][3].at[0,0].get().mean()
            loss_info = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
            metric["loss"] = {
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "ratio": loss_info[1][3],
                "ratio_0": ratio_0,
                "approx_kl": loss_info[1][4],
                "clip_frac": loss_info[1][5],
            }
            
            rng = update_state[-1]

            def callback(metric):
                # Create a completely new dictionary with only basic Python types
                safe_dict = {}
                
                try:
                    # Convert each value individually with explicit error handling
                    returns = metric["returned_episode_returns"][:, :, 0][metric["returned_episode"][:, :, 0]]
                    returns_np = np.array(returns)  # Convert to numpy
                    safe_dict["returns"] = float(returns_np.mean())
                    
                    win_rate = metric["returned_won_episode"][:, :, 0][metric["returned_episode"][:, :, 0]]
                    win_rate_np = np.array(win_rate)  # Convert to numpy
                    safe_dict["win_rate"] = float(win_rate_np.mean())
                    
                    update_steps_np = np.array(metric["update_steps"])
                    safe_dict["env_step"] = int(update_steps_np) * config["NUM_ENVS"] * config["NUM_STEPS"]
                    
                    # Process loss dictionary
                    for k, v in metric["loss"].items():
                        if v is None:
                            continue  # Skip None values
                        try:
                            v_np = np.array(v)
                            safe_dict[k] = float(v_np)
                        except Exception as e:
                            print(f"Error converting loss[{k}]: {e}")
                    
                    # Log the safe dictionary
                    wandb.log(safe_dict)
                    print("Successfully logged metrics")
                    
                except Exception as e:
                    print(f"Error in wandb logging: {e}")
                    # Try minimal logging
                    try:
                        wandb.log({"update": int(np.array(metric["update_steps"]))})
                        print("Minimal logging successful")
                    except Exception as e2:
                        print(f"Even minimal logging failed: {e2}")

            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metric)
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            init_hstate,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train


@hydra.main(version_base=None, config_path="config", config_name="ippo_rnn_smax")
def main(config):
    config = OmegaConf.to_container(config)
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "RNN"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f"{config['MAP_NAME']}_ammo_500",
    )
    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config), device=jax.devices()[0])
    out = train_jit(rng)

    train_state = out["runner_state"][0][0]  
    trained_params = train_state.params
    compute_trajectory_generalized_jsd(trained_params, config, num_steps=200)

    scenario = map_name_to_scenario("3m_vs_3m")
    env = make(
        "HeuristicEnemySMAX",
        scenario=scenario,
        use_self_play_reward=False,
        walls_cause_death=True,
        see_enemy_actions=True,
        action_type="discrete",
        observation_type="conic"
    )

    state_seq = rollout(env, trained_params, config)

    viz = SMAXVisualizer(env, state_seq)
    gif_filename = "rollout.gif"
    viz.animate(view=False, save_fname=gif_filename)
    wandb.log({"rollout_animation": wandb.Video(gif_filename, format="gif")})

    wandb.finish()

if __name__ == "__main__":
    main()
