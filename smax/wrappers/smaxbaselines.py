""" Wrappers for use with SMAX baselines. """
import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial

# from gymnax.environments import environment, spaces
from typing import Optional, Tuple, Union
from smax.environments.multi_agent_env import MultiAgentEnv, State


class SMAXWrapper(object):
    """Base class for all SMAX wrappers."""

    def __init__(self, env: MultiAgentEnv):
        self._env = env

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    # def _batchify(self, x: dict):
    #     x = jnp.stack([x[a] for a in self._env.agents])
    #     return x.reshape((self._env.num_agents, -1))

    def _batchify_floats(self, x: dict):
        return jnp.stack([x[a] for a in self._env.agents])


@struct.dataclass
class LogEnvState:
    env_state: State
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int


class LogWrapper(SMAXWrapper):
    """Log the episode returns and lengths.
    NOTE for now for envs where agents terminate at the same time.
    """

    def __init__(self, env: MultiAgentEnv, replace_info: bool = False):
        super().__init__(env)
        self.replace_info = replace_info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        obs, env_state = self._env.reset(key)
        state = LogEnvState(
            env_state,
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        ep_done = done["__all__"]
        new_episode_return = state.episode_returns + self._batchify_floats(reward)
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
            + new_episode_length * ep_done,
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = jnp.full((self._env.num_agents,), ep_done)
        return obs, state, reward, done, info


@struct.dataclass
class SMAXLogEnvState:
    env_state: State
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    won_episodes: int
    total_episodes: int


class SMAXLogWrapper(SMAXWrapper):
    def __init__(self, env: MultiAgentEnv, replace_info: bool = False):
        super().__init__(env)
        self.replace_info = replace_info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        obs, env_state = self._env.reset(key)
        state = SMAXLogEnvState(
            env_state,
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            won_episodes=0,
            total_episodes=0,
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        ep_done = done["__all__"]
        batch_reward = self._batchify_floats(reward)
        new_episode_return = state.episode_returns + self._batchify_floats(reward)
        new_episode_length = state.episode_lengths + 1
        state = SMAXLogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
            + new_episode_length * ep_done,
            won_episodes=state.won_episodes + (batch_reward[0] >= 1.0).astype(jnp.int32),
            total_episodes=state.total_episodes + (ep_done).astype(jnp.int32),
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = jnp.full((self._env.num_agents,), ep_done)
        info["win_rate"] = jnp.where(
            state.total_episodes == 0, 0, state.won_episodes / state.total_episodes
        )
        return obs, state, reward, done, info
