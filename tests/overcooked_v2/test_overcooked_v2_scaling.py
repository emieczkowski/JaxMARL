import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxmarl import make
from jaxmarl.environments.overcooked_v2.common import (
    Direction,
    DynamicObject,
    OvercookedActionsEnum,
    Position,
    StaticObject,
)
from jaxmarl.environments.overcooked_v2.layouts import Layout, overcooked_v2_layouts
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2

A = OvercookedActionsEnum
ONION = DynamicObject.ingredient(0)


def _num_pots(static_objects):
    return int(np.sum(np.asarray(static_objects) == StaticObject.POT))


def _rollout(env, steps=10):
    rng = jax.random.PRNGKey(0)
    _, state = env.reset(rng)
    for _ in range(steps):
        rng, rng_act = jax.random.split(rng)
        rng_act = jax.random.split(rng_act, env.num_agents)
        actions = {
            a: env.action_space(a).sample(rng_act[i]) for i, a in enumerate(env.agents)
        }
        _, state, _, _, _ = env.step(rng, state, actions)
    return state


@pytest.mark.parametrize("pot_cook_time", [1, 5, 50])
def test_pot_cook_time(pot_cook_time):
    layout = Layout.from_string(
        "\nWWWWW\nW A W\nW0PWW\n", possible_recipes=[[0, 0, 0]]
    )
    env = OvercookedV2(layout=layout, pot_cook_time=pot_cook_time)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        agents=state.agents.replace(
            pos=Position(x=jnp.array([2]), y=jnp.array([1])),
            dir=jnp.array([Direction.DOWN]),
            inventory=jnp.array([ONION]),
        ),
        grid=state.grid.at[2, 2, 1].set(2 * ONION),
    )
    step = jax.jit(env.step_env)

    def _pot_after(state, action):
        _, state, *_ = step(jax.random.PRNGKey(1), state, {"agent_0": jnp.array(action)})
        return state, state.grid[2, 2]

    state, pot = _pot_after(state, A.interact)
    for _ in range(pot_cook_time - 1):
        assert pot[1] & DynamicObject.COOKED == 0
        state, pot = _pot_after(state, A.stay)
    assert pot[1] & DynamicObject.COOKED != 0
    assert pot[2] == 0


@pytest.mark.parametrize("num_agents", [1, 2, 3, 6])
def test_num_agents(num_agents):
    env = OvercookedV2(layout="cramped_room", num_agents=num_agents)
    assert env.num_agents == num_agents
    assert env.agents == [f"agent_{i}" for i in range(num_agents)]

    base = overcooked_v2_layouts["cramped_room"].agent_positions
    positions = env.layout.agent_positions
    assert positions[: min(2, num_agents)] == base[:num_agents]
    assert len(set(positions)) == num_agents
    for x, y in positions:
        assert env.layout.static_objects[y, x] == StaticObject.EMPTY

    state = _rollout(env)
    assert state.agents.pos.x.shape == (num_agents,)


def test_extra_agents_keep_room_split():
    # forced_coord has two rooms with one agent each: extra agents alternate.
    layout = overcooked_v2_layouts["forced_coord"]
    rooms = layout._rooms()
    new = layout.with_num_agents(4)
    room_of = [rooms[y, x] for x, y in new.agent_positions]
    assert room_of[2] == room_of[0] and room_of[3] == room_of[1]
    assert room_of[0] != room_of[1]


@pytest.mark.parametrize("num_pots", [0, 1, 2, 4])
def test_num_pots(num_pots):
    env = OvercookedV2(layout="cramped_room", num_pots=num_pots)
    _, state = env.reset(jax.random.PRNGKey(0))
    assert _num_pots(state.grid[:, :, 0]) == num_pots
    _rollout(env)


def test_added_pots_stay_in_pot_rooms():
    # In forced_coord only the right room can reach a pot; new pots must not
    # be placed in the left room or on the shared counters between rooms.
    layout = overcooked_v2_layouts["forced_coord"]
    rooms = layout._rooms()
    pot_rooms = {
        r
        for y, x in zip(*np.nonzero(layout.static_objects == StaticObject.POT))
        for r in layout._adjacent_rooms(rooms, y, x)
    }
    new = layout.with_num_pots(4)
    assert _num_pots(new.static_objects) == 4
    for y, x in zip(*np.nonzero(new.static_objects == StaticObject.POT)):
        assert layout._adjacent_rooms(rooms, y, x) <= pot_rooms


def test_too_many_raises():
    with pytest.raises(ValueError):
        OvercookedV2(layout="cramped_room", num_agents=7)
    with pytest.raises(ValueError):
        OvercookedV2(layout="cramped_room", num_pots=20)


def test_scaling_kwargs_via_make():
    env = make(
        "overcooked_v2",
        layout="grounded_coord_simple",
        random_agent_positions=True,
        no_collision=True,
        pot_cook_time=10,
        num_agents=4,
        num_pots=3,
    )
    assert env.num_agents == 4
    assert env.pot_cook_time == 10
    assert _num_pots(env.layout.static_objects) == 3
    _rollout(env)
