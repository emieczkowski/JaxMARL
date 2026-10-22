import jax
import jax.numpy as jnp
import pytest

from jaxmarl import make
from jaxmarl.environments.overcooked_v2.common import (
    Direction,
    DynamicObject,
    OvercookedActionsEnum,
    Position,
)
from jaxmarl.environments.overcooked_v2.layouts import Layout
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2
from jaxmarl.environments.overcooked_v2.settings import POT_COOK_TIME

A = OvercookedActionsEnum
ONION = DynamicObject.ingredient(0)


def _env(layout_string, no_collision):
    layout = Layout.from_string(layout_string, possible_recipes=[[0, 0, 0]])
    return OvercookedV2(layout=layout, no_collision=no_collision)


def _step(env, state, actions):
    actions = {f"agent_{i}": jnp.array(a) for i, a in enumerate(actions)}
    _, state, *_ = jax.jit(env.step_env)(jax.random.PRNGKey(1), state, actions)
    return state


def _positions(state):
    return [tuple(p) for p in state.agents.pos.to_array().tolist()]


@pytest.mark.parametrize(
    "no_collision, expected",
    [(False, [(1, 1), (3, 1)]), (True, [(2, 1), (2, 1)])],
)
def test_move_into_same_cell(no_collision, expected):
    env = _env("\nWWWWW\nWA AW\nW   W\nWW0WW\n", no_collision)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = _step(env, state, [A.right, A.left])
    assert _positions(state) == expected


@pytest.mark.parametrize(
    "no_collision, expected",
    [(False, [(1, 1), (2, 1)]), (True, [(2, 1), (1, 1)])],
)
def test_swap_positions(no_collision, expected):
    env = _env("\nWWWWWW\nWAA  W\nWWWW0W\n", no_collision)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = _step(env, state, [A.right, A.left])
    assert _positions(state) == expected


def test_walls_still_block_with_no_collision():
    env = _env("\nWWWWW\nWA AW\nW   W\nWW0WW\n", True)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = _step(env, state, [A.up, A.right])
    assert _positions(state) == [(1, 1), (3, 1)]


def test_stacked_agents_respect_pot_capacity():
    # Two agents on the same cell, both facing a pot that already holds two
    # onions. Only one onion fits; the pot then cooks a single soup.
    env = _env("\nWWWWW\nWA AW\nW   W\nW0PWW\n", True)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        agents=state.agents.replace(
            pos=Position(x=jnp.array([2, 2]), y=jnp.array([2, 2])),
            dir=jnp.array([Direction.DOWN, Direction.DOWN]),
            inventory=jnp.array([ONION, ONION]),
        ),
        grid=state.grid.at[3, 2, 1].set(2 * ONION),
    )

    state = _step(env, state, [A.interact, A.interact])

    pot = state.grid[3, 2]
    assert DynamicObject.ingredient_count(pot[1]) == 3
    assert pot[2] == POT_COOK_TIME - 1
    assert sorted(state.agents.inventory.tolist()) == [0, ONION]

    # A cooking pot rejects further ingredients from either agent.
    state = _step(env, state, [A.interact, A.interact])
    assert DynamicObject.ingredient_count(state.grid[3, 2, 1]) == 3
    assert sorted(state.agents.inventory.tolist()) == [0, ONION]


def test_no_collision_via_make():
    env = make("overcooked_v2", layout="cramped_room", no_collision=True)
    assert env.no_collision
    rng = jax.random.PRNGKey(0)
    _, state = env.reset(rng)
    for _ in range(20):
        rng, rng_act = jax.random.split(rng)
        rng_act = jax.random.split(rng_act, env.num_agents)
        actions = {
            a: env.action_space(a).sample(rng_act[i]) for i, a in enumerate(env.agents)
        }
        _, state, _, _, _ = env.step(rng, state, actions)
