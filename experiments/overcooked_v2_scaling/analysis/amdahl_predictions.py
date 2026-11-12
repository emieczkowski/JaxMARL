"""Structural (a priori) Amdahl / roofline predictions for the order sweep.

Everything here comes from the layout and the task rules; no training results are used.
For each kitchen (layout and pot placement), number of pots P, order size K, recipe size
r, cook time T and collision setting (sweep_order.sh: scaling_kitchen, spread pots, P = K;
sweep_order_pots.sh: scaling_stations, balanced pots, P and K crossed):

Soups can go in any pot, and a pot can be reused once its soup is picked up, so an
order of K soups need not use all P pots, and with K > P some pots must be reused.

T1   Fastest single-agent completion time (env steps): the best of several schedules
     (pot by pot; filling the next pot while the previous one cooks; filling a batch of
     pots then serving them) rotating through m pots, for every m, over every choice and
     order of the m pots (or, when there are too many, the cheapest pots and chains of
     nearby cheap pots). Shortest paths include the turns needed to face an object.
W    Pure labor: the same schedules with cooking taking no time.
C    Critical path with unlimited agents: pots are worked in parallel; a pot's r
     onions are carried by r agents at once but enter the pot one interaction per step
     (one per 2 steps with collisions, since agents must swap places on the access
     cell); onion piles serve one pickup per step each (per 2 steps with collisions);
     a plate is waiting when the soup is ready; then the soup is delivered. A pot is
     refilled as soon as its soup is picked up, and soups are assigned to pots so that
     the last delivery is as early as possible.

Predictions for n agents, with serial fraction f = C / T1:
    Amdahl:    S(n) = 1 / (f + (1 - f) / n)        T(n) = C + (T1 - C) / n
    Roofline:  S(n) = min(n, T1 / C)               T(n) = max(T1 / n, C)
Both have the ceiling S_max = T1 / C; the roofline's knee is at n* = T1 / C.

Approximations: agents never block each other's paths (apart from the access-cell
rule above), starting positions are those of the 10-agent layout, and the soup is
pickable exactly T steps after the last onion is placed (checked in the env).

Usage: python amdahl_predictions.py  ->  results/order/predictions.csv
"""

import itertools
import math
import os
from collections import deque

import numpy as np
import pandas as pd

from jaxmarl.environments.overcooked_v2.common import StaticObject
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2

KITCHENS = (("scaling_kitchen", "spread"), ("scaling_stations", "balanced"))  # (layout, pot_placement)
POT_COUNTS = (1, 2, 5, 10)
ORDER_SIZES = (1, 2, 5, 10)
RECIPE_SIZES = (1, 3)
COOK_TIMES = (5, 20, 60)
MAX_AGENTS = 10
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "order", "predictions.csv")

MOVES = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # a move sets the facing even when blocked


class Kitchen:
    """Shortest paths over (x, y, facing) in the layout with P pots."""

    def __init__(self, num_pots, layout="scaling_kitchen", placement="spread"):
        env = OvercookedV2(layout=layout, num_pots=num_pots, num_agents=MAX_AGENTS, pot_placement=placement)
        self.static = np.asarray(env.layout.static_objects)
        self.height, self.width = self.static.shape
        self.starts = [(int(x), int(y)) for x, y in env.layout.agent_positions]
        cells = lambda pred: [(x, y) for y in range(self.height) for x in range(self.width) if pred(self.static[y, x])]
        self.pots = cells(lambda o: o == StaticObject.POT)
        self.onions = cells(StaticObject.is_ingredient_pile)
        self.plates = cells(lambda o: o == StaticObject.PLATE_PILE)
        self.goals = cells(lambda o: o == StaticObject.GOAL)
        assert len(self.pots) == num_pots
        self._dist = {}

    def free(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height and self.static[y, x] == StaticObject.EMPTY

    def dist_from(self, state):
        """Steps from state (x, y, facing) to every reachable state."""
        if state not in self._dist:
            dist = {state: 0}
            queue = deque([state])
            while queue:
                x, y, _ = s = queue.popleft()
                for d, (dx, dy) in enumerate(MOVES):
                    nxt = (x + dx, y + dy, d) if self.free(x + dx, y + dy) else (x, y, d)
                    if nxt not in dist:
                        dist[nxt] = dist[s] + 1
                        queue.append(nxt)
            self._dist[state] = dist
        return self._dist[state]

    def reach(self, state, targets):
        """(steps, end state) to stand facing the nearest of targets (no interaction)."""
        best = None
        for (tx, ty) in targets:
            for d, (dx, dy) in enumerate(MOVES):
                s = (tx - dx, ty - dy, d)
                if self.free(s[0], s[1]) and s in self.dist_from(state):
                    cand = (self.dist_from(state)[s], s)
                    best = cand if best is None or cand[0] < best[0] else best
        return best

    def leg(self, state, targets):
        """(steps, end state) to reach the nearest target and interact with it."""
        steps, end = self.reach(state, targets)
        return steps + 1, end


def solo_time(kitchen, r, T, plan):
    """Completion time of one agent executing plan: a list of ("fill" | "serve", pot)."""
    state = (*kitchen.starts[0], 0)
    t = 0
    ready = {}

    def do(targets):
        nonlocal t, state
        steps, state = kitchen.leg(state, targets)
        t += steps

    for action, pot in plan:
        if action == "fill":
            assert pot not in ready, "pot refilled before its soup was served"
            for _ in range(r):
                do(kitchen.onions)
                do([pot])
            ready[pot] = t + T  # interacting at step >= ready picks up the soup
        else:
            do(kitchen.plates)
            steps, state = kitchen.reach(state, [pot])
            t = max(t + steps + 1, ready.pop(pot))
            do(kitchen.goals)
    return t


def solo_plans(soups):
    """Candidate plans for serving the soups (a list of pots, one per soup) in order."""
    yield [step for pot in soups for step in (("fill", pot), ("serve", pot))]
    if all(a != b for a, b in zip(soups, soups[1:])):  # fill the next pot while one cooks
        plan = [("fill", soups[0])]
        for prev, pot in zip(soups, soups[1:]):
            plan += [("fill", pot), ("serve", prev)]
        yield plan + [("serve", soups[-1])]
    m = len(set(soups))
    for batch in range(2, m + 1):  # fill a batch of distinct pots, then serve them
        plan = []
        for i in range(0, len(soups), batch):
            chunk = soups[i : i + batch]
            plan += [("fill", p) for p in chunk] + [("serve", p) for p in chunk]
        yield plan


def best_solo(kitchen, r, T, K):
    # rank pots by the labor of one soup there, from standing at a goal (steady state)
    goal_state = kitchen.reach((*kitchen.starts[0], 0), kitchen.goals)[1]
    cost = {}
    for pot in kitchen.pots:
        state, t = goal_state, 0
        for targets in [kitchen.onions, [pot]] * r + [kitchen.plates, [pot], kitchen.goals]:
            steps, state = kitchen.leg(state, targets)
            t += steps
        cost[pot] = t
    ranked = sorted(kitchen.pots, key=cost.get)
    best = None
    for m in range(1, min(len(ranked), K) + 1):
        for pots in pot_sequences(ranked, cost, m):
            soups = [pots[i % m] for i in range(K)]
            for plan in solo_plans(soups):
                t = solo_time(kitchen, r, T, plan)
                best = t if best is None else min(best, t)
    return best


def pot_sequences(ranked, cost, m, max_orders=1000):
    """Candidate sequences of m distinct pots to rotate through: every ordering when there
    are at most max_orders, else the m cheapest pots and, from each pot, a chain that keeps
    adding the cheapest pot nearest (Manhattan) to the last one. Pots that cost the same
    (e.g. scaling_stations) differ in how far apart they are, so their choice matters."""
    if math.perm(len(ranked), m) <= max_orders:
        yield from itertools.permutations(ranked, m)
        return
    yield tuple(ranked[:m])
    for first in ranked:
        chain = [first]
        while len(chain) < m:
            last = chain[-1]
            chain.append(min((p for p in ranked if p not in chain),
                             key=lambda p: (cost[p], abs(p[0] - last[0]) + abs(p[1] - last[1]))))
        yield tuple(chain)


def critical_path(kitchen, r, T, K, collisions):
    """Completion time with unlimited agents (see module docstring)."""
    gap = 2 if collisions else 1  # steps between interactions at one pot / pile cell
    start_states = [(x, y, d) for x, y in kitchen.starts for d in range(4)]
    # onion pickups: each pile serves one agent per `gap` steps; the earliest pickup is
    # the nearest agent's walk to a pile
    first_pickup = min(kitchen.leg(s, kitchen.onions)[0] for s in start_states)
    last_pickup = first_pickup + gap * (int(np.ceil(K * r / len(kitchen.onions))) - 1)

    first_finish, cycle, tails = {}, {}, []
    for pot in kitchen.pots:
        # onion trip: from standing at a pile (just picked up) to placing in this pot
        trip = min(kitchen.leg(kitchen.reach(s, kitchen.onions)[1], [pot])[0] for s in start_states)
        fill = first_pickup + trip + gap * (r - 1)
        # plate carrier: from the best start, fetch a plate and stand facing the pot
        plate_arrival = min(
            kitchen.leg(s, kitchen.plates)[0] + kitchen.reach(kitchen.leg(s, kitchen.plates)[1], [pot])[0]
            for s in start_states
        )
        # pick up the soup when it is ready, the plate is there, and (with collisions)
        # the last onion carrier has made room on the access cell
        pick = max(fill + T, plate_arrival + 1, fill + gap)
        deliver = kitchen.leg(kitchen.reach(start_states[0], [pot])[1], kitchen.goals)[0]
        first_finish[pot] = (pick, deliver)
        # refill right after the pickup: r placements `gap` apart, cook, pick up again
        cycle[pot] = gap * r + T
        tails.append(trip + T + deliver)

    # assign soups one at a time to the pot where that soup would be delivered earliest
    next_pick = {pot: pick for pot, (pick, _) in first_finish.items()}
    finish = 0
    for _ in range(K):
        pot = min(kitchen.pots, key=lambda p: next_pick[p] + first_finish[p][1])
        finish = max(finish, next_pick[pot] + first_finish[pot][1])
        next_pick[pot] += cycle[pot]

    # shared piles: the last onion is picked up no earlier than last_pickup, and must
    # still be carried to (at best the nearest) pot, cooked and delivered
    pile_bound = last_pickup + min(tails)
    return max(finish, pile_bound)


def main():
    rows = []
    for (layout, placement), P in itertools.product(KITCHENS, POT_COUNTS):
        kitchen = Kitchen(P, layout, placement)
        for K, r, T in itertools.product(ORDER_SIZES, RECIPE_SIZES, COOK_TIMES):
            T1 = best_solo(kitchen, r, T, K)
            W = best_solo(kitchen, r, 0, K)
            for no_collision in (True, False):
                C = critical_path(kitchen, r, T, K, collisions=not no_collision)
                rows.append(dict(layout=layout, pot_placement=placement, num_pots=P, order_size=K, recipe_size=r, pot_cook_time=T, no_collision=no_collision,
                                 T1=T1, W=W, C=C, f=C / T1, S_max=T1 / C))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
