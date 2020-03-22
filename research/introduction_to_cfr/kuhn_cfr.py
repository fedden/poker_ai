from __future__ import annotations

import copy
import random

from collections import defaultdict
from functools import partial
from typing import Dict, List

import numpy as np
from tqdm import trange, tqdm

from pluribus.poker.card import Card


class Player:
    def __init__(self):
        """Initialise the strategy according the amount of actions."""
        self.hand = None
        n_actions = 2
        self.strategy = {}
        self.strategy[1]: Dict[str, np.ndarray] = defaultdict(
            partial(np.full, n_actions, 1 / n_actions)
        )
        self.strategy_sum: Dict[str, np.ndarray] = defaultdict(
            partial(np.zeros, n_actions)
        )
        self.regret: Dict[str, np.ndarray] = defaultdict(partial(np.zeros, n_actions))


class KuhnState:
    n_actions = 2
    actions = ["check", "bet"]

    def __init__(self, player_i, player_1, player_2):
        """"""
        self.player_i = player_i
        self.player_1 = player_1
        self.player_2 = player_2
        # Create the deck.
        self._deck = [
            Card(rank="2", suit="spades"),
            Card(rank="3", suit="spades"),
            Card(rank="4", suit="spades"),
        ]
        random.shuffle(self._deck)
        self.player_1.hand = self._deck[0]
        self.player_2.hand = self._deck[1]
        self._history: List[str] = []

    @property
    def payout(self) -> int:
        """"""
        if self._history == ["check", "bet", "check"]:
            payout = -1
        elif self._history == ["bet", "check"]:
            payout = 1
        m = 1 if (self.player_1.hand.rank_int > self.player_2.hand.rank_int) else -1
        if self._history == ["check", "check"]:
            payout = m
        if self._history in [["bet", "bet"], ["check", "bet", "bet"]]:
            payout = m * 2
        return payout * (1 if self.player_i == 1 else -1)

    @property
    def is_terminal(self) -> bool:
        """"""
        """Return true if the history means we are in a terminal state."""
        terminal_states = [
            ["check", "check"],
            ["check", "bet", "check"],
            ["check", "bet", "bet"],
            ["bet", "check"],
            ["bet", "bet"],
        ]
        return self._history in terminal_states

    @property
    def info_set(self) -> str:
        """"""
        if self.is_terminal:
            raise ValueError(f"Shouldn't be getting terminal history info set.")
        if len(self._history) == 0:
            hand = self.player_1.hand
        elif len(self._history) == 1:
            hand = self.player_2.hand
        else:
            hand = self.player_1.hand
            assert self._history == ["check", "bet"]
        return f"hand: {hand}, history: {self._history}"

    @property
    def player_turn(self) -> int:
        """"""
        return 2 if len(self._history) == 1 else 1

    def apply_action(self, action: str) -> KuhnState:
        """Apply an action to the game and make a new game state."""
        # Deep copy history and other vars to prevent unwanted mutations to
        # this copy of the state.
        new_state = copy.deepcopy(self)
        # Apply the action to the "future" state.
        new_state._history.append(action)
        # Ensure the players are references so we can mutate their state.
        new_state.player_1 = self.player_1
        new_state.player_2 = self.player_2
        return new_state


def cfr(iteration_i, state: KuhnState, pi1: float = 1.0, pi2: float = 1.0) -> float:
    """"""
    if state.is_terminal:
        return state.payout
    info_set = state.info_set
    player = state.player_1 if state.player_turn == 1 else state.player_2
    # if we are here, we have both actions available
    utility = np.zeros(KuhnState.n_actions)
    for action_i, action in enumerate(KuhnState.actions):
        new_state = state.apply_action(action)
        if state.player_turn == 1:
            utility[action_i] = cfr(
                iteration_i, state=new_state, pi1=player.strategy[iteration_i][info_set][action_i] * pi1, pi2=pi2,
            )
        else:
            utility[action_i] = cfr(
                iteration_i, state=new_state, pi1=pi1, pi2=player.strategy[iteration_i][info_set][action_i] * pi2,
            )
    info_set_utility = np.sum(player.strategy[iteration_i][info_set] * utility)
    if state.player_turn == state.player_i:
        if state.player_i == 1:
            pi = pi1
            neg_pi = pi2
        else:
            pi = pi2
            neg_pi = pi1
        player.regret[info_set] += neg_pi * (utility - info_set_utility)
        player.strategy_sum[info_set] += pi * player.strategy[iteration_i][info_set]
        # update the strategy_sum based on regret
        regret_sum = np.sum(np.maximum(player.regret[info_set], 0))
        if regret_sum > 0:
            player.strategy[iteration_i + 1][info_set] = (
                np.maximum(player.regret[info_set], 0) / regret_sum
            )
        else:
            player.strategy[iteration_i + 1][info_set] = np.full(KuhnState.n_actions, 0.5)
    return info_set_utility


def print_strategy(player: Player, iteration_i: int):
    """Print the strategy learned."""
    tqdm.write(f"\n\nAverage strategy at iteration {iteration_i}:")
    # Print "average" strategy_sum.
    for info_set in sorted(player.strategy_sum.keys()):
        strategy_sum = player.strategy_sum[info_set]
        total = np.sum(strategy_sum)
        if total:
            strategy_sum /= total
            strategy_sum = np.round(strategy_sum, 4)
        else:
            strategy_sum = [0.5, 0.5]
        tqdm.write(f"  {info_set}")
        tqdm.write(f"    check={strategy_sum[0]} bet={strategy_sum[1]}")


def train(n_iterations: int = 1000, print_iterations: int = 1000):
    """"""
    # Initialise players
    player_1 = Player()
    player_2 = Player()
    # If this is uncommented the players share the same strategy_sum.
    player_2.regret = player_1.regret
    player_2.strategy_sum = player_1.strategy_sum
    player_2.strategy = player_1.strategy
    # learn strategy_sum
    for iteration_i in trange(1, n_iterations, desc="train iter"):
        player_1.strategy[iteration_i + 1] = copy.deepcopy(player_1.strategy[iteration_i])
        player_2.strategy = player_1.strategy
        for player_i in [1, 2]:
            state = KuhnState(player_i=player_i, player_1=player_1, player_2=player_2)
        del player_1.strategy[iteration_i]
        if iteration_i % print_iterations == 0 and iteration_i:
            print_strategy(player=player_1, iteration_i=iteration_i)
    print_strategy(player=player_1, iteration_i=iteration_i)


if __name__ == "__main__":
    train()
