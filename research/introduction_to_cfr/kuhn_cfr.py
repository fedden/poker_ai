from __future__ import annotations

import copy
import random

from collections import defaultdict
from functools import partial
from typing import Dict, List

import numpy as np
from tqdm import trange, tqdm

from pluribus.game.card import Card


class Player:
    def __init__(self):
        """Initialise the strategy according the amount of actions."""
        self.hand = None
        n_actions = 2
        self.strategy: Dict[str, np.ndarray] = defaultdict(
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
        return self._payout(self.player_1.hand, self.player_2.hand, self._history) * (
            1 if self.player_i == 1 else -1
        )

    @property
    def is_terminal(self) -> bool:
        """"""
        return self._check_is_terminal(self._history)

    @property
    def info_set(self) -> str:
        """"""
        return self._get_information_set(
            self.player_1.hand, self.player_2.hand, self._history
        )

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

    def _payout(
        self, player_1_hand: Card, player_2_hand: Card, history: List[str]
    ) -> int:
        """"""
        if history == ["check", "bet", "check"]:
            return -1
        elif history == ["bet", "check"]:
            return 1
        m = 1 if (player_1_hand > player_2_hand) else -1
        if history == ["check", "check"]:
            return m
        if history in [["bet", "bet"], ["check", "bet", "bet"]]:
            return m * 2
        assert False

    def _check_is_terminal(self, history: List[str]) -> bool:
        """Return true if the history means we are in a terminal state."""
        terminal_states = [
            ["check", "check"],
            ["check", "bet", "check"],
            ["check", "bet", "bet"],
            ["bet", "check"],
            ["bet", "bet"],
        ]
        return history in terminal_states

    def _get_information_set(
        self, player_1_hand: Card, player_2_hand: Card, history: List[str],
    ) -> str:
        """"""
        if self.is_terminal:
            raise ValueError(f"Shouldn't be getting terminal history info set.")
        if len(history) == 0:
            hand = player_1_hand
        elif len(history) == 1:
            hand = player_2_hand
        else:
            hand = player_1_hand
            history = ["check", "bet"]
        return f"hand: {hand}, history: {history}"


def cfr(state: KuhnState, pi1: float = 1.0, pi2: float = 1.0) -> float:
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
                state=new_state, pi1=player.strategy[info_set][action_i] * pi1, pi2=pi2,
            )
        else:
            utility[action_i] = cfr(
                state=new_state, pi1=pi1, pi2=player.strategy[info_set][action_i] * pi2,
            )
    info_set_utility = np.sum(player.strategy[info_set] * utility)
    if state.player_turn == state.player_i:
        if state.player_i == 1:
            pi = pi1
            neg_pi = pi2
        else:
            pi = pi2
            neg_pi = pi1
        player.regret[info_set] += neg_pi * (utility - info_set_utility)
        player.strategy_sum[info_set] += pi * player.strategy[info_set]
        # update the strategy_sum based on regret
        regret_sum = np.sum(np.maximum(player.regret[info_set], 0))
        if regret_sum > 0:
            player.strategy[info_set] = np.maximum(player.regret[info_set], 0) / regret_sum
        else:
            player.strategy[info_set] = np.full(KuhnState.n_actions, 0.5)
    return info_set_utility


def train(n_iterations: int = 40000, print_iterations: int = 1000):
    """"""
    # Initialise players
    player_1 = Player()
    player_2 = Player()
    # If this is uncommented the players share the same strategy_sum.
    player_2.regret = player_1.regret
    player_2.strategy_sum = player_1.strategy_sum
    player_2.strategy = player_1.strategy
    # learn strategy_sum
    for iteration_i in trange(n_iterations, desc="train iter"):
        player_i = (iteration_i % 2) + 1
        state = KuhnState(player_i=player_i, player_1=player_1, player_2=player_2)
        cfr(state)
        if iteration_i % print_iterations == 0 and iteration_i:
            tqdm.write(f"\n\nAverage strategy at iteration {iteration_i}:")
            # Print "average" strategy_sum.
            for info_set in sorted(player_1.strategy_sum.keys()):
                strategy_sum = player_1.strategy_sum[info_set]
                total = np.sum(strategy_sum)
                if total:
                    strategy_sum /= total
                    strategy_sum = np.round(strategy_sum, 2)
                else:
                    strategy_sum = [0.5, 0.5]
                tqdm.write(f"  {info_set}")
                tqdm.write(f"    check={strategy_sum[0]} bet={strategy_sum[1]}")


train()
