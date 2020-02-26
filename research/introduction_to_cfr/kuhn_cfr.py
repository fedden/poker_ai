from __future__ import annotations

import copy
import random
from collections import defaultdict
from functools import partial
from typing import List

import numpy as np
from tqdm import trange

from pluribus.game.card import Card


class Player:
    """Player that learns to minimise its regrets."""

    def __init__(self, n_actions: int):
        """Initialise the strategy according the amount of actions."""
        self.strategy = defaultdict(partial(np.zeros, n_actions))
        self.strategy_sum = defaultdict(partial(np.zeros, n_actions))
        self.regret_sum = defaultdict(partial(np.zeros, n_actions))
        self.n_actions = n_actions

    def average_strategy(self, info_set: str) -> np.ndarray:
        """Property average_strategy returns the mean strategy."""
        return self._normalise(self.strategy_sum[info_set])

    def update_strategy(self, info_set: str, realisation_weight: float):
        """Inform strategy according to positive regrets."""
        # First find all positive regrets.
        self.strategy[info_set] = np.maximum(self.regret_sum[info_set], 0)
        self.strategy[info_set] = self._normalise(self.strategy[info_set])
        self.strategy[info_set] *= realisation_weight

    def update_strategy_sum(self, info_set: str):
        """Accumalate the strategy which is informed by positive regrets."""
        self.strategy_sum[info_set] += self.strategy[info_set]

    def sample_action(self, actions: List[str], info_set: str) -> int:
        """Sample according to the strategy."""
        return np.random.choice(actions, p=self.strategy[info_set])

    def _normalise(self, x: np.ndarray) -> np.ndarray:
        """Return `x` as a valid probability distribution."""
        normalising_sum = np.sum(x)
        if normalising_sum > 0:
            x /= normalising_sum
        else:
            x = np.ones_like(x) / len(x)
        return x


class KuhnState:
    n_actions = 2
    actions = ["pass", "bet"]

    def __init__(self, players: List[Player], active_player_i: int):
        """"""
        if len(players) != 2:
            raise ValueError(f"Expected 2 players but got {len(players)}.")
        self._deck = [
            Card(rank="2", suit="spades"),
            Card(rank="3", suit="spades"),
            Card(rank="4", suit="spades"),
        ]
        random.shuffle(self._deck)
        self._hand = dict(active=self._deck[0], opponent=self._deck[1])
        self._history: List[str] = []
        self._players = dict(
            active=players[active_player_i], opponent=players[active_player_i + 1 % 2]
        )

    @property
    def is_terminal(self) -> bool:
        """"""
        return False

    @property
    def is_chance(self) -> bool:
        """"""
        return len(self._history) % 2 == 1

    @property
    def opponent_player(self) -> Player:
        """"""
        return self._players["opponent"]

    @property
    def active_player(self) -> Player:
        """"""
        return self._players["active"]

    @property
    def active_player_hand(self) -> List[Card]:
        """"""
        return self._hand["active"]

    @property
    def active_player_info_set(self) -> str:
        hand_str: str = self.active_player_hand.rank
        history_str: str = ", ".join(self._history)
        return f"hand=[{hand_str}], actions=[{history_str}]"

    @property
    def payoff(self) -> int:
        """"""
        if not self.is_terminal:
            raise ValueError("Both players have not had atleast one action.")
        terminal_pass = self._history[-1] == "pass"
        double_bet = self._history[-2:] == ["bet", "bet"]
        double_pass = self._history == ["pass", "pass"]
        active_player_wins = self._hand["active"] > self._hand["opponent"]
        if terminal_pass and double_pass:
            return 1 if active_player_wins else -1
        elif terminal_pass:
            return 1
        elif double_bet:
            return 2 if active_player_wins else -2

    def apply_action(self, action: str) -> KuhnState:
        """"""
        new_state = self
        # Deep copy history to prevent unwanted mutations.
        new_state._history = copy.deepcopy(self._history)
        new_state._history.append(action)
        return new_state


def cfr(state: KuhnState, active_player_pi: float):
    if state.is_terminal:
        return state.payoff
    elif state.is_chance:
        # Sample the opponent's strategy.
        info_set = state.opponent_player_info_set
        action = state.opponent_player.sample_action(KuhnState.actions, info_set)
        new_state = state.apply_action(action)
        cfr(new_state, active_player_pi)
    # Otherwise execution continues, computing the active player information
    # set representation by concatenating the active players card with the
    # history of all player actions.
    info_set = state.active_player_info_set
    utility = np.zeros(KuhnState.n_actions)
    active_strategy = state.active_player.strategy[info_set]
    for action_i, action in enumerate(KuhnState.actions):
        new_state = state.apply_action(action)
        new_active_player_pi = active_strategy[action_i]
        utility[action_i] += cfr(new_state, new_active_player_pi)
    # Each action probability multiplied by the corresponding returned action
    # utility is accumulated to the utility for playing to this node for the
    # current player.
    info_set_utility = np.sum(state.active_player.strategy[info_set] * utility)
    regret = utility - info_set_utility
    state.active_player.regret_sum[info_set] += active_player_pi * regret
    # TODO(fedden): Realisation weight shouldn't be 1.0 here.
    state.update_strategy(info_set, 1.0)


utility = 0
n_iterations = 1
players = [Player(n_actions=KuhnState.n_actions), Player(n_actions=KuhnState.n_actions)]
for iteration_i in trange(n_iterations):
    active_player_i = iteration_i % 2
    state = KuhnState(players=players, active_player_i=active_player_i)
    cfr(state, 1.0)
