import random
from typing import List

import numpy as np
from tqdm import trange

from pluribus.game.card import Card


class Player:
    """Player that learns to minimise its regrets."""

    def __init__(self, n_actions: int):
        """Initialise the strategy according the amount of actions."""
        self.strategy = np.zeros(n_actions)
        self.strategy_sum = np.zeros(n_actions)
        self.regret_sum = np.zeros(n_actions)
        self.n_actions = n_actions

    @property
    def average_strategy(self) -> np.ndarray:
        """Property average_strategy returns the mean strategy."""
        return self._normalise(self.strategy_sum)

    def update_strategy(self, realisation_weight: float):
        """Inform strategy according to positive regrets."""
        # First find all positive regrets.
        self.strategy = np.maximum(self.regret_sum, 0)
        self.strategy = self._normalise(self.strategy)
        self.strategy *= realisation_weight

    def update_strategy_sum(self):
        """Accumalate the strategy which is informed by positive regrets."""
        self.strategy_sum += self.strategy

    def sample_action(self) -> int:
        """Sample according to the strategy."""
        actions = np.arange(self.n_actions)
        return np.random.choice(actions, p=self.strategy)

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

    def __init__(self, players: List[Player], active_player_i: int):
        """"""
        self._players = players
        self._active_player_i = active_player_i
        self._pass = 0
        self._bet = 1
        self._deck = [
            Card(rank="2", suit="spades"),
            Card(rank="3", suit="spades"),
            Card(rank="4", suit="spades"),
        ]
        random.shuffle(self._deck)

    @property
    def is_terminal(self):
        """"""
        return False

    @property
    def is_chance(self):
        """"""
        return False

    def sample_action(self):
        """"""
        pass

    def apply_action(self, action):
        """"""
        pass


def payoff_for_terminal_states(history):
    if len(history) <= 1:
        raise ValueError("Both players have not had atleast one action.")


def cfr(state: KuhnState):
    if state.is_terminal:
        return state.utility(player)
    elif state.is_chance:
        action = state.sample_action()
        new_state: KuhnState = state.apply_action(action)
        cfr(new_state,)


utility = 0
n_iterations = 1
players = [Player(n_actions=KuhnState.n_actions), Player(n_actions=KuhnState.n_actions)]
for iteration_i in trange(n_iterations):
    active_player_i = iteration_i % 2
    state = KuhnState(players=players, active_player_i=active_player_i)
    cfr(state)
