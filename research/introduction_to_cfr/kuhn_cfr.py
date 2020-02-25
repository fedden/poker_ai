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
        if len(players) != 2:
            raise ValueError(f"Expected 2 players but got {len(players)}.")
        self._actions = ["pass", "bet"]
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
        return False

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

    def sample_action(self) -> str:
        """"""
        return random.choice(self._actions)

    def apply_action(self, action: str):
        """"""
        self._history.append(action)
        return self


def cfr(state: KuhnState):
    if state.is_terminal:
        return state.payoff
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
