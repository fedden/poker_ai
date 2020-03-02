from __future__ import annotations

import copy
import random
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm, trange

from pluribus.game.card import Card


class Player:
    """Player that learns to minimise its regrets."""

    def __init__(self, n_actions: int):
        """Initialise the strategy according the amount of actions."""
        self.strategy: Dict[str, np.ndarray] = defaultdict(
            partial(np.full, n_actions, 1 / n_actions)
        )
        self.strategy_sum: Dict[str, np.ndarray] = defaultdict(
            partial(np.zeros, n_actions)
        )
        self.regret_sum: Dict[str, np.ndarray] = defaultdict(
            partial(np.zeros, n_actions)
        )
        self.n_actions = n_actions

    @property
    def info_sets(self) -> List[str]:
        """Return the info sets that we have strategies for."""
        return sorted(list(self.strategy.keys()))

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

    def sample_action(self, actions: List[str], info_set: str) -> Tuple[str, float]:
        """Sample according to the strategy.

        Returns action name and probability of taking such an action.
        """
        actions_ints = np.arange(self.n_actions)
        try:
            action_i = np.random.choice(actions_ints, p=self.strategy[info_set])
        except ValueError as e:
            print(f"strategy: {self.strategy[info_set]}")
            raise e
        prob_i = self.strategy[info_set][action_i]
        action_str = actions[action_i]
        return action_str, prob_i

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
    actions = ["check", "bet"]

    def __init__(self, players: List[Player], active_player_i: int):
        """Initialise the deck and hands, history and players.

        Only one player can be active in a game, and the other is the opponent.
        """
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
            active=players[active_player_i],
            opponent=players[(active_player_i + 1) % 2],
        )

    @property
    def is_terminal(self) -> bool:
        """Should the game finish?"""
        return self._check_is_terminal(self._history)

    @property
    def is_chance(self) -> bool:
        """Is it the opponents turn?"""
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
    def active_player_hand(self) -> Card:
        """"""
        return self._hand["active"]

    @property
    def opponent_player_hand(self) -> Card:
        """"""
        return self._hand["opponent"]

    @property
    def active_player_info_set(self) -> str:
        """Return the active players info set."""
        return self._get_info_set(self.active_player_hand)

    @property
    def opponent_player_info_set(self) -> str:
        """Return the active players info set."""
        return self._get_info_set(self.active_player_hand)

    @property
    def payoff(self) -> int:
        """Get the utility/reward for the active agent playing."""
        if len(self._history) < 2:
            raise ValueError(f"History not long enough yet {self._history}")
        if not self._check_is_terminal(self._history):
            raise ValueError(f"Unexpected history {self._history}")
        terminal_check = self._history[-1] == "check"
        double_bet = self._history[-2:] == ["bet", "bet"]
        double_check = self._history == ["check", "check"]
        active_player_wins = self._hand["active"] > self._hand["opponent"]
        if terminal_check and double_check:
            return 1 if active_player_wins else -1
        elif terminal_check:
            return 1
        elif double_bet:
            return 2 if active_player_wins else -2
        else:
            raise ValueError(f"Unexpected payoff state.")

    def apply_action(self, action: str) -> KuhnState:
        """Apply an action to the game and make a new game state."""
        # Deep copy history and other vars to prevent unwanted mutations to
        # this copy of the state.
        new_state = copy.deepcopy(self)
        # Apply the action to the "future" state.
        new_state._history.append(action)
        # Ensure the players are references so we can mutate their state.
        new_state._players = self._players
        return new_state

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

    def _get_info_set(self, hand: Card) -> str:
        """Gets string infomation set for a given hand (of one card)."""
        if self._check_is_terminal(self._history):
            raise ValueError(f"Shouldn't be getting terminal history info set.")
        hand_str: str = self.active_player_hand.rank
        history_str: str = ", ".join(self._history)
        return f"hand=[{hand_str}], actions=[{history_str}]"


def cfr(
    state: KuhnState, active_player_pi: float = 1.0, opponent_player_pi: float = 1.0
) -> Optional[int]:
    """Depth-wise recursive CFR."""
    if state.is_terminal:
        return state.payoff
    elif state.is_chance:
        # Sample the opponent's strategy.
        info_set: str = state.opponent_player_info_set
        action, probability = state.opponent_player.sample_action(
            KuhnState.actions, info_set
        )
        new_state: KuhnState = state.apply_action(action)
        return cfr(new_state, active_player_pi, opponent_player_pi * probability)
    else:
        # Otherwise execution continues, computing the active player
        # information set representation by concatenating the active players
        # card with the history of all player actions.
        info_set: str = state.active_player_info_set
        utility: np.ndarray = np.zeros(KuhnState.n_actions)
        for action_i, action in enumerate(KuhnState.actions):
            new_state: KuhnState = state.apply_action(action)
            probability: float = state.active_player.strategy[info_set][action_i]
            utility[action_i] = cfr(
                new_state, active_player_pi * probability, opponent_player_pi
            )
        # Each action probability multiplied by the corresponding returned
        # action utility is accumulated to the utility for playing to this node
        # for the current player.
        info_set_utility: float = np.sum(
            state.active_player.strategy[info_set] * utility
        )
        regret: np.ndarray = utility - info_set_utility
        state.active_player.regret_sum[info_set] += opponent_player_pi * regret
        state.active_player.update_strategy(info_set, active_player_pi)
        state.active_player.update_strategy_sum(info_set)


def train(n_iterations: int) -> List[Player]:
    """Train two agents with self-play."""
    players: List[Player] = [
        Player(n_actions=KuhnState.n_actions),
        Player(n_actions=KuhnState.n_actions),
    ]
    for iteration_i in trange(n_iterations):
        active_player_i: int = iteration_i % 2
        state: KuhnState = KuhnState(players=players, active_player_i=active_player_i)
        cfr(state)
    return players


def print_players_strategy(players: List[Player]):
    """Print the players learned strategy."""
    for player_i, player in enumerate(players):
        tqdm.write(f"player {player_i} strategy:")
        for info_set in player.info_sets:
            average_strategy = np.round(player.average_strategy(info_set), 2)
            tqdm.write(f" * info set <{info_set}> strategy: {average_strategy}")
        tqdm.write("")


if __name__ == "__main__":
    players: List[Player] = train(n_iterations=10000)
    print_players_strategy(players)
    print("Finished!")
