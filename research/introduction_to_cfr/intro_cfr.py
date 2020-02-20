from typing import List, Tuple

import numpy as np


class Agent:
    """Agent that learns to minimise it's regrets."""

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

    def update_strategy(self):
        """Inform strategy according to positive regrets."""
        # First find all positive regrets.
        self.strategy = np.maximum(self.regret_sum, 0)
        self.strategy = self._normalise(self.strategy)

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


def compute_action_utility(opponent_action: int) -> np.ndarray:
    """Computes action utility relative to opponent's move.

    Action utility is draw (0), win (1), lose (-1)
    Actions are rock (0), paper (1), scissors (2)
    We roll the utility by the opponent action, so if opponent plays scissors (2),
    we would roll the `action_utility` array from [0, 1, -1] by (2) to [1, -1, 0].
    So if now the player (not the oppoenent) plays paper (1), then the score is -1,
    otherwise if the player plays rock (0) they get a score of 1, and 0 if they play
    scissors as they have drawn.
    """
    action_utility = np.roll([0, 1, -1], opponent_action)
    return action_utility


def train(n_iterations: int) -> Tuple[Agent, Agent]:
    """Get two learned optimal strategies for two-player rock-paper-scissors."""
    n_actions = 3
    player_a = Agent(n_actions)
    player_b = Agent(n_actions)
    players = [player_a, player_b]
    for _ in range(n_iterations):
        for player in players:
            player.update_strategy()
            player.update_strategy_sum()
        action_a = player_a.sample_action()
        action_b = player_b.sample_action()
        # Compute the utility of the action compared to the opponent.
        actions_1 = [action_a, action_b]
        actions_2 = [action_b, action_a]
        for player, this_action, opponent_action in zip(players, actions_1, actions_2):
            action_utility = compute_action_utility(opponent_action)
            player.regret_sum += action_utility - action_utility[this_action]
    return player_a, player_b


n_iterations = 1000
player_a, player_b = train(n_iterations=n_iterations)
print(
    f"""
Trained for {n_iterations} iterations.

Following are the learned strategies for two players playing rock paper scissors.
The players should have learned a nash-equilibrium for this game.
For the game rock, paper, scissors this means playing equal probability (1/3).

Player a's strategy: {player_a.average_strategy}
Player b's strategy: {player_b.average_strategy}
"""
)
