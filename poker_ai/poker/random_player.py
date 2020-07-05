import logging
from typing import List

import numpy as np

from poker_ai.poker.player import Player
from poker_ai.poker.pot import Pot
from poker_ai.poker.state import PokerGameState

logger = logging.getLogger(__name__)


class RandomPlayer(Player):
    """Complete a dummy agent largely for development purposes.
    Extends the `poker_ai.game.player.Player` class so inherits all of that
    functionality.
    The agent will make a move based on the probabilities set in the
    constructor, so you can weight the chances of it taking various actions for
    a given turn.
    """

    def __init__(
            self,
            name: str,
            initial_chips: int,
            pot: Pot,
            fold_probability: float = 0.1,
            raise_probability: float = 0.1,
            call_probability: float = 0.8):
        """Construct the random player."""
        super().__init__(name=name, initial_chips=initial_chips, pot=pot)
        self.fold_probability = fold_probability
        self.raise_probability = raise_probability
        self.call_probability = call_probability
        prob_sum = fold_probability + raise_probability + call_probability
        if not np.isclose(prob_sum, 1.0):
            raise ValueError(f'Probabilities passed must sum to one.')

    def _random_move(self, players: List[Player]):
        """Make a random move."""
        dice_roll = np.random.sample()
        bound_1 = self.fold_probability
        bound_2 = self.fold_probability + self.raise_probability
        if 0.0 < dice_roll <= bound_1:
            return self.fold()
        elif bound_1 < dice_roll <= bound_2:
            return self.raise_to(100)
        else:
            return self.call(players=players)

    def take_action(self, game_state: PokerGameState) -> PokerGameState:
        action = self._random_move(players=game_state.table.players)
        logger.debug(f'{self.name} {action}')
        return PokerGameState(game_state, game_state.table, self, action, False)
