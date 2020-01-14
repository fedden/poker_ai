from __future__ import annotations

from typing import List, TYPE_CHECKING

import numpy as np

from pluribus.game.actions import Call, Fold, Raise
from pluribus.game.state import PokerGameState

if TYPE_CHECKING:
    from pluribus.game.cards import Card


class Player:
    """Base class for all poker-playing agents.

    A poker player has a name, holds chips to bet with, and has private cards
    to play with. The amount of contributions to the pot for a given hand of
    poker are stored cumulative, as the total pot to cash out is just the sum
    of all players' contributions.
    """

    def __init__(self, name: str, initial_chips: int):
        """Instanciate a player."""
        self.name: str = name
        self.chips: int = initial_chips
        self.cards: List[Card] = []
        self._is_active = True
        self._total_in_pot = 0

    def __repr__(self):
        """"""
        return f'<Player name="{self.name}" chips={self.chips}>'

    def payout(self, chips: int):
        """Pay-out chips earned or lost in the last hand and reset the pot."""
        self.chips += chips
        self._total_in_pot = 0

    def fold(self):
        """Deactivate player for this hand by folding cards."""
        self.is_active = False
        return Fold()

    def call(self, players: List[Player]):
        """Call the highest bet among all active players."""
        if self.is_all_in:
            return Call()
        else:
            amount_to_call = max(p.bet_so_far for p in players)
            self.add_to_pot(amount_to_call)
            return Call()

    def raise_to(self, amount: int):
        """Raise your bet to a certain amount."""
        if self.chips - amount < 0:
            # We can't bet more than we have.
            amount = self.chips
        self.add_to_pot(amount)
        _raise = Raise()
        _raise(amount)
        return _raise

    def add_to_pot(self, amount: int):
        """Add to the amount put into the pot by this player."""
        self._total_in_pot += amount
        self.chips -= amount

    def add_private_card(self, card: Card):
        """Add a private card to this player."""
        self.cards.append(card)

    def take_action(self, game_state: PokerGameState) -> PokerGameState:
        """All poker strategy is implemented here.

        Smart agents have to implement this method to compete. To take an
        action, agents receive the current game state and have to emit the next
        state.
        """
        action = self._random_move(players=game_state.table.players)
        return PokerGameState(game_state, game_state.table, self, action)

    def _random_move(self, players: List[Player]):
        """Random move to make FOR DEVELOPMENT PURPOSES"""
        # TODO(fedden): Delete this method.
        dice_roll = np.random.sample()
        if 0.0 < dice_roll < 0.05:
            # 5% chance to fold.
            return self.fold()
        elif 0.05 < dice_roll < 0.10:
            # 10% chance to raise.
            return self.raise_to(100)
        else:
            # 85% chance to call.
            return self.call(players=players)

    @property
    def is_active(self) -> bool:
        """Getter for if the player is playing or not."""
        return self._is_active

    @is_active.setter
    def is_active(self, x):
        """Setter for if the player is playing or not."""
        self._is_active = x

    @property
    def is_all_in(self) -> bool:
        """"""
        return self._is_active and self.chips == 0

    @property
    def bet_so_far(self) -> int:
        """Returns the amount this player has be so far."""
        return self._total_in_pot
