from __future__ import annotations

import logging
import uuid
from typing import List, TYPE_CHECKING

import numpy as np

from pluribus.game.actions import Call, Fold, Raise
from pluribus.game.state import PokerGameState

if TYPE_CHECKING:
    from pluribus.game.cards import Card
    from pluribus.game.pot import Pot


logger = logging.getLogger(__name__)


class Player:
    """Base class for all poker-playing agents.

    A poker player has a name, holds chips to bet with, and has private cards
    to play with. The n_chips of contributions to the pot for a given hand of
    poker are stored cumulative, as the total pot to cash out is just the sum
    of all players' contributions.
    """

    def __init__(self, name: str, initial_chips: int, pot: Pot):
        """Instanciate a player."""
        self.name: str = name
        self.n_chips: int = initial_chips
        self.cards: List[Card] = []
        self._is_active = True
        self._id = int(uuid.uuid4().hex, 16)
        self.pot = pot

    def __hash__(self):
        """Make player hashable so we can index the pot like `pot[player]`."""
        return self._id

    def __repr__(self):
        """"""
        return '<Player name="{}" n_chips={:05d} n_bet_chips={:05d} ' \
            'folded={}>'.format(
                self.name,
                self.n_chips,
                self.n_bet_chips,
                int(not self.is_active))

    def add_chips(self, chips: int):
        """Add chips."""
        self.n_chips += chips

    def fold(self):
        """Deactivate player for this hand by folding cards."""
        self.is_active = False
        return Fold()

    def call(self, players: List[Player]):
        """Call the highest bet among all active players."""
        if self.is_all_in:
            return Call()
        else:
            n_chips_to_call = max(p.n_bet_chips for p in players)
            self.add_to_pot(n_chips_to_call)
            return Call()

    def raise_to(self, n_chips: int):
        """Raise your bet to a certain n_chips."""
        n_chips = self.add_to_pot(n_chips)
        raise_action = Raise()
        raise_action(n_chips)
        return raise_action

    def _try_to_make_full_bet(self, n_chips: int):
        """Ensures no bet is greater than the n_chips of chips left."""
        if self.n_chips - n_chips < 0:
            # We can't bet more than we have.
            n_chips = self.n_chips
        return n_chips

    def add_to_pot(self, n_chips: int):
        """Add to the n_chips put into the pot by this player."""
        # TODO(fedden): This code is called by engine.py for the small and big
        #               blind. What if the player can't actually add the blind?
        #               What do the rules stipulate in these circumstances.
        #               Ensure that this is sorted.
        n_chips = self._try_to_make_full_bet(n_chips)
        self.pot.add_chips(self, n_chips)
        self.n_chips -= n_chips
        return n_chips

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
        logger.debug(f'{self.name} {action}')
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
        """Return if the player is all in or not."""
        return self._is_active and self.n_chips == 0

    @property
    def n_bet_chips(self) -> int:
        """Returns the n_chips this player has bet so far."""
        return self.pot[self]
