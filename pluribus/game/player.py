from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from pluribus.game.actions import Call, Fold, Raise
    from pluribus.game.cards import Card
    from pluribus.game.state import PokerGameState


class Player:
    """Base class for all poker-playing agents.

    A poker player has a name, holds chips to bet with, and has private cards
    to play with. The amount of contributions to the pot for a given hand of
    poker are stored cumulative, as the total pot to cash out is just the sum
    of all players' contributions.
    """

    def __init__(self, name: str, initial_chips: int):
        """"""
        self.name: str = name
        self.chips: int = initial_chips
        self.cards: list[Card] = []
        self._is_active = True
        self._total_in_pot = 0

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
        # TODO how to handle all-ins? for later
        amount_to_call = max(p.bet_so_far() for p in players)
        self.bet(amount_to_call)
        return Call()

    def raise_to(self, amount: int):
        """Raise your bet to a certain amount."""
        self.bet(amount)
        _raise = Raise()
        _raise(amount)
        return _raise

    def bet(self, amount: int):
        """Add to the amount put into the pot by this player."""
        self._total_in_pot += amount

    def add_private_card(self, card: Card):
        """Add a private card to this player."""
        self.cards.append(card)

    def take_action(self, game_state: PokerGameState) -> PokerGameState:
        """All poker strategy is implemented here.

        Smart agents have to implement this method to compete. To take an
        action, agents receive the current game state and have to emit the next
        state.
        """
        raise NotImplementedError
        # previous = game_state.previous_state
        # table = previous.table
        # action = Fold()

        # return PokerGameState(game_state, table, self, action)

    @property
    def is_active(self) -> bool:
        """Returns if the player is playing or not."""
        return self._is_active

    @property
    def bet_so_far(self) -> int:
        """Returns the amount this player has be so far."""
        return self._total_in_pot
