from pluribus.game.actions import *
from pluribus.game.cards import Card
from pluribus.game.game import PokerGameState


class Player:
    """Base class for all poker-playing agents. A poker
    player has a name, holds chips to bet with, and has
    private cards to play with.

    The amount of contributions to the pot for a
    given hand of poker are stored cumulative, as the total
    pot to cash out is just the sum of all players' contributions.
    """
    def __init__(self, name:str, initial_chips: int):
        self.name: str = name
        self.chips: int = initial_chips
        self.cards: list[Card] = []
        self._is_active = True
        self._total_in_pot = 0

    def payout(self, chips: int):
        """Pay-out chips earned (or lost) in the last
        hand and reset the pot.
        """
        self.chips += chips
        self._total_in_pot = 0

    def fold(self):
        """Deactivate player for this hand by folding cards."""
        self.is_active = False
        return Fold()

    def call(self, players):
        """Call the highest bet among all active players."""
        #TODO how to handle all-ins? for later
        amount_to_call = max(p.bet_so_far() for p in players)
        self.bet(amount_to_call)
        return Call()

    def raise_to(self, amount: int):
        """Raise your bet to a certain amount.
        """
        self.bet(amount)
        raize = Raise()
        raize(amount)
        return raize

    def is_active(self) -> bool:
        return self._is_active

    def bet(self, amount: int):
        self._total_in_pot += amount

    def bet_so_far(self):
        return self._total_in_pot

    def add_private_card(self, card: Card):
        self.cards.append(card)

    def take_action(self, game_state: PokerGameState) -> PokerGameState:
        """All poker strategy is implemented here. Smart agents have
        to implement this method to compete. To take an action, agents
        receive the current game state and have to emit the next state.
        """
        raise NotImplementedError
        # previous = game_state.previous_state
        # table = previous.table
        # action = Fold()

        # return PokerGameState(game_state, table, self, action)
