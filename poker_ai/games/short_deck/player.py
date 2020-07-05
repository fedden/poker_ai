from poker_ai.poker.player import Player
from poker_ai.poker.pot import Pot


class ShortDeckPokerPlayer(Player):
    """Inherits from Player which will interface easily with the PokerEngine.

    This class should manage the state of the players personal pot, and the
    private cards that are dealt to the player. Also manages whether this
    player has folded or not.
    """

    def __init__(self, player_i: int, initial_chips: int, pot: Pot):
        """Instanciate a player."""
        super().__init__(
            name=f"player_{player_i}", initial_chips=initial_chips, pot=pot,
        )
        self.is_turn = False
