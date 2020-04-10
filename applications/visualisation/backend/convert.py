from typing import Any, Dict

from pluribus.games.short_deck.player import ShortDeckPokerPlayer
from pluribus.games.short_deck.state import ShortDeckPokerState
from pluribus.poker.card import Card
from pluribus.poker.pot import Pot


def to_player_dict(
    player_i: int, player: ShortDeckPokerPlayer, pot: Pot,
) -> Dict[str, Any]:
    """Create dictionary to describe player for frontend."""
    return {
        "name": player.name,
        "color": colours[player_i],
        "bank": player.n_chips,
        "onTable": pot[player],
        "hasCards": True,
    }


def to_card_dict(card: Card) -> Dict[str, str]:
    """Create dictionary to describe card for frontend."""
    return {
        "f": {"spades": "P", "diamonds": "D", "clubs": "C", "hearts": "H",}[card.suit],
        "v": card.rank[0].upper(),
    }
