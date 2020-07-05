from typing import Any, Dict

from poker_ai.games.short_deck.player import ShortDeckPokerPlayer
from poker_ai.poker.card import Card

_colours = ["cyan", "lightcoral", "crimson", "#444", "forestgreen", "goldenrod", "gold"]
_suit_lut = {"spades": "P", "diamonds": "D", "clubs": "C", "hearts": "H"}


def to_player_dict(player_i: int, player: ShortDeckPokerPlayer) -> Dict[str, Any]:
    """Create dictionary to describe player for frontend."""
    return {
        "name": player.name,
        "color": _colours[player_i],
        "bank": player.n_chips,
        "onTable": player.pot[player],
        "hasCards": True,
    }


def to_card_dict(card: Card) -> Dict[str, str]:
    """Create dictionary to describe card for frontend."""
    return {
        "f": _suit_lut[card.suit],
        "v": card.rank[0].upper(),
    }
