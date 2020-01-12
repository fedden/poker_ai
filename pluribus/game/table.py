from __future__ import annotations

from typing import List, TYPE_CHECKING

from pluribus.game.dealer import Dealer

if TYPE_CHECKING:
    from pluribus.game.player import Player
    from pluribus.game.cards import Card


class PokerTable:
    """On a poker table a minimum of two players and one dealer are seated.

    You also find the community cards on it and can see the current pot size.
    Each player is responisble for handling his own cards privately.
    """

    def __init__(self, players: List[Player]):
        """"""
        self.players = players
        self.dealer = Dealer()
        self.community_cards: List[Card] = []

        self.num_games = 0
        self.num_players = len(players)
        assert self.num_players >= 2

    def set_players(self, players: List[Player]):
        """"""
        self.players = players

    def add_community_card(self, card: Card):
        """"""
        self.community_cards.append(card)
