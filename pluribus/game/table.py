from .player import Player
from .dealer import Dealer
from .cards import Card


class PokerTable:
    """On a poker table at least two players
    and precisely one dealer are seated. You
    also find the community cards on it
    and can see the current pot size. Each player
    is responisble for handling his own cards
    privately.
    """

    def __init__(self, players: list[Player]):
        self.players = players
        self.dealer = Dealer()
        self.community_cards: list[Card] = []

        self.num_games = 0
        self.num_players = len(players)
        assert self.num_players >= 2

    def set_players(self, players: list[Player]):
        self.players = players

    def add_community_card(self, card: Card):
        self.community_cards.append(card)
