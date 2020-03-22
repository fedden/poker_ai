from __future__ import annotations

from typing import List, TYPE_CHECKING

from pluribus.poker.deck import Deck

if TYPE_CHECKING:
    from pluribus.poker.table import PokerTable
    from pluribus.poker.player import Player


class Dealer:
    """The dealer is in charge of handling the cards on a poker table."""

    def __init__(self):
        self.deck = Deck()
        self.deck.shuffle()

    def deal_card(self):
        return self.deck.pick()

    def use_fresh_deck(self):
        self.deck = Deck()
        self.deck.shuffle()

    def deal_private_cards(self, players: List[Player]):
        for _ in range(2):
            for player in players:
                card = self.deal_card()
                player.add_private_card(card)

    def deal_community_cards(self, table: PokerTable, num_cards: int):
        assert num_cards > 0
        # TODO(fedden): Do we need to burn a card like in casinos? Check the
        #               papers supplimentary materials.
        for _ in range(num_cards):
            card = self.deal_card()
            table.add_community_card(card)

    def deal_flop(self, table: PokerTable):
        return self.deal_community_cards(table, 3)

    def deal_turn(self, table: PokerTable):
        return self.deal_community_cards(table, 1)

    def deal_river(self, table: PokerTable):
        return self.deal_community_cards(table, 1)
