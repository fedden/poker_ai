from __future__ import annotations

from typing import List, TYPE_CHECKING

from poker_ai.poker.deck import Deck

if TYPE_CHECKING:
    from poker_ai.poker.table import PokerTable
    from poker_ai.poker.player import Player
    from poker_ai.poker.card import Card


class Dealer:
    """The dealer is in charge of handling the cards on a poker table."""

    def __init__(self, **deck_kwargs):
        self.deck = Deck(**deck_kwargs)

    def deal_card(self) -> Card:
        """Return a completely random card."""
        return self.deck.pick(random=True)

    def deal_private_cards(self, players: List[Player]):
        """Deal private card to players.

        Parameters
        ----------
        players : list of Player
            The players to deal the private cards to.
        """
        for _ in range(2):
            for player in players:
                card: Card = self.deal_card()
                player.add_private_card(card)

    def deal_community_cards(self, table: PokerTable, n_cards: int):
        """Deal public cards."""
        if n_cards <= 0:
            raise ValueError(
                f"Positive n of cards must be specified, but got {n_cards}"
            )
        for _ in range(n_cards):
            card: Card = self.deal_card()
            table.add_community_card(card)

    def deal_flop(self, table: PokerTable):
        """Deal the flop public cards to the `table`."""
        return self.deal_community_cards(table, 3)

    def deal_turn(self, table: PokerTable):
        """Deal the turn public cards to the `table`."""
        return self.deal_community_cards(table, 1)

    def deal_river(self, table: PokerTable):
        """Deal the river public cards to the `table`."""
        return self.deal_community_cards(table, 1)
