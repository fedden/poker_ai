from __future__ import annotations

import random
from typing import List

import numpy as np

from poker_ai.poker.card import Card, get_all_suits

default_include_suits: List[str] = list(get_all_suits())
default_include_ranks: List[int] = list(range(2, 15))


class Deck:
    """Class to manage the deck."""

    def __init__(
        self,
        include_suits: List[str] = default_include_suits,
        include_ranks: List[int] = default_include_ranks,
    ):
        """Construct the deck of cards."""
        self._include_suits = include_suits
        self._include_ranks = include_ranks
        self.reset()

    def __len__(self) -> int:
        """Return overall length of the deck."""
        return len(self._cards_in_deck) + len(self._dealt_cards)

    def reset(self):
        """Reset the deck and shuffle it, ready for use."""
        self._cards_in_deck: List[Card] = [
            Card(rank, suit)
            for suit in self._include_suits
            for rank in self._include_ranks
        ]
        self._dealt_cards: List[Card] = []
        random.shuffle(self._cards_in_deck)

    def pick(self, random: bool = True) -> Card:
        """Return a card from the deck.

        Parameters
        ----------
        random : bool
            If this is true, return a completely random card, else return the
            next card in the deck.

        Returns
        -------
        card : Card
            The card that was picked.
        """
        if not len(self._cards_in_deck):
            raise ValueError("Deck is empty - please use Deck.reset()")
        elif random:
            index: int = np.random.randint(len(self._cards_in_deck), size=None)
        else:
            index: int = len(self._cards_in_deck) - 1
        card: Card = self._cards_in_deck.pop(index)
        self._dealt_cards.append(card)
        return card

    def remove(self, card):
        """Remove a specific card from the deck"""
        if card in self._cards_in_deck:
            self._cards_in_deck.remove(card)
            self._dealt_cards.append(card)
