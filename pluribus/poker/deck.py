from __future__ import annotations

import random

from pluribus.poker.card import Card, get_all_suits


class Deck:
    def __init__(self):
        self._cards = [
            Card(rank, suit)
            for suit in get_all_suits() for rank in range(2, 15)
        ]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        return self._cards[position]

    def __setitem__(self, position, card):
        self._cards[position] = card

    def shuffle(self):
        random.shuffle(self._cards)

    def pick(self):
        return self._cards.pop()
