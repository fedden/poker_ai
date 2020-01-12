from __future__ import annotations

import collections
import random


__all__ = [
    "Card",
    "Deck",
]


def get_all_suits():
    """"""
    return {"spades", "diamonds", "clubs", "hearts"}


def get_all_ranks():
    """"""
    return [
        "2", "3", "4", "5", "6", "7", "8", "9", "10",
        "jack", "queen", "king", "ace"
    ]


class Card:
    def __init__(self, rank, suit):
        if not isinstance(rank, int):
            raise ValueError(
                f'rank should be int but was type: {type(rank)}.')
        if rank < 2 or rank > 14:
            raise ValueError(
                f'rank should be between 2 and 14 (inclusive) but was {rank}')
        if suit not in get_all_suits():
            raise ValueError(f'suit {suit} must be in {get_all_suits()}')
        self._rank = rank
        self._suit = suit

    @property
    def rank(self):
        return self._rank_to_str(self._rank)

    @property
    def suit(self):
        return self._suit

    def _rank_to_str(self, rank):
        return {
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "10",
            11: "jack",
            12: "queen",
            13: "king",
            14: "ace"
        }[rank]

    def _suit_to_icon(self, suit):
        return {
            "hearts": "♥", "diamonds": "♦", "clubs": "♣", "spades": "♠"
        }[suit]

    def __repr__(self):
        icon = self._suit_to_icon(self.suit)
        return f'<Card card=[{self.rank} of {self.suit} {icon}]>'


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
