import collections
import random

__all__ = [
    'Card',
    'Deck',
]


Card = collections.namedtuple('Card', ['rank', 'suit'])


def card_to_string(card):
    return card.rank + card.suit

# Monkey patch the string representation of a card
Card.__repr__ = card_to_string


Ranks = [str(n) for n in range(2, 11)] + list('JQKA')


Suits = {
    's': 'spades',
    'd': 'diamonds',
    'c': 'clubs',
    'h': 'hearts'
}  


class Deck: 
    def __init__(self):
        self._cards = [Card(rank, suit) for suit in Suits.keys() for rank in Ranks]
    
    def __len__(self):
        return len(self._cards)
    
    def __getitem__(self, position):
        return self._cards[position]

    def __setitem__(self, position, card):
        self._cards[position] = card

    def shuffle(self):
        random.shuffle(self)

    def pick(self):
        return self._cards.pop()
