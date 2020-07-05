import random

from poker_ai.poker.card import Card, get_all_suits


def random_card(suit=None, rank=None):
    """Get random card."""
    if rank is None:
        rank = random.randint(2, 14)
    if suit is None:
        suit = random.choice(list(get_all_suits()))
    card = Card(rank, suit)
    return card, rank


def test_card_equality():
    """Ensure cards are less than."""
    n_iterations = 1000
    for _ in range(n_iterations):
        same_suit = random.choice(list(get_all_suits()))
        card_a, rank_a = random_card(suit=same_suit)
        card_b, rank_b = random_card(suit=same_suit)
        if rank_a == rank_b:
            assert card_a == card_b
            assert int(card_a) == int(card_b)
        else:
            assert card_a != card_b
            assert int(card_a) != int(card_b)
