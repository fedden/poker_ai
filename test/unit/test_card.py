import random

from pluribus.poker.card import Card, get_all_suits


def random_card():
    """Get random card."""
    rank = random.randint(2, 14)
    suit = random.choice(list(get_all_suits()))
    card = Card(rank, suit)
    return card, rank


def test_card_boolean_ops():
    """Ensure cards are less than."""
    n_iterations = 1000
    for _ in range(n_iterations):
        card_a, rank_a = random_card()
        card_b, rank_b = random_card()
        if rank_a < rank_b:
            assert card_a < card_b
        if rank_a > rank_b:
            assert card_a > card_b
        if rank_a >= rank_b:
            assert card_a >= card_b
        if rank_a <= rank_b:
            assert card_a <= card_b
        if rank_a == rank_b:
            assert card_a == card_b
        if rank_a != rank_b:
            assert card_a != card_b
