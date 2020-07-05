from poker_ai.poker.dealer import Dealer
from poker_ai.poker.card import Card


def test_dealer_1():
    include_ranks = [10, 11, 12, 13, 14]
    dealer = Dealer(include_ranks=include_ranks)
    deck_size = len(dealer.deck._cards_in_deck)
    assert deck_size == len(include_ranks * 4)
    for i in range(1, deck_size + 1):
        card: Card = dealer.deal_card()
        del card
        deck_size = len(dealer.deck._cards_in_deck)
        assert deck_size == len(include_ranks * 4) - i
        assert len(dealer.deck._dealt_cards) == i
