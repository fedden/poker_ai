"""
Quick and easy script for getting lossless abstraction for starting hands
This Script only applies to a Deck of 20 cards (ie; make sure line 58 of information_abstraction.py is removing 2-9)
cd into clustering and run python preflop_lossless_short_deck.py
--It will fail if you aren't in the clustering directory as it will output to clustering/data/preflop_lossless.pkl

Next Steps/Future Enhancements:
- Make dynamic based on deck size
- Move on to full lossless for all combos: https://github.com/kdub0/hand-isomorphism
"""

from typing import List
import dill as pickle

from information_abstraction import ShortDeck


def make_starting_hand_lossless(starting_hand: List[int], short_deck: ShortDeck) -> int:
    """

    :param starting_hand: list of two ints that represent the evaluation of a card
    :param short_deck: class ShortDeck that specifies the deck and allows to get card combos
    :return: index for lossless abstraction of prefop hands
    """
    ranks = []
    suits = []
    for card_eval in starting_hand:
        card = short_deck._evals_to_cards[card_eval]
        ranks.append(card.rank_int)
        suits.append(card.suit)
    if len(set(suits)) == 1:
        suited = True
    else:
        suited = False
    if all(c_rank == 14 for c_rank in ranks):
        return 0
    elif all(c_rank == 13 for c_rank in ranks):
        return 1
    elif all(c_rank == 12 for c_rank in ranks):
        return 2
    elif all(c_rank == 11 for c_rank in ranks):
        return 3
    elif all(c_rank == 10 for c_rank in ranks):
        return 4
    elif 14 in ranks and 13 in ranks:
        return 5 if suited else 15
    elif 14 in ranks and 12 in ranks:
        return 6 if suited else 16
    elif 14 in ranks and 11 in ranks:
        return 7 if suited else 17
    elif 14 in ranks and 10 in ranks:
        return 8 if suited else 18
    elif 13 in ranks and 12 in ranks:
        return 9 if suited else 19
    elif 13 in ranks and 11 in ranks:
        return 10 if suited else 20
    elif 13 in ranks and 10 in ranks:
        return 11 if suited else 21
    elif 12 in ranks and 11 in ranks:
        return 12 if suited else 22
    elif 12 in ranks and 10 in ranks:
        return 13 if suited else 23
    elif 11 in ranks and 10 in ranks:
        return 14 if suited else 24


if __name__ == "__main__":
    short_deck = ShortDeck()
    # making sure this is 20 card deck with 2-9 removed
    allowed_ranks = {10, 11, 12, 13, 14}
    found_ranks = set([c.rank_int for c in short_deck._cards])
    assert found_ranks == allowed_ranks

    # getting combos and indexing with lossless abstraction
    starting_hands = short_deck.get_card_combos(2)
    preflop_lossless = {}
    for starting_hand in starting_hands:
        starting_hand = sorted(starting_hand, reverse=True)
        preflop_lossless[tuple(starting_hand)] = make_starting_hand_lossless(
            starting_hand, short_deck
        )

    # dumping to preflop_lossless.pkl in data folder
    location = "data/preflop_lossless.pkl"
    with open(location, "wb") as file:
        pickle.dump(preflop_lossless, file)
    print(f"Dumped Data to {location}")
