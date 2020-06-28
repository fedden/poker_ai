from typing import Dict, Tuple


def make_starting_hand_lossless(starting_hand, short_deck) -> int:
    """

    :param starting_hand: list of two ints that represent the evaluation of a card
    :param short_deck: class ShortDeck that specifies the deck and allows to get card combos
    :return: index for lossless abstraction of prefop hands
    """
    ranks = []
    suits = []
    for card in starting_hand:
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


def compute_preflop_lossless_abstraction(builder) -> Dict[Tuple[int, int], int]:
    """Compute the preflop abstraction dictionary.

    Only works for the short deck presently.
    """
    # making sure this is 20 card deck with 2-9 removed
    allowed_ranks = {10, 11, 12, 13, 14}
    found_ranks = set([c.rank_int for c in builder._cards])
    if found_ranks != allowed_ranks:
        raise ValueError(
            f"Preflop lossless abstraction only works for a short deck with "
            f"ranks [10, jack, queen, king, ace]. What was specified="
            f"{found_ranks} doesn't equal what is allowed={allowed_ranks}"
        )
    # getting combos and indexing with lossless abstraction
    preflop_lossless: Dict[Tuple[int, int], int] = {}
    for starting_hand in builder.starting_hands:
        starting_hand = tuple(sorted(starting_hand, reverse=True))
        preflop_lossless[starting_hand] = make_starting_hand_lossless(
            starting_hand, builder
        )
    return preflop_lossless
