from typing import Dict, Tuple, List
import operator
import math

from poker_ai.poker.card import Card


def nCr(n: int, r: int):
    """Combinations helper: n choose r."""
    if n < r:
        # Following the convention that if n < r, return 0
        return 0
    f = math.factorial
    return f(n) / f(r) / f(n-r)


def rank_set(cards: List[Card], idx=0):
    """Recursive function for gicing unique id to hoe cards."""
    if len(cards) == 1:
        # Offsetting by 2 since ranks go from 2-14.
        return int(idx + cards[0].rank_int - 2)
    # Offsetting by 2 since ranks go from 2-14.
    idx += nCr(cards[0].rank_int - 2, len(cards))
    return rank_set(cards=cards[1:], idx=idx)


def compute_preflop_lossless_abstraction(builder) -> Dict[Tuple[int, int], int]:
    """Compute the preflop abstraction dictionary.

    Only works for the short deck presently.
    """
    # Making sure this is 20 card deck with 2-9 removed
#    allowed_ranks = {10, 11, 12, 13, 14}
 #   found_ranks = set([c.rank_int for c in builder._cards])
  #  if found_ranks != allowed_ranks:
#        raise ValueError(
 #           f"Preflop lossless abstraction only works for a short deck with "
  #          f"ranks [10, jack, queen, king, ace]. What was specified="
   #         f"{found_ranks} doesn't equal what is allowed={allowed_ranks}"
    #    )
    # Getting combos and indexing with lossless abstraction
    preflop_lossless: Dict[Tuple[Card, Card], int] = {}
    for starting_hand in builder.starting_hands:
        starting_hand = sorted(
            list(starting_hand),
            key=operator.attrgetter("eval_card"),
            reverse=True
        )
        preflop_lossless[tuple(starting_hand)] = rank_set(starting_hand)
    return preflop_lossless
