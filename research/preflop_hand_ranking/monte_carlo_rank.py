import collections
from typing import Dict, List

import numpy as np

from pluribus.game.card import Card
from pluribus.game.deck import Deck
from pluribus.game.evaluation import Evaluator
from pluribus.game.evaluation import EvaluationCard


class PreflopMatrixEvaluator:
    """"""

    def __init__(self):
        """Initialise class."""
        self._deck = Deck()
        self._evaluator = Evaluator()

    def _deal_cards(self, n_players: int):
        """Deal a hand for each player and a table of cards."""
        self._deck.shuffle()
        hands = collections.defaultdict(list)
        card_i = 0
        for _ in range(2):
            for player_i in range(n_players):
                hands[player_i].append(self._deck[card_i])
                card_i += 1
        table = []
        # "Burn" card.
        card_i += 1
        for _ in range(3):
            table.append(self._deck[card_i])
        for _ in range(2):
            # "Burn" card.
            card_i += 1
            table.append(self._deck[card_i])
        return hands, table

    def _to_eval_cards(self, cards: List[Card]) -> List[EvaluationCard]:
        """Convert to cards suitable for deuces."""
        return [card.eval_card for card in cards]

    def _compute_preflop_rankings(self, n_players: int) -> Dict[int, List[List[Card]]]:
        """"""
        # Get random hands and table.
        hands, table = self._deal_cards(n_players)
        table = self._to_eval_cards(table)
        # Get ranking of hands.
        ranks_to_hands = collections.defaultdict(list)
        for player_i in range(n_players):
            eval_hand = self._to_eval_cards(hands[player_i])
            import ipdb; ipdb.set_trace()
            rank = self._evaluator.evaluate(cards=eval_hand, board=table)
            ranks_to_hands[rank].append(hands[player_i])
        score = len(set(ranks_to_hands.keys()))
        # Ensure the ranks go from n_players, ..., 3, 2, 1 as rank gets worse.
        normalised_rank_to_hands = collections.defaultdict(list)
        for rank in sorted(ranks_to_hands.keys()):
            for hand in ranks_to_hands[rank]:
                normalised_rank_to_hands[score].append(hand)
            score -= 1
        return normalised_rank_to_hands

    def _compute_delta_matrix(self, rank_to_hands: Dict[int, List[List[Card]]]):
        """"""
        matrix = np.zeros(shape=(13, 13))
        for rank, hands in rank_to_hands.items():
            for hand in hands:
                matrix[hand[0].rank_int - 2, hand[1].rank_int - 2] += rank
                matrix[hand[1].rank_int - 2, hand[0].rank_int - 2] += rank
        return matrix

    def next(self, n_players: int) -> np.ndarray:
        """"""
        rank_to_hands = self._compute_preflop_rankings(n_players)
        delta_matrix = self._compute_delta_matrix(rank_to_hands)
        return delta_matrix


evaluator = PreflopMatrixEvaluator()
print(evaluator.next(6))