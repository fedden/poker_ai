import logging
from typing import List
from itertools import combinations

import numpy as np
from joblib import Memory
from tqdm import tqdm

from poker_ai.poker.card import Card
from poker_ai.poker.deck import get_all_suits


cache_path = "./clustering_cache"
memory = Memory(cache_path, verbose=0)
log = logging.getLogger("poker_ai.clustering.runner")


class CardCombos:
    """This class stores combinations of cards (histories) per street."""
    def __init__(
        self,
        low_card_rank: int,
        high_card_rank: int,
        disk_cache: bool,
    ):
        super().__init__()
        # Setup caching.
        if disk_cache:
            self.get_card_combos = memory.cache(self.get_card_combos)
            self.create_info_combos = memory.cache(self.create_info_combos)
        # Sort for caching.
        suits: List[str] = sorted(list(get_all_suits()))
        ranks: List[int] = sorted(list(range(low_card_rank, high_card_rank)))
        self._cards = np.array(
            [Card(rank, suit) for suit in suits for rank in ranks]
        )
        self.starting_hands = self.get_card_combos(2)
        self.flop = self.create_info_combos(
            self.starting_hands, self.get_card_combos(3)
        )
        log.info("created flop")
        self.turn = self.create_info_combos(
            self.starting_hands, self.get_card_combos(4)
        )
        log.info("created turn")
        self.river = self.create_info_combos(
            self.starting_hands, self.get_card_combos(5)
        )
        log.info("created river")

    def get_card_combos(self, num_cards: int) -> np.ndarray:
        """
        Get the card combinations for a given street.

        Parameters
        ----------
        num_cards : int
            Number of cards you want returned

        Returns
        -------
            Combos of cards (Card) -> np.array
        """
        return np.array([c for c in combinations(self._cards, num_cards)])

    def create_info_combos(
        self, start_combos: np.array, publics: np.array
    ) -> np.ndarray:
        """Combinations of private info(hole cards) and public info (board).

        Uses the logic that a AsKsJs on flop with a 10s on turn is the same
        as AsKs10s on flop and Js on turn. That logic is used within the
        literature as well as the logic where those two are different.

        Parameters
        ----------
        start_combos : np.array
            Starting combination of cards (beginning with hole cards)
        publics : np.array
            Public cards being added
        Returns
        -------
            Combinations of private information (hole cards) and public
            information (board)
        """
        if publics.shape[1] == 3:
            betting_stage = "flop"
        elif publics.shape[1] == 4:
            betting_stage = "turn"
        elif publics.shape[1] == 5:
            betting_stage = "river"
        else:
            betting_stage = "unknown"
        our_cards: List[Card] = []
        for combos in tqdm(
            start_combos,
            dynamic_ncols=True,
            desc=f"Creating {betting_stage} info combos",
        ):
            for public_combo in publics:
                if not np.any(np.isin(combos, public_combo)):
                    our_cards.append(np.concatenate((combos, public_combo)))
        return np.array(our_cards)
