import logging
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List

import click
import ipdb
import joblib
import numpy as np
from joblib import Memory
from sklearn.cluster import KMeans
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import concurrent.futures

from poker_ai.poker.card import Card
from poker_ai.poker.deck import get_all_suits
from poker_ai.poker.evaluation import Evaluator


cache_path = "./clustering_cache"
memory = Memory(cache_path, verbose=0)
log = logging.getLogger("poker_ai.clustering.runner")


class GameUtility:
    """
    This class takes care of some game related functions
    """

    def __init__(self, our_hand: np.ndarray, board: np.ndarray, cards: np.ndarray):
        """"""
        self._evaluator = Evaluator()
        # TODO: this is what takes forever, find a better way
        unavailable_cards = np.concatenate([board, our_hand], axis=0)
        self.available_cards = np.array(
            [c for c in cards if c not in unavailable_cards]
        )
        self.our_hand = our_hand
        self.board = board

    def evaluate_hand(self, hand: np.ndarray) -> int:
        """
        takes a hand
        :param hand: list of two integers Card.eval_card
        :return: evaluation of hand
        """
        try:
            return self._evaluator.evaluate(
                board=self.board.astype(np.int).tolist(),
                cards=hand.astype(np.int).tolist(),
            )
        except KeyError:
            ipdb.set_trace()

    def get_winner(self) -> int:
        """Get the winner.

        Returns
        -------
            int of win (0), lose (1) or tie (2) - this is an index in the
            expected hand strength array
        """
        our_hand_rank = self.evaluate_hand(self.our_hand)
        opp_hand_rank = self.evaluate_hand(self.opp_hand)
        if our_hand_rank > opp_hand_rank:
            return 0
        elif our_hand_rank < opp_hand_rank:
            return 1
        else:
            return 2

    @property
    def opp_hand(self) -> List[int]:
        """Get random card.

        :return: two cards for the opponent (Card.eval_card)
        """
        return np.random.choice(self.available_cards, 2, replace=False)


class InfoSets:
    """
    This class stores combinations of cards (histories) per street (for flop, turn, river)
    # TODO: should this be isomorphic/lossless to reduce the program run time?
    """

    def __init__(self, disk_cache: bool = False):
        super().__init__()
        # Setup caching.
        if disk_cache:
            self.get_card_combos = memory.cache(self.get_card_combos)
            self.create_info_combos = memory.cache(self.create_info_combos)
        # Sort for caching.
        suits: List[str] = sorted(list(get_all_suits()))
        ranks: List[int] = sorted(list(range(12, 15)))
        self._cards = np.array([Card(rank, suit) for suit in suits for rank in ranks])
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
        :param num_cards: number of cards you want returned
        :return: combos of cards (Card.eval_card) -> np.array
        """
        return np.array([c for c in combinations(self._cards, num_cards)])

    def create_info_combos(
        self, start_combos: np.array, publics: np.array
    ) -> np.ndarray:
        """Combinations of private info(hole cards) and public info (board).

        Uses the logic that a AsKsJs on flop with a 10s on turn is different
        than AsKs10s on flop and Js on turn. That logic is used within the
        literature.

        :param start_combos: starting combination of cards (beginning with hole cards)
        :param publics: np.array of public combinations being added
        :return: Combinations of private information (hole cards) and public information (board)
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


class CardInfoLutBuilder(InfoSets):
    """
    Stores info buckets for each street when called
    # TODO: create key to access these from a dictionary, store more efficiently somehow
    # TODO: change cluster to num_clusters=200 for full deck
    """

    def __init__(self, save_dir: str = ""):
        """"""
        super().__init__()
        self.card_info_lut_path: Path = Path(save_dir) / "card_info_lut.joblib"
        try:
            self.card_info_lut: Dict[str, Any] = joblib.load(self.card_info_lut_path)
        except FileNotFoundError:
            self.card_info_lut: Dict[str, Any] = {}

    def compute(self):
        """Compute all clusters and save to card_info_lut dictionary.

        Will attempt to load previous progress and will save after each cluster
        is computed.
        """
        log.info("Starting computation of clusters.")
        start = time.time()
        if "river" not in self.card_info_lut:
            self.card_info_lut["river"] = self._compute_river_clusters()
            joblib.dump(self.card_info_lut, self.card_info_lut_path)
        if "turn" not in self.card_info_lut:
            self.card_info_lut["turn"] = self._compute_turn_clusters()
            joblib.dump(self.card_info_lut, self.card_info_lut_path)
        if "flop" not in self.card_info_lut:
            self.card_info_lut["flop"] = self._compute_flop_clusters()
            joblib.dump(self.card_info_lut, self.card_info_lut_path)
        end = time.time()
        log.info(f"Finished computation of clusters - took {end - start} seconds.")

    def _compute_river_clusters(self):
        """"""
        log.info("Starting computation of river clusters.")
        start = time.time()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            self._river_ehs = list(
                tqdm(
                    executor.map(
                        self.process_river_ehs,
                        self.river,
                        chunksize=len(self.river) // 160,
                    ),
                    total=len(self.river),
                )
            )
        self._river_centroids, self._river_clusters = self.cluster(
            num_clusters=50, X=self._river_ehs
        )
        end = time.time()
        log.info(
            f"Finished computation of river clusters - took {end - start} seconds."
        )
        return self._river_clusters

    def _compute_turn_clusters(self):
        """"""
        log.info("Starting computation of turn clusters.")
        start = time.time()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            self._turn_ehs_distributions = list(
                tqdm(
                    executor.map(
                        self.process_turn_ehs_distributions,
                        self.turn,
                        chunksize=len(self.turn) // 160,
                    ),
                    total=len(self.turn),
                )
            )
        self._turn_centroids, self._turn_clusters = self.cluster(
            num_clusters=50, X=self._turn_ehs_distributions
        )
        end = time.time()
        log.info(f"Finished computation of turn clusters - took {end - start} seconds.")
        return self._turn_clusters

    def _compute_flop_clusters(self):
        """"""
        log.info("Starting computation of flop clusters.")
        start = time.time()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            self._flop_potential_aware_distributions = list(
                tqdm(
                    executor.map(
                        self.process_flop_potential_aware_distributions,
                        self.flop,
                        chunksize=len(self.flop) // 160,
                    ),
                    total=len(self.flop),
                )
            )
        self._flop_centroids, self._flop_clusters = self.cluster(
            num_clusters=50, X=self._flop_potential_aware_distributions
        )
        end = time.time()
        log.info(f"Finished computation of flop clusters - took {end - start} seconds.")
        return self._flop_clusters

    @staticmethod
    def simulate_get_ehs(game: GameUtility, num_simulations: int = 2) -> List[float]:
        """
        # TODO: probably want to increase simulations..
        :param game: GameState for help with determining winner and sampling opponent hand
        :param num_simulations: how many simulations you want to do
        :return: [win_rate, loss_rate, tie_rate]
        """
        ehs: np.ndarray = np.zeros(3)
        for _ in range(num_simulations):
            idx: int = game.get_winner()
            # increment win rate for winner/tie
            ehs[idx] += 1 / num_simulations
        return ehs

    def simulate_get_turn_ehs_distributions(
        self,
        available_cards: List[int],
        the_board: List[int],
        our_hand: List[int],
        num_simulations: int = 2,
    ) -> np.array:
        """
        # TODO num_simulations should be higher

        :param available_cards: list of available cards on the turn
        :param the_board: the board as of the turn
        :param our_hand: cards our hand (Card.eval_card)
        :param num_simulations: int of simulations
        :return: array of counts for each cluster the turn fell into by the river after simulations
        """
        turn_ehs_distribution = np.zeros(len(self._river_centroids))
        # sample river cards and run a simulation
        for _ in range(num_simulations):
            river_card = np.random.choice(available_cards, 1, replace=False)
            board = np.append(the_board, river_card)
            game = GameUtility(our_hand=our_hand, board=board, cards=self._cards)
            ehs = self.simulate_get_ehs(game)
            # get EMD for expected hand strength against each river centroid
            # to which does it belong?
            for idx, river_centroid in enumerate(self._river_centroids):
                # TODO: do we need a faster implementation of this?
                emd = wasserstein_distance(ehs, river_centroid)
                if idx == 0:
                    min_idx = idx
                    min_emd = emd
                else:
                    if emd < min_emd:
                        min_idx = idx
                        min_emd = emd
            # now increment the cluster to which it belongs -
            turn_ehs_distribution[min_idx] += 1 / num_simulations
        return turn_ehs_distribution

    def process_river_ehs(self, public: List[int]) -> List[float]:
        """

        :param num_print: number of simulations of opponents cards for calculating ehs
        :return: np.ndarray of arrays containing [win_rate, loss_rate, tie_rate]
        """
        our_hand = public[:2]
        board = public[2:7]
        # get expected hand strength
        game = GameUtility(our_hand=our_hand, board=board, cards=self._cards)
        return self.simulate_get_ehs(game)

    @staticmethod
    def get_available_cards(
        cards: np.ndarray, unavailable_cards: np.ndarray
    ) -> np.ndarray:
        """Get all cards that are available."""
        # Turn into set for O(1) lookup speed.
        unavailable_cards = set(unavailable_cards.tolist())
        return np.array([c for c in cards if c not in unavailable_cards])

    def process_turn_ehs_distributions(self, public: List[int]) -> List[float]:
        """

        :param num_print: frequency at which to print
        :return: np.ndarray of distribution aware turn distributions
        """
        available_cards: np.ndarray = self.get_available_cards(
            cards=self._cards, unavailable_cards=public
        )
        # sample river cards and run a simulation
        turn_ehs_distribution = self.simulate_get_turn_ehs_distributions(
            available_cards, the_board=public[2:6], our_hand=public[:2]
        )
        return turn_ehs_distribution

    def process_flop_potential_aware_distributions(
        self, public: List[int], num_simulations: int = 2
    ) -> np.ndarray:
        """

        :param num_print: frequency at which to print
        :param num_simulations: number of simulations
        :return: ndarray of potential aware histograms
        """
        available_cards: np.ndarray = self.get_available_cards(
            cards=self._cards, unavailable_cards=public
        )
        potential_aware_distribution_flop = [0] * len(self._turn_centroids)
        for j in range(num_simulations):
            # randomly generating turn
            turn_card = np.random.choice(available_cards, 1, replace=False)
            our_hand = public[:2]
            board = public[2:5]
            the_board = np.append(board, turn_card).tolist()
            # getting available cards
            available_cards_turn = [
                x for x in available_cards if x != turn_card[0]
            ]  # TODO: get better implementation of this
            turn_ehs_distribution = self.simulate_get_turn_ehs_distributions(
                available_cards_turn, the_board=the_board, our_hand=our_hand
            )
            for idx, turn_centroid in enumerate(self._turn_centroids):
                # earth mover distance
                emd = wasserstein_distance(turn_ehs_distribution, turn_centroid)
                if idx == 0:
                    min_idx = idx
                    min_emd = emd
                else:
                    if emd < min_emd:
                        min_idx = idx
                        min_emd = emd
            # ok, now increment the cluster to which it belongs -
            potential_aware_distribution_flop[min_idx] += 1 / num_simulations
            # object for storing flop potential aware expected hand strength distributions
        return potential_aware_distribution_flop

    @staticmethod
    def cluster(num_clusters: int, X: np.array):
        km = KMeans(
            n_clusters=num_clusters,
            init="random",  # would be 200 in our example
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0,
        )
        y_km = km.fit_predict(X)
        # centers to be used for r - 1 (ie; the previous round)
        centroids = km.cluster_centers_
        return centroids, y_km


@click.command()
def cluster():
    """Run clustering."""
    builder = CardInfoLutBuilder()
    builder.compute()


if __name__ == "__main__":
    cluster()
