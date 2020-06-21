import logging
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List
import concurrent.futures

import click
import ipdb
import joblib
import numpy as np
from joblib import Memory
from sklearn.cluster import KMeans
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from poker_ai.poker.card import Card
from poker_ai.poker.deck import get_all_suits
from poker_ai.poker.evaluation import Evaluator


cache_path = "./clustering_cache"
memory = Memory(cache_path, verbose=0)
log = logging.getLogger("poker_ai.clustering.runner")


class GameUtility:
    """This class takes care of some game related functions."""

    def __init__(self, our_hand: np.ndarray, board: np.ndarray, cards: np.ndarray):
        self._evaluator = Evaluator()
        unavailable_cards = np.concatenate([board, our_hand], axis=0)
        self.available_cards = np.array(
            [c for c in cards if c not in unavailable_cards]
        )
        self.our_hand = our_hand
        self.board = board

    def evaluate_hand(self, hand: np.array) -> int:
        """
        Evaluate a hand.

        Parameters
        ----------
        hand : np.array
            Hand to evaluate.

        Returns
        -------
            Evaluation of hand
        """
        return self._evaluator.evaluate(
            board=self.board.astype(np.int).tolist(),
            cards=hand.astype(np.int).tolist(),
        )

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

        Returns
        -------
            Two cards for the opponent (Card)
        """
        return np.random.choice(self.available_cards, 2, replace=False)


class CardCombos:
    """This class stores combinations of cards (histories) per street."""
    def __init__(self, disk_cache: bool = True):
        super().__init__()
        # Setup caching.
        if disk_cache:
            self.get_card_combos = memory.cache(self.get_card_combos)
            self.create_info_combos = memory.cache(self.create_info_combos)
        # Sort for caching.
        suits: List[str] = sorted(list(get_all_suits()))
        ranks: List[int] = sorted(list(range(12, 15)))
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
        num_cards : Number of cards you want returned

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


class CardInfoLutBuilder(CardCombos):
    """
    Stores info buckets for each street when called

    Attributes
    ----------
    card_info_lut : Dict[str, Any]
        Lookup table of card combinations per betting round to a cluster id.
    centroids : Dict[str, Any]
        Centroids per betting round for use in clustering previous rounds by
        earth movers distance.
    """
    def __init__(self, save_dir: str = ""):
        super().__init__()
        self.card_info_lut_path: Path = Path(save_dir) / "card_info_lut.joblib"
        self.centroid_path: Path = Path(save_dir) / "centroids.joblib"
        try:
            self.card_info_lut: Dict[str, Any] = joblib.load(self.card_info_lut_path)
            self.centroids: Dict[str, Any] = joblib.load(self.centroid_path)
        except FileNotFoundError:
            self.centroids: Dict[str, Any] = {}
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
            joblib.dump(self.centroids, self.centroid_path)
        if "turn" not in self.card_info_lut:
            self.card_info_lut["turn"] = self._compute_turn_clusters()
            joblib.dump(self.card_info_lut, self.card_info_lut_path)
            joblib.dump(self.centroids, self.centroid_path)
        if "flop" not in self.card_info_lut:
            self.card_info_lut["flop"] = self._compute_flop_clusters()
            joblib.dump(self.card_info_lut, self.card_info_lut_path)
            joblib.dump(self.centroids, self.centroid_path)
        end = time.time()
        log.info(f"Finished computation of clusters - took {end - start} seconds.")

    def _compute_river_clusters(self):
        """Compute river clusters and create lookup table."""
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
        self.centroids["river"], self._river_clusters = self.cluster(
            num_clusters=50, X=self._river_ehs
        )
        end = time.time()
        log.info(
            f"Finished computation of river clusters - took {end - start} seconds."
        )
        return self.create_card_lookup(self._river_clusters, self.river)

    def _compute_turn_clusters(self):
        """Compute turn clusters and create lookup table."""
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
        self.centroids["turn"], self._turn_clusters = self.cluster(
            num_clusters=50, X=self._turn_ehs_distributions
        )
        end = time.time()
        log.info(f"Finished computation of turn clusters - took {end - start} seconds.")
        return self.create_card_lookup(self._turn_clusters, self.turn)

    def _compute_flop_clusters(self):
        """Compute flop clusters and create lookup table."""
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
        self.centroids["flop"], self._flop_clusters = self.cluster(
            num_clusters=50, X=self._flop_potential_aware_distributions
        )
        end = time.time()
        log.info(f"Finished computation of flop clusters - took {end - start} seconds.")
        return self.create_card_lookup(self._flop_clusters, self.flop)

    @staticmethod
    def simulate_get_ehs(
        game: GameUtility,
        num_simulations: int = 2
    ) -> np.array:
        """
        Get expected hand strength object.

        Parameters
        ----------
        game : GameUtility
            GameState for help with determining winner and sampling opponent hand
        num_simulations : nt
            How many simulations you want to do

        Returns
        -------
        ehs : np.array
            [win_rate, loss_rate, tie_rate]
        """
        ehs: np.ndarray = np.zeros(3)
        for _ in range(num_simulations):
            idx: int = game.get_winner()
            # increment win rate for winner/tie
            ehs[idx] += 1 / num_simulations
        return ehs

    def simulate_get_turn_ehs_distributions(
        self,
        available_cards: np.array,
        the_board: np.array,
        our_hand: np.array,
        num_simulations: int = 2,
    ) -> np.array:
        """
        Get histogram of frequencies that a given turn situation resulted in a
        certain cluster id after a river simulation.

        Parameters
        ----------
        available_cards : np.array
            Array of available cards on the turn
        the_board : np.array
            The board as of the turn
        our_hand : np.array
            Cards our hand (Card)
        num_simulations: int
            Number of simulations

        Returns
        -------
        turn_ehs_distribution : np.array
            Array of counts for each cluster the turn fell into by the river
            after simulations
        """
        turn_ehs_distribution = np.zeros(len(self.centroids["river"]))
        # sample river cards and run a simulation
        for _ in range(num_simulations):
            river_card = np.random.choice(available_cards, 1, replace=False)
            board = np.append(the_board, river_card)
            game = GameUtility(our_hand=our_hand, board=board, cards=self._cards)
            ehs = self.simulate_get_ehs(game)
            # get EMD for expected hand strength against each river centroid
            # to which does it belong?
            for idx, river_centroid in enumerate(self.centroids["river"]):
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
        Get the expected hand strength for a particular card combo.

        Parameters
        ----------
        public : List[float]
            Cards to process

        Returns
        -------
            Expected hand strength
        """
        our_hand = public[:2]
        board = public[2:7]
        # get expected hand strength
        game = GameUtility(our_hand=our_hand, board=board, cards=self._cards)
        return self.simulate_get_ehs(game)

    @staticmethod
    def get_available_cards(
        cards: np.array, unavailable_cards: np.array
    ) -> np.array:
        """
        Get all cards that are available.

        Parameters
        ----------
        cards : np.array
        unavailable_cards : np.array
            Cards that are not available.

        Returns
        -------
            Available cards
        """
        # Turn into set for O(1) lookup speed.
        unavailable_cards = set(unavailable_cards.tolist())
        return np.array([c for c in cards if c not in unavailable_cards])

    def process_turn_ehs_distributions(self, public: List[int]) -> List[float]:
        """
        Get the potential aware turn distribution for a particular card combo.

        Parameters
        ----------
        public : List[float]
            Cards to process

        Returns
        -------
            Potential aware turn distributions
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
        Get the potential aware flop distribution for a particular card combo.

        Parameters
        ----------
        public : List[float]
            Cards to process

        Returns
        -------
            Potential aware flop distributions
        """
        available_cards: np.ndarray = self.get_available_cards(
            cards=self._cards, unavailable_cards=public
        )
        potential_aware_distribution_flop = [0] * len(self.centroids["turn"])
        for j in range(num_simulations):
            # randomly generating turn
            turn_card = np.random.choice(available_cards, 1, replace=False)
            our_hand = public[:2]
            board = public[2:5]
            the_board = np.append(board, turn_card).tolist()
            # getting available cards
            available_cards_turn = [
                x for x in available_cards if x != turn_card[0]
            ]
            turn_ehs_distribution = self.simulate_get_turn_ehs_distributions(
                available_cards_turn, the_board=the_board, our_hand=our_hand
            )
            for idx, turn_centroid in enumerate(self.centroids["turn"]):
                # earth mover distance
                emd = wasserstein_distance(turn_ehs_distribution, turn_centroid)
                if idx == 0:
                    min_idx = idx
                    min_emd = emd
                else:
                    if emd < min_emd:
                        min_idx = idx
                        min_emd = emd
            # Now increment the cluster to which it belongs.
            potential_aware_distribution_flop[min_idx] += 1 / num_simulations
        return potential_aware_distribution_flop

    @staticmethod
    def cluster(num_clusters: int, X: np.array):
        km = KMeans(
            n_clusters=num_clusters,
            init="random",
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0,
        )
        y_km = km.fit_predict(X)
        # Centers to be used for r - 1 (ie; the previous round)
        centroids = km.cluster_centers_
        return centroids, y_km

    @staticmethod
    def create_card_lookup(clusters: np.array, card_combos: np.array) -> Dict:
        """
        Create lookup table.

        Parameters
        ----------
        clusters : np.array
            Array of cluster ids.
        card_combos : np.array
            The card combos to which the cluster ids belong.

        Returns
        -------
        lossy_lookup : Dict
            Lookup table for finding cluster ids.
        """
        log.info("Creating lookup table.")
        lossy_lookup = {}
        for i, card_combo in enumerate(tqdm(card_combos)):
            lossy_lookup[tuple(card_combo)] = clusters[i]
        return lossy_lookup


@click.command()
def cluster():
    """Run clustering."""
    builder = CardInfoLutBuilder()
    builder.compute()


if __name__ == "__main__":
    cluster()
