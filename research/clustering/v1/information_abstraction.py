"""
Script to get information abstraction buckets for flop, turn, river
(the pre-flop information buckets are just the 169 lossless hands)

Important Run Notes
--Cd into research/clustering, the program will try to output to data/information_abstraction.py
----If you are not that directory the program will fail
--Run with `python information_abstraction.py`
--Budget an hour to run with a 10 card deck, you may want to cmd + F "num_simulations" and reduce the defaults to test

This is a naive implementation of https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8459/8487

Notes on running for deck size 20 on MacBook Pro with 16 GB RAM
- Creating combinations is relatively quick, I decided to do reduced combination space (considering AsKsJs|Qs
the same as AsKsJs|Qs) - not sure how this will affect equilibrium finding, but should work ok for a "toy" product
at first
- FLOP: 155040 combos (20C2 * 18C3), runtime ~6 hrs, dict from flop_lossy.pkl .02GB
- TURN: 581400 combos (20C2 * 18C4), runtime ~10 hrs, dict from turn_lossy.pkl .005 GB
- RIVER: 1627920 combos (20C2 * 18C5), runtime ~12 hrs, dict from river_lossy.pkl .08 GB

river ehs, from information_abstraction.pkl: '_flop_potential_aware_distributions': .04GB
flop potential aware dist, from information_abstraction.pkl: '_turn_ehs_distributions':.06GB
turn ehs distributions, from information_abstraction.pkl: 'river_ehs': 0.23256

All in for 28 hrs, will need to work on some improvements for clustering 52 card deck..

Next Steps/Future Enhancements
- Try rolling out to full short deck (36 cards) using multi-processing
- Implement isomorphisms to canonicalize hands (estimated 24x reduction)
- Switch to non-naive implementation where vectors are tuples of (index,weight) or use sparse representation
- Switch to https://www.cs.cmu.edu/~sandholm/hierarchical.aamas15.pdf for parallelization of blueprint algo (?)
-- This will make 52 card game combos tractable as well
- Split up output objects in order to keep less in memory
- Hard Code opponent clusters and us OHS instead of EHS: http://www.ifaamas.org/Proceedings/aamas2013/docs/p271.pdf
- Adjust cluster sizes to ~200 with 52 card game
- If we decide to go with this algo, we might consider the optimization for estimating EMD:
--https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8459/8487
"""
import random
import time
from itertools import combinations
from typing import List

import dill as pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from pluribus.poker.card import Card
from pluribus.poker.deck import get_all_suits
from pluribus.poker.evaluation import Evaluator


class ShortDeck:
    """
    Extends Deck - A smaller Deck based on the number of cards requested
    --not sure how well it extends beyond 10 atm
    TODO: maybe I should just use _cards rather than _evals? but, _evals directly might have better performance?

    """

    def __init__(self):
        super().__init__()

        self._cards = [
            Card(rank, suit) for suit in get_all_suits() for rank in range(10, 15)
        ]  # hardcoding removal of 2-9
        self._evals = [c.eval_card for c in self._cards]
        self._evals_to_cards = {i.eval_card: i for i in self._cards}

    def get_card_combos(self, num_cards: int) -> np.ndarray:
        """

        :param num_cards: number of cards you want returned
        :return: combos of cards (Card.eval_card) -> np.array
        """
        return np.asarray(list(combinations(self._evals, num_cards)))


class GameUtility:
    """
    This class takes care of some game related functions
    """

    def __init__(self, our_hand: List[int], board: List[int], cards: List[int]):

        self._evaluator = Evaluator()
        # TODO: this is what takes forever, find a better way
        self.available_cards = [x for x in cards if x not in board + our_hand]
        self.our_hand = our_hand
        self.board = board

    def evaluate_hand(self, hand: List[int]) -> int:
        """
        takes a hand
        :param hand: list of two integers Card.eval_card
        :return: evaluation of hand
        """
        return self._evaluator.evaluate(board=self.board, cards=hand)

    def get_winner(self) -> int:
        """

        :return: int of win (0), lose (1) or tie (2) - this is an index in the expected hand strength array
        """
        our_hand_rank = self.evaluate_hand(self.our_hand)
        opp_hand_rank = self.evaluate_hand(self.opp_hand)
        if our_hand_rank > opp_hand_rank:  # maybe some mod magic here
            return 0
        elif our_hand_rank < opp_hand_rank:
            return 1
        elif our_hand_rank == opp_hand_rank:
            return 2

    @property
    def opp_hand(self) -> List[int]:
        """

        :return: two cards for the opponent (Card.eval_card)
        """
        return random.sample(self.available_cards, 2)


class InfoSets(ShortDeck):
    """
    This class stores combinations of cards (histories) per street (for flop, turn, river)
    # TODO: should this be isomorphic/lossless to reduce the program run time?
    """

    def __init__(self):
        super().__init__()

        self.starting_hands = self.get_card_combos(2)
        self.flop = self.create_info_combos(
            self.starting_hands, self.get_card_combos(3)
        )
        self.turn = self.create_info_combos(self.starting_hands, self.get_card_combos(4))  # will this work??
        self.river = self.create_info_combos(self.starting_hands, self.get_card_combos(5))  # will this work??

    @staticmethod
    def create_info_combos(start_combos: np.array, publics: np.array) -> np.ndarray:
        """
        Combinations of private information (hole cards) and public information (board)
        Uses the logic that a AsKsJs on flop with a 10s on turn is different than AsKs10s on flop and Js on turn
        That logic is used within the literature

        :param start_combos: starting combination of cards (beginning with hole cards)
        :param publics: np.array of public combinations being added
        :return: Combinations of private information (hole cards) and public information (board)
        """
        our_cards = []
        for combos in tqdm(start_combos):
            for public_combo in publics:
                # TODO: need a way to create these combos with better performance?
                if not np.any(np.isin(combos, public_combo)):
                    our_cards.append(np.concatenate((combos, public_combo)))
        return np.array(our_cards)


class InfoBucketMaker(InfoSets):
    """
    Stores info buckets for each street when called
    # TODO: create key to access these from a dictionary, store more efficiently somehow
    # TODO: change cluster to num_clusters=200 for full deck
    """

    def __init__(self):
        super().__init__()

        overarching_start = time.time()
        start = time.time()
        self._river_ehs = self.get_river_ehs(num_print=1000)
        self._river_centroids, self._river_clusters = self.cluster(
            num_clusters=50, X=self._river_ehs
        )
        end = time.time()
        print(f"Finding River EHS Took {end - start} Seconds")

        start = time.time()
        self._turn_ehs_distributions = self.get_turn_ehs_distributions(num_print=100)
        self._turn_centroids, self._turn_clusters = self.cluster(
            num_clusters=50, X=self._turn_ehs_distributions
        )
        end = time.time()
        print(f"Finding Turn EHS Distributions Took {end - start} Seconds")

        start = time.time()
        self._flop_potential_aware_distributions = self.get_flop_potential_aware_distributions(
            num_print=100
        )
        self._flop_centroids, self._flop_clusters = self.cluster(
            num_clusters=50, X=self._flop_potential_aware_distributions
        )
        end = time.time()
        print(f"Finding Flop Potential Aware Distributions Took {end - start} Seconds")
        overarching_end = time.time()

        print(f"Whole Process Took {overarching_end - overarching_start} Seconds")

    def __call__(self):
        # TODO: switch to log
        self.dump_data(location="data/information_abstraction.pkl")
        self.print_cluster_example(
            X=self._river_ehs,
            clusters=self._river_clusters,
            cluster_name="Expected Hand Strength on River",
            cluster_id=4,
        )
        self.print_cluster_example(
            X=self._turn_ehs_distributions,
            clusters=self._turn_clusters,
            cluster_name="Expected Hand Strength Distribution on Turn",
            cluster_id=4,
        )
        self.print_cluster_example(
            X=self._turn_ehs_distributions,
            clusters=self._turn_clusters,
            cluster_name="Potential Aware Distribution on Flop",
            cluster_id=4,
        )
        self.plot_river_clusters()

    @staticmethod
    def simulate_get_ehs(game: GameUtility, num_simulations: int = 10) -> List[float]:
        """
        # TODO: probably want to increase simulations..
        :param game: GameState for help with determining winner and sampling opponent hand
        :param num_simulations: how many simulations you want to do
        :return: [win_rate, loss_rate, tie_rate]
        """
        ehs = [0] * 3
        for _ in range(num_simulations):

            idx = game.get_winner()

            # increment win rate for winner/tie
            ehs[idx] += 1 / num_simulations

        return ehs

    def simulate_get_turn_ehs_distributions(
        self,
        available_cards: List[int],
        the_board: List[int],
        our_hand: List[int],
        num_simulations: int = 5,
    ) -> np.array:
        """
        # TODO num_simulations should be higher

        :param available_cards: list of available cards on the turn
        :param the_board: the board as of the turn
        :param our_hand: cards our hand (Card.eval_card)
        :param num_simulations: int of simulations
        :return: array of counts for each cluster the turn fell into by the river after simulations
        """
        turn_ehs_distribution = [0] * len(self._river_centroids)

        # sample river cards and run a simulation
        for _ in range(num_simulations):

            river_card = random.sample(available_cards, 1)
            board = list(the_board)  # copy list
            board = board + river_card

            game = GameUtility(our_hand=our_hand, board=board, cards=self._evals)
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

    def get_river_ehs(self, num_print: int) -> np.ndarray:
        """

        :param num_print: number of simulations of opponents cards for calculating ehs
        :return: np.ndarray of arrays containing [win_rate, loss_rate, tie_rate]
        """
        start = time.time()
        river_ehs = [0] * len(self.river)

        # iterate over possible boards/hole cards
        for i, public in enumerate(tqdm(self.river)):

            our_hand = list(public[:2])
            board = list(public[2:7])

            # get expected hand strength
            game = GameUtility(our_hand=our_hand, board=board, cards=self._evals)
            river_ehs[i] = self.simulate_get_ehs(game)

            if i % num_print == 0:
                tqdm.write(
                    f"Finding River Expected Hand Strength, iteration {i} of {len(self.river)}"
                )
        end = time.time()
        print(f"Finding River Expected Hand Strength Took {end - start} Seconds")
        return np.array(river_ehs)

    def get_turn_ehs_distributions(self, num_print: int) -> np.ndarray:
        """

        :param num_print: frequency at which to print
        :return: np.ndarray of distribution aware turn distributions
        """
        start = time.time()
        turn_ehs_distributions = [0] * len(self.turn)

        for i, public in enumerate(tqdm(self.turn)):
            available_cards = [
                x
                for x in self._evals
                if x not in public  # TODO need better implementation of this
            ]

            # sample river cards and run a simulation
            turn_ehs_distribution = self.simulate_get_turn_ehs_distributions(
                available_cards, the_board=list(public[2:6]), our_hand=list(public[:2])
            )

            turn_ehs_distributions[i] = turn_ehs_distribution
            if i % num_print == 0:
                tqdm.write(
                    f"Finding Turn Distribution Aware Histograms, iteration {i} of {len(self.turn)}"
                )
        end = time.time()
        print(f"Finding Turn Distribution Aware Histograms Took {end - start} Seconds")
        return np.array(turn_ehs_distributions)

    def get_flop_potential_aware_distributions(
        self, num_print: int, num_simulations: int = 5
    ) -> np.ndarray:
        """

        :param num_print: frequency at which to print
        :param num_simulations: number of simulations
        :return: ndarray of potential aware histograms
        """
        start = time.time()
        potential_aware_distribution_flops = [0] * len(self.flop)

        for i, public in enumerate(tqdm(self.flop)):
            available_cards = [
                x for x in self._evals if x not in public
            ]  # TODO: find better implementation of this

            potential_aware_distribution_flop = [0] * len(self._turn_centroids)
            for j in range(num_simulations):

                # randomly generating turn
                turn_card = random.sample(available_cards, 1)

                our_hand = list(public[:2])
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
            potential_aware_distribution_flops[i] = potential_aware_distribution_flop
            if i % num_print == 0:
                tqdm.write(
                    f"Finding Flop Potential Aware Histogram, iteration {i} of {len(self.flop)}"
                )
        end = time.time()
        print(f"Finding Flop Potential Aware Distributions Took {end - start} Seconds")
        return np.array(potential_aware_distribution_flops)

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

    @staticmethod
    def print_cluster_example(
        X: np.ndarray, clusters: np.ndarray, cluster_name: str, cluster_id: int = 4
    ):
        """

        :param X
        :param clusters: cluster centroids
        :param cluster_name: string to be printed
        :param cluster_id: id to look at (just an example - can inspect dumped object)
        :return: just prints
        """
        print(f"####{cluster_name} Example for Cluster Number {str(cluster_id)}:")
        print(X[clusters == cluster_id])

    def plot_river_clusters(self):
        """

        :return: plot of river ehs - colors represent different clusters
        """
        colors = {
            0: "y",
            1: "b",
            2: "g",
            3: "r",
            4: "c",
            5: "m",
            6: "y",
            7: "b",
            8: "w",
            9: "#7A68A6",
            10: "#FFB5B8",
            11: "#fdb462",
            12: "#8b8b8b",
            13: "#bc82bd",
            14: "#8EBA42",
            15: "#467821",
            16: "#fdb462",
            17: "#8d67a8",
            18: "#cbcbcb",
            19: "#b3de69",
            20: "#0a0a0a",
        }

        X = self._river_ehs
        y_km = self._river_clusters
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for m, zlow, zhigh in [("o", -50, -25), ("^", -30, -5)]:
            # plot the centroids
            for i in range(len(self._river_centroids)):
                ax.scatter(
                    X[y_km == i, 0],
                    X[y_km == i, 2],
                    X[y_km == i, 1],
                    s=20,
                    marker="o",
                    c=colors[i],
                )

        ax.set_zlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)

        ax.set_xlabel("Win Rate")
        ax.set_ylabel("Tie Rate")
        ax.set_zlabel("Loss Rate")

        plt.show()

    def dump_data(self, location: str = "data/information_abstraction_3.pkl"):
        """
        Should be in research/clustering or it will fail
        :param location: string for location and file name off the data
        :return: dumps object
        """
        with open(location, "wb") as file:
            pickle.dump(self.__dict__, file)
        print(f"Dumped Data to {location}")


if __name__ == "__main__":
    info_bucket = InfoBucketMaker()
    info_bucket()
