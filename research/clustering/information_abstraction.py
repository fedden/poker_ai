from pluribus.game.deck import Deck
from pluribus.game.evaluation import Evaluator

import numpy as np
from itertools import combinations
import random
from sklearn.cluster import KMeans
from typing import List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import wasserstein_distance


class ShortDeck(Deck):
    """
    Extends Deck - smaller Deck based on the number of cards requested
    # TODO: maybe I should just use _cards rather than _evals?
    """
    def __init__(self, num_cards: int):

        super().__init__()

        self.shuffle()

        self._cards = self._cards[:num_cards]

        self._evals = [c.eval_card for c in self._cards]

        self._evals_to_cards = {i.eval_card: i for i in self._cards}

    def get_card_combos(self, num_cards: int) -> np.array:
        """

        :param num_cards: number of cards you want returned
        :return: combos of cards (Card.eval_card) -> np.array
        """
        return np.asarray(list(combinations(self._evals, num_cards)))


class GameUtility:
    """
    Using this to take care of game related functions
    """
    def __init__(self, our_hand: List[int], board: List[int], cards: List[int]):

        self._evaluator = Evaluator()

        # TODO: this is what takes forever I think, find a better way
        self.available_cards = [x for x in cards if x not in board + our_hand]

        self.our_hand = our_hand

        self.board = board

    def evaluate_hand(self, hand: List[int]) -> int:
        """
        takes a hand
        :param hand: two integers Card.eval_card
        :return: evaluation of hand
        """
        return self._evaluator.evaluate(
            board=self.board,
            cards=hand
        )

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
    def __init__(self, num_cards):
        super().__init__(num_cards)

        self.starting_hands = self.get_card_combos(2)
        self.flop = self.create_info_combos(self.starting_hands, self.get_card_combos(3))
        self.turn = self.create_info_combos(self.flop, self.get_card_combos(1))
        self.river = self.create_info_combos(self.turn, self.get_card_combos(1))

    @staticmethod
    def create_info_combos(start_combos: np.array, publics: np.array) -> np.array:
        """
        Combinations of private information (hole cards) and public information (board)
        Uses the logic that a AsKsJs on flop with a 10s on turn is different than AsKs10s on flop and Js on turn

        :param start_combos: starting combination of cards (beginning with hole cards)
        :param publics: np.array of public combinations being added
        :return: Combinations of private information (hole cards) and public information (board) -> np.array
        """
        our_cards = []
        for combos in start_combos:
            for public_combo in publics:
                # TODO: maybe a way to create these combos with better performance?
                if not np.any(np.isin(combos, public_combo)):
                    our_cards.append(np.concatenate((combos, public_combo)))
        return np.array(our_cards)


class InfoBucketMaker(InfoSets):
    def __init__(self, num_cards):
        super().__init__(num_cards)

        self._river_ehs = self.get_river_ehs(num_print=1000)
        self._river_centroids, self._river_clusters = self.cluster(num_clusters=15, X=self._river_ehs)
        self._turn_ehs_distributions = self.get_turn_turn_ehs_distributions(num_print=100)
        self._turn_centroids, self._turn_clusters = self.cluster(num_clusters=15, X=self._turn_ehs_distributions)

    def __call__(self):
        # TODO: Add a dump to pkl function
        # add a print hands function to inspect the cluster performance
        self.plot_river_clusters()

    @staticmethod
    def simulate_get_ehs(game: GameUtility, num_simulations: int) -> List[float]:
        ehs = [0] * 3
        for _ in range(num_simulations):  # probably want to increase this number?

            idx = game.get_winner()

            # increment win rate for winner/tie
            ehs[idx] += 1 / num_simulations

        return ehs

    def get_river_ehs(self, num_print: int):
        """

        :param num_print: number of simulations of opponents cards for calculating ehs
        :return: ehss -> np.array of arrays containing [win_rate, loss_rate, tie_rate]
        """
        river_ehs = [0] * len(self.river)
        for i, public in enumerate(self.river):
            our_hand = list(public[:2])
            board = list(public[2:7])

            game = GameUtility(our_hand=our_hand, board=board, cards=self._evals)
            river_ehs[i] = self.simulate_get_ehs(game, 5)

            if i % num_print == 0:
                print(f"Finding River Expected Hand Strength, iteration {i} of {len(self.river)}")
        return np.array(river_ehs)

    def get_turn_turn_ehs_distributions(self, num_print: int):

        # TODO: pull the inner loop out for use in the flop as well
        turn_ehs_distributions = [0] * len(self.turn)
        for i, public in enumerate(self.turn):
            available_cards = [x for x in self._evals if x not in public]  # this is probably not a good idea
            random.shuffle(available_cards)

            # sample river cards and run a simulation
            ehs_distribution = np.zeros(len(self._river_centroids))
            for j in range(15):
                # probably want to increase this number?
                # it's too small maybe for this toy problem

                river_card = random.sample(available_cards, 1)
                our_hand = list(public[:2])
                board = public[2:6]
                board = np.append(board, river_card).tolist()
                # if sample with river then error (because obvi)

                game = GameUtility(our_hand=our_hand, board=board, cards=self._evals)
                ehs = self.simulate_get_ehs(game, 5)

                # get EMD for expected hand strength against each river centroid
                # to which does it belong?
                for idx, river_centroid in enumerate(self._river_centroids):
                    emd = wasserstein_distance(ehs, river_centroid)

                    if idx == 0:
                        min_idx = idx
                        min_emd = emd
                    else:
                        if emd < min_emd:
                            min_idx = idx
                            min_emd = emd

                # ok, now increment the cluster to which it belongs -
                ehs_distribution[min_idx] += 1 / 15  # could also probs be just a regular old integer

            turn_ehs_distributions.append(ehs_distribution)
            if i % num_print == 0:
                print(f"Finding Turn Distribution Aware Histogram, iteration {i} of {len(self.turn)}")

        return turn_ehs_distributions

    # TODO: add def get_turn_flop_ehs_distributions(self, num_print: int):

    @staticmethod
    def cluster(num_clusters: int, X: np.array):
        # simple kmeans algo - should I write from scratch?
        # you might need to adjust the centers number since the deck is being shuffled if
        # you run it from the top, but you can also get the data out of the data folder
        # no more than 20 clusters or graph won't work below, FYI

        km = KMeans(
            n_clusters=num_clusters, init='random',  # would be 200 in our example
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        y_km = km.fit_predict(X)  # simple kmeans algo - should I write from scratch?

        # centers to be used to get data for EMD
        centroids = km.cluster_centers_

        return centroids, y_km

    def plot_river_clusters(self):
        colors = {
            0: 'y',
            1: 'b',
            2: 'g',
            3: 'r',
            4: 'c',
            5: 'm',
            6: 'y',
            7: 'b',
            8: 'w',
            9: '#7A68A6',
            10: '#FFB5B8',
            11: '#fdb462',
            12: '#8b8b8b',
            13: '#bc82bd',
            14: '#8EBA42',
            15: '#467821',
            16: '#fdb462',
            17: '#8d67a8',
            18: '#cbcbcb',
            19: '#b3de69',
            20: '#0a0a0a'
        }

        X = self._river_ehs
        y_km = self._river_clusters
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
            # plot the centroids
            for i in range(len(self._river_centroids)):
                ax.scatter(
                    X[y_km == i, 0], X[y_km == i, 2], X[y_km == i, 1],
                    s=20, marker='o',
                    c=colors[i])

        ax.set_zlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)

        ax.set_xlabel('Win Rate')
        ax.set_ylabel('Tie Rate')
        ax.set_zlabel('Loss Rate')

        plt.show()


if __name__ == "__main__":
    info_bucket = InfoBucketMaker(10)
    info_bucket()

