"""Script to calulate the value of pre-flop hands for n_players

Examples:

Call this file like so to get the help:
```bash
$ python monte_carlo_rank.py --help

Usage: monte_carlo_rank.py [OPTIONS]

  Multithreaded monte carlo pre-flop hand equity calculation.

  Over `n_threads` threads, rank the pre-flop hands according to which  lead
  to winning results. Run for a maximum of `n_iterations` iterations,  this
  should be a big number!

Options:
  --n_threads INTEGER      Number of threads.
  --n_iterations INTEGER   Number of iterations.
  --save_path TEXT         Save path.
  --max_n_players INTEGER  Maximum number of players in a game.
  --n_ranks INTEGER        Number of ranks in a deck of cards.
  --print_n_steps INTEGER  How many steps until print.
  --help                   Show this message and exit.
```

Call this file like so to compute rankings of pre-flop hands:
```bash
python monte_carlo_rank.py --n_threads 4 --max_n_players 6
```
"""
import collections
import pickle
import queue
from typing import Dict, List
from threading import Thread

import click
import numpy as np
from tqdm import tqdm, trange

from pluribus.poker.card import Card
from pluribus.poker.deck import Deck
from pluribus.poker.evaluation import Evaluator
from pluribus.poker.evaluation import EvaluationCard


class PreflopMatrixEvaluator:
    """Generates poker scenarios and returns a matrix of results."""

    def __init__(self, n_ranks=13):
        """Initialise class."""
        self._deck = Deck()
        self._evaluator = Evaluator()
        self._n_ranks = n_ranks

    def __call__(self, n_players: int) -> np.ndarray:
        """Get new delta matrix containing information about which player won."""
        # Compute the
        rank_to_hands = self._compute_preflop_rankings(n_players)
        delta_matrix = self._compute_delta_matrix(rank_to_hands)
        return delta_matrix

    @property
    def n_ranks(self) -> int:
        """Return the number of ranks in a deck of cards."""
        return self._n_ranks

    def _deal_cards(self, n_players: int):
        """Deal a hand for each player and a table of cards.

        We never pop cards from the deck otherwise we'd have to reconstruct the deck
        each iteration which would add the the computational expense.
        """
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
            card_i += 1
        for _ in range(2):
            # "Burn" card.
            card_i += 1
            table.append(self._deck[card_i])
        return hands, table

    def _to_eval_cards(self, cards: List[Card]) -> List[EvaluationCard]:
        """Convert cards representation to one suitable for deuces."""
        return [card.eval_card for card in cards]

    def _compute_preflop_rankings(self, n_players: int) -> Dict[int, List[List[Card]]]:
        """Generate random hands (one for each player) and a table and rank."""
        # Get random hands and table.
        hands, table = self._deal_cards(n_players)
        eval_table = self._to_eval_cards(table)
        # Get ranking of hands.
        ranks_to_hands = collections.defaultdict(list)
        for player_i in range(n_players):
            eval_hand = self._to_eval_cards(hands[player_i])
            rank = self._evaluator.evaluate(cards=eval_hand, board=eval_table)
            ranks_to_hands[rank].append(hands[player_i])
        # The maximum score is the number of unique ranks.
        score = len(set(ranks_to_hands.keys()))
        # Ensure the ranks go from `score`, ..., 3, 2, 1 as rank gets worse.
        normalised_rank_to_hands = collections.defaultdict(list)
        for rank in sorted(ranks_to_hands.keys()):
            for hand in ranks_to_hands[rank]:
                normalised_rank_to_hands[score].append(hand)
            score -= 1
        # Return a dictionary of hands. The keys are int, and relate to how well
        # the hand(s) did. If the hands drew, they will share the same key. The
        # bigger the key, the better the hand.
        return dict(normalised_rank_to_hands)

    def _compute_delta_matrix(self, rank_to_hands: Dict[int, List[List[Card]]]):
        """Write the results to a numpy matrix mirroed along the diagonal."""
        delta_matrix = np.zeros(shape=(self.n_ranks, self.n_ranks))
        for rank, hands in rank_to_hands.items():
            for hand in hands:
                # Get the range to go between 0 and 13.
                rank_0 = hand[0].rank_int - 2
                rank_1 = hand[1].rank_int - 2
                # Write rank to zeroed matrix.
                delta_matrix[rank_0, rank_1] = rank
                delta_matrix[rank_1, rank_0] = rank
        return delta_matrix


def delta_matrix_worker(
    n_ranks: int,
    min_n_players: int,
    max_n_players: int,
    sentinal_queue: queue.Queue,
    delta_matrix_queue: queue.Queue,
):
    """Generate delta matrix and push to queue."""
    evaluator = PreflopMatrixEvaluator(n_ranks=n_ranks)
    while True:
        try:
            sentinal = sentinal_queue.get(block=False)
            if sentinal == "terminate":
                break
        except queue.Empty:
            # Handle empty queue here
            pass
        # Read from sentinal queue, if any sentinals then quit!
        for n_players in range(min_n_players, max_n_players):
            delta_matrix = evaluator(n_players=n_players)
            delta_matrix_queue.put(dict(
                n_players=n_players,
                delta_matrix=delta_matrix
            ))


@click.command()
@click.option(
    '--n_threads', default=1, help='Number of threads.'
)
@click.option(
    '--n_iterations', default=10000000, help='Number of iterations.'
)
@click.option(
    '--save_path', default="./results.pickle", help='Save path.'
)
@click.option(
    '--max_n_players', default=6, help='Maximum number of players in a game.'
)
@click.option(
    '--n_ranks', default=13, help='Number of ranks in a deck of cards.'
)
@click.option(
    '--print_n_steps', default=10000, help='How many steps until print.'
)
def multithreaded_matrix_summation(
    n_threads: int,
    n_iterations: int,
    save_path: str,
    max_n_players: int = 6,
    n_ranks: int = 13,
    print_n_steps: int = 10000,
):
    """Multithreaded monte carlo pre-flop hand equity calculation.

    Over `n_threads` threads, rank the pre-flop hands according to which
    lead to winning results. Run for a maximum of `n_iterations` iterations,
    this should be a big number!
    """
    # How we communicate to our workers.
    delta_matrix_queue = queue.Queue()
    sentinal_queue = queue.Queue()
    threads = []
    min_n_players = 2
    args = (n_ranks, min_n_players, max_n_players + 1, sentinal_queue, delta_matrix_queue)
    for _ in range(n_threads):
        thread = Thread(target=delta_matrix_worker, args=args)
        threads.append(thread)
        thread.start()
    # Create a matrix of rankings for each hand for every number of
    # players.
    matrices = {
        n_players: np.zeros(shape=(n_ranks, n_ranks))
        for n_players in range(min_n_players, max_n_players + 1)
    }
    np.set_printoptions(precision=2)
    try:
        for i in trange(n_iterations):
            result = delta_matrix_queue.get()
            n_players = result["n_players"]
            matrices[n_players] += result["delta_matrix"]
            if i > 0 and i % print_n_steps == 0:
                tqdm.write(f"\nStep {i} reached.")
                for n_players, matrix in matrices.items():
                    tqdm.write(
                        f"> For {n_players} players, the normalised matrix "
                        f"looks like:\n{matrix / np.max(matrix)}"
                    )
    except KeyboardInterrupt:
        print(f"Control-c detected, quitting looping at iteration {i}.")
        pass
    # Save the rankings.
    with open(save_path, 'wb') as f:
        pickle.dump(matrices, f, pickle.HIGHEST_PROTOCOL)
    print(f"Saved matrix to {save_path}")
    for thread in threads:
        sentinal_queue.put("terminate")
        thread.join()
    print("Program terminated.")


if __name__ == "__main__":
    multithreaded_matrix_summation()
