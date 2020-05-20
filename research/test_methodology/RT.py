from typing import List

import dill as pickle

from RT_cfr import train
from pluribus.games.short_deck.agent import TrainedAgent
from pluribus.poker.card import Card


if __name__ == "__main__":
    # public_cards = [Card("ace", "spades"), Card("queen", "spades"),
    #   Card("queen", "hearts")]
    public_cards: List[Card] = []
    # we load a (trained) strategy
    agent1 = TrainedAgent("../blueprint_algo/results_2020_05_10_21_36_47_291425")
    # the better strategy
    # offline_strategy = joblib.load('/Users/colin/Downloads/offline_strategy_285800.gz')
    action_sequence = ["raise", "call", "call", "call", "call"]
    agent_output = train(
        agent1.offline_strategy, public_cards, action_sequence, 40, 6, 6, 3, 2, 6
    )
    with open("testing2.pkl", "wb") as file:
        pickle.dump(agent_output, file)
    import ipdb
    ipdb.set_trace()
