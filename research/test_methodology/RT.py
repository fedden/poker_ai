from typing import Dict
import joblib
import sys

import dill as pickle

from RT_cfr import *
from pluribus.games.short_deck.state import *
from pluribus.poker.card import Card


if __name__ == "__main__":
    public_cards = [Card("ace", "spades"), Card("queen", "spades"), Card("queen", "hearts")]
    # we load a (trained) strategy
    #agent1 = TrainedAgent("../blueprint_algo/results_2020_05_10_21_36_47_291425")
    # sorta hacky, but I loaded the average strategy above, now I'm replacing with
    # the better strategy
    offline_strategy = joblib.load('/Users/colin/Downloads/offline_strategy_285800.gz')
    print(sys.getsizeof(offline_strategy))
    # agent1.offline_strategy = offline_strategy
    # print(sys.getsizeof(agent1.offline_strategy))
    action_sequence = ["raise", "call", "raise", "call", "call", "call"]
    agent_output = train(offline_strategy, public_cards, action_sequence, 2400, 900, 900, 3, 2, 900)  # TODO: back to 50
    with open('realtime-strategy-off-better-1500.pkl', 'wb') as file:
        pickle.dump(agent_output, file)
    import ipdb;
    ipdb.set_trace()
