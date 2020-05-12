from typing import Dict

from RT_cfr import *
from pluribus.games.short_deck.state import *
from pluribus.poker.card import Card


if __name__ == "__main__":
    utils.random.seed(38)
    public_cards = [Card("ace", "spades"), Card("jack", "spades"), Card("queen", "hearts")]
    # we load a (trained) strategy
    agent1 = TrainedAgent("../blueprint_algo/results_2020_05_10_21_36_47_291425")
    action_sequence = ["raise", "call", "raise"]
    import ipdb
    ipdb.set_trace()
    agent_output = train(agent1.offline_strategy, public_cards, action_sequence, 100, 20, 20, 3)
    import ipdb;
    ipdb.set_trace()
