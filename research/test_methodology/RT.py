from typing import List
import joblib

import dill as pickle

from RT_cfr import train
from pluribus.poker.card import Card
from pluribus.games.short_deck.agent import Agent


if __name__ == "__main__":
    # public_cards = [Card("ace", "spades"), Card("queen", "spades"),
    #   Card("queen", "hearts")]
    public_cards: List[Card] = []
    # we load a (trained) strategy
    agent = Agent(regret_dir='test_strategy/strategy_100.gz')
    action_sequence = ["raise", "call", "call", "call", "call"]
    agent_output, offline_strategy = train(
        'test_strategy/unnormalized_output/offline_strategy_100.gz',
        'test_strategy/strategy_100.gz', public_cards, action_sequence,
        40, 6, 6, 3, 2, 6, 10
    )
    with open("testing2.pkl", "wb") as file:
        pickle.dump(agent_output, file)
    import ipdb
    ipdb.set_trace()
