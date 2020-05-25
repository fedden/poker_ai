from typing import List
import joblib

from RT_cfr import rts
from pluribus.poker.card import Card


if __name__ == "__main__":
    # We can set public cards or not
    public_cards = [Card("king", "spades"), Card("queen", "clubs"),
                    Card("jack", "hearts")]
    # public_cards: List[Card] = []
    # Action sequence must be in old form (one list, includes skips)
    action_sequence = ["raise", "call", "call", "call"]
    agent_output, offline_strategy = rts(
        'test_strategy/unnormalized_output/offline_strategy_100.gz',
        'test_strategy/strategy_100.gz', public_cards, action_sequence,
        140, 6, 6, 3, 2, 6, 10
    )
    save_path = "test_strategy/unnormalized_output/"
    last_regret = {info_set: dict(strategy) for info_set, strategy in agent_output.regret.items()}
    joblib.dump(offline_strategy, save_path + 'rts_output.gz', compress="gzip")
    joblib.dump(last_regret, save_path + 'last_regret.gz', compress="gzip")
    import ipdb;
    ipdb.set_trace()
