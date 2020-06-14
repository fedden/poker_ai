from typing import List
import joblib

from RT_cfr import rts
from pluribus.poker.card import Card


if __name__ == "__main__":
    # We can set public cards or not
    public_cards = [Card("ace", "diamonds"), Card("king", "clubs"),
                    Card("jack", "spades"), Card("10", "hearts"),
                    Card("10", "spades")]
    # Action sequence must be in old form (one list, includes skips)
    action_sequence = ["raise", "raise", "raise", "call", "call",
                       "raise", "raise", "raise", "call", "call",
                       "raise", "raise", "raise", "call", "call", "call"]
    agent_output, offline_strategy = rts(
        'test_strategy2/unnormalized_output/offline_strategy_1500.gz',
        'test_strategy2/strategy_1500.gz', public_cards, action_sequence,
        1400, 1, 1, 3, 1, 1, 20
    )
    save_path = "test_strategy2/unnormalized_output/"
    last_regret = {
        info_set: dict(strategy)
        for info_set, strategy in agent_output.regret.items()
    }
    joblib.dump(offline_strategy, save_path + 'rts_output.gz', compress="gzip")
    joblib.dump(last_regret, save_path + 'last_regret.gz', compress="gzip")
    import ipdb;
    ipdb.set_trace()
