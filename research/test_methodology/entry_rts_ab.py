import numpy as np
import json
import joblib

from RT_cfr import rts
from bot_test import agent_test
from pluribus.poker.deck import Deck


check = joblib.load('test_strategy2/unnormalized_output/offline_strategy_1500.gz')
histories = np.random.choice(list(check.keys()), 2)
action_sequences = []
public_cards_lst = []
community_card_dict = {
    "pre_flop": 0,
    "flop": 3,
    "turn": 4,
    "river": 5,
}
ranks = list(range(10, 14 + 1))
deck = Deck(include_ranks=ranks)
for history in histories:
    history_dict = json.loads(history)
    history_lst = history_dict['history']
    action_sequence = []
    betting_rounds = []
    for x in history_lst:
        action_sequence += list(x.values())[0]
        betting_rounds += list(x.keys())
    action_sequences.append(action_sequence)
    final_betting_round = list(betting_rounds)[-1]
    n_cards = community_card_dict[final_betting_round]
    cards_in_deck = deck._cards_in_deck
    public_cards = np.random.choice(cards_in_deck, n_cards)
    public_cards_lst.append(list(public_cards))

for i in range(0, len(action_sequences)):
#    try:
    public_cards = public_cards_lst[i].copy()
    action_sequence = action_sequences[i].copy()
    agent_output, offline_strategy = rts(
        'test_strategy2/unnormalized_output/offline_strategy_1500.gz',
        'test_strategy2/strategy_1500.gz', public_cards, action_sequence,
        1500, 400, 400, 3, 400, 400, 20
    )
    save_path = "test_strategy2/unnormalized_output/"
    last_regret = {info_set: dict(strategy) for info_set, strategy in agent_output.regret.items()}
    joblib.dump(offline_strategy, save_path + f'rts_output{i+35}.gz', compress="gzip")
    joblib.dump(last_regret, save_path + f'last_regret{i+35}.gz', compress="gzip")

    public_cards = public_cards_lst[i].copy()
    action_sequence = action_sequences[i].copy()
    strat_path = "test_strategy2/unnormalized_output/"
    agent_test(
        hero_strategy_path=strat_path + f"rts_output{i+35}.gz",
        opponent_strategy_path=strat_path + "offline_strategy_1500.gz",
        real_time_est=True,
        public_cards=public_cards,
        action_sequence=action_sequence,
        n_inner_iters=25,
        n_outter_iters=250,
        hero_count=0,
        hero_total_count=0,
    )
#    except:
#        print(f"ERROR on {action_sequences[i]}, {public_cards_lst[i]}")
#        pass
