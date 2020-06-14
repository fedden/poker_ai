import numpy as np
import json
import joblib
import sys
from typing import List

import click

from agent_test import agent_test
from pluribus.poker.deck import Deck
sys.path.append('research/rts')
from RT_cfr import rts


@click.command()
@click.option("--offline_strategy_path", help=".")
@click.option("--last_regret_path", help=".")
@click.option("--n_iterations", default=1500, help=".")
@click.option("--lcfr_threshold", default=400, help=".")
@click.option("--discount_interval", default=400, help=".")
@click.option("--n_players", default=3, help=".")
@click.option("--update_interval", default=400, help=".")
@click.option("--update_threshold", default=400, help=".")
@click.option("--dump_int", default=20, help=".")
@click.option("--save_dir", help=".")
@click.option("--n_inner_iters", default=25, help=".")
@click.option("--n_outer_iters", default=150, help=".")
def rts_ab_test(
    offline_strategy_path: str,
    last_regret_path: str,
    n_iterations: int,
    lcfr_threshold: int,
    discount_interval: int,
    n_players: int,
    update_interval: int,
    update_threshold: int,
    dump_int: int,
    save_dir: str,
    n_inner_iters: int,
    n_outer_iters: int,
    ranks: List[int] = list(range(10, 14 + 1)),
):
    check = joblib.load(offline_strategy_path)
    histories = np.random.choice(list(check.keys()), 2)
    action_sequences = []
    public_cards_lst = []
    community_card_dict = {
        "pre_flop": 0,
        "flop": 3,
        "turn": 4,
        "river": 5,
    }
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
        if action_sequences:
            final_betting_round = list(betting_rounds)[-1]
        else:
            final_betting_round = "pre_flop"
        n_cards = community_card_dict[final_betting_round]
        cards_in_deck = deck._cards_in_deck
        public_cards = list(
            np.random.choice(cards_in_deck, n_cards)
        )
        public_cards_lst.append(public_cards)

    for i in range(0, len(action_sequences)):
        public_cards = public_cards_lst[i].copy()
        action_sequence = action_sequences[i].copy()
        agent_output, offline_strategy = rts(
            offline_strategy_path,
            last_regret_path,
            public_cards,
            action_sequence,
            n_iterations=n_iterations,
            lcfr_threshold=lcfr_threshold,
            discount_interval=discount_interval,
            n_players=n_players,
            update_interval=update_interval,
            update_threshold=update_threshold,
            dump_int=dump_int
        )
        last_regret = {
            info_set: dict(strategy)
            for info_set, strategy in agent_output.regret.items()
        }
        joblib.dump(offline_strategy, save_dir + f'rts_output{i}.gz', compress="gzip")
        joblib.dump(last_regret, save_dir + f'last_regret{i}.gz', compress="gzip")

        public_cards = public_cards_lst[i].copy()
        action_sequence = action_sequences[i].copy()
        agent_test(
            hero_strategy_path=save_dir + f"rts_output{i}.gz",
            opponent_strategy_path=offline_strategy_path,
            real_time_est=True,
            public_cards=public_cards,
            action_sequence=action_sequence,
            n_inner_iters=n_inner_iters,
            n_outer_iters=n_outer_iters,
            hero_count=0,
            hero_total_count=0,
        )

if __name__ == "__main__":
    rts_ab_test()
