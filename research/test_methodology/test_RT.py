from typing import List, Dict, DefaultDict
import joblib
import collections

from tqdm import trange
import numpy as np
from scipy import stats

from pluribus.games.short_deck.state import ShortDeckPokerState, new_game
from pluribus.poker.card import Card


def _calculate_strategy(
        state: ShortDeckPokerState,
        I: str,
        strategy: DefaultDict[str, DefaultDict[str, float]]
) -> str:
    sigma = collections.defaultdict(lambda: collections.defaultdict(lambda: 1 / 3))
    try:
        # If strategy is empty, go to other block
        if sigma[I] == {}:
            raise KeyError
        sigma[I] = strategy[I].copy()
        norm = sum(sigma[I].values())
        for a in sigma[I].keys():
            sigma[I][a] /= norm
        a = np.random.choice(
            list(sigma[I].keys()), 1, p=list(sigma[I].values()),
        )[0]
    except KeyError:
        p = 1 / len(state.legal_actions)
        probabilities = np.full(len(state.legal_actions), p)
        a = np.random.choice(state.legal_actions, p=probabilities)
        sigma[I] = {action: p for action in state.legal_actions}
    return a


# Load unnormalized strategy before rts
offline_path = 'test_strategy/unnormalized_output/offline_strategy_100.gz'
offline_strategy = joblib.load(offline_path)
# Load unnormalized strategy with rts
offline_rts_path = 'test_strategy/unnormalized_output/rts_output.gz'
offline_strategy_rts = joblib.load(offline_rts_path)

public_cards: List[Card] = []
action_sequence = ["raise", "call", "call", "call", "call"]
# Loading game state we used RTS on
state: ShortDeckPokerState = new_game(
    3, real_time_test=True, public_cards=public_cards
)
# Load current game state, either strategy should be identical for this
current_game_state: ShortDeckPokerState = state.load_game_state(
    offline_strategy, action_sequence
)
n_inner_iters = 100
n_outter_iters = 30
EVs = np.array([])
# We don't need the offline strategy for search..
for _ in trange(1, n_outter_iters):
    EV = np.array([])  # Expected value for player 0 (hero)
    for t in trange(1, n_inner_iters + 1, desc="train iter"):
        for p_i in range(3):
            # Deal hole cards based on bayesian updating of hole card probs
            state: ShortDeckPokerState = current_game_state.deal_bayes()
            while True:
                player_not_in_hand = not state.players[p_i].is_active
                if state.is_terminal or player_not_in_hand:
                    EV = np.append(EV, state.payout[p_i])
                    break
                if state.player_i == p_i:
                    random_action: str = _calculate_strategy(state, state.info_set,
                                                             offline_strategy_rts)
                else:
                    random_action: str = _calculate_strategy(state, state.info_set,
                                                             offline_strategy)
                state = state.apply_action(random_action)
    EVs = np.append(EVs, EV.mean())
print(f"Average EV after 30 sets of 100 games: {EVs.mean()}")
t_stat = (EVs.mean() - 0) / (EVs.std() / np.sqrt(n_outter_iters))
print(f"T statistic = {t_stat}")
p_val = stats.t.sf(np.abs(t_stat), n_outter_iters - 1)
print(f"P-Value: {p_val}")
import ipdb;
ipdb.set_trace()
