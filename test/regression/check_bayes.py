import joblib
import collections
import json
from typing import DefaultDict

import numpy as np
from tqdm import trange

from pluribus.poker.deck import Deck
from pluribus.games.short_deck.state import ShortDeckPokerState, new_game


def _calculate_strategy(
        state: ShortDeckPokerState,
        I: str,
        strategy: DefaultDict[str, DefaultDict[str, float]],
) -> str:
    sigma = collections.defaultdict(lambda: collections.defaultdict(lambda: 1 / 3))
    try:
        # If strategy is empty, go to other block
        sigma[I] = strategy[I].copy()
        if sigma[I] == {}:
            raise KeyError
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


n = 10000
n_players = 3
inner_iters = 1000

strategy_dir = "research/test_methodology/test_strategy2/"
strategy_path = "unnormalized_output/offline_strategy_1500.gz"
check = joblib.load(strategy_dir + strategy_path)
histories = np.random.choice(list(check.keys()), n)
action_sequences = []
public_cards_lst = []
community_card_dict = {
    "pre_flop": 0,
    "flop": 3,
    "turn": 4,
    "river": 5,
}
# Shorter deck for more reasonable simulation time..
ranks = list(range(12, 14 + 1))
deck = Deck(include_ranks=ranks)
found = 0
for idx, history in enumerate(histories):
    if idx % 100 == 0:
        print(idx)
    history_dict = json.loads(history)
    history_lst = history_dict["history"]
    if history_lst == []:
        continue
    action_sequence = []
    betting_rounds = []
    for x in history_lst:
        action_sequence += list(x.values())[0]
        betting_rounds += list(x.keys())
    try:
        final_betting_round = list(betting_rounds)[-1]
    except:
        import ipdb;
        ipdb.set_trace()
    # Hacking this for now, keeping the simulation small..
    if len(action_sequence) > 2:
        continue
    action_sequences.append(action_sequence)
    n_cards = community_card_dict[final_betting_round]
    cards_in_deck = deck._cards_in_deck
    public_cards = np.random.choice(cards_in_deck, n_cards, replace=False)
    public_cards_lst.append(list(public_cards))
    found += 1
    if found == 2:
        break
    # Assuming we find 2 action sequences a=out of 1000

store_hand_probs = {}
for i in trange(0, len(action_sequences)):
    public_cards = public_cards_lst[i].copy()
    # will need to check for this bug later..
#    if not public_cards:
#        import ipdb;
#        ipdb.set_trace()
    action_sequence = action_sequences[i].copy()
    state: ShortDeckPokerState = new_game(
        n_players,
        real_time_test=True,
        public_cards=public_cards,
    )
    current_game_state: ShortDeckPokerState = state.load_game_state(
        offline_strategy=check, action_sequence=action_sequence
    )
    new_state = current_game_state.deal_bayes()

    this_hand_probs = new_state._starting_hand_probs.copy()
    for p_i in this_hand_probs.keys():
        for starting_hand in this_hand_probs[p_i].keys():
            x = this_hand_probs[p_i][starting_hand]
            this_hand_probs[p_i][starting_hand] = {'deal_bayes':x, 'sim':None}

    action_sequence = action_sequences[i].copy()
    public_cards = public_cards_lst[i].copy()
    info_set_lut = {}
    cont = True
    actions = []
    tries = 0
    success = 0
    hand_dict = {0: {}, 1: {}, 2: {}}
    while cont:
        state: ShortDeckPokerState = new_game(
            n_players,
            info_set_lut,
            real_time_test=True,
            public_cards=public_cards
        )
        info_set_lut = state.info_set_lut
        while True:
            count = 0
            if tries == 1000: # definitely a hack need to be careful about this
                              # value
                for p_i in state.players:
                    hole_cards = tuple(state.players[p_i].cards)
                    try:
                        hand_dict[p_i][hole_cards] += 0
                    except KeyError:
                        hand_dict[p_i][hole_cards] = 0
            random_action = _calculate_strategy(state, state.info_set, check)
            if random_action != action_sequence[count]:
                tries += 1
                break
            new_state = state.apply_action(random_action)
            actions.append(random_action)
            if actions == action_sequence:
                for p_i in state.players:
                    hole_cards = tuple(state.players[p_i].cards)
                    try:
                        hand_dict[p_i][hole_cards] += 1
                    except KeyError:
                        hand_dict[p_i][hole_cards] = 1
                success += 1
                break
            count += 1
        if success == 1:
            break
    import ipdb;
    ipdb.set_trace()
