import random
import numpy as np

# kuhn poker time
# 3 cards
# 3*2 = 6 hands

HANDS = [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]

# there are 12 information sets
# p1(1st) 1 {2, 3}
# p1(1st) 2 {1, 3}
# p1(1st) 3 {1, 2}
# p2(2nd) 1, p1 pass {2, 3}
# p2(2nd) 1, p1 bet  {2, 3}
# p2(2nd) 2, p1 pass {1, 3}
# p2(2nd) 2, p1 bet  {1, 3}
# p2(2nd) 3, p1 pass {1, 2}
# p2(2nd) 3, p1 bet  {1, 2}
# p1(3rd) 1, p1 pass, p2 bet  {2,3}
# p1(3rd) 2, p1 pass, p2 bet  {1,3}
# p1(3rd) 3, p1 pass, p2 bet  {1,2}

ISETS = [
    "1",
    "2",
    "3",  # round 1
    "P1",
    "P2",
    "P3",
    "B1",
    "B2",
    "B3",  # round 2
    "PB1",
    "PB2",
    "PB3",  # round 3
]

# terminal history states
TERMINAL = ["PP", "PBP", "PBB", "BP", "BB"]
ACTIONS = ["P", "B"]
N_ACTIONS = 2


def payout(player_1_hand, player_2_hand, history):
    if history == "PBP":
        return -1
    elif history == "BP":
        return 1
    m = 1 if (player_1_hand > player_2_hand) else -1
    if history == "PP":
        return m
    if history in ["BB", "PBB"]:
        return m * 2
    assert False


def get_information_set(player_1_hand, player_2_hand, history):
    assert history not in TERMINAL
    if history == "":
        return str(player_1_hand)
    elif len(history) == 1:
        return history + str(player_2_hand)
    else:
        return "PB" + str(player_1_hand)
    assert False


def cfr(player_1, player_2, history, player_i, pi1, pi2):
    if history in TERMINAL:
        return payout(player_1.hand, player_2.hand, history) * (
            1 if player_i == 1 else -1
        )
    info_set = get_information_set(player_1.hand, player_2.hand, history)
    ph = 2 if len(history) == 1 else 1
    if ph == 1:
        regret = player_1.regret
        strategy_sum = player_1.strategy_sum
        strategy = player_1.strategy
    else:
        regret = player_2.regret
        strategy_sum = player_2.strategy_sum
        strategy = player_2.strategy
    # if we are here, we have both actions available
    utility = np.zeros(N_ACTIONS)
    for action_i, action in enumerate(ACTIONS):
        if ph == 1:
            utility[action_i] = cfr(
                player_1=player_1,
                player_2=player_2,
                history=history + action,
                player_i=player_i,
                pi1=strategy[info_set][action_i] * pi1,
                pi2=pi2,
            )
        else:
            utility[action_i] = cfr(
                player_1=player_1,
                player_2=player_2,
                history=history + action,
                player_i=player_i,
                pi1=pi1,
                pi2=strategy[info_set][action_i] * pi2,
            )
    info_set_utility = np.sum(strategy[info_set] * utility)
    if ph == player_i:
        if player_i == 1:
            pi = pi1
            pnegi = pi2
        else:
            pi = pi2
            pnegi = pi1
        regret[info_set] += pnegi * (utility - info_set_utility)
        strategy_sum[info_set] += pi * strategy[info_set]
        # update the strategy_sum based on regret
        rsum = np.sum(np.maximum(regret[info_set], 0))
        if rsum > 0:
            strategy[info_set] = np.maximum(regret[info_set], 0) / rsum
        else:
            strategy[info_set] = np.full(N_ACTIONS, 0.5)
    return info_set_utility


class Player:
    def __init__(self):
        self.hand = None
        self.regret = {info_set: np.zeros(N_ACTIONS) for info_set in ISETS}
        self.strategy_sum = {info_set: np.zeros(N_ACTIONS) for info_set in ISETS}
        self.strategy = {info_set: np.full(N_ACTIONS, 0.5) for info_set in ISETS}


def train(n_iterations: int = 20000):
    # Initialise players
    player_1 = Player()
    player_2 = Player()
    # If this is uncommented the players share the same strategy_sum.
    player_2.regret = player_1.regret
    player_2.strategy_sum = player_1.strategy_sum
    player_2.strategy = player_1.strategy
    # learn strategy_sum
    for _ in range(n_iterations):
        for player_i in [1, 2]:
            hands = random.choice(HANDS)
            player_1.hand = hands[0]
            player_2.hand = hands[1]
            cfr(
                player_1=player_1,
                player_2=player_2,
                history="",
                player_i=player_i,
                pi1=1,
                pi2=1,
            )
    # Print "average" strategy_sum.
    for k, v in player_1.strategy_sum.items():
        norm = sum(list(v))
        print("%3s: P:%.4f B:%.4f" % (k, v[0] / norm, v[1] / norm))
    # https://en.wikipedia.org/wiki/Kuhn_poker#Optimal_strategy_sum


train()
