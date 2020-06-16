import copy
import random
from typing import Tuple, Dict

import numpy as np
from tqdm import trange


HANDS = [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]

# there are 12 information sets
# p1(1st) 1 {2,3}
# p1(1st) 2 {1,3}
# p1(1st) 3 {1,2}
# p2(2nd) 1, p1 pass {2,3}
# p2(2nd) 1, p1 bet  {2,3}
# p2(2nd) 2, p1 pass {1,3}
# p2(2nd) 2, p1 bet  {1,3}
# p2(2nd) 3, p1 pass {1,2}
# p2(2nd) 3, p1 bet  {1,2}
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


def payout(rs: Tuple[int, int], h: str) -> int:
    """
    :param rs: realstate, a tuple of two ints, first is card for p1, second p2
    :param h: the action sequences without the card information
    :return: expected value (at least at that moment in the game)
    """
    if h == "PBP":
        return -100
    elif h == "BP":
        return 100
    m = 100 if (rs[0] > rs[1]) else -100
    if h == "PP":
        return m
    if h in ["BB", "PBB"]:
        return m * 2
    assert False


def get_information_set(rs: Tuple[int, int], h: str) -> str:
    """
    :param rs: realstate, a tuple of two ints, first is card for p1, second p2
    :param h: the action sequences without the card information
    :return: infoset
    """
    assert h not in TERMINAL
    if h == "":
        return str(rs[0])
    elif len(h) == 1:
        return h + str(rs[1])
    else:
        return "PB" + str(rs[0])
    assert False


def update_strategy(rs: Tuple[int, int], h: str, i: int):
    """
    :param rs: realstate, a tuple of two ints, first is card for p1, second p2
    :param h: the action sequences without the card information
    :param i: the player, i = 1 is always first to act and i = 2 is always
        second to act, but they take turns who updates the strategy (only one
        strategy)
    :return: nothing, updates action count in the strategy of actions chosen
        according to sigma, this simple choosing of actions is what allows the
        algorithm to build up preference for one action over another in a given
        spot
    """
    ph = 2 if len(h) == 1 else 1  # this is always the case no matter what i is

    if (
        h in TERMINAL
    ):  # or if p_i is not in the hand or if betting round is > 0, strategy is
        # only
        return
    # elif h is chance_node:  -- we don't care about chance nodes here
    #   sample action from strategy for h
    #   update_strategy(rs, h + a, i)
    elif ph == i:
        I = get_information_set(rs, h)
        # calculate regret
        calculate_strategy(regret, sigma, I)
        # choose an action based of sigma
        a = np.random.choice(list(sigma[t][I].keys()), 1, p=list(sigma[t][I].values()))[
            0
        ]
        strategy[I][a] += 1
        # so strategy is counts based on sigma, this takes into account the
        # reach probability
        # so there is no need to pass around that pi guy..
        update_strategy(rs, h + a, i)
    else:
        for a in ACTIONS:
            # not actually updating the strategy for p_i != i, only one i at a
            # time
            update_strategy(rs, h + a, i)


def calculate_strategy(
    regret: Dict[str, Dict[str, float]],
    sigma: Dict[int, Dict[str, Dict[str, float]]],
    I: str,
):
    """

    :param regret: dictionary of regrets, I is key, then each action at I, with
        values being regret
    :param sigma: dictionary of strategy updated by regret, iteration is key,
        then I is key, then each action with prob
    :param I:
    :return: doesn't return anything, just updates sigma
    """
    rsum = sum([max(x, 0) for x in regret[I].values()])
    for a in ACTIONS:
        if rsum > 0:
            sigma[t + 1][I][a] = max(regret[I][a], 0) / rsum
        else:
            sigma[t + 1][I][a] = 1 / len(ACTIONS)


def cfr(rs: Tuple[int, int], h: str, i: int, t: int) -> float:
    """
    regular cfr algo

    :param rs: realstate, a tuple of two ints, first is card for p1, second p2
    :param h: the action sequences without the card information
    :param i: player
    :param t: iteration
    :return: expected value for node for player i
    """
    ph = 2 if len(h) == 1 else 1  # this is always the case no matter what i is

    if h in TERMINAL:
        return payout(rs, h) * (1 if i == 1 else -1)
    # elif p_i not in hand:
    #   cfr()
    # TODO: this will be needed for No Limit Hold'Em
    # elif h is chance_node:  -- we don't care about chance nodes here
    #   sample action from strategy for h
    #   cfr()
    elif ph == i:
        I = get_information_set(rs, h)
        # calculate strategy
        calculate_strategy(regret, sigma, I)
        vo = 0.0
        voa = {}
        for a in ACTIONS:
            voa[a] = cfr(rs, h + a, i, t)
            vo += sigma[t][I][a] * voa[a]
        for a in ACTIONS:
            regret[I][a] += voa[a] - vo
            # do not need update the strategy based on regret, strategy does
            # that with sigma
        return vo
    else:
        Iph = get_information_set(rs, h)
        calculate_strategy(regret, sigma, Iph)
        a = np.random.choice(
            list(sigma[t][Iph].keys()), 1, p=list(sigma[t][Iph].values())
        )[0]
        return cfr(rs, h + a, i, t)


def cfrp(rs: Tuple[int, int], h: str, i: int, t: int):
    """
    pruning cfr algo, might need to adjust only pruning if not final betting
        round and if not terminal node

    :param rs: realstate, a tuple of two ints, first is card for p1, second p2
    :param h: the action sequences without the card information
    :param i: player
    :param t: iteration
    :return: expected value for node for player i
    """
    ph = 2 if len(h) == 1 else 1

    if h in TERMINAL:
        return payout(rs, h) * (1 if i == 1 else -1)
    # elif p_i not in hand:
    #   cfrp()
    # elif h is chance_node:
    #   sample action from strategy for h
    #   cfrp()
    elif ph == i:
        I = get_information_set(rs, h)
        # calculate strategy
        calculate_strategy(regret, sigma, I)
        vo = 0.0
        voa = {}
        explored = {}  # keeps tracked of items that can be skipped
        for a in ACTIONS:
            if regret[I][a] > C:
                voa[a] = cfrp(rs, h + a, i, t)
                explored[a] = True
                vo += sigma[t][I][a] * voa[a]
            else:
                explored[a] = False
        for a in ACTIONS:
            if explored[a]:
                regret[I][a] += voa[a] - vo
                # do not need update the strategy based on regret, strategy
                # does that with sigma
        return vo
    else:
        Iph = get_information_set(rs, h)
        calculate_strategy(regret, sigma, Iph)
        a = np.random.choice(
            list(sigma[t][Iph].keys()), 1, p=list(sigma[t][Iph].values())
        )[0]
        return cfrp(rs, h + a, i, t)


if __name__ == "__main__":
    # init tables
    regret = {}
    strategy = {}
    for I in ISETS:
        regret[I] = {k: 0 for k in ACTIONS}
        strategy[I] = {k: 0 for k in ACTIONS}

    sigma = {1: {}}
    for I in ISETS:
        sigma[1][I] = {k: 1 / len(ACTIONS) for k in ACTIONS}

    # algorithm constants
    strategy_interval = 100
    LCFR_threshold = 4000
    discount_interval = 100
    prune_threshold = 2000
    C = -20000  # somewhat arbitrary

    for t in trange(1, 20000):
        sigma[t + 1] = copy.deepcopy(sigma[t])
        for i in [1, 2]:  # fixed position i
            h = ""
            rs = random.choice(HANDS)
            if t % strategy_interval == 0:
                update_strategy(rs, h, i)
            if t > prune_threshold:
                if random.uniform(0, 1) < 0.05:
                    cfr(rs, h, i, t)
                else:
                    cfrp(rs, h, i, t)
            else:
                cfr(rs, h, i, t)
        if t < LCFR_threshold & t % discount_interval == 0:
            d = (t / discount_interval) / ((t / discount_interval) + 1)
            for I in ISETS:
                for a in ACTIONS:
                    regret[I][a] *= d
                    strategy[I][a] *= d
        del sigma[t]

    for k, v in strategy.items():
        norm = sum(list(v.values()))
        print("%3s: P:%.4f B:%.4f" % (k, v["P"] / norm, v["B"] / norm))
        # https://en.wikipedia.org/wiki/Kuhn_poker#Optimal_strategy
        # generally close, converges to 1s much faster
        # reducing C should allow for better convergence, but slower..

# code for normalizing strategy
# normalized_strategy = {}
# for I in strategy.keys():
#     d = strategy[I]
#     factor = 1.0/sum(list(d.values()))
#     normalized_d = {k: v*factor for k, v in d.items()}
#     normalized_strategy[I] = normalized_d
#
# print(normalized_strategy)
