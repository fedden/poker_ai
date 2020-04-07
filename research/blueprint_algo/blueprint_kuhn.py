"""
DEBUGGING TRIALS/NOTES

-Is blueprint kuhn blueprint mockup incorrect in any way?
--Firstly, I wanted to check if the logic of the blueprint MCCFR algo, as applied to the kuhn
poker game, had any inaccuracies. After stepping through the logic, I do not believe this is
the case. The algorithm runs as expected, first going to the depths of the tree. It samples actions
for the player that is NOT ph based on the regrets up to that point. It searches each action
if i == ph. Once it finds the expected value of a node, it compares this to the value
actually generated and updates regret accordingly. In the following seed (not currently set),
you can see that when t == 11, if you step through the CFR function, you will see an example of player 2
folding the best hand to a bet (a mistake) and the regret jumps to a positive regret for
I == '1' for player 1. This accounts for the algorithm's optimizations, as the sampling for player 2
will allow for occasions where player 2 samples actions that do not work in their favor
(which is the nature of imperfect information games). In this example, player 2 allowed player 1 to
bluff, and added positive regret for betting with the the worst hand. The players approach a
nash equilibrium in this way.

-Should action samplings done from iteration t+1?
--There are some cases where player two could immediately benefit from an updated strategy
within the same iteration where player 1 had played the same infoset when they were p_i.
However, I believe this would bias results as play would (though perhaps VERY slightly) favor player 2

-Other learnings/Next Steps
--Next I will step through the three player game to check the game logic/updating of
regret/sigma
--Now it makes sense to me why we need to dump sigma every x iterations. The reason being,
sigma (really regret might be the more important value as sigma isn't always updated) changes drastically
from time to time. In the case mentioned above, sigma went from being normally p: 1, B: 0 for I = '1'
to then being P: .5, B: .5.
--To this effect, if you run the script as it currently exists, you will generate a plot
of the amount of time a particular move has a sigma value of 1. The behavior is interesting.
I first noticed it because I felt like there was a disproportionate amount of certain probabilities
in the sigma object after completing 20000 iterations. Although
it seems to have a consistent proportion of time that sigma is == 1 for this infoset and action,
the amount of time sigma spends at a value less than one happens in spurts, whose durations become longer
(but happen less frequently) over time. This is not a complete analysis of the behavior, but it
does illustrate the importance of properly setting the parameters before running.
--Further, if I remember correctly, for three player poker, we are looking at about 70 action
sequences (is that right??) for each of the 190 (after lossless compression) hands. We must choose
an appropriate proportion of updates/iters such that the probability of visiting each infoset in the
preflop round is good enough to produce a decent strategy.


"""
import copy
import random
from typing import Tuple, Dict

import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt


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

    :param rs: realstate, a tuple of two ints, first is card for player one, second player 2
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

    :param rs: realstate, a tuple of two ints, first is card for player one, second player 2
    :param h: the action sequences without the card information
    :return: I: information set, which contains all h that the p_i cannot distinguish
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

    :param rs: realstate, a tuple of two ints, first is card for player one, second player 2
    :param h: the action sequences without the card information
    :param i: the player, i = 1 is always first to act and i = 2 is always second to act, but they take turns who
        updates the strategy (only one strategy)
    :return: nothing, updates action count in the strategy of actions chosen according to sigma, this simple choosing of
        actions is what allows the algorithm to build up preference for one action over another in a given spot
    """
    # print("UPDATE FUNCTION")
    # import ipdb
    # ipdb.set_trace()
    ph = 2 if len(h) == 1 else 1  # this is always the case no matter what i is

    if (
        h in TERMINAL
    ):  # or if p_i is not in the hand or if betting round is > 0, strategy is only
        return  # updated in betting round 1 for Pluribus, but I am doing all rounds in this example
    # elif h is chance_node:  -- we don't care about chance nodes here, but we will for No Limit
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
        # if a == 'B' and I == '1':
        #     import ipdb
        #     ipdb.set_trace()
        strategy[I][a] += 1
        # so strategy is counts based on sigma, this takes into account the reach probability
        # so there is no need to pass around that pi guy..
        update_strategy(rs, h + a, i)
    else:
        for a in ACTIONS:
            # not actually updating the strategy for p_i != i, only one i at a time
            update_strategy(rs, h + a, i)


def calculate_strategy(
    regret: Dict[str, Dict[str, float]],
    sigma: Dict[int, Dict[str, Dict[str, float]]],
    I: str,
):
    """

    :param regret: dictionary of regrets, I is key, then each action at I, with values being regret
    :param sigma: dictionary of strategy updated by regret, iteration is key, then I is key, then each action with prob
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

    :param rs: realstate, a tuple of two ints, first is card for player one, second player 2
    :param h: the action sequences without the card information
    :param i: player
    :param t: iteration
    :return: expected value for node for player i
    """
    # if i == 1 and rs[0] == 1 and h == "":
    #     print("CFR")
    #     import ipdb
    #     ipdb.set_trace()
    ph = 2 if len(h) == 1 else 1  # this is always the case no matter what i is

    if h in TERMINAL:
        return payout(rs, h) * (1 if i == 1 else -1)
    # elif p_i not in hand:
    #   cfr()
    # TODO: this will be needed for No Limit Hold'Em, but in two player the player is always in the hand
    # elif h is chance_node:  -- we don't care about chance nodes here, but we will for No Limit
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
            # do not need update the strategy based on regret, strategy does that with sigma
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
    pruning cfr algo, might need to adjust only pruning if not final betting round and if not terminal node

    :param rs: realstate, a tuple of two ints, first is card for player one, second player 2
    :param h: the action sequences without the card information
    :param i: player
    :param t: iteration
    :return: expected value for node for player i
    """
    # if t == 13:
    #     print("CFRP")
    #     import ipdb
    #     ipdb.set_trace()
    ph = 2 if len(h) == 1 else 1

    if h in TERMINAL:
        return payout(rs, h) * (1 if i == 1 else -1)
    # elif p_i not in hand:
    #   cfrp()
    # TODO: this will be needed for No Limit Hold'Em, but in two player the player is always in the hand
    # elif h is chance_node:  -- we don't care about chance nodes here, but we will for No Limit
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
                # do not need update the strategy based on regret, strategy does that with sigma
        return vo
    else:
        Iph = get_information_set(rs, h)
        calculate_strategy(regret, sigma, Iph)
        a = np.random.choice(
            list(sigma[t][Iph].keys()), 1, p=list(sigma[t][Iph].values())
        )[0]
        return cfrp(rs, h + a, i, t)


if __name__ == "__main__":
    #np.random.seed(20)

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
    strategy_interval = 50
    LCFR_threshold = 4000
    discount_interval = 4000
    prune_threshold = 2000
    C = -20000  # somewhat arbitrary

    # algorithm presented here, pg.16:
    # https://science.sciencemag.org/content/sci/suppl/2019/07/10/science.aay2400.DC1/aay2400-Brown-SM.pdf
    #count = 0
    counts = []
    for t in trange(1, 80000):
        counts.append(sigma[t]['1']['P'])  # as an example of a value that hangs out at 1
        sigma[t + 1] = copy.deepcopy(sigma[t])
        for i in [1, 2]:  # fixed position i
            h = ""
            rs_idx = np.random.choice(len(HANDS), 1)[0]
            rs = HANDS[rs_idx]
            print(rs)
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

    counts = np.array(counts)
    print("proportion of time value was 1: ", len(np.where(counts==1)[0])/t)
    plt.plot(range(0, len(counts)), counts)
    plt.show()

    import ipdb
    ipdb.set_trace()
    for k, v in strategy.items():
        norm = sum(list(v.values()))
        print("%3s: P:%.4f B:%.4f" % (k, v["P"] / norm, v["B"] / norm))
        # https://en.wikipedia.org/wiki/Kuhn_poker#Optimal_strategy
        # generally close, converges to 1s much faster
        # reducing C should allow for better convergence, but slower..
