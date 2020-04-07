
# okay, it's not "AI"
# see http://modelai.gettysburg.edu/2013/cfr/cfr.pdf

# kuhn poker time
# 3 cards
# 3*2 = 6 hands

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

ISETS = ["1", "2", "3",  # round 1
         "P1", "P2", "P3", "B1", "B2", "B3",  # round 2
         "PB1", "PB2", "PB3"]  # round 3

# terminal history states
TERMINAL = ["PP", "PBP", "PBB", "BP", "BB"]
ACTIONS = ["P", "B"]


def payout(rs, h):
    if h == "PBP":
        return -1
    elif h == "BP":
        return 1
    m = 1 if (rs[0] > rs[1]) else -1
    if h == "PP":
        return m
    if h in ["BB", "PBB"]:
        return m * 2
    assert False


def get_information_set(rs, h):
    assert h not in TERMINAL
    if h == "":
        return str(rs[0])
    elif len(h) == 1:
        return h + str(rs[1])
    else:
        return "PB" + str(rs[0])
    assert False


def cfr(rs, h, i, t, pi1, pi2):
    # rs = realstate
    # h = history
    # i = player
    # t = timestep

    if h in TERMINAL:
        return payout(rs, h) * (1 if i == 1 else -1)
    I = get_information_set(rs, h)
    ph = 2 if len(h) == 1 else 1

    # if we are here, we have both actions available
    vo = 0.0
    voa = {}
    for a in ACTIONS:
        if ph == 1:
            voa[a] = cfr(rs, h + a, i, t, sigma[t][I][a] * pi1, pi2)
        else:
            voa[a] = cfr(rs, h + a, i, t, pi1, sigma[t][I][a] * pi2)
        vo += sigma[t][I][a] * voa[a]
    if ph == i:
        if i == 1:
            pi = pi1
            pnegi = pi2
        else:
            pi = pi2
            pnegi = pi1
        for a in ACTIONS:
            regret[I][a] += pnegi * (voa[a] - vo)
            strategy[I][a] += pi * sigma[t][I][a]
        # update the strategy based on regret
        rsum = sum([max(x, 0) for x in regret[I].values()])
        for a in ACTIONS:
            if rsum > 0:
                sigma[t + 1][I][a] = max(regret[I][a], 0) / rsum
            else:
                sigma[t + 1][I][a] = 0.5
    return vo



# init tables
regret = {}
strategy = {}
for I in ISETS:
    regret[I] = {k: 0 for k in ACTIONS}
    strategy[I] = {k: 0 for k in ACTIONS}

sigma = {}
sigma[1] = {}
for I in ISETS:
    sigma[1][I] = {k: 0.5 for k in ACTIONS}

# learn strategy
import copy
import random

for t in range(1, 20000):
    sigma[t + 1] = copy.deepcopy(sigma[t])
    for i in [1, 2]:
        cfr(random.choice(HANDS), "", i, t, 1, 1)
    del sigma[t]

# print "average" strategy
for k, v in strategy.items():
    norm = sum(list(v.values()))
    print("%3s: P:%.4f B:%.4f" % (k, v['P'] / norm, v['B'] / norm))
    # https://en.wikipedia.org/wiki/Kuhn_poker#Optimal_strategy

