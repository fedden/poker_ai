"""
"""
import copy
import random
from typing import Tuple, Dict, Any
import collections
import datetime
import json
from pathlib import Path

import numpy as np
from tqdm import trange
import joblib
import yaml
import click


HANDS = [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]

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

strategy = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
regret = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))


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


def update_strategy(rs: Tuple[int, int], h: str, i: int, sigma):
    """

    :param rs: realstate, a tuple of two ints, first is card for player one, second player 2
    :param h: the action sequences without the card information
    :param i: the player, i = 1 is always first to act and i = 2 is always second to act, but they take turns who
        updates the strategy (only one strategy)
    :return: nothing, updates action count in the strategy of actions chosen according to sigma, this simple choosing of
        actions is what allows the algorithm to build up preference for one action over another in a given spot
    """
    ph = 2 if len(h) == 1 else 1  # this is always the case no matter what i is

    betting_round = len(h)
    if (
        h in TERMINAL or betting_round > 0
    ):  # or if p_i is not in the hand or if betting round is > 0, strategy is only
        return
    # elif h is chance_node:  -- we don't care about chance nodes here, but we will for No Limit
    #   sample action from strategy for h
    #   update_strategy(rs, h + a, i)
    elif ph == i:
        I = get_information_set(rs, h)
        # calculate regret
        calculate_strategy(regret, sigma, I)
        # choose an action based of sigma
        try:
            a = np.random.choice(list(sigma[I].keys()), 1, p=list(sigma[I].values()))[0]
        except ValueError:
            p = 1 / len(ACTIONS)
            probabilities = np.full(len(ACTIONS), p)
            a = np.random.choice(ACTIONS, p=probabilities)
        strategy[I][a] += 1
        # so strategy is counts based on sigma, this takes into account the reach probability
        # so there is no need to pass around that pi guy..
        update_strategy(rs, h + a, i, sigma)
    else:
        for a in ACTIONS:
            # not actually updating the strategy for p_i != i, only one i at a time
            update_strategy(rs, h + a, i, sigma)


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
            sigma[I][a] = max(regret[I][a], 0) / rsum
        else:
            sigma[I][a] = 1 / len(ACTIONS)


def cfr(rs: Tuple[int, int], h: str, i: int, t: int, sigma: Dict) -> float:
    """
    regular cfr algo

    :param rs: realstate, a tuple of two ints, first is card for player one, second player 2
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
            voa[a] = cfr(rs, h + a, i, t, sigma)
            vo += sigma[I][a] * voa[a]
        for a in ACTIONS:
            regret[I][a] += voa[a] - vo
            # do not need update the strategy based on regret, strategy does that with sigma
        return vo
    else:
        Iph = get_information_set(rs, h)
        calculate_strategy(regret, sigma, Iph)
        try:
            a = np.random.choice(
                list(sigma[Iph].keys()), 1, p=list(sigma[Iph].values())
            )[0]
        except ValueError:
            p = 1 / len(ACTIONS)
            probabilities = np.full(len(ACTIONS), p)
            a = np.random.choice(ACTIONS, p=probabilities)
        return cfr(rs, h + a, i, t, sigma)


def cfrp(rs: Tuple[int, int], h: str, i: int, t: int, sigma: Dict, c: int):
    """
    pruning cfr algo, might need to adjust only pruning if not final betting round and if not terminal node

    :param rs: realstate, a tuple of two ints, first is card for player one, second player 2
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
            if regret[I][a] > c:
                voa[a] = cfrp(rs, h + a, i, t, sigma, c)
                explored[a] = True
                vo += sigma[I][a] * voa[a]
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
        try:
            a = np.random.choice(
                list(sigma[Iph].keys()), 1, p=list(sigma[Iph].values())
            )[0]
        except ValueError:
            p = 1 / len(ACTIONS)
            probabilities = np.full(len(ACTIONS), p)
            a = np.random.choice(ACTIONS, p=probabilities)
        return cfrp(rs, h + a, i, t, sigma, c)


def to_dict(**kwargs) -> Dict[str, Any]:
    """Hacky method to convert weird collections dicts to regular dicts."""
    return json.loads(json.dumps(copy.deepcopy(kwargs)))


def _create_dir() -> Path:
    """Create and get a unique dir path to save to using a timestamp."""
    time = str(datetime.datetime.now())
    for char in ":- .":
        time = time.replace(char, "_")
    path: Path = Path(f"./results_toy_{time}")
    path.mkdir(parents=True, exist_ok=True)
    return path


@click.command()
@click.option("--strategy_interval", default=100, help=".")
@click.option("--n_iterations", default=20000, help=".")
@click.option("--lcfr_threshold", default=4000, help=".")
@click.option("--discount_interval", default=100, help=".")
@click.option("--prune_threshold", default=2000, help=".")
@click.option("--c", default=-20000, help=".")
@click.option("--dump_iteration", default=100, help=".")
@click.option("--update_threshold", default=2000, help=".")
def train(
    strategy_interval: int,
    n_iterations: int,
    lcfr_threshold: int,
    discount_interval: int,
    prune_threshold: int,
    c: int,
    dump_iteration: int,
    update_threshold: int,
):

    # Get the values passed to this method, save this.
    config: Dict[str, int] = {**locals()}
    save_path: Path = _create_dir()
    with open(save_path / "config.yaml", "w") as steam:
        yaml.dump(config, steam)

    for t in trange(1, n_iterations):
        for i in [1, 2]:  # fixed position i
            sigma = collections.defaultdict(
                lambda: collections.defaultdict(lambda: 1 / 2)
            )
            h = ""
            rs = random.choice(HANDS)
            if t > update_threshold and t % strategy_interval == 0:
                update_strategy(rs, h, i, sigma)
            if t > prune_threshold:
                if random.uniform(0, 1) < 0.05:
                    cfr(rs, h, i, t, sigma)
                else:
                    cfrp(rs, h, i, t, sigma, c)
            else:
                cfr(rs, h, i, t, sigma)
            del sigma
        if t < lcfr_threshold & t % discount_interval == 0:
            d = (t / discount_interval) / ((t / discount_interval) + 1)
            for I in ISETS:
                for a in ACTIONS:
                    regret[I][a] *= d
                    strategy[I][a] *= d
        if (t > update_threshold) & (t % dump_iteration == 0):
            to_persist = to_dict(regret=regret)
            joblib.dump(to_persist, save_path / f"strategy_{t}.gz", compress="gzip")

    to_persist = to_dict(strategy=strategy)
    joblib.dump(to_persist, save_path / "strategy.gz", compress="gzip")

    for k, v in strategy.items():
        norm = sum(list(v.values()))
        print("%3s: P:%.4f B:%.4f" % (k, v["P"] / norm, v["B"] / norm))


if __name__ == "__main__":
    train()
