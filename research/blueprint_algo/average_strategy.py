import joblib
from os import listdir
from os.path import isfile, join
import collections
from typing import Dict

import click

ACTIONS = ["P", "B"]


# TODO: pull this from main script
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
    return sigma


def average_strategy(directory: str):
    files = [x for x in listdir(directory) if isfile(join(directory, x))]

    offline_strategy = collections.defaultdict(
        lambda: collections.defaultdict(lambda: 0)
    )
    strategy_tmp = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))

    for idx, f in enumerate(files):
        if f in ["config.yaml", "strategy.gz"]:
            continue

        regret_dict = joblib.load(directory + "/" + f)["regret"]
        sigma = collections.defaultdict(lambda: collections.defaultdict(lambda: 1 / 2))

        for info_set, regret in sorted(regret_dict.items()):
            sigma = calculate_strategy(regret_dict, sigma, info_set)

        for info_set, strategy in sigma.items():
            for action, probability in strategy.items():
                try:
                    strategy_tmp[info_set][action] += probability
                except KeyError:
                    strategy_tmp[info_set][action] = probability

    for info_set, strategy in sorted(strategy_tmp.items()):
        norm = sum(list(strategy.values()))
        for action, probability in strategy.items():
            try:
                offline_strategy[info_set][action] += probability / norm
            except KeyError:
                offline_strategy[info_set][action] = probability / norm

    return offline_strategy


def print_strategy(offline_strategy: Dict):
    for k, v in offline_strategy.items():
        norm = sum(list(v.values()))
        print("%3s: P:%.4f B:%.4f" % (k, v["P"] / norm, v["B"] / norm))


@click.command()
@click.option("--directory", default="results_toy_2020_04_25_17_26_07_905950", help=".")
def train(directory: str):
    offline_strategy = average_strategy(directory)
    print_strategy(offline_strategy)


if __name__ == "__main__":
    train()
