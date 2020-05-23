import os
import collections
from typing import Dict
import joblib


class Agent:
    # TODO(fedden): Note from the supplementary material, the data here will
    #               need to be lower precision: "To save memory, regrets were
    #               stored using 4-byte integers rather than 8-byte doubles.
    #               There was also a ﬂoor on regret at -310,000,000 for every
    #               action. This made it easier to unprune actions that were
    #               initially pruned but later improved. This also prevented
    #               integer overﬂows".
    def __init__(self, regret_dir=None):
        self.strategy = collections.defaultdict(
            lambda: collections.defaultdict(lambda: 0)
        )
        if regret_dir:
            offline_strategy = joblib.load(regret_dir)
            self.regret = collections.defaultdict(
                lambda: collections.defaultdict(lambda: 0),
                offline_strategy['regret']
            )
        else:
            self.regret = collections.defaultdict(
                lambda: collections.defaultdict(lambda: 0)
            )
        self.tmp_regret = collections.defaultdict(
            lambda: collections.defaultdict(lambda: 0)
        )

        def reset_new_regret(self):
            """Remove regret from temporary storage"""
            del self.tmp_regret
            self.tmp_regret = collections.defaultdict(
                lambda: collections.defaultdict(lambda: 0)
            )


# TODO: Change to Leon's newest iteration on this method
class TrainedAgent(Agent):
    """
    Agent who has been trained
    Points to a folder whose strategies is calculated from regret and then averaged
    """

    def __init__(self, directory: str):
        super().__init__()
        self.offline_strategy = self._load_regret(directory)

    # TODO: the following could use a refactor, just getting through this
    #    rather quickly
    def _calculate_strategy(
        self,
        regret: Dict[str, Dict[str, float]],
        sigma: Dict[int, Dict[str, Dict[str, float]]],
        I: str,
    ):
        """
        Get strategy from regret
        """
        rsum = sum([max(x, 0) for x in regret[I].values()])
        ACTIONS = regret[I].keys()  # TODO: this is hacky, might be a better way
        for a in ACTIONS:
            if rsum > 0:
                sigma[I][a] = max(regret[I][a], 0) / rsum
            else:
                sigma[I][a] = 1 / len(ACTIONS)
        return sigma

    def _average_strategy(self, directory: str):
        files = [
            x
            for x in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, x))
        ]

        offline_strategy: Dict = collections.defaultdict(
            lambda: collections.defaultdict(lambda: 0)
        )
        strategy_tmp = collections.defaultdict(
            lambda: collections.defaultdict(lambda: 0)
        )

        for idx, f in enumerate(files):
            if f in ["config.yaml", "strategy.gz"]:
                continue

            regret_dict = joblib.load(directory + "/" + f)["regret"]
            sigma = collections.defaultdict(
                lambda: collections.defaultdict(lambda: 1 / 3)
            )

            for info_set, regret in sorted(regret_dict.items()):
                sigma = self._calculate_strategy(regret_dict, sigma, info_set)

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

    def _load_regret(self, directory: str):
        return self._average_strategy(directory)
