import collections
import joblib


class Agent:
    def __init__(self, regret_path=None):
        self.strategy = collections.defaultdict(
            lambda: collections.defaultdict(lambda: 0)
        )
        if regret_path:
            offline_strategy = joblib.load(regret_path)
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
