import multiprocessing as mp
from pathlib import Path
from typing import Optional, Union

import joblib

manager = mp.Manager()


class Agent:
    # TODO(fedden): Note from the supplementary material, the data here will
    #               need to be lower precision: "To save memory, regrets were
    #               stored using 4-byte integers rather than 8-byte doubles.
    #               There was also a ﬂoor on regret at -310,000,000 for every
    #               action. This made it easier to unprune actions that were
    #               initially pruned but later improved. This also prevented
    #               integer overﬂows".

    def __init__(self, agent_path: Optional[Union[str, Path]] = None):
        """Create agent, optionally initialise to agent specified at path."""
        self.strategy = manager.dict()
        self.regret = manager.dict()
        if agent_path is not None:
            saved_agent = joblib.load(agent_path)
            # Assign keys manually because I don't trust the manager proxy.
            for info_set, value in saved_agent["regret"].items():
                self.regret[info_set] = value
            for info_set, value in saved_agent["strategy"].items():
                self.strategy[info_set] = value
