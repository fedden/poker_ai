import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict

import joblib
import numpy as np

import ai
from agent import Agent
from pluribus import utils
from pluribus.games.short_deck import state

log = logging.getLogger("sync.worker")


class Worker(mp.Process):
    """"""

    def __init__(
        self,
        job_queue: mp.Queue,
        status_queue: mp.Queue,
        locks: Dict[str, mp.synchronize.Lock],
        agent: Agent,
        info_set_lut: state.InfoSetLookupTable,
        n_players: int,
        prune_threshold: int,
        c: int,
        lcfr_threshold: int,
        discount_interval: int,
        update_threshold: int,
        dump_iteration: int,
        save_path: Path,
    ):
        """"""
        super().__init__(group=None, name=None, args=(), kwargs={}, daemon=None)
        self._job_queue: mp.Queue = job_queue
        self._status_queue: mp.Queue = status_queue
        self._locks = locks
        self._n_players = n_players
        self._prune_threshold = prune_threshold
        self._agent = agent
        self._c = c
        self._lcfr_threshold = lcfr_threshold
        self._discount_interval = discount_interval
        self._update_threshold = update_threshold
        self._dump_iteration = dump_iteration
        self._save_path = save_path
        self._info_set_lut: state.InfoSetLookupTable = info_set_lut
        self._setup_new_game()

    def run(self):
        """"""
        while True:
            # Get the name of the method and the key word arguments needed for
            # the method.
            self._update_status("idle")
            name, kwargs = self._job_queue.get(block=True)
            if name == "terminate":
                break
            elif name == "cfr":
                function = self._cfr
            elif name == "discount":
                function = self._discount
            elif name == "update_strategy":
                function = self._update_strategy
            elif name == "serialise_agent":
                function = self._serialise_agent
            else:
                raise ValueError(f"Unrecognised function name: {name}")
            self._update_status(name)
            function(**kwargs)
            # Notify the job queue that the task is done.
            self._job_queue.task_done()

    def _cfr(self, t, i):
        """Search over random game and calculate the strategy."""
        self._setup_new_game()
        use_pruning = np.random.uniform() < 0.95
        if t > self._prune_threshold and use_pruning:
            ai.cfr(self._agent, self._state, i, t, self._locks)
        else:
            ai.cfrp(self._agent, self._state, i, t, self._c, self._locks)

    def _discount(self, t):
        """Discount previous regrets and strategy."""
        # TODO(fedden): Is discount_interval actually set/managed in
        #               minutes here? In Algorithm 1 this should be managed
        #               in minutes using perhaps the time module, but here
        #               it appears to be being managed by the iterations
        #               count.
        discount_factor = (t / self._discount_interval) / (
            (t / self._discount_interval) + 1
        )
        for info_set in self._agent.regret.keys():
            for action in self._agent.regret[info_set].keys():
                self._agent.regret[info_set][action] *= discount_factor
                self._agent.strategy[info_set][action] *= discount_factor

    def _update_strategy(self, t, i):
        """Update the strategy."""
        ai.update_strategy(self._agent, self._state, i, t, self._locks)

    def _serialise_agent(self, t):
        """Write agent to file."""
        # dump the current
        # strategy (sigma) throughout training and then take an average.
        # This allows for estimation of expected value in leaf nodes later
        # on using modified versions of the blueprint strategy
        to_persist = utils.io.to_dict(
            strategy=self._agent.strategy, regret=self._agent.regret
        )
        joblib.dump(to_persist, self._save_path / f"strategy_{t}.gz", compress="gzip")

    def _update_status(self, status, log_status: bool = False):
        """Update the status of this worker by posting it to the server."""
        if log_status:
            log.info(f"{self.name} updating status to {status}")
        self._status_queue.put((self.name, status), block=True)

    def _setup_new_game(self):
        """Setup up new poker game."""
        self._state: state.ShortDeckPokerState = state.new_game(
            self._n_players, self._info_set_lut,
        )
