import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, Optional, Union

import joblib
import yaml
from tqdm import trange

from agent import Agent
from pluribus import utils
from pluribus.games.short_deck import state
from worker import Worker

log = logging.getLogger("sync.server")
manager = mp.Manager()


class Server:
    """"""

    def __init__(
        self,
        strategy_interval: int,
        n_iterations: int,
        lcfr_threshold: int,
        discount_interval: int,
        prune_threshold: int,
        c: int,
        n_players: int,
        print_iteration: int,
        dump_iteration: int,
        update_threshold: int,
        n_processes: int = mp.cpu_count() - 1,
        seed: Optional[int] = None,
        pickle_dir: Union[str, Path] = ".",
    ):
        """Set up the optimisation server."""
        config: Dict[str, int] = {**locals()}
        self._save_path: Path = utils.create_dir()
        with open(self._save_path / "config.yaml", "w") as steam:
            yaml.dump(config, steam)
        log.info("saved config")
        if seed is not None:
            utils.random.seed(42)
        self._n_iterations = n_iterations
        self._lcfr_threshold = lcfr_threshold
        self._discount_interval = discount_interval
        self._update_threshold = update_threshold
        self._dump_iteration = dump_iteration
        self._n_players = n_players
        self._info_set_lut: state.InfoSetLookupTable = utils.io.load_info_set_lut(
            pickle_dir,
        )
        log.info("Loaded lookup table.")
        self._queue: mp.Queue = mp.Queue(maxsize=n_processes)
        self._worker_status: Dict[str, bool] = manager.dict()
        self._agent: Agent = Agent()
        self._workers: Dict[str, Worker] = dict()
        for _ in range(n_processes):
            worker = Worker(
                queue=self._queue,
                agent=self._agent,
                info_set_lut=self._info_set_lut,
                n_players=n_players,
                prune_threshold=prune_threshold,
                c=c,
                lcfr_threshold=lcfr_threshold,
                discount_interval=discount_interval,
                update_threshold=update_threshold,
                dump_iteration=dump_iteration,
                save_path=self._save_path,
                worker_status=self._worker_status,
            )
            self._workers[worker.name] = worker
            self._workers[worker.name].start()
            log.info(f"started worker [bold green]{worker.name}[/]")

    def search(self):
        """Perform MCCFR and train the agent."""
        import ipdb

        ipdb.set_trace()
        for t in trange(1, self._n_iterations + 1, desc="train iter"):
            for i in range(self._n_players):
                self._send_job("cfr", t=t, i=i)
                if t < self._lcfr_threshold & t % self._discount_interval == 0:
                    self._wait_until_all_workers_are_idle()
                    self._send_job("discount", t=t)
                if t > self._update_threshold and t % self._dump_iteration == 0:
                    self._wait_until_all_workers_are_idle()
                    self._send_job("serialise_agent", t=t)

    def terminate(self):
        """Kill all workers."""
        for _ in self._workers.values():
            name = "terminate"
            kwargs = dict()
            self._queue.put((name, kwargs), block=True)
            log.info("sending sentinel to worker")
        for name, worker in self._workers.items():
            worker.join()
            log.info(f"worker {name} joined.")

    def serialise_agent(self):
        """Write agent to file."""
        to_persist = utils.io.to_dict(strategy=self._agent.strategy, regret=self._agent.regret)
        joblib.dump(to_persist, self._save_path / "strategy.gz", compress="gzip")
        utils.io.print_strategy(self._agent.strategy)

    def _send_job(self, name, **kwargs):
        """Send job of type `name` with arguments `kwargs` to worker pool."""
        self._queue.put((name, kwargs), block=True)

    def _wait_until_all_workers_are_idle(self, sleep_secs=0.5):
        """Blocks until all workers have finished their current job."""
        while any(status != "idle" for status in self._worker_status.values()):
            time.sleep(sleep_secs)
