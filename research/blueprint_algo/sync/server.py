import logging
import multiprocessing as mp
import time
import threading
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
        self._save_path: Path = utils.io.create_dir()
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
        self._strategy_interval = strategy_interval
        self._info_set_lut: state.InfoSetLookupTable = utils.io.load_info_set_lut(
            pickle_dir,
        )
        log.info("Loaded lookup table.")
        self._job_queue: mp.JoinableQueue = mp.JoinableQueue(maxsize=n_processes)
        self._status_queue: mp.Queue = mp.Queue()
        self._worker_status: Dict[str, str] = dict()
        self._agent: Agent = Agent()
        self._workers: Dict[str, Worker] = dict()
        locks: Dict[str, mp.Lock] = dict(regret=manager.Lock(), strategy=manager.Lock())
        for _ in range(n_processes):
            worker = Worker(
                job_queue=self._job_queue,
                status_queue=self._status_queue,
                locks=locks,
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
            )
            self._workers[worker.name] = worker
        for name, worker in self._workers.items():
            worker.start()
            log.info(f"started worker {name}")

    def search(self):
        """Perform MCCFR and train the agent."""
        for t in trange(1, self._n_iterations + 1, desc="train iter"):
            for i in range(self._n_players):
                if t > self._update_threshold and t % self._strategy_interval == 0:
                    self._syncronised_job("update_strategy", t=t, i=i)
                self._send_job("cfr", t=t, i=i)
                if t < self._lcfr_threshold & t % self._discount_interval == 0:
                    self._syncronised_job("discount", t=t)
                if t > self._update_threshold and t % self._dump_iteration == 0:
                    self._syncronised_job("serialise_agent", t=t)

    def terminate(self):
        """Kill all workers."""
        # Wait for all workers to finish their current jobs.
        self._job_queue.join()
        # Ensure all workers are idle.
        self._wait_until_all_workers_are_idle()
        # Send the terminate command to all workers.
        for _ in self._workers.values():
            name = "terminate"
            kwargs = dict()
            self._job_queue.put((name, kwargs), block=True)
            log.info("sending sentinel to worker")
        for name, worker in self._workers.items():
            worker.join()
            log.info(f"worker {name} joined.")

    def serialise_agent(self):
        """Write agent to file."""
        to_persist = utils.io.to_dict(
            strategy=self._agent.strategy, regret=self._agent.regret
        )
        joblib.dump(to_persist, self._save_path / "strategy.gz", compress="gzip")
        utils.io.print_strategy(self._agent.strategy)

    def _send_job(self, job_name: str, **kwargs):
        """Send job of type `name` with arguments `kwargs` to worker pool."""
        self._job_queue.put((job_name, kwargs), block=True)

    def _syncronised_job(self, job_name: str, **kwargs):
        """Only perform this job with one process."""
        # Wait for all enqueued jobs to be completed.
        self._job_queue.join()
        # Wait for all workers to become idle.
        self._wait_until_all_workers_are_idle()
        log.info(f"Sending synchronised {job_name} to workers")
        log.info(self._worker_status)
        # Send the job to a single worker.
        self._send_job(job_name, **kwargs)
        # Wait for the synchronised job to be completed.
        self._job_queue.join()
        # The status update of the worker starting the job should be flushed
        # first.
        name_a, status = self._status_queue.get(block=True)
        assert status == job_name, f"expected {job_name} but got {status}"
        # Next get the status update of the job being completed.
        name_b, status = self._status_queue.get(block=True)
        assert status == "idle", f"status should be idle but was {status}"
        assert name_a == name_b, f"{name_a} != {name_b}"

    def _wait_until_all_workers_are_idle(self, sleep_secs=0.5):
        """Blocks until all workers have finished their current job."""
        while True:
            # Read all status updates.
            while not self._status_queue.empty():
                worker_name, status = self._status_queue.get(block=False)
                self._worker_status[worker_name] = status
            # Are all workers idle, all workers statues obtained, if so, stop
            # waiting.
            all_idle = all(status == "idle" for status in self._worker_status.values())
            all_statuses_obtained = len(self._worker_status) == len(self._workers)
            if all_idle and all_statuses_obtained:
                break
            time.sleep(sleep_secs)
