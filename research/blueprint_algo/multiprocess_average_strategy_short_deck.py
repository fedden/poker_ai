import collections
import glob
import logging
import os
import re
import multiprocessing as mp
from typing import Dict, List, Tuple, Union

import click
import joblib
from rich.logging import RichHandler
from tqdm import tqdm

FORMAT = "%(message)s"
logging.basicConfig(
    format=FORMAT,
    datefmt="[%X] ",
    handlers=[RichHandler()],
    level=logging.INFO,
)

log = logging.getLogger("average_strategy")


def calculate_strategy(this_info_sets_regret: Dict[str, float]) -> Dict[str, float]:
    """Calculate the strategy based on the current information sets regret."""
    # TODO: Could we instanciate a state object from an info set?
    actions = this_info_sets_regret.keys()
    regret_sum = sum([max(regret, 0) for regret in this_info_sets_regret.values()])
    if regret_sum > 0:
        strategy: Dict[str, float] = {
            action: max(this_info_sets_regret[action], 0) / regret_sum
            for action in actions
        }
    else:
        default_probability = 1 / len(actions)
        strategy: Dict[str, float] = {action: default_probability for action in actions}
    return strategy


def try_to_int(text: str) -> Union[str, int]:
    """Attempt to return int."""
    return int(text) if text.isdigit() else text


def natural_key(text):
    """Sort with natural numbers."""
    return [try_to_int(c) for c in re.split(r"(\d+)", text)]


def _load_regret_from_file(dump_path: str) -> Tuple[Dict[str, Dict[str, float]], str]:
    try:
        agent = joblib.load(dump_path)
    except Exception as e:
        tqdm.write(f"Failed to load file at {dump_path} because:{e}")
        agent = {}
    regret = agent.get("regret", {})
    return regret, dump_path


class Worker(mp.Process):
    """"""

    def __init__(self, job_queue, strategy_queue):
        """"""
        super().__init__(group=None, name=None, args=(), kwargs={}, daemon=None)
        # The offline strategy for all information sets.
        self._offline_strategy: Dict[str, Dict[str, float]] = collections.defaultdict(
            lambda: collections.defaultdict(lambda: 0.0)
        )
        self._strategy_queue = strategy_queue
        self._job_queue = job_queue
        log.info(f"setup {self.name}")

    def run(self):
        """"""
        log.info(f"started {self.name}")
        while True:
            file_path = self._job_queue.get(block=True)
            if file_path is None:
                log.info(f"{self.name} got sentinel")
                break
            log.info(f"{self.name} loading {file_path}")
            regret, _ = _load_regret_from_file(file_path)
            # Sum probabilities from computed strategy..
            for info_set, this_info_sets_regret in sorted(regret.items()):
                strategy = calculate_strategy(this_info_sets_regret)
                for action, probability in strategy.items():
                    self._offline_strategy[info_set][action] += probability
            self._job_queue.task_done()
        log.info(f"{self.name} queueing offline_strategy")
        self._strategy_queue.put(self._offline_strategy)
        log.info(f"{self.name} terminated")


def average_strategy(
    all_file_paths: List[str], n_jobs: int = 1
) -> Dict[str, Dict[str, float]]:
    """Compute the mean strategy over all timesteps."""
    n_processes = 32
    # The offline strategy for all information sets.
    job_queue = mp.JoinableQueue()
    for file_path in all_file_paths:
        job_queue.put(file_path)
    # Create the strategy queue and workers.
    strategy_queue = mp.Queue(maxsize=1)
    workers = []
    for _ in range(n_processes):
        worker = Worker(job_queue, strategy_queue)
        workers.append(worker)
    for worker in workers:
        worker.start()
    # Wait until job queue is empty.
    job_queue.join()
    log.info(f"sending sentinels")
    # Send sentinels.
    for _ in range(n_processes):
        job_queue.put(None)
    job_queue.join()
    # Combine all offline strategies.
    offline_strategy: Dict[str, Dict[str, float]] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: 0.0)
    )
    for _ in range(n_processes):
        worker_offline_strategy = strategy_queue.get(block=True)
        for info_set, strategy in worker_offline_strategy.items():
            for action, probability in strategy.items():
                offline_strategy[info_set][action] += probability
    # Normalise summed probabilities.
    # for info_set, this_info_sets_strategy in offline_strategy.items():
    #    norm = sum(this_info_sets_strategy.values())
    #    for action in this_info_sets_strategy.keys():
    #        offline_strategy[info_set][action] /= norm
    # Return regular dict, not defaultdict.
    return {info_set: dict(strategy) for info_set, strategy in offline_strategy.items()}


@click.command()
@click.option(
    "--results_dir_path", default=".", help="the location of the agent file dumps."
)
@click.option(
    "--write_dir_path", default=".", help="where to save the offline strategy"
)
@click.option("--n_jobs", default=1, help="How many parallel processes to read files")
def cli(results_dir_path: str, write_dir_path: str, n_jobs: int):
    """Compute the strategy and write to file."""
    # Find all files to load.
    all_file_paths = glob.glob(os.path.join(results_dir_path, "agent*.gz"))
    if not all_file_paths:
        raise ValueError(f"No agent dumps could be found at: {results_dir_path}")
    # Sort the file paths in the order they were created.
    all_file_paths = sorted(all_file_paths, key=natural_key)
    import ipdb

    ipdb.set_trace()
    offline_strategy = average_strategy(all_file_paths, n_jobs)
    # Save dictionary to compressed file.
    latest_file = os.path.basename(all_file_paths[-1])
    latest_iteration: int = int(re.findall(r"\d+", latest_file)[0])
    save_file: str = f"offline_strategy_{latest_iteration}.gz"
    joblib.dump(offline_strategy, os.path.join(write_dir_path, save_file))


if __name__ == "__main__":
    cli()
