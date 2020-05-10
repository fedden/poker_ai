import collections
import glob
import os
from typing import Dict

import click
import joblib
from tqdm import tqdm


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


def average_strategy(results_dir_path: str) -> Dict[str, Dict[str, float]]:
    """Compute the mean strategy over all timesteps."""
    # Find all files to load.
    all_file_paths = glob.glob(os.path.join(results_dir_path, "agent*.gz"))
    if not all_file_paths:
        raise ValueError(f"No agent dumps could be found at: {results_dir_path}")
    # The offline strategy for all information sets.
    offline_strategy: Dict[str, Dict[str, float]] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: 0.0)
    )
    # Sum up all strategies.
    for dump_path in tqdm(all_file_paths, desc="loading dumps"):
        # Load file.
        try:
            agent = joblib.load(dump_path)
        except Exception as e:
            tqdm.write(f"Failed to load file at {dump_path} because:{e}")
            agent = {}
        regret = agent.get("regret", {})
        # Sum probabilities from computed strategy..
        for info_set, this_info_sets_regret in sorted(regret.items()):
            strategy = calculate_strategy(this_info_sets_regret)
            for action, probability in strategy.items():
                offline_strategy[info_set][action] += probability
    # Normalise summed probabilities.
    for info_set, this_info_sets_strategy in offline_strategy.items():
        norm = sum(this_info_sets_strategy.values())
        for action in this_info_sets_strategy.keys():
            offline_strategy[info_set][action] /= norm
    # Return regular dict, not defaultdict.
    return {info_set: dict(strategy) for info_set, strategy in offline_strategy.items()}


@click.command()
@click.option(
    "--results_dir_path", default=".", help="the location of the agent file dumps."
)
@click.option(
    "--write_dir_path", default=".", help="where to save the offline strategy"
)
def cli(results_dir_path: str, write_dir_path: str):
    """Compute the strategy and write to file."""
    offline_strategy = average_strategy(results_dir_path)
    # Save dictionary to compressed file.
    joblib.dump(offline_strategy, os.path.join(write_dir_path, "offline_strategy.gz"))


if __name__ == "__main__":
    cli()
