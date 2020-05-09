from pathlib import Path
from typing import Dict

import click
import yaml

from pluribus import utils
from server import Server


@click.command()
@click.option("--strategy_interval", default=2, help=".")
@click.option("--n_iterations", default=10, help=".")
@click.option("--lcfr_threshold", default=80, help=".")
@click.option("--discount_interval", default=1000, help=".")
@click.option("--prune_threshold", default=4000, help=".")
@click.option("--c", default=-20000, help=".")
@click.option("--n_players", default=3, help=".")
@click.option("--print_iteration", default=10, help=".")
@click.option("--dump_iteration", default=10, help=".")
@click.option("--update_threshold", default=0, help=".")
@click.option("--sync_update_strategy", default=False, help=".")
@click.option("--sync_cfr", default=False, help=".")
@click.option("--sync_discount", default=False, help=".")
@click.option("--sync_serialise_agent", default=False, help=".")
def search(
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
    sync_update_strategy: bool,
    sync_cfr: bool,
    sync_discount: bool,
    sync_serialise_agent: bool,
):
    """Train agent."""
    # Write config to file, and create directory to save results in.
    config: Dict[str, int] = {**locals()}
    save_path: Path = utils.io.create_dir()
    with open(save_path / "config.yaml", "w") as steam:
        yaml.dump(config, steam)
    # Create the server that controls/coordinates the workers.
    server = Server(
        strategy_interval=strategy_interval,
        n_iterations=n_iterations,
        lcfr_threshold=lcfr_threshold,
        discount_interval=discount_interval,
        prune_threshold=prune_threshold,
        c=c,
        n_players=n_players,
        print_iteration=print_iteration,
        dump_iteration=dump_iteration,
        update_threshold=update_threshold,
        save_path=save_path,
        pickle_dir="..",
    )
    # Minimise the regret!
    server.search(
        sync_update_strategy=sync_update_strategy,
        sync_cfr=sync_cfr,
        sync_discount=sync_discount,
        sync_serialise_agent=sync_serialise_agent,
    )
    server.terminate()
    server.serialise_agent()


if __name__ == "__main__":
    search()
