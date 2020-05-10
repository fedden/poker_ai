"""Script for using multiprocessing to train agent.

CLI Use
-------

Below you can run `python main.py --help` to get the following description of
the two commands available in the CLI, `resume` and `search`:
```
Usage: main.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  resume  Continue training agent from config loaded from file.
  search  Train agent from scratch.
```

More information on the `search` command can be obtained by running the command
`python main.py search --help`. This will then return the following args that
can be set to guide the agent:
```
Usage: main.py search [OPTIONS]

  Train agent from scratch.

Options:
  --strategy_interval INTEGER  .
  --n_iterations INTEGER       .
  --lcfr_threshold INTEGER     .
  --discount_interval INTEGER  .
  --prune_threshold INTEGER    .
  --c INTEGER                  .
  --n_players INTEGER          .
  --dump_iteration INTEGER     .
  --update_threshold INTEGER   .
  --pickle_dir TEXT            .
  --sync_update_strategy / --async_update_strategy
  --sync_cfr / --async_cfr
  --sync_discount / --async_discount
  --sync_serialise / --async_serialise
  --help                       Show this message and exit.
```
"""
from pathlib import Path
from typing import Dict

import click
import joblib
import yaml

from pluribus import utils
from server import Server


@click.group()
def cli():
    pass


@cli.command()
@click.option("--server_config_path", default="./server.gz", help=".")
def resume(server_config_path: str):
    """Continue training agent from config loaded from file."""
    try:
        config = joblib.load(server_config_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Server config file not found at the path: {server_config_path}\n "
            f"Please set the path to a valid file dumped by a previous session."
        )
    server = Server.from_dict(config)
    server.search()
    server.terminate()


@cli.command()
@click.option("--strategy_interval", default=2, help=".")
@click.option("--n_iterations", default=10, help=".")
@click.option("--lcfr_threshold", default=80, help=".")
@click.option("--discount_interval", default=1000, help=".")
@click.option("--prune_threshold", default=4000, help=".")
@click.option("--c", default=-20000, help=".")
@click.option("--n_players", default=3, help=".")
@click.option("--dump_iteration", default=10, help=".")
@click.option("--update_threshold", default=0, help=".")
@click.option("--pickle_dir", default="../blueprint_algo", help=".")
@click.option("--sync_update_strategy/--async_update_strategy", default=False, help=".")
@click.option("--sync_cfr/--async_cfr", default=False, help=".")
@click.option("--sync_discount/--async_discount", default=False, help=".")
@click.option("--sync_serialise/--async_serialise", default=False, help=".")
def search(
    strategy_interval: int,
    n_iterations: int,
    lcfr_threshold: int,
    discount_interval: int,
    prune_threshold: int,
    c: int,
    n_players: int,
    dump_iteration: int,
    update_threshold: int,
    pickle_dir: str,
    sync_update_strategy: bool,
    sync_cfr: bool,
    sync_discount: bool,
    sync_serialise: bool,
):
    """Train agent from scratch."""
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
        dump_iteration=dump_iteration,
        update_threshold=update_threshold,
        save_path=save_path,
        pickle_dir=pickle_dir,
        sync_update_strategy=sync_update_strategy,
        sync_cfr=sync_cfr,
        sync_discount=sync_discount,
        sync_serialise=sync_serialise,
    )
    server.search()
    server.terminate()


if __name__ == "__main__":
    cli()
