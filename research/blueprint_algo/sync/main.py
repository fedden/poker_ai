import click

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
):
    """Train agent."""
    # Get the values passed to this method, save this.
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
        # n_processes=1,
        seed=42,
        pickle_dir="..",
    )
    server.search()
    server.terminate()
    server.serialise_agent()


if __name__ == "__main__":
    search()
