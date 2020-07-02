"""
Usage: poker_ai cluster [OPTIONS]

  Run clustering.

Options:
  --low_card_rank INTEGER        The starting hand rank from 2 through 14 for
                                 the deck we want to cluster. We recommend
                                 starting small.
  --high_card_rank INTEGER       The starting hand rank from 2 through 14 for
                                 the deck we want to cluster. We recommend
                                 starting small.
  --n_river_clusters INTEGER     The number of card information buckets we
                                 would like to create for the river. We
                                 recommend to start small.
  --n_turn_clusters INTEGER      The number of card information buckets we
                                 would like to create for the turn. We
                                 recommend to start small.
  --n_flop_clusters INTEGER      The number of card information buckets we
                                 would like to create for the flop. We
                                 recommend to start small.
  --n_simulations_river INTEGER  The number of opponent hand simulations we
                                 would like to run on the river. We recommend
                                 to start small.
  --n_simulations_turn INTEGER   The number of river card hand simulations we
                                 would like to run on the turn. We recommend
                                 to start small.
  --n_simulations_flop INTEGER   The number of turn card hand simulations we
                                 would like to run on the flop. We recommend
                                 to start small.
  --save_dir TEXT                Path to directory to save card info lookup
                                 table and betting stage centroids.
  --help                         Show this message and exit.
"""
import click

from poker_ai.clustering.card_info_lut_builder import CardInfoLutBuilder


@click.command()
@click.option(
    "--low_card_rank",
    default=10,
    help=(
        "The starting hand rank from 2 through 14 for the deck we want to "
        "cluster. We recommend starting small."
    )
)
@click.option(
    "--high_card_rank",
    default=14,
    help=(
        "The starting hand rank from 2 through 14 for the deck we want to "
        "cluster. We recommend starting small."
    )
)
@click.option(
    "--n_river_clusters",
    default=50,
    help=(
        "The number of card information buckets we would like to create for "
        "the river. We recommend to start small."
    )
)
@click.option(
    "--n_turn_clusters",
    default=50,
    help=(
        "The number of card information buckets we would like to create for "
        "the turn. We recommend to start small."
    )
)
@click.option(
    "--n_flop_clusters",
    default=50,
    help=(
        "The number of card information buckets we would like to create for "
        "the flop. We recommend to start small."
    )
)
@click.option(
    "--n_simulations_river",
    default=6,
    help=(
        "The number of opponent hand simulations we would like to run on the "
        "river. We recommend to start small."
    )
)
@click.option(
    "--n_simulations_turn",
    default=6,
    help=(
        "The number of river card hand simulations we would like to run on the "
        "turn. We recommend to start small."
    )
)
@click.option(
    "--n_simulations_flop",
    default=6,
    help=(
        "The number of turn card hand simulations we would like to run on the "
        "flop. We recommend to start small."
    )
)
@click.option(
    "--save_dir",
    default="",
    help=(
        "Path to directory to save card info lookup table and betting stage "
        "centroids."
    )
)
def cluster(
    low_card_rank: int,
    high_card_rank: int,
    n_river_clusters: int,
    n_turn_clusters: int,
    n_flop_clusters: int,
    n_simulations_river: int,
    n_simulations_turn: int,
    n_simulations_flop: int,
    save_dir: str,
):
    """Run clustering."""
    builder = CardInfoLutBuilder(
        n_simulations_river,
        n_simulations_turn,
        n_simulations_flop,
        low_card_rank,
        high_card_rank,
        save_dir
    )
    builder.compute(
        n_river_clusters,
        n_turn_clusters,
        n_flop_clusters,
    )


if __name__ == "__main__":
    cluster()
