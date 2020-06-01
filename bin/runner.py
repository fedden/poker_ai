import click

from poker_ai.ai.runner import train


@click.group()
def cli():
    """The CLI for the poker_ai package that groups the various scripts.

    Select {train, ...} and set the subsequent options to build your model.
    """
    pass


cli.add_command(train)
