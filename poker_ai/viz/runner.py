import os
from collections import defaultdict
from typing import List, Dict, Any, TypedDict, Tuple
import itertools
import logging
from pathlib import Path

import click
import joblib
import json
import pandas as pd
from tqdm import tqdm
from flask import Flask, render_template


log = logging.getLogger(__name__)

# Flask settings for the viz.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
template_folder = os.path.join(repo_root, "viz")
app = Flask(__name__, template_folder=template_folder)


class StrategySizeLevelDict:
    """
    Object used for storing the size and max level of a strategy for use
    in the strategy viz.
    """

    def __init__(self):
        self.size = defaultdict(int)
        self.max_level = 0


class InfoSet(TypedDict):
    """Info set for a given state. """

    cards_cluster: int
    betting_stage: str
    history: List[Dict[str, List[str]]]


def to_start_of_betting_round_str(info_set_str: str) -> Tuple[str, List[str]]:
    """
    Convert the info set string to a string that represents the start of the betting round
    for use in the DAG vizualizations. Also return the history of actions for
    the current betting stage.

    Example Useage.
    >>> strategy = joblib.load('path/to/strategy_dir/agent.joblib')
    >>> strategy['strategy'].keys()
    Make sure to user '' around the infoset so it can be parsed by your
    shell.

    Eg;
    >>> info_set_str = '{
    >>>    "cards_cluster":9,
    >>>    "betting_stage":"turn",
    >>>    "history":[
    >>>        {"pre_flop":["raise","raise","raise","call","call"]},
    >>>        {"flop":["raise","raise","fold","call"]},
    >>>        {"turn":["raise","raise"]}
    >>>    ]
    >>> }'
    >>> to_start_of_betting_round_str(info_set_str)

    Params
    ------
    info_set_str : str
        The info set string for a given state. As retrieved from the strategy.

    Returns
    -------
    top_of_betting_round : str
        The string representing the start of the betting round for the given state.
        Roughly `{betting_stage}_card_cluster_{cluster}{action_history}`.
    history : List[str]
        The history of actions for the current betting stage.
    """
    # For understanding which betting stage is later than the another.
    comp_lookup: Dict[str, int] = {"pre_flop": 0, "flop": 1, "turn": 2, "river": 3}

    info_set: InfoSet = json.loads(info_set_str)
    cluster = info_set["cards_cluster"]
    betting_stage = info_set["betting_stage"]

    histories_lst: List[Dict[str, List[str]]] = info_set["history"]
    history: List[str] = []
    action_history = ""

    # Loop over betting rounds and add the actions to either 1) the
    # `top_of_betting_round` string or 2) the `history` list.
    # `top_of_betting_round` string represents the state at the top of
    # the betting round for an info set. The remaining actions past and
    # including the current betting stage are kept in the `history` list.
    for past_betting_stage_dict in histories_lst:
        past_betting_stage = list(past_betting_stage_dict.keys())[0]
        if comp_lookup[betting_stage] > comp_lookup[past_betting_stage]:
            action_history += "_" + "-".join(list(past_betting_stage_dict.values())[0])
        else:
            for actions in past_betting_stage_dict.values():
                history.extend(actions)
    top_of_betting_round = f"{betting_stage}_hand_cluster_{cluster}{action_history}"

    return top_of_betting_round, history


def _get_bot_dag_data(strategy_path: Path) -> Dict[str, StrategySizeLevelDict]:
    """
    Loop over the strategy and create a lookup of stats from bot's state strategies.
    We will use this to create a DAG.

    Params
    ------
    strategy_path : Path
        The path to the strategy file.
    """
    bot_dag_data: Dict[str, StrategySizeLevelDict] = defaultdict(StrategySizeLevelDict)
    strategy = joblib.load(strategy_path)

    for info_set_str, action_to_probabilities in tqdm(strategy["strategy"].items()):
        norm = sum(list(action_to_probabilities.values()))

        for next_action in ["call", "fold", "raise"]:
            probability = action_to_probabilities.get(next_action, 0)

            start_of_betting_round, history = to_start_of_betting_round_str(
                info_set_str
            )
            history.append(next_action)

            path = [start_of_betting_round, *history]
            level = len(path) - 1

            # Get a lookup for a path at this state.
            # Size is scale 0-1000, the strategy is converted to a probability then
            # scaled to 0-1000.
            bot_dag_data[start_of_betting_round].size[os.path.join(*path)] = int(
                1000 * probability / norm
            )
            bot_dag_data[start_of_betting_round].max_level = max(
                bot_dag_data[start_of_betting_round].max_level, level
            )
    return bot_dag_data


def _generate_action_combos(base_path, level):
    """
    Helper function to generate all possible action combinations for a given
    level of the DAG. Plus it eliminates invalid paths.

    Params
    ------
    base_path : str
        Usually this would be the `start_of_betting_round` string.
    level : int
        The level of the strategy as based on the start of the betting round.

    Returns
    -------
    valid_paths : List[str]
        A list of valid paths for the given level. Removes sequences of three
        folds prior to the last action.
    """
    actions = ["fold", "raise", "call"]
    combinations = itertools.product(actions, repeat=level)
    paths = [f"{base_path}/{'/'.join(p)}" for p in combinations]
    valid_paths = [
        path
        for path in paths
        if (path.count("fold") - (1 if path.endswith("fold") else 0)) <= 2
    ]
    return valid_paths


def _get_betting_round_action_combos(betting_round_state_viz, max_generate_level):
    """
    For a given betting round, and the max level in the bot's strategy for that
    betting round, generate all possible paths for the betting round.

    Params
    ------
    betting_round_state_viz : str
        The string representing the start of the betting round for the given
        state. This will be the root node of the vizualizations.
    max_generate_level : int
        The max level in the bot's strategy for the given betting round.
    """
    betting_round_action_combos: Dict[str, List[Any]] = defaultdict(list)
    # Size is scale 0-1000, the strategy is converted to a probability then
    # scaled to 0-1000.
    betting_round_action_combos["size"].append(1000)
    betting_round_action_combos["path"].append(betting_round_state_viz)
    betting_round_action_combos["is_player"] = ["not_player"]
    # Get all possible paths for the current state.
    for level in range(1, max_generate_level + 1):
        paths = _generate_action_combos(betting_round_state_viz, level)
        betting_round_action_combos["path"].extend(paths)
        betting_round_action_combos["size"].extend([0] * len(paths))
        betting_round_action_combos["is_player"].extend(["not_player"] * len(paths))
    return betting_round_action_combos


@click.command()
@click.option(
    "--strategy_dir",
    type=click.Path(exists=True),
    help="Path to the strategy directory.",
    required=False,
    default=None,
)
@click.option(
    "--info_set_str",
    type=str,
    help=(
        """
        Infoset to vizualize, you can copy one from the output of the
        strategy.

        >>> strategy = joblib.load('path/to/strategy_dir/agent.joblib')
        >>> strategy['strategy'].keys()
        Make sure to user '' around the infoset so it can be parsed by your
        shell.

        Eg;
        ```
        '{
            "cards_cluster":9,
            "betting_stage":"turn",
            "history":[
                {"pre_flop":["raise","raise","raise","call","call"]},
                {"flop":["raise","raise","fold","call"]},
                {"turn":["raise","raise"]}
            ]
        }'
        ```
        """
    ),
)
@click.option(
    "--max_depth",
    type=int,
    default=7,
    help="Max depth of the viz.  Recommend 7 or less, as the tree grows exponentially (~^3)",
)
@click.option("--host", default="127.0.0.1", help="The interface to bind to.")
@click.option("--port", default=8888, help="The port to bind to.")
def create_viz(strategy_dir, info_set_str, max_depth, host, port):
    """
    Create a vizualization of a bot's strategy in your browser.

    Prerequisites
    -------------
    - You must have a strategy trained via the `poker_ai cluster` and `poker_ai
    train` commands in addition to the repo's installation instructions.
    Though, there is test data included for the `poker_ai viz` command when the
    `strategy_dir` is None.
    - You'll need to get an infoset to vizualize, you can copy one from the
    output of the strategy.

    >>> strategy = joblib.load('path/to/strategy_dir/agent.joblib')
    >>> strategy['strategy'].keys()
    Make sure to user '' around the infoset so it can be parsed by your
    shell.

    Eg;
    ```
    '{
        "cards_cluster":9,
        "betting_stage":"turn",
        "history":[
            {"pre_flop":["raise","raise","raise","call","call"]},
            {"flop":["raise","raise","fold","call"]},
            {"turn":["raise","raise"]}
        ]
    }'

    Run `poker_ai viz --help` for more information about the accepted
    arguments.
    ```
    TODO update the one used for the test data also in the dataset and argument
    default.
    """
    betting_round_state_viz, _ = to_start_of_betting_round_str(info_set_str)

    # Use the strategy provided by the user, otherwise, we'll serve the
    # templated one.
    if strategy_dir is not None:
        strategy_dir = Path(strategy_dir)
        log.info(f"Checking for cached strategy data in {strategy_dir}.")
        # Let's check the cache first. No need to reparse the same strategy a
        # bunch.
        bot_dag_cache_path: Path = strategy_dir / "bot_dag_cache.joblib"
        if bot_dag_cache_path.exists():
            log.info("Cache found, loading from cache.")
            bot_dag_data: Dict[str, StrategySizeLevelDict] = joblib.load(
                bot_dag_cache_path
            )
        else:
            log.info("Cache not found, parsing strategy.")
            bot_dag_data: Dict[str, StrategySizeLevelDict] = _get_bot_dag_data(
                strategy_dir / "agent.joblib"
            )
            log.info("Caching strategy data.")
            joblib.dump(bot_dag_data, bot_dag_cache_path)
        log.info(
            "Done parsing strategy. Generating missing states the bot did not traverse."
        )
        max_generate_level = bot_dag_data[betting_round_state_viz].max_level
        betting_round_action_combos = _get_betting_round_action_combos(
            betting_round_state_viz, max_generate_level
        )

        # Update the size of the nodes based on the bot's strategy.
        for i, path in enumerate(betting_round_action_combos["path"]):
            if path in bot_dag_data[betting_round_state_viz].size:
                betting_round_action_combos["size"][i] = bot_dag_data[
                    betting_round_state_viz
                ].size[path]
                # Used for coloring the vizualization to see the states the bot has
                # a strategy for.
                betting_round_action_combos["is_player"][i] = "player"

        strategy_viz_dataset_path = (
            Path(template_folder) / "static/strategy_viz_dataset.csv"
        )
        strategy_viz_df = pd.DataFrame(data=betting_round_action_combos)
        strategy_viz_df.to_csv(strategy_viz_dataset_path, index=False)
    else:
        log.info("Using test data from the repo.")
        pass

    @app.route("/")
    def index():
        return render_template("index.html", max_depth=max_depth)

    app.run(host=host, port=port)


if __name__ == "__main__":
    create_viz()
