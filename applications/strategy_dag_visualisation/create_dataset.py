import os
from collections import defaultdict
from typing import List, Dict, Any, TypedDict, Tuple
import itertools

import joblib
import json
import pandas as pd
from tqdm import tqdm


# Will be cli args.
betting_round_viz = "flop_hand_cluster_0_call-call-call"
strategy_path = "../../_2023_04_01_18_29_31_951511/agent.joblib"


class StrategySizeLevelDict:
    def __init__(self):
        self.size = defaultdict(int)
        self.max_level = 0


class InfoSet(TypedDict):
    cards_cluster: int
    betting_stage: str
    history: List[Dict[str, List[str]]]


def to_start_of_betting_round_str(info_set_str: str) -> Tuple[str, List[str]]:
    """
    Convert the info set string to a string that represents the start of the betting round
    for use in the DAG vizualizations. Also return the history of actions for
    the current betting stage.

    TODO: Doc string, show example for fast lookup of a particular betting
    round.
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


def _get_bot_dag_data(strategy_path: str) -> Dict[str, StrategySizeLevelDict]:
    """
    Loop over the strategy and create a lookup of stats from bot's state strategies.
    We will use this to create a DAG.

    TODO
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


def _get_betting_round_action_combos(betting_round_viz, max_generate_level):
    """
    For a given betting round, and the max level in the bot's strategy for that
    betting round, generate all possible paths for the betting round.
    """
    betting_round_action_combos: Dict[str, List[Any]] = defaultdict(list)
    betting_round_action_combos["size"].append(1000)
    betting_round_action_combos["path"].append(betting_round_viz)
    betting_round_action_combos["is_player"] = ["not_player"]
    # Get all possible paths for the current state.
    for level in range(1, max_generate_level + 1):
        paths = _generate_action_combos(betting_round_viz, level)
        betting_round_action_combos["path"].extend(paths)
        betting_round_action_combos["size"].extend([0] * len(paths))
        betting_round_action_combos["is_player"].extend(["not_player"] * len(paths))
    return betting_round_action_combos


bot_dag_data = _get_bot_dag_data(strategy_path)
max_generate_level = bot_dag_data[betting_round_viz].max_level
betting_round_action_combos = _get_betting_round_action_combos(
    betting_round_viz, max_generate_level
)

for i, path in enumerate(betting_round_action_combos["path"]):
    if path in bot_dag_data[betting_round_viz].size:
        betting_round_action_combos["size"][i] = bot_dag_data[betting_round_viz].size[
            path
        ]
        betting_round_action_combos["is_player"][i] = "player"

strategy_viz_df = pd.DataFrame(data=betting_round_action_combos)
strategy_viz_df.to_csv("strategy_viz_dataset.csv", index=False)
