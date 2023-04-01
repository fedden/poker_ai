import os
from collections import defaultdict
from typing import List, Dict, Any
import itertools

import joblib
import json
import pandas as pd
from tqdm import tqdm


def to_current_state_str(betting_stage: str, cluster: int, action_history: str) -> str:
    return f"{betting_stage}_hand_cluster_{cluster}{action_history}"


class NestedDict:
    def __init__(self):
        self.size = defaultdict(int)
        self.max_level = 0


strategy_path = "../../_2023_04_01_18_29_31_951511/agent.joblib"
dag_data: Dict[str, NestedDict] = defaultdict(NestedDict)
strategy = joblib.load(strategy_path)
comp_lookup: Dict[str, int] = {"pre_flop": 0, "flop": 1, "turn": 2, "river": 3}
for info_set_str, action_to_probabilities in tqdm(strategy["strategy"].items()):
    norm = sum(list(action_to_probabilities.values()))
    for next_action in ["call", "fold", "raise"]:
        probability = action_to_probabilities.get(next_action, 0)
        info_set = json.loads(info_set_str)
        cluster = info_set["cards_cluster"]
        betting_stage = info_set["betting_stage"]
        histories_lst: List[Dict[str, List[str]]] = info_set["history"]
        history: List[str] = []
        action_history = ""
        for past_betting_stage_dict in histories_lst:
            past_betting_stage = list(past_betting_stage_dict.keys())[0]
            if comp_lookup[betting_stage] > comp_lookup[past_betting_stage]:
                action_history += "_" + "-".join(
                    list(past_betting_stage_dict.values())[0]
                )
            else:
                for actions in past_betting_stage_dict.values():
                    # betting_stage_action = [
                    #     list(betting_stage.keys())[0] + "_" + a for a in actions
                    # ]
                    history.extend(actions)
        history.append(next_action)
        current_state = to_current_state_str(betting_stage, cluster, action_history)
        if (
            current_state
            == "river_hand_cluster_9_raise-raise-raise-call-call_raise-raise-fold-call_call-raise-skip-raise-raise-skip-call_raise-raise"
        ):
            import ipdb

            ipdb.set_trace()
        path = [current_state, *history]
        level = len(path) - 1
        # Get a lookup for a path at this state.
        dag_data[current_state].size[os.path.join(*path)] = int(
            1000 * probability / norm
        )
        dag_data[current_state].max_level = max(
            dag_data[current_state].max_level, level
        )


def generate_combinations(base_path, level):
    combinations = itertools.product(actions, repeat=level)
    paths = [f"{base_path}/{'/'.join(p)}" for p in combinations]
    valid_paths = [
        path
        for path in paths
        if (path.count("fold") - (1 if path.endswith("fold") else 0)) <= 2
    ]
    return valid_paths


actions = ["fold", "raise", "call"]
current_state = "flop_hand_cluster_0_call-call-call"
current_state_folds = current_state.count("fold")
import ipdb

ipdb.set_trace()
max_generate_level = dag_data[current_state].max_level
data: Dict[str, List[Any]] = defaultdict(list)
data["size"].append(1000)
data["path"].append(current_state)
data["is_player"] = ["not_player"]
# Get all possible paths for the current state.
for level in range(1, max_generate_level + 1):
    paths = generate_combinations(current_state, level)
    data["path"].extend(paths)
    data["size"].extend([0] * len(paths))
    data["is_player"].extend(["not_player"] * len(paths))

for i, path in enumerate(data["path"]):
    if path in dag_data[current_state].size:
        data["size"][i] = dag_data[current_state].size[path]
        data["is_player"][i] = "player"

df = pd.DataFrame(data=data)
df.to_csv("dataset.csv", index=False)
