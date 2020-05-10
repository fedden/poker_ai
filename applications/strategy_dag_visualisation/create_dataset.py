import os
from collections import defaultdict

import joblib
import pandas as pd
from tqdm import tqdm


def to_hand_str(cluster: int) -> str:
    return f"hand_cluster_{cluster}"


strategy_path = "./strategy_2160.gz"
dag_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
data = joblib.load(strategy_path)
for info_set, action_to_probabilities in tqdm(sorted(data["strategy"].items())):
    norm = sum(list(action_to_probabilities.values()))
    for action in ["call", "fold", "raise"]:
        probability = action_to_probabilities.get(action, 0)
        cluster_str, history_str = info_set.split(" ", 1)
        cluster = int(cluster_str[14:-1])
        history = [h for h in eval(history_str[8:]) if h != "skip"]
        history.append(action)
        path = [to_hand_str(cluster), *history]
        level = len(path)
        dag_data[cluster][level]["size"].append(int(1000 * probability / norm))
        dag_data[cluster][level]["path"].append(os.path.join(*path))

cluster = 1
data = defaultdict(list)
data["size"].append(1000)
data["path"].append(to_hand_str(cluster))
for level, size_and_path in sorted(dag_data[cluster].items()):
    size = size_and_path["size"]
    path = size_and_path["path"]
    path, size = (list(t) for t in zip(*sorted(zip(path, size))))
    data["size"] += size
    data["path"] += path
df = pd.DataFrame(data=data)
df.to_csv("dataset.csv", index=False)
