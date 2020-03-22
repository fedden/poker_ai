"""
Simple script for converting card combos and clusters into a dictionary where a tuple of cards are the keys
  and the cluster id is the value

Cd into clustering to run, it will attempt to drop off files in clustering/data
"""
import dill as pickle
from tqdm import tqdm

with open("data/information_abstraction.pkl", "rb") as file:
    data = pickle.load(file)

if __name__ == "__main__":
    rounds = ["flop", "turn", "river"]
    for round in rounds:
        print(f"Getting indices for {round}")
        card_combos = data[round]
        clusters = data["_" + round + "_clusters"]
        lossy_lookup = {}
        for i, card_combo in enumerate(tqdm(card_combos)):
            lossy_lookup[tuple(card_combo)] = clusters[i]
        location = "data/" + round + "_lossy.pkl"
        with open(location, "wb") as file:
            pickle.dump(lossy_lookup, file)
        print(f"Dumped {round} data to {location}")
