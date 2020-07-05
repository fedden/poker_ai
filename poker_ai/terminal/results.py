import collections
import os
from typing import Dict, Any

import numpy as np
import yaml

from poker_ai.games.short_deck.state import ShortDeckPokerState


class UserResults:
    """Class to store user results."""

    def __init__(self):
        """"""
        save_dir = os.path.expanduser("~/.poker")
        os.makedirs(save_dir, exist_ok=True)
        self._file_path = os.path.join(save_dir, "results.yaml")
        try:
            with open(self._file_path, "r") as stream:
                self._results: Dict[str, Any] = yaml.safe_load(stream=stream)
        except FileNotFoundError:
            self._results: Dict[str, Any] = {
                "stats": {},
                "results": [],
            }

    def add_result(
        self,
        strategy_path: str,
        agent: str,
        state: ShortDeckPokerState,
        og_name_to_name: Dict[str, str],
    ):
        """
        Adds results to file.

        Parameters
        ----------
        strategy_path : str
            Path to the strategy.
        agent : Agent
            Trainable entity that stores regret and unnormalized strategy.
        state : ShortDeckPokerState
            Current state of the game.
        og_name_to_name : Dict[str, str]
        """
        ai_key = f"{agent}_{os.path.basename(strategy_path)}"
        players = []
        for player_i, player in enumerate(state.players):
            name = og_name_to_name[player.name]
            player_info_dict = dict(
                name=name,
                args=dict(
                    cards=[c.to_dict() for c in player.cards],
                    value=state.payout[player_i],
                    is_big_blind=player.is_big_blind,
                    is_small_blind=player.is_small_blind,
                    is_dealer=player.is_dealer,
                ),
            )
            players.append(player_info_dict)
        result_entry = dict(
            ai_key=ai_key,
            players=players,
            community_cards=[c.to_dict() for c in state.community_cards],
        )
        self._results["results"].append(result_entry)
        self._compute_human_stats()
        self._write_to_file()

    def _compute_human_stats(self):
        """"""
        values = collections.defaultdict(lambda: collections.defaultdict(list))
        for result_entry in self._results["results"]:
            ai_key = result_entry["ai_key"]
            for player in result_entry["players"]:
                if player["name"].lower() == "human":
                    if player["args"]["is_big_blind"]:
                        key = "BB"
                    elif player["args"]["is_small_blind"]:
                        key = "SB"
                    elif player["args"]["is_dealer"]:
                        key = "D"
                    else:
                        raise NotImplementedError("")
                    values[ai_key][key].append(player["args"]["value"])
                    break
        self._results["stats"] = {
            ai_key: {
                p: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for p, v in positions_to_values.items()
            }
            for ai_key, positions_to_values in values.items()
        }

    def _write_to_file(self):
        """"""
        with open(self._file_path, "w") as stream:
            yaml.safe_dump(self._results, stream=stream, default_flow_style=False)
