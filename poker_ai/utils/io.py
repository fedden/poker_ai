import copy
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from poker_ai.games.short_deck import state

log = logging.getLogger("poker_ai.utils.io")


class NumpyJSONEncoder(json.JSONEncoder):
    """Handle those pesky numpy arrays on serialisation."""

    def default(self, obj):
        """Method to handle the conversion of numpy types to Python types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyJSONEncoder, self).default(obj)


def load_info_set_lut(lut_path: Union[str, Path], pickle_dir: bool):
    """Load the info set abstraction lookup table."""
    info_set_lut = state.ShortDeckPokerState.load_card_lut(lut_path, pickle_dir)
    return info_set_lut


def to_dict(**kwargs) -> Dict[str, Any]:
    """Hacky method to convert weird collections dicts to regular dicts."""
    return json.loads(json.dumps(copy.deepcopy(kwargs)))


def print_strategy(strategy: Dict[str, Dict[str, int]]):
    """Print strategy."""
    for info_set, action_to_probabilities in sorted(strategy.items()):
        norm = sum(list(action_to_probabilities.values()))
        log.info(f"{info_set}")
        for action, probability in action_to_probabilities.items():
            log.info(f"  - {action}: {probability / norm:.2f}")


def create_dir(dir_name: str = "results") -> Path:
    """Create and get a unique dir path to save to using a timestamp."""
    time = str(datetime.datetime.now())
    for char in ":- .":
        time = time.replace(char, "_")
    path: Path = Path(f"./{dir_name}_{time}")
    path.mkdir(parents=True, exist_ok=True)
    return path
