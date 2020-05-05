import copy
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict

from pluribus.games.short_deck import state

log = logging.getLogger("pluribus.utils.io")


def load_info_set_lut(path: str) -> state.InfoSetLookupTable:
    """Load the info set abstraction lookup table."""
    info_set_lut = state.ShortDeckPokerState.load_pickle_files(path)
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


def create_dir() -> Path:
    """Create and get a unique dir path to save to using a timestamp."""
    time = str(datetime.datetime.now())
    for char in ":- .":
        time = time.replace(char, "_")
    path: Path = Path(f"./results_{time}")
    path.mkdir(parents=True, exist_ok=True)
    return path
