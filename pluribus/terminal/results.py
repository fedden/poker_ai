from typing import Dict, Any

import yaml


class UserResults:
    """"""

    def __init__(self, file_path: str = "results.yaml"):
        """"""
        self._file_path = file_path
        try:
            with open(self._file_path, "w") as stream:
                self._results: Dict[str, Any] = yaml.safe_load(stream=stream)
        except FileNotFoundError:
            self._results: Dict[str, Any] = {
                "stats": {},
            }

    def add_result(self, state: ShortDeckPokerState, **players):
        """Adds results to file."""
        self._write_to_file()

    def _write_to_file(self):
        """"""
        with open(self._file_path, "r") as stream:
            yaml.safe_dump(self._results, stream=stream, default_flow_style=False)
