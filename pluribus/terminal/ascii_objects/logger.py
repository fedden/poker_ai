from collections import deque
from datetime import datetime

from blessed import Terminal


class AsciiLogger:
    """"""

    def __init__(self, term: Terminal):
        """"""
        self._log_queue: deque = deque()
        self._term = term
        self.height = None

    def clear(self):
        """"""
        self._log_queue: deque = deque()

    def info(self, *args):
        """"""
        if self.height is None:
            raise ValueError("Logger.height must be set before logging.")
        x: str = " ".join(map(str, args))
        str_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log_queue.append(f"{self._term.skyblue1(str_time)} {x}")
        if len(self._log_queue) > self.height:
            self._log_queue.popleft()

    def __str__(self) -> str:
        """"""
        if self.height is None:
            raise ValueError("Logger.height must be set before logging.")
        n_logs = len(self._log_queue)
        start = max(n_logs - self.height, 0)
        lines = [self._log_queue[i] for i in range(start, n_logs)]
        return "\n".join(lines)
