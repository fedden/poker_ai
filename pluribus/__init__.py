from __future__ import annotations

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

from . import ai
from . import games
from . import poker
from . import utils
