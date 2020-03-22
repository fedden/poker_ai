from __future__ import annotations

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

from . import ai
from . import games
from . import poker
from . import utils
