import numpy as np
import random


def seed(seed: int = 42):
    """Set random seed for all libraries used to make program deterministic."""
    np.random.seed(seed)
    random.seed(seed)
