from __future__ import annotations

import copy
import collections
import datetime
import json
import random
from pathlib import Path
from typing import Any, Dict
import logging

logging.basicConfig(filename="test.txt", level=logging.DEBUG)

import click
import joblib
import numpy as np
import yaml
from tqdm import tqdm, trange

from pluribus import utils
from pluribus.games.short_deck.player import ShortDeckPokerPlayer
from pluribus.games.short_deck.state import ShortDeckPokerState
from pluribus.poker.pot import Pot

action_dict = collections.defaultdict(
    lambda: []
)


def cfr(state: ShortDeckPokerState, lst, t) -> float:
    """
    """
    if lst:
        num_raises = len([x for x in lst if x == 'raise'])
    else:
        num_raises = 0
    if state.is_terminal or state.betting_round > 0 or num_raises == 3:
        print(lst)
        lst.pop()
        return lst

    for a in state.legal_actions:
        print(state._n_raises, state.legal_actions)
        lst.append(a)
        new_state: ShortDeckPokerState = state.apply_action(a)

        lst = cfr(new_state, lst, t)




def new_game(n_players: int, info_set_lut: Dict[str, Any] = {}) -> ShortDeckPokerState:
    """Create a new game of short deck poker."""
    pot = Pot()
    players = [
        ShortDeckPokerPlayer(player_i=player_i, initial_chips=10000, pot=pot)
        for player_i in range(n_players)
    ]

    # Don't reload massive files, it takes ages.
    state = ShortDeckPokerState(players=players, load_pickle_files=False)
    state.info_set_lut = info_set_lut

    return state


def create_action_sequences(action_dict):
    n_players = 3
    """create combos of action sequences"""
    state: ShortDeckPokerState = new_game(n_players)
    t = 0
    lst = []
    cfr(state, lst, t)
    print(action_dict)


if __name__ == "__main__":
    create_action_sequences(action_dict)



