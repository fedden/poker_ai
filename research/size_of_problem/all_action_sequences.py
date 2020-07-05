"""
Similar file to action_sequences.py, except I'm removing the round threshold to
generate all action sequences.
"""
from typing import Tuple, Dict
import sys

import dill as pickle

from pluribus.games.short_deck.player import ShortDeckPokerPlayer
from pluribus.games.short_deck.state import ShortDeckPokerState
from pluribus.poker.pot import Pot


class ActionSequences:
    def __init__(self):
        self.action_combos = []
        self.action_combo = []


def generate_all_action_sequences(
    state: ShortDeckPokerState,
    action_sequences: ActionSequences,
    n_players: int
):
    """
    DFS to return action combos
    """
    if state.is_terminal:
        lst = action_sequences.action_combo.copy()
        action_sequences.action_combos.append(lst)
        nodes_found = len(action_sequences.action_combos)
        if nodes_found % 1000 == 0:
            print(f"Found {nodes_found} of ceiling {133**4}")
            size_of_lst = sys.getsizeof(action_sequences.action_combos)
            print(f"Size of list: {size_of_lst}")
        action_sequences.action_combo.pop()
        return action_sequences.action_combo

    for a in state.legal_actions:
        action_sequences.action_combo.append(a)
        new_state: ShortDeckPokerState = state.apply_action(a)

        action_sequences.action_combo = generate_all_action_sequences(
            new_state, action_sequences, n_players
        )
    if action_sequences.action_combo:
        action_sequences.action_combo.pop()
    return action_sequences.action_combo


def new_game(
    n_players: int,
    small_blind: int = 50,
    big_blind: int = 100,
    initial_chips: int = 10000,
) -> Tuple[ShortDeckPokerState, Pot]:
    """Create a new game."""
    pot = Pot()
    players = [
        ShortDeckPokerPlayer(player_i=player_i, pot=pot, initial_chips=initial_chips)
        for player_i in range(n_players)
    ]
    state = ShortDeckPokerState(
        players=players,
        load_pickle_files=False,
        small_blind=small_blind,
        big_blind=big_blind,
    )
    return state


def create_action_sequences(n_players: int, action_sequences):
    """Create combos of action sequences"""
    state = new_game(n_players)
    generate_all_action_sequences(state, action_sequences, n_players)


if __name__ == "__main__":
    action_sequences = ActionSequences()
    print("3 Players:")
    create_action_sequences(3, action_sequences)
    print(f"# of nodes: {len(action_sequences.action_combos)}")
    with open("all_action_sequences.pkl", "wb") as file:
        pickle.dump(action_sequences.action_combos, file)
