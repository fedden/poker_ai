"""
DFS for finding possible action sequences using the ShortDeckPoker state, but this
is application independent, so it should be usable for other games that could be modeled
as tree.

In this case each round of poker has the same combinations of possible action sequences
where n_players is equal to the number of players at the start of the round. I'm just generating
the preflop round combos for this reason, but they apply to each round.

For 3 players and 2 players the space of possible action sequences is small enough to search
for testing the ShortDeckPokerState class. However, if we move to more players, we may have
to switch to validating with action sequence generation logic.
"""
from typing import Tuple, Dict

import dill as pickle

from pluribus.games.short_deck.player import ShortDeckPokerPlayer
from pluribus.games.short_deck.state import ShortDeckPokerState
from pluribus.poker.pot import Pot


class ActionSequences:
    def __init__(self):
        self.action_combos = {2: [], 3: []}
        self.action_combo = []


def generate_preflop_action_sequences(
    state: ShortDeckPokerState,
    action_sequences: ActionSequences,
    n_players: int
):
    """
    DFS to return action combos
    """
    if state.is_terminal or state.betting_round > 0:
        lst = action_sequences.action_combo.copy()
        print(lst)
        action_sequences.action_combos[n_players].append(lst)
        action_sequences.action_combo.pop()
        return action_sequences.action_combo

    for a in state.legal_actions:
        action_sequences.action_combo.append(a)
        new_state: ShortDeckPokerState = state.apply_action(a)

        action_sequences.action_combo = generate_preflop_action_sequences(
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
    generate_preflop_action_sequences(state, action_sequences, n_players)


if __name__ == "__main__":
    action_sequences = ActionSequences()
    print("3 Players:")
    create_action_sequences(3, action_sequences)
    print("2 Players:")
    create_action_sequences(2, action_sequences)
    with open("action_sequences.pkl", "wb") as file:
        pickle.dump(action_sequences.action_combos, file)
