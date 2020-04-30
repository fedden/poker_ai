from typing import List, Tuple

from pluribus.games.short_deck.player import ShortDeckPokerPlayer
from pluribus.games.short_deck.state import ShortDeckPokerState
from pluribus.poker.pot import Pot


def generate_preflop_action_sequences(state: ShortDeckPokerState, lst: List):
    """
    DFS to print action combos
    """
    if state.is_terminal or state.betting_round > 0:
        print(lst)
        lst.pop()
        # PREFLOP_ACTION_SEQUENCES.append(state._history)
        return lst

    for a in state.legal_actions:
        lst.append(a)
        new_state: ShortDeckPokerState = state.apply_action(a)

        lst = generate_preflop_action_sequences(new_state, lst)
        # generate_preflop_action_sequences(new_state, lst)
    if lst:
        lst.pop()
    return lst


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


def create_action_sequences(n_players: int = 3):
    """Create combos of action sequences"""
    state = new_game(n_players)
    lst = []
    generate_preflop_action_sequences(state, lst)


if __name__ == "__main__":
    create_action_sequences()
