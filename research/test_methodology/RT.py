from typing import Dict

from pluribus.games.short_deck.state import *


# this is a decent function for loading up a particular action sequence
def get_game_state(state: ShortDeckPokerState, action_sequence: list):
    """Follow through the action sequence provided to get current node"""
    if not action_sequence:
        return state
    a = action_sequence.pop(0)
    state.apply_action(a)
    if a == "skip":
        a = action_sequence.pop(0)
        state.apply_action(a)
    new_state = state.apply_action(a)
    return get_game_state(new_state, action_sequence)


# added some flags for RT
def new_rt_game(
    n_players: int, offline_strategy: Dict, real_time=True
) -> ShortDeckPokerState:
    """Create a new game of short deck poker."""
    pot = Pot()
    players = [
        ShortDeckPokerPlayer(player_i=player_i, initial_chips=10000, pot=pot)
        for player_i in range(n_players)
    ]
    state = ShortDeckPokerState(
        players=players, offline_strategy=offline_strategy, real_time=real_time
    )
    return state


if __name__ == "__main__":
    # we load a (trained) strategy
    agent1 = TrainedAgent("../blueprint_algo/results_2020_05_10_21_36_47_291425")
    action_sequence = ["raise", "call", "raise"]
    state: ShortDeckPokerState = new_rt_game(3, agent1.offline_strategy)
    # load up a particular action sequence
    current_game_state = get_game_state(state, action_sequence)
    # decided to make this a one time method rather than something that always updates
    # reason being: we won't need it except for a few choice nodes
    current_game_state.update_hole_cards_bayes()

    import ipdb
    ipdb.set_trace()
    # TODO: need function for dealing starting hands according to the bayesian updates
    # TODO: need a function to load a particular flop
    # TODO: use that game state and run CFR to update strategy (easy as passing that state to cfr)
