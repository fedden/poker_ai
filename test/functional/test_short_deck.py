import collections
import copy
import random
from typing import List, Tuple, Optional

import pytest
import numpy as np
import dill as pickle

from poker_ai.games.short_deck.state import ShortDeckPokerState
from poker_ai.games.short_deck.player import ShortDeckPokerPlayer
from poker_ai.poker.card import Card
from poker_ai.poker.pot import Pot
from poker_ai.utils.random import seed


def _new_game(
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
        load_card_lut=False,
        small_blind=small_blind,
        big_blind=big_blind,
    )
    return state, pot


def _load_action_sequences(directory):
    with open(directory, "rb") as file:
        action_sequences = pickle.load(file)
    return action_sequences


def test_short_deck_1():
    """Test the short deck poker game state works as expected."""
    n_players = 3
    state, _ = _new_game(n_players=n_players)
    # Call for all players.
    player_i_order = [2, 0, 1]
    for i in range(n_players):
        assert state.current_player.name == f"player_{player_i_order[i]}"
        assert len(state.legal_actions) == 3
        assert state.betting_stage == "pre_flop"
        state = state.apply_action(action_str="call")
    assert state.betting_stage == "flop"
    # Fold for all but last player.
    for player_i in range(n_players - 1):
        assert state.current_player.name == f"player_{player_i}"
        assert len(state.legal_actions) == 3
        assert state.betting_stage == "flop"
        state = state.apply_action(action_str="fold")
    # Only one player left, so game state should be terminal.
    assert state.is_terminal, "state was not terminal"
    assert state.betting_stage == "terminal"


def test_short_deck_2():
    """Test the short deck poker game state works as expected."""
    n_players = 3
    state, _ = _new_game(n_players=3)
    player_i_order = [2, 0, 1]
    # Call for all players.
    for i in range(n_players):
        assert state.current_player.name == f"player_{player_i_order[i]}"
        assert len(state.legal_actions) == 3
        assert state.betting_stage == "pre_flop"
        state = state.apply_action(action_str="call")
    # Raise for all players.
    for player_i in range(n_players):
        assert state.current_player.name == f"player_{player_i}"
        assert len(state.legal_actions) == 3
        assert state.betting_stage == "flop"
        state = state.apply_action(action_str="raise")
    # Call for all players and ensure all players have chipped in the same..
    for player_i in range(n_players - 1):
        assert state.current_player.name == f"player_{player_i}"
        assert len(state.legal_actions) == 2
        assert state.betting_stage == "flop"
        state = state.apply_action(action_str="call")
    # Raise for all players.
    for player_i in range(n_players):
        assert state.current_player.name == f"player_{player_i}"
        assert len(state.legal_actions) == 3
        assert state.betting_stage == "turn"
        state = state.apply_action(action_str="raise")
    # Call for all players and ensure all players have chipped in the same..
    for player_i in range(n_players - 1):
        assert state.current_player.name == f"player_{player_i}"
        assert len(state.legal_actions) == 2
        assert state.betting_stage == "turn"
        state = state.apply_action(action_str="call")
    # Fold for all but last player.
    for player_i in range(n_players - 1):
        assert state.current_player.name == f"player_{player_i}"
        assert len(state.legal_actions) == 3
        assert state.betting_stage == "river"
        state = state.apply_action(action_str="fold")
    # Only one player left, so game state should be terminal.
    assert state.is_terminal, "state was not terminal"
    assert state.betting_stage == "terminal"


@pytest.mark.parametrize(
    "n_players",
    [
        pytest.param(0, marks=pytest.mark.xfail),
        pytest.param(1, marks=pytest.mark.xfail),
        2,
        3,
        4,
    ],
)
def test_short_deck_3(n_players: int):
    """Check the state fails when the wrong number of players are provided.

    Test the short deck poker game state works as expected - make sure the
    order of the players is correct - for the pre-flop it should be
    [-1, -2, 0, 1, ..., -3].
    """
    state, _ = _new_game(n_players=n_players)
    order = list(range(n_players))
    player_i_order = {
        "pre_flop": order[2:] + order[:2],
        "flop": order,
        "turn": order,
        "river": order,
    }
    prev_stage = ""
    while state.betting_stage in player_i_order:
        if state.betting_stage != prev_stage:
            # If there is a new betting stage, reset the target player index
            # counter.
            order_i = 0
            prev_stage = state.betting_stage
        target_player_i = player_i_order[state.betting_stage][order_i]
        assert (
            state.current_player.name == f"player_{target_player_i}"
        ), f"{state.current_player.name} != player_{target_player_i}"
        assert (
            state.player_i == target_player_i
        ), f"{state.player_i} != {target_player_i}"
        # All players call to keep things simple.
        state = state.apply_action("call")
        order_i += 1


@pytest.mark.parametrize("n_players", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("small_blind", [50, 200])
@pytest.mark.parametrize("big_blind", [100, 1000])
def test_pre_flop_pot(n_players: int, small_blind: int, big_blind: int):
    """Test preflop the state is set up for player 2 to start betting."""
    state, pot = _new_game(
        n_players=n_players, small_blind=small_blind, big_blind=big_blind,
    )
    n_bet_chips = sum(p.n_bet_chips for p in state.players)
    target = small_blind + big_blind
    assert state.player_i == 0 if n_players == 2 else 2
    assert state.betting_stage == "pre_flop"
    assert (
        n_bet_chips == target
    ), f"small and big blind have not bet! {n_bet_chips} == {target}"
    assert (
        n_bet_chips == pot.total
    ), f"small and big blind have are not in pot! {n_bet_chips} == {pot.total}"


def test_flops_are_random():
    """Ensure across multiple runs that the flop varies."""

    def _get_flop(state: ShortDeckPokerState) -> List[Card]:
        """Get the public cards for the flop stage."""
        save_state = copy.deepcopy(state)
        while save_state.betting_stage != "flop":
            # accounting for when we hit a terminal node before the flop
            if save_state.betting_stage == "terminal":
                return _get_flop(state)
            action: Optional[str] = random.choice(save_state.legal_actions)
            save_state = save_state.apply_action(action)
        return save_state._table.community_cards

    seed(42)
    state, _ = _new_game(n_players=3, small_blind=50, big_blind=100)
    n_iterations = 5
    # We'll store the public cards from the flop as eval_cards (ints).
    flops: List[Tuple[int, int, int]] = []
    for _ in range(n_iterations):
        flop = _get_flop(state)
        flops.append(tuple([card.eval_card for card in flop]))
    flop_occurances = collections.Counter(flops)
    # Ensure that we have not had the same flop `n_iterations` number of times
    # repeatedly.
    assert len(flop_occurances) > 1 and len(flop_occurances.most_common()) != 1


@pytest.mark.parametrize("n_players", [2, 3])
def test_call_action_sequence(n_players):
    """
    Make sure we never see an action sequence of "raise", "call", "call" in the same
    round with only two players. There would be a similar analog for more than two players,
    but this should aid in initially finding the bug.
    """
    # Seed the random number generation so things are procedural/reproducable.
    seed(42)
    # example of a bad sequence in a two-handed game in one round
    bad_seq = ["raise", "call", "call"]
    # Run some number of random iterations.
    for _ in range(200):
        state, _ = _new_game(n_players=n_players, small_blind=50, big_blind=100)
        betting_round_dict = collections.defaultdict(list)
        while state.betting_stage not in {"show_down", "terminal"}:
            uniform_probability: float = 1 / len(state.legal_actions)
            probabilities = np.full(len(state.legal_actions), uniform_probability)
            random_action: str = np.random.choice(state.legal_actions, p=probabilities)
            if state._poker_engine.n_active_players == 2:
                betting_round_dict[state.betting_stage].append(random_action)
                no_fold_action_history: List[str] = [
                    action
                    for action in betting_round_dict[state.betting_stage]
                    if action != "skip"
                ]
                # Loop through the action history and make sure the bad
                # sequence has not happened.
                for i in range(len(no_fold_action_history)):
                    history_slice = no_fold_action_history[i : i + len(bad_seq)]
                    assert history_slice != bad_seq
            state = state.apply_action(random_action)


@pytest.mark.parametrize("n_players", [2, 3])
def test_action_sequence(n_players: int):
    """
    Check each round against validated action sequences to ensure the state class is
    working correctly.
    """
    # Seed the random number generation so things are procedural/reproducable.
    seed(42)
    directory = "research/size_of_problem/action_sequences.pkl"
    action_sequences = _load_action_sequences(directory)
    for i in range(200):
        state, _ = _new_game(n_players=n_players, small_blind=50, big_blind=100)

        betting_stage_dict = {
            "pre_flop": {"action_sequence": [], "n_players": 0},
            "flop": {"action_sequence": [], "n_players": 0},
            "turn": {"action_sequence": [], "n_players": 0},
            "river": {"action_sequence": [], "n_players": 0},
        }
        betting_stage = None

        while state.betting_stage not in {"show_down", "terminal"}:
            if betting_stage != state.betting_stage:
                betting_stage_dict[state.betting_stage][
                    "n_players"
                ] = state.n_players_started_round
                betting_stage = state.betting_stage
            uniform_probability: float = 1 / len(state.legal_actions)
            probabilities = np.full(len(state.legal_actions), uniform_probability)
            random_action: str = np.random.choice(state.legal_actions, p=probabilities)

            betting_stage_dict[state.betting_stage]["action_sequence"].append(
                random_action
            )
            state = state.apply_action(random_action)

        for betting_stage in betting_stage_dict.keys():
            if betting_stage_dict[betting_stage]["action_sequence"]:
                n_players_started_round = betting_stage_dict[betting_stage]["n_players"]
                action_sequence = betting_stage_dict[betting_stage]["action_sequence"]
                possible_sequences = action_sequences[n_players_started_round]

                assert action_sequence in possible_sequences


def test_skips(n_players: int = 3):
    """
    Check each round to make sure that skips are mod number of players and appended on
    the skipped player's turn
    """
    # Seed the random number generation so things are procedural/reproducable.
    seed(42)
    for _ in range(500):
        state, _ = _new_game(n_players=n_players, small_blind=50, big_blind=100)

        while True:
            uniform_probability: float = 1 / len(state.legal_actions)
            probabilities = np.full(len(state.legal_actions), uniform_probability)
            random_action: str = np.random.choice(state.legal_actions, p=probabilities)
            state = state.apply_action(random_action)

            if state.betting_stage in {"show_down", "terminal"}:
                break

        # TODO: wrap this in a for loop, add history as property of poker state class
        preflop_actions = state._history["pre_flop"]
        preflop_folds = [i for i, x in enumerate(preflop_actions) if x == "fold"]
        # accounting for rotation preflop
        preflop_fold_players = [(x + 2) % 3 for x in preflop_folds]

        if state._history["flop"] is not None:
            flop_actions = state._history["flop"]
            flop_folds = [i for i, x in enumerate(flop_actions) if x == "fold"]

        if state._history["turn"] is not None:
            turn_actions = state._history["turn"]
            turn_folds = [i for i, x in enumerate(turn_actions) if x == "fold"]

        for stage in state._history.keys():
            if state._history[stage] is not None:
                if stage == "flop":
                    fold_players = preflop_fold_players
                if stage == "turn":
                    fold_players = preflop_fold_players + flop_folds
                if stage == "river":
                    fold_players = preflop_fold_players + flop_folds + turn_folds

                actions = state._history[stage]
                folds = [i for i, x in enumerate(actions) if x == "fold"]

                for fold_idx in folds:
                    for i, action in enumerate(actions[fold_idx:]):
                        # i greater than 0 because 0 is a fold
                        if i > 0 and i % n_players == 0:
                            assert action == "skip"

                if stage != "pre_flop":
                    for fold_idx in fold_players:
                        # i can be 0 because folds happened in a previous round
                        for i, action in enumerate(actions[fold_idx:]):
                            if i % n_players == 0:
                                assert action == "skip"
