import pytest


def test_short_deck_1():
    """Test the short deck poker game state works as expected."""
    from pluribus.games.short_deck.player import ShortDeckPokerPlayer
    from pluribus.games.short_deck.state import ShortDeckPokerState
    from pluribus.poker.pot import Pot

    n_players = 3
    pot = Pot()
    players = [
        ShortDeckPokerPlayer(player_i=player_i, pot=pot, initial_chips=10000)
        for player_i in range(n_players)
    ]
    state = ShortDeckPokerState(players=players, load_pickle_files=False)
    # Call for all players.
    player_i_order = [1, 2, 0]
    for i in range(n_players):
        assert state.current_player.name == f"player_{player_i_order[i]}"
        assert len(state.legal_actions) == 3
        assert state._betting_stage == "pre_flop"
        state = state.apply_action(action_str="call")
    # Fold for all but last player.
    for player_i in range(n_players - 1):
        assert state.current_player.name == f"player_{player_i}"
        assert len(state.legal_actions) == 3
        assert state._betting_stage == "flop"
        state = state.apply_action(action_str="fold")
    # Only one player left, so game state should be terminal.
    assert state.is_terminal, "state was not terminal"
    assert state._betting_stage == "terminal"


def test_short_deck_2():
    """Test the short deck poker game state works as expected."""
    from pluribus.games.short_deck.player import ShortDeckPokerPlayer
    from pluribus.games.short_deck.state import ShortDeckPokerState
    from pluribus.poker.pot import Pot

    n_players = 3
    pot = Pot()
    players = [
        ShortDeckPokerPlayer(player_i=player_i, pot=pot, initial_chips=10000)
        for player_i in range(n_players)
    ]
    state = ShortDeckPokerState(players=players, load_pickle_files=False)
    player_i_order = [1, 2, 0]
    # Call for all players.
    for i in range(n_players):
        assert state.current_player.name == f"player_{player_i_order[i]}"
        assert len(state.legal_actions) == 3
        assert state._betting_stage == "pre_flop"
        state = state.apply_action(action_str="call")
    # Raise for all players.
    for player_i in range(n_players):
        assert state.current_player.name == f"player_{player_i}"
        assert len(state.legal_actions) == 3
        assert state._betting_stage == "flop"
        state = state.apply_action(action_str="raise")
    # Call for all players and ensure all players have chipped in the same..
    for player_i in range(n_players):
        assert state.current_player.name == f"player_{player_i}"
        assert len(state.legal_actions) == 2
        assert state._betting_stage == "flop"
        state = state.apply_action(action_str="call")
    # Raise for all players.
    for player_i in range(n_players):
        assert state.current_player.name == f"player_{player_i}"
        assert len(state.legal_actions) == 3
        assert state._betting_stage == "turn"
        state = state.apply_action(action_str="raise")
    # Call for all players and ensure all players have chipped in the same..
    for player_i in range(n_players):
        assert state.current_player.name == f"player_{player_i}"
        assert len(state.legal_actions) == 2
        assert state._betting_stage == "turn"
        state = state.apply_action(action_str="call")
    # Fold for all but last player.
    for player_i in range(n_players - 1):
        assert state.current_player.name == f"player_{player_i}"
        assert len(state.legal_actions) == 3
        assert state._betting_stage == "river"
        state = state.apply_action(action_str="fold")
    # Only one player left, so game state should be terminal.
    assert state.is_terminal, "state was not terminal"
    assert state._betting_stage == "terminal"


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
    from pluribus.games.short_deck.player import ShortDeckPokerPlayer
    from pluribus.games.short_deck.state import ShortDeckPokerState
    from pluribus.poker.pot import Pot

    pot = Pot()
    players = [
        ShortDeckPokerPlayer(player_i=player_i, pot=pot, initial_chips=10000)
        for player_i in range(n_players)
    ]
    state = ShortDeckPokerState(players=players, load_pickle_files=False)
    order = list(range(n_players))
    player_i_order = {
        "pre_flop": order[-2:] + order[:-2],
        "flop": order,
        "turn": order,
        "river": order,
    }
    prev_stage = ""
    while state._betting_stage in player_i_order:
        if state._betting_stage != prev_stage:
            # If there is a new betting stage, reset the target player index
            # counter.
            order_i = 0
            prev_stage = state._betting_stage
        target_player_i = player_i_order[state._betting_stage][order_i]
        assert (
            state.current_player.name == f"player_{target_player_i}"
        ), f"{state.current_player.name} != player_{target_player_i}"
        assert (
            state.player_i == target_player_i
        ), f"{state.player_i} != {target_player_i}"
        # All players call to keep things simple.
        state = state.apply_action("call")
        order_i += 1
