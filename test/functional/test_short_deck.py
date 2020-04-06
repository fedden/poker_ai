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
    for player_i in range(n_players):
        assert state.current_player.name == f"player_{player_i}"
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
    # Call for all players.
    for player_i in range(n_players):
        assert state.current_player.name == f"player_{player_i}"
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
