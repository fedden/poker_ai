def test_short_deck():
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
    import ipdb; ipdb.set_trace()
    state = ShortDeckPokerState(players=players)
    # Call for all players.
    for player_i in range(n_players):
        assert state.current_player.name == f"player_{player_i}"
        assert len(state.legal_actions) == 3
        state = state.apply_action(action_str="call")
    import ipdb

    ipdb.set_trace()


test_short_deck()
