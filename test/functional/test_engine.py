import pytest


@pytest.mark.parametrize("n_players", [2, 4, 6])
def test_hand(n_players):
    """Test a hand can be played."""
    from poker_ai import utils
    from poker_ai.poker.random_player import RandomPlayer
    from poker_ai.poker.table import PokerTable
    from poker_ai.poker.engine import PokerEngine
    from poker_ai.poker.pot import Pot
    utils.random.seed(42)
    initial_chips_amount = 10000
    small_blind_amount = 10
    big_blind_amount = 50
    pot = Pot()
    players = [
        RandomPlayer(
            name=f'player {player_i}',
            initial_chips=initial_chips_amount,
            pot=pot)
        for player_i in range(n_players)
    ]
    table = PokerTable(players=players, pot=pot)
    engine = PokerEngine(
        table=table,
        small_blind=small_blind_amount,
        big_blind=big_blind_amount)
    engine.play_one_round()
