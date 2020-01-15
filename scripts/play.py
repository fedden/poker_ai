from pluribus.ai.dummy import RandomPlayer
from pluribus.game.table import PokerTable
from pluribus.game.engine import PokerEngine
from pluribus.game.pot import Pot


initial_chips_amount = 10000
small_blind_amount = 10
big_blind_amount = 50

pot = Pot()
players = [
    RandomPlayer(
        name=f'player {player_i}',
        initial_chips=initial_chips_amount,
        pot=pot)
    for player_i in range(6)
]
table = PokerTable(players=players, pot=pot)
engine = PokerEngine(
    table=table,
    small_blind=small_blind_amount,
    big_blind=big_blind_amount)
engine.play_one_round()
