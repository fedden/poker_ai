from pluribus.game.table import PokerTable
from pluribus.game.hand import PokerHand
from pluribus.game.player import Player


initial_chips_amount = 10000
small_blind_amount = 10
big_blind_amount = 50

players = [
    Player(name=f'player {player_i}', initial_chips=initial_chips_amount)
    for player_i in range(6)
]
table = PokerTable(players=players)
hand = PokerHand(
    table=table,
    small_blind=small_blind_amount,
    big_blind=big_blind_amount)
hand.play()
