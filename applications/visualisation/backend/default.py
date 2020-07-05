from poker_ai.games.short_deck.player import ShortDeckPokerPlayer
from poker_ai.games.short_deck.state import ShortDeckPokerState
from poker_ai.poker.pot import Pot


def default_state_to_visualise() -> ShortDeckPokerState:
    """"""
    pot = Pot()
    n_players = 3
    players = [
        ShortDeckPokerPlayer(player_i=player_i, initial_chips=10000, pot=pot)
        for player_i in range(n_players)
    ]
    return ShortDeckPokerState(
        players=players, pickle_dir="../../research/blueprint_algo/"
    )


