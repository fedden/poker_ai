from __future__ import annotations


class PokerGameState:
    """Poker game state is encoded as immutable data structure.

    At each point in time a poker game is described by the information on the
    table and the player whose turn it is taking an action, plus all previous
    states.
    """

    def __init__(self, previous_state, table, player, action):
        self.previous_state = previous_state
        self.table = table
        self.player = player
        self.action = action

    def __str__(self):
        return "foo"

    @classmethod
    def new_hand(cls, table):
        return PokerGameState(None, table, None, None)
