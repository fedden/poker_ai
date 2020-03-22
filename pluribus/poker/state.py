from __future__ import annotations


class PokerGameState:
    """Poker game state is encoded as an immutable data structure.

    At each point in time a poker game is described by the information on the
    table and the player whose turn it is taking an action, plus all previous
    states.
    """

    def __init__(self, previous_state, table, player, action, is_terminal):
        self._previous_state = previous_state
        self._table = table
        self._player = player
        self._action = action

    def __repr__(self):
        """Make a nice printable node."""
        name = "<PokerGameState prev_state={} table={} player={} action={}>"
        return name.format(
            self._previous_state, self._table, self._player, self._action
        )

    @classmethod
    def new_hand(cls, table):
        return PokerGameState(
            previous_state=None,
            table=table,
            player=None,
            action=None,
            is_terminal=False,
        )

    @property
    def table(self):
        """"""
        return self._table

    @property
    def is_terminal(self):
        """"""
        return self._is_terminal

    @property
    def is_chance_node(self):
        """"""
        return False

    @property
    def current_player(self):
        """Get the current player."""
        return self._player

    @property
    def utility(self):
        """"""
        if self.is_terminal:
            utility = [-1.0 for _ in range(self._table.n_players)]
            # TODO(fedden): How to route the winner index here.
            utility[winner_i] = 1.0
            return utility
        return [0.0 for _ in range(self._table.n_players)]
