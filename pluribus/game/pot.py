from __future__ import annotations

import collections
import uuid

from pluribus.game.player import Player


class Pot:
    """Class to manage the bets from all players."""

    def __init__(self):
        """Construct the pot, and initialise the counter."""
        self._player_contributions = collections.Counter()
        self._uid = str(uuid.uuid4().hex)
        self.reset()

    def __repr__(self):
        """"""
        contributions = list(self._player_contributions.values())
        return f"<Pot contributions={contributions}>"

    def __getitem__(self, player: Player):
        """Get a players contribution to the pot."""
        if not isinstance(player, Player):
            raise ValueError(
                f'Index the pot with the player to get the contribution.')
        return self._player_contributions[player]

    def reset(self):
        """Reset the pot."""
        self._player_contributions = collections.Counter()

    def add_chips(self, player: Player, n_chips: int):
        """Add chips to the pot, from a player."""
        if n_chips < 0:
            raise ValueError(f'Negative chips cannot be added to the pot.')
        self._player_contributions[player] += n_chips

    @property
    def uid(self):
        """Get a unique identifier for this pot."""
        return self._uid

    @property
    def total(self):
        """Return the total in the pot from all players."""
        return sum(self._player_contributions.values())

