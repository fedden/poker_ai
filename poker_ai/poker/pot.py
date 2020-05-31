from __future__ import annotations

import collections
import uuid

from poker_ai.poker.player import Player


class Pot:
    """"""

    def __init__(self):
        """"""
        self._pot = collections.Counter()
        self._uid = str(uuid.uuid4().hex)

    def __repr__(self):
        """Nicer way to print a Pot object."""
        return f"<Pot n_chips={self.total}>"

    def __getitem__(self, player: Player):
        """Get a players contribution to the pot."""
        if not isinstance(player, Player):
            raise ValueError(
                f'Index the pot with the player to get the contribution.')
        return self._pot[player]

    def add_chips(self, player: Player, n_chips: int):
        """Add chips to the pot, from a player for a given round."""
        self._pot[player] += n_chips

    def reset(self):
        """Reset the pot."""
        self._pot = collections.Counter()

    @property
    def side_pots(self):
        """Compute the side pots."""
        side_pots = []
        if not len(self._pot):
            return []
        pot = {k: v for k, v in self._pot.items()}
        while len(pot):
            side_pots.append({})
            min_n_chips = min(pot.values())
            players_to_pop = []
            for player, n_chips in pot.items():
                side_pots[-1][player] = min_n_chips
                pot[player] -= min_n_chips
                if pot[player] == 0:
                    players_to_pop.append(player)
            for player in players_to_pop:
                pot.pop(player)
        return side_pots

    @property
    def uid(self):
        """Get a unique identifier for this pot."""
        return self._uid

    @property
    def total(self):
        """Return the total in the pot from all players."""
        return sum(self._pot.values())
