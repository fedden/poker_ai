from __future__ import annotations

import collections
import uuid

from pluribus.game.player import Player


class Pot:
    """Class to manage the bets from all players."""

    def __init__(self):
        """Construct the pot, and initialise the counter."""
        self._side_pots = [{}]
        self._uid = str(uuid.uuid4().hex)
        self.reset()

    def __repr__(self):
        """Nicer way to print a Pot object."""
        return f"<Pot n_chips={self.total}>"

    def __getitem__(self, player: Player):
        """Get a players contribution to the pot."""
        if not isinstance(player, Player):
            raise ValueError(
                f'Index the pot with the player to get the contribution.')
        return sum(pot.get(player, 0) for pot in self.side_pots)

    def add_chips(self, player: Player, n_chips: int):
        """Add chips to the pot, from a player for a given round."""
        if n_chips < 0:
            raise ValueError(f'Negative chips cannot be added to the pot.')
        if player in self._side_pots[-1]:
            # We have already bet with this player, make a new side pot.
            self._add_new_side_pot(player=player, n_chips=n_chips)
        elif all(c == n_chips for c in self._side_pots[-1].values()):
            # All the other players bets are equal to this bet, so add to
            # current side pot.
            self._side_pots[-1][player] = n_chips
        else:
            # Else player is not in the current side pot, and the amount is
            # different to the side current values in the side pot.
            self._split_side_pot(player=player, n_chips=n_chips)

    def reset(self):
        """Reset the pot."""
        self._side_pots = [{}]

    def _add_new_side_pot(self, player: Player, n_chips: int):
        """Make a new side pot."""
        self._side_pots.append({player: n_chips})

    def _split_side_pot(self, player: Player, n_chips: int):
        """Split the current side pot and make a new one."""
        original_n_chips = list(self._side_pots[-1].values())[0]
        original_players = list(self._side_pots[-1].keys())
        smallest_n_chips = min(original_n_chips, n_chips)
        self._side_pots[-1][player] = smallest_n_chips
        self._side_pots[-1] = {
            p: smallest_n_chips for p in original_players + [player]
        }
        n_chips_diff = abs(original_n_chips - n_chips)
        if original_n_chips > n_chips:
            diff_players = original_players
        else:
            diff_players = [player]
        self._side_pots.append({p: n_chips_diff for p in diff_players})

    @property
    def side_pots(self):
        """Returns all side pots."""
        # Collapse side pots as there may be duplicates with the same keys."""
        summed_side_pots = collections.Counter()
        for side_pot in self._side_pots:
            if side_pot:
                key = tuple(side_pot.keys())
                value = list(side_pot.values())[0]
                summed_side_pots[key] += value
        combined_side_pots = [{}]
        for key, value in summed_side_pots.items():
            for k in key:
                combined_side_pots[-1][k] = value
            combined_side_pots.append({})
        # Overwrite side_pots with the collapsed version.
        self._side_pots = combined_side_pots
        return self._side_pots

    @property
    def uid(self):
        """Get a unique identifier for this pot."""
        return self._uid

    @property
    def total(self):
        """Return the total in the pot from all players."""
        return sum(sum(p.values()) for p in self.side_pots)
