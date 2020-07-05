from __future__ import annotations

from typing import List, TYPE_CHECKING

from poker_ai.poker.dealer import Dealer

if TYPE_CHECKING:
    from poker_ai.poker.player import Player
    from poker_ai.poker.pot import Pot
    from poker_ai.poker.cards import Card


class PokerTable:
    """On a poker table a minimum of two players and one dealer are seated.

    You also find the community cards on it and can see the current pot size.
    Each player is responisble for handling his own cards privately.
    """

    def __init__(self, players: List[Player], pot: Pot, **deck_kwargs):
        """Construct the table."""
        self.players: List[Player] = players
        self.total_n_chips_on_table: int = sum(p.n_chips for p in self.players)
        self.pot: Pot = pot
        self.dealer: Dealer = Dealer(**deck_kwargs)
        self.community_cards: List[Card] = []
        self.n_games: int = 0
        if self.n_players < 2:
            raise ValueError(f'Must be atleast two players on the table.')
        if not all(p.pot.uid == self.pot.uid for p in self.players):
            raise ValueError(f'Players and table point to different pots.')

    @property
    def n_players(self) -> int:
        """How many players are on the table?"""
        return len(self.players)

    def set_players(self, players: List[Player]):
        """Set the players."""
        self.players = players
        if not all(p.pot.uid == self.pot.uid for p in self.players):
            raise ValueError(f'Players and table point to different pots.')

    def add_community_card(self, card: Card):
        """Add a public card to the table for all players to use."""
        self.community_cards.append(card)

    def __repr__(self):
        """Get a nice print out in the debugger for the table."""
        player_names = [player.name for player in self.players]
        return f"<PokerTable players={player_names}>"
