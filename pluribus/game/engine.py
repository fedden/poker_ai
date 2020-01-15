from __future__ import annotations

import collections
import copy
import logging
import operator
from typing import Dict, List, TYPE_CHECKING

from pluribus.game.evaluation.evaluator import Evaluator
from pluribus.game.state import PokerGameState

if TYPE_CHECKING:
    from pluribus.game.player import Player
    from pluribus.game.table import PokerTable


logger = logging.getLogger(__name__)


class PokerEngine:
    """Instance to represent the lifetime of a full poker hand.

    A hand of poker is played at a table by playing for betting rounds:
    pre-flop, flop, turn and river. Small blind and big blind can be set per
    hand, but should generally not change during a session on the table.
    """

    def __init__(self, table: PokerTable, small_blind: int, big_blind: int):
        """"""
        self.table = table
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.evaluator = Evaluator()
        self.state = PokerGameState.new_hand(self.table)
        self.wins_and_losses = []

    def play_one_round(self):
        """"""
        self.table.pot.reset()
        self.assign_order_to_players()
        self.assign_blinds()
        self.table.dealer.deal_private_cards(self.table.players)
        self.betting_round(first_round=True)
        self.table.dealer.deal_flop(self.table)
        self.betting_round()
        self.table.dealer.deal_turn(self.table)
        self.betting_round()
        self.table.dealer.deal_river(self.table)
        self.betting_round()
        # From the active players on the table, compute the winners.
        ranked_player_groups = self.rank_players_by_best_hand()
        payouts = self.compute_payouts(ranked_player_groups)
        self.payout_players(payouts)
        logger.debug("Winnings computation complete. Players:")
        for player in self.table.players:
            logger.debug(f"{player}")
        # TODO(fedden): What if someone runs out of chips here?
        self.move_blinds()

    def _get_players_in_pot(self, player_group, pot):
        """Return the players in the pot, ordered by hand played."""
        return sorted(
            [player for player in player_group if player in pot],
            key=operator.attrgetter("order"),
        )

    def _process_side_pot(self, player_group, pot):
        """Check if this list of players contributed to this side pot."""
        payouts = collections.Counter()
        players_in_pot = self._get_players_in_pot(player_group, pot)
        n_players = len(players_in_pot)
        if not n_players:
            return {}
        n_total = sum(pot.values())
        n_per_player = n_total // n_players
        n_remainder = n_total - n_players * n_per_player
        for player in players_in_pot:
            payouts[player] += n_per_player
        for i in range(n_remainder):
            payouts[players_in_pot[i]] += 1
        return payouts

    def compute_payouts(self, ranked_player_groups: List[Player]):
        """Find the highest ranked players for each sidepot and get winnings"""
        payouts = collections.Counter()
        for pot in self.table.pot.side_pots:
            for player_group in ranked_player_groups:
                pot_payouts = self._process_side_pot(player_group, pot)
                if pot_payouts:
                    payouts.update(pot_payouts)
                    break
        return payouts

    def payout_players(self, payouts: Dict[Player, int]):
        """Remove money from the pot and pay the winning players the chips."""
        self.table.pot.reset()
        for player, winnings in payouts.items():
            player.add_chips(winnings)

    def rank_players_by_best_hand(self) -> List[List[Player]]:
        """Rank all players hands and return the players in order of rank."""
        # The cards that can be passed to the evaluator object from the table.
        table_cards = [card.eval_card for card in self.table.community_cards]
        # For every active player...
        grouped_players = collections.defaultdict(list)
        for player in self.table.players:
            if player.is_active:
                # Get evaluator friendly cards.
                hand_cards = [card.eval_card for card in player.cards]
                # Rank of the best hand - lower is better.
                rank = self.evaluator.evaluate(table_cards, hand_cards)
                hand_class = self.evaluator.get_rank_class(rank)
                hand_desc = self.evaluator.class_to_string(hand_class).lower()
                logger.debug(f'f"Rank #{rank} {player} {hand_desc}')
                grouped_players[rank].append(player)
        # Group players by hand ranks, going from best to worst. We group
        # incase multiple players have identically ranked hands - these players
        # should be in the same list together.
        ranked_player_groups: List[List[Player]] = []
        for rank in sorted(grouped_players.keys()):
            ranked_player_groups.append(grouped_players[rank])
        return ranked_player_groups

    def assign_order_to_players(self):
        """"""
        for player_i, player in enumerate(self.table.players):
            player.order = player_i

    def assign_blinds(self):
        """Assign the blinds to the players."""
        self.table.players[0].add_to_pot(self.small_blind)
        self.table.players[1].add_to_pot(self.big_blind)
        logger.debug(f"Assigned blinds to players {self.table.players[:2]}")

    def move_blinds(self):
        """Rotate the table's player list.

        This is so that the next player in line gets the small blind and the
        right to act first in the next hand.
        """
        players = copy.deepcopy(self.table.players)
        players.append(players.pop(0))
        logger.debug(f"Rotated players from {self.table.players} to {players}")
        self.table.set_players(players)

    def betting_round(self, first_round=False):
        """Computes the round(s) of betting.

        Until the current betting round is complete, all active players take
        actions in the order they were placed at the table. A betting round
        lasts until all players either call the highest placed bet or fold.
        """
        if first_round:
            players = self.table.players[2:] + self.table.players[:2]
        else:
            players = self.table.players
        if self.n_players_with_moves > 1:
            # Ensure for the first move we do one round of betting.
            first_round = True
            logger.debug("Started round of betting.")
            while first_round or self.more_betting_needed:
                # For every active player compute the move, but big and small
                # blind move last..
                for player in players:
                    if player.is_active:
                        self.state = player.take_action(self.state)
                first_round = False
                logger.debug(f"> Betting iter, total: {sum(self.all_bets)}")
            logger.debug(
                f"Finished round of betting, {self.n_active_players} active "
                f"players, {self.n_all_in_players} all in players."
            )
        else:
            logger.debug("Skipping betting as no players are free to bet.")
        self._post_betting_analysis()

    def _post_betting_analysis(self):
        """Log objects and run checks at the end of each round of betting."""
        logger.debug(f"Pot at the end of betting: {self.table.pot}")
        logger.debug("Players at the end of betting:")
        for player in self.table.players:
            logger.debug(f"{player}")
        total_n_chips = self.table.pot.total + sum(
            p.n_chips for p in self.table.players
        )
        n_chips_correct = total_n_chips == self.table.total_n_chips_on_table
        pot_correct = self.table.pot.total == sum(
            p.n_bet_chips for p in self.table.players
        )
        if not n_chips_correct or not pot_correct:
            raise ValueError(
                "Bad logic - total n_chips are not the same as at the start "
                "of the game"
            )

    @property
    def n_players_with_moves(self):
        """Returns the amount of players that can freely make a move."""
        return sum(p.is_active and not p.is_all_in for p in self.table.players)

    @property
    def n_active_players(self):
        """Returns the number of active players."""
        return sum(p.is_active for p in self.table.players)

    @property
    def n_all_in_players(self):
        """Return the amount of players that are active and that are all in."""
        return sum(p.is_active and p.is_all_in for p in self.table.players)

    @property
    def all_bets(self) -> list[int]:
        """Returns all bets made by the players."""
        return [p.n_bet_chips for p in self.table.players]

    @property
    def more_betting_needed(self) -> bool:
        """Returns if more bets are required to terminate betting.

        If all active players have settled, i.e everyone has called the highest
        bet or folded, the current betting round is complete, else, more
        betting is required from the active players that are not all in.
        """
        active_complete_bets = []
        for player in self.table.players:
            if player.is_active and not player.is_all_in:
                active_complete_bets.append(player.n_bet_chips)
        all_bets_equal = all(
            [x == active_complete_bets[0] for x in active_complete_bets]
        )
        return not all_bets_equal
