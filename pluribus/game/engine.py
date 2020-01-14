from __future__ import annotations

import copy
import logging
from operator import itemgetter
from typing import List, TYPE_CHECKING

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
        self.table.pot.reset()
        self.assign_blinds()
        self.table.dealer.deal_private_cards(self.table.players)
        self.betting_round()
        self.table.dealer.deal_flop(self.table)
        self.betting_round()
        self.table.dealer.deal_turn(self.table)
        self.betting_round()
        self.table.dealer.deal_river(self.table)
        self.betting_round()
        # From the active players on the table, compute the winners.
        winners = self.evaluate_hand()
        self.compute_payouts(winners)
        # TODO(fedden): What if someone runs out of chips here?
        self.move_blinds()

    def compute_payouts(self, winners: list[Player]):
        if not winners:
            raise ValueError("At least one player must be in winners.")
        # TODO(fedden): This doesn't take care of all-ins and what if everyone
        #               folds? Needs work.

        pot = sum(self.all_bets)
        n_winners = len(winners)
        pot_contribution = pot / n_winners
        payouts = []
        for player in self.table.players:
            win = 0 if player not in winners else pot_contribution
            loss = player.n_bet_chips
            payout = win - loss
            payouts.append(payout)
            player.payout(payout)
        self.wins_and_losses = payouts

    def evaluate_hand(self) -> list[Player]:
        """Get the winner(s) from the showdown hands on the table."""
        # The cards that can be passed to the evaluator object from the table.
        table_cards = [card.eval_card for card in self.table.community_cards]
        # For every active player...
        player_results = []
        for player in self.table.players:
            if player.is_active:
                # Get evaluator friendly cards.
                hand_cards = [card.eval_card for card in player.cards]
                # Rank of the best hand - lower is better.
                rank = self.evaluator.evaluate(table_cards, hand_cards)
                hand_class = self.evaluator.get_rank_class(rank)
                hand_desc = self.evaluator.class_to_string(hand_class).lower()
                player_results.append(
                    dict(player=player, rank=rank, hand_desc=hand_desc)
                )
        # Sort results by rank.
        player_results = sorted(player_results, key=itemgetter("rank"))
        # The first definitely won, but did anyone draw? Use the rank to find
        # out.
        winning_rank = player_results[0]["rank"]
        winners: List[Player] = [player_results[0]["player"]]
        for result in player_results[1:]:
            if result["rank"] > winning_rank:
                break
            winners.append(result["player"])
        return winners

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

    def betting_round(self):
        """Computes the round(s) of betting.

        Until the current betting round is complete, all active players take
        actions in the order they were placed at the table. A betting round
        lasts until all players either call the highest placed bet or fold.
        """
        if self.n_players_with_moves > 1:
            # Ensure for the first move we do one round of betting.
            first_round = True
            logger.debug("Started round of betting.")
            while first_round or self.more_betting_needed:
                # For every active player compute the move, but big and small
                # blind move last..
                for player in self.table.players[2:] + self.table.players[:2]:
                    if player.is_active:
                        self.state = player.take_action(self.state)
                        import ipdb; ipdb.set_trace()
                first_round = False
                logger.debug(
                    f"  Betting iteration, bet total: {sum(self.all_bets)}")
            logger.debug(
                f"Finished round of betting, {self.n_active_players} active "
                f"players, {self.n_all_in_players} all in players.")
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
            p.n_chips for p in self.table.players)
        n_chips_correct = total_n_chips == self.table.total_n_chips_on_table
        pot_correct = self.table.pot.total == sum(
            p.n_bet_chips for p in self.table.players)
        if not n_chips_correct or not pot_correct:
            raise ValueError(
                'Bad logic - total n_chips are not the same as at the start '
                'of the game')

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
