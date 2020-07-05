"""Assert that the correct payout is made for some hand designed scenarios."""


from typing import List

from poker_ai.poker.card import Card
from poker_ai.poker.table import PokerTable
from poker_ai.poker.engine import PokerEngine
from poker_ai.poker.pot import Pot
from poker_ai.poker.random_player import RandomPlayer


def _scenario_helper(
    winner_indices: List[int], player_cards: List[List[Card]], board_cards: List[Card]
):
    """Facillitates a round of texas hold'em.

    We pass both the player cards and the winners indices, and we check the
    engine found that the winners were the beneficiaries of the winnings.
    """
    n_players = len(player_cards)
    initial_chips = 100
    pot = Pot()
    # Construct players that will never fold or call, but only RAISE.
    players = [
        RandomPlayer(
            name=f"random player {player_i}",
            initial_chips=initial_chips,
            pot=pot,
            fold_probability=0.0,
            raise_probability=1.0,
            call_probability=0.0,
        )
        for player_i in range(n_players)
    ]
    table = PokerTable(players=players, pot=pot)
    engine = PokerEngine(table=table, small_blind=50, big_blind=10)
    engine.round_setup()
    engine._all_dealing_and_betting_rounds()
    # Inject cards here.
    for player, cards in zip(players, player_cards):
        player.cards = cards
    engine.table.community_cards = board_cards
    # We have now done rounds of betting with players that are desperate to
    # raise, and we also know exactly what cards they have because they have
    # been rigged. We should be able to easily calculate and predict who has
    # gained, so check players that have made money agaist `winner_indices`.
    engine.compute_winners()
    for winner_i in winner_indices:
        assert (
            players[winner_i].n_chips > initial_chips
        ), f"winner {winner_i} did not gain as expected."
    for loser_i in [i for i in range(n_players) if i not in winner_indices]:
        assert (
            players[loser_i].n_chips < initial_chips
        ), f"loser {loser_i} did not lose as expected"
    engine._round_cleanup()


def test_scenario_a():
    """Player 0 and 1 should share winnings."""
    winner_indices = [0, 1]
    player_cards = [
        [Card(rank="ace", suit="hearts"), Card(rank="8", suit="diamonds")],
        [Card(rank="ace", suit="spades"), Card(rank="3", suit="diamonds")],
        [Card(rank="2", suit="spades"), Card(rank="4", suit="diamonds")],
    ]
    board_cards = [
        Card(rank="ace", suit="clubs"),
        Card(rank="king", suit="diamonds"),
        Card(rank="king", suit="hearts"),
        Card(rank="8", suit="spades"),
        Card(rank="3", suit="spades"),
    ]
    _scenario_helper(
        winner_indices=winner_indices,
        player_cards=player_cards,
        board_cards=board_cards,
    )


def test_scenario_b():
    """Player 3 should win winnings."""
    winner_indices = [3]
    player_cards = [
        [Card(rank="2", suit="clubs"), Card(rank="3", suit="clubs")],
        [Card(rank="4", suit="diamonds"), Card(rank="6", suit="clubs")],
        [Card(rank="3", suit="hearts"), Card(rank="king", suit="diamonds")],
        [Card(rank="6", suit="spades"), Card(rank="6", suit="hearts")],
        [Card(rank="3", suit="spades"), Card(rank="8", suit="diamonds")],
        [Card(rank="ace", suit="spades"), Card(rank="3", suit="diamonds")],
    ]
    board_cards = [
        Card(rank="6", suit="diamonds"),
        Card(rank="jack", suit="diamonds"),
        Card(rank="10", suit="hearts"),
        Card(rank="ace", suit="hearts"),
        Card(rank="jack", suit="hearts"),
    ]
    _scenario_helper(
        winner_indices=winner_indices,
        player_cards=player_cards,
        board_cards=board_cards,
    )


def test_scenario_c():
    """Player 3 should win winnings."""
    winner_indices = [5]
    player_cards = [
        [Card(rank="2", suit="clubs"), Card(rank="3", suit="clubs")],
        [Card(rank="4", suit="diamonds"), Card(rank="6", suit="clubs")],
        [Card(rank="3", suit="hearts"), Card(rank="king", suit="diamonds")],
        [Card(rank="6", suit="spades"), Card(rank="6", suit="hearts")],
        [Card(rank="3", suit="spades"), Card(rank="8", suit="diamonds")],
        [Card(rank="ace", suit="spades"), Card(rank="3", suit="diamonds")],
    ]
    board_cards = [
        Card(rank="ace", suit="diamonds"),
        Card(rank="ace", suit="clubs"),
        Card(rank="king", suit="clubs"),
        Card(rank="7", suit="diamonds"),
        Card(rank="ace", suit="hearts"),
    ]
    _scenario_helper(
        winner_indices=winner_indices,
        player_cards=player_cards,
        board_cards=board_cards,
    )
