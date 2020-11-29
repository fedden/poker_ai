import pytest
from poker_ai.poker.evaluation.eval_card import EvaluationCard
from poker_ai.poker.evaluation.evaluator import Evaluator

# All ranks used in tests are taken from http://suffe.cool/poker/7462.html
# A low rank is better

POSSIBLE_SUITS = ["c", "d", "h", "s"]


@pytest.mark.parametrize("suit", POSSIBLE_SUITS)
def test_evaluator_royal_flush(suit):
    board = [
        EvaluationCard.new(f"A{suit}"),
        EvaluationCard.new(f"K{suit}"),
        EvaluationCard.new(f"Q{suit}"),
    ]
    hand = [EvaluationCard.new(f"J{suit}"), EvaluationCard.new(f"T{suit}")]
    evaluator = Evaluator()
    rank = evaluator.evaluate(board, hand)
    hand_class_int = evaluator.get_rank_class(rank)
    hand_class_str = evaluator.class_to_string(hand_class_int).lower()
    if hand_class_int != 1:
        raise ValueError
    if hand_class_str != "straight flush":
        raise ValueError
    if rank != 1:
        raise ValueError


@pytest.mark.parametrize("suit", POSSIBLE_SUITS)
def test_evaluator_straight_flush(suit):
    board = [
        EvaluationCard.new(f"9{suit}"),
        EvaluationCard.new(f"K{suit}"),
        EvaluationCard.new(f"Q{suit}"),
    ]
    hand = [EvaluationCard.new(f"J{suit}"), EvaluationCard.new(f"T{suit}")]
    evaluator = Evaluator()
    rank = evaluator.evaluate(board, hand)
    hand_class_int = evaluator.get_rank_class(rank)
    hand_class_str = evaluator.class_to_string(hand_class_int).lower()
    if hand_class_int != 1:
        raise ValueError
    if hand_class_str != "straight flush":
        raise ValueError
    if rank != 2:
        raise ValueError


@pytest.mark.parametrize("suit", POSSIBLE_SUITS)
@pytest.mark.parametrize("hand", [{"cardRanks": "AKQJT", "expectedRank": 1}, {"cardRanks": "KQJT9", "expectedRank": 2},
                                  {"cardRanks": "QJT98", "expectedRank": 3}, {"cardRanks": "JT987", "expectedRank": 4},
                                  {"cardRanks": "T9876", "expectedRank": 5}, {"cardRanks": "98765", "expectedRank": 6},
                                  {"cardRanks": "87654", "expectedRank": 7}, {"cardRanks": "76543", "expectedRank": 8},
                                  {"cardRanks": "65432", "expectedRank": 9}])
def test_evaluate_with_five_cards_returns_correct_rank_for_straight_flushes(suit, hand):
    cards = [
        EvaluationCard.new(f"{hand['cardRanks'][0]}{suit}"),
        EvaluationCard.new(f"{hand['cardRanks'][1]}{suit}"),
        EvaluationCard.new(f"{hand['cardRanks'][2]}{suit}"),
        EvaluationCard.new(f"{hand['cardRanks'][3]}{suit}"),
        EvaluationCard.new(f"{hand['cardRanks'][4]}{suit}"),
    ]
    evaluator = Evaluator()

    assert evaluator.evaluate(cards[:2], cards[2:]) == hand['expectedRank']


@pytest.mark.parametrize("hand", [{"typeFourOfKind": "A", "kicker": "2", "expectedRank": 22},
                                  {"typeFourOfKind": "K", "kicker": "3", "expectedRank": 33},
                                  {"typeFourOfKind": "Q", "kicker": "4", "expectedRank": 44},
                                  {"typeFourOfKind": "J", "kicker": "5", "expectedRank": 55},
                                  {"typeFourOfKind": "T", "kicker": "6", "expectedRank": 66},
                                  {"typeFourOfKind": "9", "kicker": "7", "expectedRank": 77},
                                  {"typeFourOfKind": "8", "kicker": "7", "expectedRank": 89},
                                  {"typeFourOfKind": "7", "kicker": "8", "expectedRank": 101},
                                  {"typeFourOfKind": "6", "kicker": "9", "expectedRank": 112},
                                  {"typeFourOfKind": "5", "kicker": "T", "expectedRank": 123},
                                  {"typeFourOfKind": "4", "kicker": "J", "expectedRank": 134},
                                  {"typeFourOfKind": "3", "kicker": "Q", "expectedRank": 145},
                                  {"typeFourOfKind": "2", "kicker": "K", "expectedRank": 156}])
def test_evaluate_with_five_cards_returns_correct_rank_for_four_of_a_kind(hand):
    cards = [
        EvaluationCard.new(f"{hand['typeFourOfKind']}d"),
        EvaluationCard.new(f"{hand['typeFourOfKind']}c"),
        EvaluationCard.new(f"{hand['typeFourOfKind']}h"),
        EvaluationCard.new(f"{hand['typeFourOfKind']}s"),
        EvaluationCard.new(f"{hand['kicker']}d"),
    ]
    evaluator = Evaluator()

    assert evaluator.evaluate(cards[:2], cards[2:]) == hand['expectedRank']


@pytest.mark.parametrize("hand", [{"tripleRank": "A", "pairRank": "2", "expectedRank": 178},
                                  {"tripleRank": "K", "pairRank": "A", "expectedRank": 179},
                                  {"tripleRank": "Q", "pairRank": "A", "expectedRank": 191},
                                  {"tripleRank": "J", "pairRank": "A", "expectedRank": 203},
                                  {"tripleRank": "T", "pairRank": "A", "expectedRank": 215},
                                  {"tripleRank": "9", "pairRank": "A", "expectedRank": 227},
                                  {"tripleRank": "8", "pairRank": "A", "expectedRank": 239},
                                  {"tripleRank": "7", "pairRank": "A", "expectedRank": 251},
                                  {"tripleRank": "6", "pairRank": "A", "expectedRank": 263},
                                  {"tripleRank": "5", "pairRank": "A", "expectedRank": 275},
                                  {"tripleRank": "4", "pairRank": "A", "expectedRank": 287},
                                  {"tripleRank": "3", "pairRank": "A", "expectedRank": 299},
                                  {"tripleRank": "2", "pairRank": "A", "expectedRank": 311}])
def test_evaluate_with_five_cards_returns_correct_rank_for_full_house(hand):
    cards = [
        EvaluationCard.new(f"{hand['tripleRank']}d"),
        EvaluationCard.new(f"{hand['tripleRank']}c"),
        EvaluationCard.new(f"{hand['tripleRank']}h"),
        EvaluationCard.new(f"{hand['pairRank']}s"),
        EvaluationCard.new(f"{hand['pairRank']}d"),
    ]
    evaluator = Evaluator()

    assert evaluator.evaluate(cards[:2], cards[2:]) == hand['expectedRank']


@pytest.mark.parametrize("suit", POSSIBLE_SUITS)
@pytest.mark.parametrize("hand", [{"cardRanks": "AKQJ9", "expectedRank": 323},
                                  {"cardRanks": "A9842", "expectedRank": 760},
                                  {"cardRanks": "75432", "expectedRank": 1599}])
def test_evaluate_with_five_cards_returns_correct_rank_for_flush(suit, hand):
    cards = [
        EvaluationCard.new(f"{hand['cardRanks'][0]}{suit}"),
        EvaluationCard.new(f"{hand['cardRanks'][1]}{suit}"),
        EvaluationCard.new(f"{hand['cardRanks'][2]}{suit}"),
        EvaluationCard.new(f"{hand['cardRanks'][3]}{suit}"),
        EvaluationCard.new(f"{hand['cardRanks'][4]}{suit}"),
    ]
    evaluator = Evaluator()

    assert evaluator.evaluate(cards[:2], cards[2:]) == hand['expectedRank']


@pytest.mark.parametrize("suits", ["cdshd", "dsshc"])
@pytest.mark.parametrize("hand", [{"cardRanks": "AKQJT", "expectedRank": 1600},
                                  {"cardRanks": "98765", "expectedRank": 1605},
                                  {"cardRanks": "A5432", "expectedRank": 1609}])
def test_evaluate_with_five_cards_returns_correct_rank_for_straight(suits, hand):
    cards = [
        EvaluationCard.new(f"{hand['cardRanks'][0]}{suits[0]}"),
        EvaluationCard.new(f"{hand['cardRanks'][1]}{suits[1]}"),
        EvaluationCard.new(f"{hand['cardRanks'][2]}{suits[2]}"),
        EvaluationCard.new(f"{hand['cardRanks'][3]}{suits[3]}"),
        EvaluationCard.new(f"{hand['cardRanks'][4]}{suits[4]}"),
    ]
    evaluator = Evaluator()

    assert evaluator.evaluate(cards[:2], cards[2:]) == hand['expectedRank']


@pytest.mark.parametrize("hand", [{"tripleRank": "A", "kickers": "JT", "expectedRank": 1631},
                                  {"tripleRank": "K", "kickers": "JT", "expectedRank": 1697},
                                  {"tripleRank": "Q", "kickers": "AK", "expectedRank": 1742},
                                  {"tripleRank": "J", "kickers": "AK", "expectedRank": 1808},
                                  {"tripleRank": "T", "kickers": "AK", "expectedRank": 1874},
                                  {"tripleRank": "9", "kickers": "AK", "expectedRank": 1940},
                                  {"tripleRank": "8", "kickers": "AK", "expectedRank": 2006},
                                  {"tripleRank": "7", "kickers": "AK", "expectedRank": 2072},
                                  {"tripleRank": "6", "kickers": "AK", "expectedRank": 2138},
                                  {"tripleRank": "5", "kickers": "AK", "expectedRank": 2204},
                                  {"tripleRank": "4", "kickers": "AK", "expectedRank": 2270},
                                  {"tripleRank": "3", "kickers": "AK", "expectedRank": 2336},
                                  {"tripleRank": "2", "kickers": "AK", "expectedRank": 2402}])
def test_evaluate_with_five_cards_returns_correct_rank_for_three_of_a_kind(hand):
    cards = [
        EvaluationCard.new(f"{hand['tripleRank']}s"),
        EvaluationCard.new(f"{hand['tripleRank']}h"),
        EvaluationCard.new(f"{hand['tripleRank']}c"),
        EvaluationCard.new(f"{hand['kickers'][0]}d"),
        EvaluationCard.new(f"{hand['kickers'][1]}c"),
    ]
    evaluator = Evaluator()

    assert evaluator.evaluate(cards[:2], cards[2:]) == hand['expectedRank']


@pytest.mark.parametrize("hand", [{"pairRank1": "A", "pairRank2": "K", "kicker": "2", "expectedRank": 2478},
                                  {"pairRank1": "J", "pairRank2": "T", "kicker": "A", "expectedRank": 2831},
                                  {"pairRank1": "8", "pairRank2": "4", "kicker": "3", "expectedRank": 3137},
                                  {"pairRank1": "4", "pairRank2": "2", "kicker": "5", "expectedRank": 3313},
                                  {"pairRank1": "3", "pairRank2": "2", "kicker": "4", "expectedRank": 3325}])
def test_evaluate_with_five_cards_returns_correct_rank_for_two_pairs(hand):
    cards = [
        EvaluationCard.new(f"{hand['pairRank1']}s"),
        EvaluationCard.new(f"{hand['pairRank1']}h"),
        EvaluationCard.new(f"{hand['pairRank2']}c"),
        EvaluationCard.new(f"{hand['pairRank2']}d"),
        EvaluationCard.new(f"{hand['kicker']}c"),
    ]
    evaluator = Evaluator()

    assert evaluator.evaluate(cards[:2], cards[2:]) == hand['expectedRank']


@pytest.mark.parametrize("hand", [{"pairRank": "A", "kickers": "K32", "expectedRank": 3380},
                                  {"pairRank": "K", "kickers": "432", "expectedRank": 3765},
                                  {"pairRank": "Q", "kickers": "K32", "expectedRank": 3865},
                                  {"pairRank": "J", "kickers": "K32", "expectedRank": 4085},
                                  {"pairRank": "T", "kickers": "K32", "expectedRank": 4305},
                                  {"pairRank": "9", "kickers": "K32", "expectedRank": 4525},
                                  {"pairRank": "8", "kickers": "K32", "expectedRank": 4745},
                                  {"pairRank": "7", "kickers": "K32", "expectedRank": 4965},
                                  {"pairRank": "6", "kickers": "K32", "expectedRank": 5185},
                                  {"pairRank": "5", "kickers": "K32", "expectedRank": 5405},
                                  {"pairRank": "4", "kickers": "K32", "expectedRank": 5625},
                                  {"pairRank": "3", "kickers": "K42", "expectedRank": 5845},
                                  {"pairRank": "2", "kickers": "K43", "expectedRank": 6065}])
def test_evaluate_with_five_cards_returns_correct_rank_for_one_pair(hand):
    cards = [
        EvaluationCard.new(f"{hand['pairRank']}s"),
        EvaluationCard.new(f"{hand['pairRank']}h"),
        EvaluationCard.new(f"{hand['kickers'][0]}c"),
        EvaluationCard.new(f"{hand['kickers'][1]}d"),
        EvaluationCard.new(f"{hand['kickers'][2]}c"),
    ]
    evaluator = Evaluator()

    assert evaluator.evaluate(cards[:2], cards[2:]) == hand['expectedRank']


@pytest.mark.parametrize("hand", [{"ranks": "AK982", "expectedRank": 6299},
                                  {"ranks": "J7532", "expectedRank": 7335},
                                  {"ranks": "85432", "expectedRank": 7458},
                                  {"ranks": "75432", "expectedRank": 7462}])
def test_evaluate_with_five_cards_returns_correct_rank_for_highest_card(hand):
    cards = [
        EvaluationCard.new(f"{hand['ranks'][0]}c"),
        EvaluationCard.new(f"{hand['ranks'][1]}d"),
        EvaluationCard.new(f"{hand['ranks'][2]}c"),
        EvaluationCard.new(f"{hand['ranks'][3]}s"),
        EvaluationCard.new(f"{hand['ranks'][4]}h"),
    ]
    evaluator = Evaluator()

    assert evaluator.evaluate(cards[:2], cards[2:]) == hand['expectedRank']


@pytest.mark.parametrize("parameters", [
    {"handCards": "A3", "handSuits": "dc", "communityCards": "A2K3", "communitySuits": "dhss", "expectedRank": 2578},
    {"handCards": "K3", "handSuits": "dc", "communityCards": "2456", "communitySuits": "chss", "expectedRank": 1608},
    {"handCards": "25", "handSuits": "dc", "communityCards": "3632", "communitySuits": "chss", "expectedRank": 3323},
    {"handCards": "78", "handSuits": "dc", "communityCards": "5426", "communitySuits": "chss", "expectedRank": 1606},
    {"handCards": "JT", "handSuits": "dc", "communityCards": "AA4T", "communitySuits": "chss", "expectedRank": 2503}])
def test_evaluate_with_six_cards_returns_lowest_rank(parameters):
    cards = [
        EvaluationCard.new(f"{parameters['handCards'][0]}{parameters['handSuits'][0]}"),
        EvaluationCard.new(f"{parameters['handCards'][1]}{parameters['handSuits'][1]}"),
        EvaluationCard.new(f"{parameters['communityCards'][0]}{parameters['communitySuits'][0]}"),
        EvaluationCard.new(f"{parameters['communityCards'][1]}{parameters['communitySuits'][1]}"),
        EvaluationCard.new(f"{parameters['communityCards'][2]}{parameters['communitySuits'][2]}"),
        EvaluationCard.new(f"{parameters['communityCards'][3]}{parameters['communitySuits'][3]}"),
    ]
    evaluator = Evaluator()

    assert evaluator.evaluate(cards[:2], cards[2:]) == parameters['expectedRank']
