import pytest


@pytest.mark.parametrize("suit", ["c", "d", "h", "s"])
def test_evaluator_royal_flush(suit):
    from poker_ai.poker.evaluation.eval_card import EvaluationCard
    from poker_ai.poker.evaluation.evaluator import Evaluator
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


@pytest.mark.parametrize("suit", ["c", "d", "h", "s"])
def test_evaluator_straight_flush(suit):
    from poker_ai.poker.evaluation.eval_card import EvaluationCard
    from poker_ai.poker.evaluation.evaluator import Evaluator
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
