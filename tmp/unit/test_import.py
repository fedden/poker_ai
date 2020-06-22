def test_import():
    """Test the imports work"""
    import poker_ai
    from poker_ai import ai, poker
    from poker_ai.ai import runner
    from poker_ai.terminal import runner
    from poker_ai.games.short_deck import player, state
    from poker_ai.poker import actions, card, dealer, deck, engine, player
    from poker_ai.poker import state, table, evaluation
    from poker_ai.poker.evaluation import eval_card, evaluator, lookup
