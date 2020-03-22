def test_import():
    """Test the imports work"""
    import pluribus
    from pluribus import ai, poker
    from pluribus.ai import pluribus
    from pluribus.games.short_deck import player, state
    from pluribus.poker import actions, card, dealer, deck, engine, player
    from pluribus.poker import state, table, evaluation
    from pluribus.poker.evaluation import eval_card, evaluator, lookup
