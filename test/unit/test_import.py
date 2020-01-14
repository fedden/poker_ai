def test_import():
    """Test the imports work"""
    import pluribus
    from pluribus import ai, game
    from pluribus.ai import pluribus
    from pluribus.game import actions, card, dealer, deck, engine, player
    from pluribus.game import state, table, evaluation
    from pluribus.game.evaluation import eval_card, evaluator, lookup
