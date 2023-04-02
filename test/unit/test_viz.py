import os

from pathlib import Path
from collections import defaultdict

from poker_ai.viz.runner import (
    _get_bot_dag_data,
    to_start_of_betting_round_str,
    _generate_action_combos,
    _get_betting_round_action_combos,
    StrategySizeLevelDict,
)


def test_get_bot_dag_data():
    # Test setup
    current_dir = os.path.dirname(__file__)
    strategy_dir = os.path.join(current_dir, "test_data")
    strategy_path = Path(strategy_dir) / "agent.joblib"

    # Exercise the code under test
    bot_dag_data = _get_bot_dag_data(strategy_path)

    # Assertions
    assert isinstance(bot_dag_data, defaultdict)
    assert len(bot_dag_data) > 0
    assert all(isinstance(x, StrategySizeLevelDict) for x in bot_dag_data.values())


def test_to_start_of_betting_round_str():
    # Test setup
    info_set_str = """
        {
            "cards_cluster":1,
            "betting_stage":"flop",
            "history":[{"pre_flop":["call","call","call"]},{"flop":["call","call","call"]}]
        }
        """
    expected_output = (
        "flop_hand_cluster_1_call-call-call",
        ["call", "call", "call"],
    )

    # Exercise the code under test
    output = to_start_of_betting_round_str(info_set_str)

    # Assertions
    assert output == expected_output


def test_generate_action_combos():
    # Test setup
    base_path = "flop_hand_cluster_1_pre_flop-call-call-call-fold-fold"
    level = 2
    expected_output = [
        "flop_hand_cluster_1_pre_flop-call-call-call-fold-fold/raise/fold",
        "flop_hand_cluster_1_pre_flop-call-call-call-fold-fold/raise/raise",
        "flop_hand_cluster_1_pre_flop-call-call-call-fold-fold/raise/call",
        "flop_hand_cluster_1_pre_flop-call-call-call-fold-fold/call/raise",
        "flop_hand_cluster_1_pre_flop-call-call-call-fold-fold/call/call",
        "flop_hand_cluster_1_pre_flop-call-call-call-fold-fold/call/fold",
    ]

    # Exercise the code under test
    output = _generate_action_combos(base_path, level)

    # Assertions
    assert set(output) == set(expected_output)


def test_get_betting_round_action_combos():
    betting_round_state_viz = "flop_hand_cluster_1"
    max_generate_level = 2
    expected_paths = [
        "flop_hand_cluster_1",
        "flop_hand_cluster_1/fold",
        "flop_hand_cluster_1/call",
        "flop_hand_cluster_1/raise",
        "flop_hand_cluster_1/call/call",
        "flop_hand_cluster_1/call/fold",
        "flop_hand_cluster_1/call/raise",
        "flop_hand_cluster_1/fold/call",
        "flop_hand_cluster_1/fold/fold",
        "flop_hand_cluster_1/fold/raise",
        "flop_hand_cluster_1/raise/call",
        "flop_hand_cluster_1/raise/fold",
        "flop_hand_cluster_1/raise/raise",
    ]
    expected_sizes = [1000] + [0] * 12
    action_combos = _get_betting_round_action_combos(
        betting_round_state_viz, max_generate_level
    )
    assert set(action_combos["path"]) == set(expected_paths)
    assert action_combos["size"] == expected_sizes
