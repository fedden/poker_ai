import os
import pickle
import shlex
from typing import List

import pytest
from click.testing import CliRunner

from poker_ai.cli.runner import cli

os.environ["TESTING_SUITE"] = "1"
pickle_dir = os.environ.get("LUT_DIR", os.path.abspath("research/blueprint_algo/"))


@pytest.mark.parametrize("strategy_interval", [1])
@pytest.mark.parametrize("n_iterations", [5])
@pytest.mark.parametrize("lcfr_threshold", [0])
@pytest.mark.parametrize("discount_interval", [1])
@pytest.mark.parametrize("prune_threshold", [1])
@pytest.mark.parametrize("c", [0])
@pytest.mark.parametrize("n_players", [2])
@pytest.mark.parametrize("dump_iteration", [1])
@pytest.mark.parametrize("update_threshold", [0])
def test_train_multiprocess_async(
    strategy_interval: int,
    n_iterations: int,
    lcfr_threshold: int,
    discount_interval: int,
    prune_threshold: int,
    c: int,
    n_players: int,
    dump_iteration: int,
    update_threshold: int,
):
    """Test we can call the syncronous multiprocessing training CLI."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        cli_str: str = f"""train start              \
            --strategy_interval {strategy_interval} \
            --n_iterations {n_iterations}           \
            --lcfr_threshold {lcfr_threshold}       \
            --discount_interval {discount_interval} \
            --prune_threshold {prune_threshold}     \
            --c {c}                                 \
            --n_players {n_players}                 \
            --dump_iteration {dump_iteration}       \
            --update_threshold {update_threshold}   \
            --pickle_dir  {pickle_dir}              \
            --multi_process                         \
            --async_update_strategy                 \
            --async_cfr                             \
            --async_discount                        \
            --async_serialise                       \
            --nickname test
        """
        cli_args: List[str] = shlex.split(cli_str)
        result = runner.invoke(cli, cli_args, catch_exceptions=True)


@pytest.mark.parametrize("strategy_interval", [1])
@pytest.mark.parametrize("n_iterations", [5])
@pytest.mark.parametrize("lcfr_threshold", [0])
@pytest.mark.parametrize("discount_interval", [1])
@pytest.mark.parametrize("prune_threshold", [1])
@pytest.mark.parametrize("c", [0])
@pytest.mark.parametrize("n_players", [2])
@pytest.mark.parametrize("dump_iteration", [1])
@pytest.mark.parametrize("update_threshold", [0])
def test_train_multiprocess_sync(
    strategy_interval: int,
    n_iterations: int,
    lcfr_threshold: int,
    discount_interval: int,
    prune_threshold: int,
    c: int,
    n_players: int,
    dump_iteration: int,
    update_threshold: int,
):
    """Test we can call the syncronous multiprocessing training CLI."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        cli_str: str = f"""train start              \
            --strategy_interval {strategy_interval} \
            --n_iterations {n_iterations}           \
            --lcfr_threshold {lcfr_threshold}       \
            --discount_interval {discount_interval} \
            --prune_threshold {prune_threshold}     \
            --c {c}                                 \
            --n_players {n_players}                 \
            --dump_iteration {dump_iteration}       \
            --update_threshold {update_threshold}   \
            --pickle_dir  {pickle_dir}              \
            --multi_process                         \
            --sync_update_strategy                  \
            --sync_cfr                              \
            --sync_discount                         \
            --sync_serialise                        \
            --nickname test
        """
        cli_args: List[str] = shlex.split(cli_str)
        result = runner.invoke(cli, cli_args, catch_exceptions=True)


@pytest.mark.parametrize("strategy_interval", [1])
@pytest.mark.parametrize("n_iterations", [5])
@pytest.mark.parametrize("lcfr_threshold", [0])
@pytest.mark.parametrize("discount_interval", [1])
@pytest.mark.parametrize("prune_threshold", [1])
@pytest.mark.parametrize("c", [0])
@pytest.mark.parametrize("n_players", [2])
@pytest.mark.parametrize("dump_iteration", [1])
@pytest.mark.parametrize("update_threshold", [0])
def test_train_singleprocess(
    strategy_interval: int,
    n_iterations: int,
    lcfr_threshold: int,
    discount_interval: int,
    prune_threshold: int,
    c: int,
    n_players: int,
    dump_iteration: int,
    update_threshold: int,
):
    """Test we can call the syncronous multiprocessing training CLI."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        cli_str: str = f"""train start              \
            --strategy_interval {strategy_interval} \
            --n_iterations {n_iterations}           \
            --lcfr_threshold {lcfr_threshold}       \
            --discount_interval {discount_interval} \
            --prune_threshold {prune_threshold}     \
            --c {c}                                 \
            --n_players {n_players}                 \
            --dump_iteration {dump_iteration}       \
            --update_threshold {update_threshold}   \
            --pickle_dir  {pickle_dir}              \
            --single_process                        \
            --nickname test
        """
        cli_args: List[str] = shlex.split(cli_str)
        result = runner.invoke(cli, cli_args, catch_exceptions=True)


# TODO(fedden): Figure out a way to test the terminal game.
#  from os import kill, getpid
#  from multiprocessing import Queue, Process
#  from time import sleep
#  from threading import Timer
#  from signal import SIGINT
#  def test_terminal():
#      """Test we can call the Terminal game."""
#      n_secs_to_run: int = 5
#      queue: Queue = Queue()
#
#      runner = CliRunner()
#      cli_str: str = "play --pickle_dir . --debug_quick_start"
#      cli_args: List[str] = shlex.split(cli_str)
#
#      def background():
#          """Use a killable background process."""
#          Timer(n_secs_to_run, lambda: kill(getpid(), SIGINT)).start()
#          result = runner.invoke(cli, cli_args, catch_exceptions=False)
#          queue.put(result)
#
#      process = Process(target=background)
#      process.start()
#      while process.is_alive():
#          sleep(0.1)
#      else:
#          result = queue.get()
#      import ipdb
#
#      ipdb.set_trace()
#      assert result["exit_code"] == 0
#      assert (
#          "Results can be inconsistent, as execution was terminated" in results["output"]
#      )
