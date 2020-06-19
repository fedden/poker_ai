"""
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict

import click
import joblib
import yaml
from tqdm import tqdm, trange

from poker_ai.ai.agent import Agent
from poker_ai.ai import ai
from poker_ai import utils
from poker_ai.games.short_deck.state import new_game, ShortDeckPokerState


def print_strategy(strategy: Dict[str, Dict[str, int]]):
    """Print strategy."""
    for info_set, action_to_probabilities in sorted(strategy.items()):
        norm = sum(list(action_to_probabilities.values()))
        tqdm.write(f"{info_set}")
        for action, probability in action_to_probabilities.items():
            tqdm.write(f"  - {action}: {probability / norm:.2f}")


def simple_search(
    config: Dict[str, int],
    save_path: Path,
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
    """Train agent."""
    utils.random.seed(42)
    agent = Agent(use_manager=False)
    info_set_lut = {}
    for t in trange(1, n_iterations + 1, desc="train iter"):
        if t == 2:
            logging.disable(logging.DEBUG)
        for i in range(n_players):  # fixed position i
            # Create a new state.
            state: ShortDeckPokerState = new_game(n_players, info_set_lut)
            info_set_lut = state.info_set_lut
            if t > update_threshold and t % strategy_interval == 0:
                ai.update_strategy(agent=agent, state=state, i=i, t=t)
            if t > prune_threshold:
                if random.uniform(0, 1) < 0.05:
                    ai.cfr(agent=agent, state=state, i=i, t=t)
                else:
                    ai.cfrp(agent=agent, state=state, i=i, t=t, c=c)
            else:
                ai.cfr(agent=agent, state=state, i=i, t=t)
        if t < lcfr_threshold & t % discount_interval == 0:
            d = (t / discount_interval) / ((t / discount_interval) + 1)
            for I in agent.regret.keys():
                for a in agent.regret[I].keys():
                    agent.regret[I][a] *= d
                    agent.strategy[I][a] *= d
        if (t > update_threshold) & (t % dump_iteration == 0):
            # dump the current strategy (sigma) throughout training and then
            # take an average. This allows for estimation of expected value in
            # leaf nodes later on using modified versions of the blueprint
            # strategy.
            ai.serialise(
                agent=agent, save_path=save_path, t=t, server_state=config,
            )

    print_strategy(agent.strategy)


if __name__ == "__main__":
    train()
