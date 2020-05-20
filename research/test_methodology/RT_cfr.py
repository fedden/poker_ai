from __future__ import annotations

import logging
import sys

from tqdm import trange
import numpy as np

from pluribus import utils
from pluribus.games.short_deck.state import ShortDeckPokerState, new_game
from pluribus.games.short_deck.agent import Agent
sys.path.append('../blueprint_algo')
from blueprint_short_deck_poker import calculate_strategy, cfr, cfrp


def update_strategy(agent: Agent, state: ShortDeckPokerState, ph_test_node: int):
    """
    Update strategy for test node only
    """
    ph = state.player_i
    if ph == ph_test_node:
        I = state.info_set
        # calculate regret
        sigma = calculate_strategy(agent.regret, I, state)
        # choose an action based of sigma
        try:
            a = np.random.choice(list(sigma[I].keys()), 1, p=list(sigma[I].values()))[0]
        except ValueError:
            p = 1 / len(state.legal_actions)
            probabilities = np.full(len(state.legal_actions), p)
            a = np.random.choice(state.legal_actions, p=probabilities)
            sigma[I] = {action: p for action in state.legal_actions}
        # Increment the action counter.
        agent.strategy[I][a] += 1
        return
    else:
        return

def train(
    offline_strategy: Dict,
    public_cards: list,
    action_sequence: list,
    n_iterations: int,
    lcfr_threshold: int,
    discount_interval: int,
    n_players: int,
    update_interval: int,
    update_threshold: int,
):
    """Train agent."""
    # TODO: fix the seed
    utils.random.seed(36)
    agent = Agent()

    state: ShortDeckPokerState = new_game(3, real_time_test=True,
                                          public_cards=public_cards)
    current_game_state: ShortDeckPokerState = state.load_game_state(
        offline_strategy,
        action_sequence
    )
    del offline_strategy
    ph_test_node = current_game_state.player_i
    for t in trange(1, n_iterations + 1, desc="train iter"):
        print(t)
        if t == 2:
            logging.disable(logging.DEBUG)
        for i in range(n_players):  # fixed position i
            # Create a new state.
            state: ShortDeckPokerState = current_game_state.deal_bayes()
            if t % update_interval == 0 and t > update_threshold:
                update_strategy(agent, state, ph_test_node)
            cfr(agent, state, i, t)
        if t < lcfr_threshold & t % discount_interval == 0:
            d = (t / discount_interval) / ((t / discount_interval) + 1)
            for I in agent.regret.keys():
                for a in agent.regret[I].keys():
                    agent.regret[I][a] *= d
                    agent.strategy[I][a] *= d

    return agent
