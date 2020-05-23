from __future__ import annotations

import logging
import sys
import collections
from typing import Dict
import copy
import json
import joblib

from tqdm import trange
import numpy as np

from pluribus import utils
from pluribus.games.short_deck.state import ShortDeckPokerState, new_game
from pluribus.games.short_deck.agent import Agent

sys.path.append("../blueprint_algo")




def to_dict(**kwargs) -> Dict[str, Any]:
    """Hacky method to convert weird collections dicts to regular dicts."""
    return json.loads(json.dumps(copy.deepcopy(kwargs)))


def normalize_strategy(this_info_sets_regret: Dict[str, float]) -> Dict[str, float]:
    """Calculate the strategy based on the current information sets regret."""
    # TODO: Could we instanciate a state object from an info set?
    actions = this_info_sets_regret.keys()
    regret_sum = sum([max(regret, 0) for regret in this_info_sets_regret.values()])
    if regret_sum > 0:
        strategy: Dict[str, float] = {
            action: max(this_info_sets_regret[action], 0) / regret_sum
            for action in actions
        }
    elif this_info_sets_regret == {}:
        # Don't return strategy if no strategy was made
        # during training
        strategy: Dict[str, float] = {}
    elif regret_sum == 0:
        # Regret is negative, we learned something
        default_probability = 1 / len(actions)
        strategy: Dict[str, float] = {action: default_probability for action in actions}
    return strategy


def update_strategy(agent: Agent, state: ShortDeckPokerState, ph_test_node: int):
    """
    Update strategy for test node only
    """
    ph = state.player_i
    if ph == ph_test_node:
        I = state.info_set
        # calculate regret
        sigma = calculate_strategy(agent.regret, I, state, offline_strategy,
                                    unnormalized_strategy)
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


def calculate_strategy(
    regret: Dict[str, Dict[str, float]],
    I: str,
    state: ShortDeckPokerState,
):
    """

    :param regret: dictionary of regrets, I is key, then each action at I, with values being regret
    :param sigma: dictionary of strategy updated by regret, iteration is key, then I is key, then each action with prob
    :param I:
    :param state: the game state
    :return: doesn't return anything, just updates sigma
    """
    sigma = collections.defaultdict(lambda: collections.defaultdict(lambda: 1 / 3))
    rsum = sum([max(x, 0) for x in regret[I].values()])
    for a in state.legal_actions:
        if rsum > 0:
            sigma[I][a] = max(regret[I][a], 0) / rsum
        else:
            sigma[I][a] = 1 / len(state.legal_actions)
    return sigma


def cfr(agent: Agent, state: ShortDeckPokerState, i: int, t: int) -> float:
    """
    regular cfr algo

    :param state: the game state
    :param i: player
    :param t: iteration
    :return: expected value for node for player i
    """
    ph = state.player_i

    player_not_in_hand = not state.players[i].is_active
    if state.is_terminal or player_not_in_hand:
        return state.payout[i]

    # NOTE(fedden): The logic in Algorithm 1 in the supplementary material
    #               instructs the following lines of logic, but state class
    #               will already skip to the next in-hand player.
    # elif p_i not in hand:
    #   cfr()
    # NOTE(fedden): According to Algorithm 1 in the supplementary material,
    #               we would add in the following bit of logic. However we
    #               already have the game logic embedded in the state class,
    #               and this accounts for the chance samplings. In other words,
    #               it makes sure that chance actions such as dealing cards
    #               happen at the appropriate times.
    # elif h is chance_node:
    #   sample action from strategy for h
    #   cfr()

    elif ph == i:
        I = state.info_set
        # calculate strategy
        if agent.tmp_regret[I] == {}:
            agent.tmp_regret[I] == agent.regret[I].copy()
        sigma = calculate_strategy(agent.tmp_regret, I, state)

        vo = 0.0
        voa = {}
        for a in state.legal_actions:
            new_state: ShortDeckPokerState = state.apply_action(a)
            voa[a] = cfr(agent, new_state, i, t)
            vo += sigma[I][a] * voa[a]

        for a in state.legal_actions:
            agent.tmp_regret[I][a] += voa[a] - vo

        return vo
    else:
        Iph = state.info_set
        if agent.tmp_regret[Iph] == {}:
            agent.tmp_regret[Iph] == agent.regret[Iph].copy()
        sigma = calculate_strategy(agent.regret, Iph, state)

        try:
            a = np.random.choice(
                list(sigma[Iph].keys()), 1, p=list(sigma[Iph].values()),
            )[0]
        except KeyError:
            p = 1 / len(state.legal_actions)
            probabilities = np.full(len(state.legal_actions), p)
            a = np.random.choice(state.legal_actions, p=probabilities)
            sigma[Iph] = {action: p for action in state.legal_actions}

        new_state: ShortDeckPokerState = state.apply_action(a)
        return cfr(agent, new_state, i, t)


def train(
    offline_strategy_path: str,
    regret_path: str,
    public_cards: list,
    action_sequence: list,
    n_iterations: int,
    lcfr_threshold: int,
    discount_interval: int,
    n_players: int,
    update_interval: int,
    update_threshold: int,
    dump_int: int,
):
    """Train agent."""
    # TODO: fix the seed
    utils.random.seed(36)
    agent = Agent(regret_dir=regret_path)

    offline_strategy = joblib.load(offline_strategy_path)
    state: ShortDeckPokerState = new_game(
        3, real_time_test=True, public_cards=public_cards
    )
    current_game_state: ShortDeckPokerState = state.load_game_state(
        offline_strategy, action_sequence
    )
    del offline_strategy
    # ph_test_node = current_game_state.player_i
    for t in trange(1, n_iterations + 1, desc="train iter"):
        print(t)
        if t == 2:
            logging.disable(logging.DEBUG)
        for i in range(n_players):  # fixed position i
            # Create a new state.
            state: ShortDeckPokerState = current_game_state.deal_bayes()
#            if t % update_interval == 0 and t > update_threshold:
#                update_strategy(agent, state, ph_test_node)
            cfr(agent, state, i, t)
        if t < lcfr_threshold & t % discount_interval == 0:
            d = (t / discount_interval) / ((t / discount_interval) + 1)
            for I in agent.tmp_regret.keys():
                for a in agent.tmp_regret[I].keys():
                    agent.tmp_regret[I][a] *= d

    offline_strategy = joblib.load(offline_strategy_path)
    # Adding the regret back to the regret dict
    for I in agent.tmp_regret.keys():
        if agent.tmp_regret != {}:
            agent.regret[I] = agent.tmp_regret[I].copy()

    # Add the unnormalized strategy into the original
    for info_set, this_info_sets_regret in sorted(agent.tmp_regret.items()):
        # If this_info_sets_regret == {}, we do nothing
        strategy = normalize_strategy(this_info_sets_regret)
        if info_set not in offline_strategy:
            offline_strategy[info_set] = {a: 0 for a in strategy.keys()}
        for action, probability in strategy.items():
            offline_strategy[info_set][action] += t / dump_int * probability
    return agent, offline_strategy
