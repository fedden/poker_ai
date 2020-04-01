"""
call python blueprint_short_deck_poker.py in the research/blueprint_algo directory

The following code is an attempt at mocking up the pseudo code for the blueprint algorithm found in Pluribus.
-- That pseudo code can be found here, pg. 16:
---- https://science.sciencemag.org/content/sci/suppl/2019/07/10/science.aay2400.DC1/aay2400-Brown-SM.pdf

Additionally, this blueprint mockup has previously been applied to Kuhn poker.
-- https://github.com/fedden/pluribus-poker-AI/blob/develop/research/blueprint_algo/blueprint_kuhn.py

This code was left in a functional style that mimics code that has been pushed to the feature/add-kuhn-poker branch.
That code was originally inspired by the two links below, I believe, pg 12:
-- http://modelai.gettysburg.edu/2013/cfr/cfr.pdf
-- https://github.com/geohot/ai-notebooks/blob/8bdd5ca2d9560f978ea79bb9d0cb8d5acf3dee4b/cfr_kuhn_poker.ipynb

The code-style choice was made in order to make converting the findings to an OOP style easier and more
consistent by working from a familiar style (meaning, similar variable names, etc.. to the notebook link above),
while not introducing (more) bugs in a pre-mature conversion to work in our Pluribus framework.

The differences between the following code and blueprint_kuhn.py:
-- Applied blueprint mockup to Limit Holdem with three players with 20 card deck
---- This choice of a contrived short deck game
---- is in order to apply game logic that introduces similar complexities as No Limit but on a much smaller game tree
-- Now only storing actions in strategy/sigma/regret as we "discover" the infoset
-- Incorporates card combination clustering from the following files:
---- https://github.com/fedden/pluribus-poker-AI/blob/develop/research/clustering/information_abstraction.py
---- https://github.com/fedden/pluribus-poker-AI/blob/develop/research/clustering/information_abstraction_multiprocess.py

The following would still need to be done:
-- Debug (search through TODOs)
-- Decide if the blueprint algo needs to be altered to be used for rounds past the first round (Pluribus only uses the
blueprint for the preflop round)
---- We plan to use the blueprint algo for all rounds in order to have some intelligence without needing to search
---- This will allow for us to host the bot cheaply (although will sacrifice some itelligence)
-- Decide if chance sampling needs to be treated differently than it's currently being implemented
-- Adjust game logic if needed
-- Stop using global variables where possible - a lot of potential for subtle bugs
-- Use minutes instead of iterations for the prune_threshold and discount interval
-- Only apply pruning to non-terminal nodes and non-last round of betting as described in the supplementary materials:
---- https://science.sciencemag.org/content/sci/suppl/2019/07/10/science.aay2400.DC1/aay2400-Brown-SM.pdf (pg.14 - 15)
"""
from __future__ import annotations

import copy
import collections
import logging
import random
from typing import Any, Dict

import numpy as np
from tqdm import tqdm, trange

from pluribus import utils
from pluribus.games.short_deck.player import ShortDeckPokerPlayer
from pluribus.games.short_deck.state import ShortDeckPokerState
from pluribus.poker.pot import Pot


logger = logging.getLogger(__name__)


# TODO: In general, wondering how important this function is if we are to use
# the blueprint algo for more than the preflop round? Would using just sigma
# allow for a more complete rendering of strategies for infosets?
def update_strategy(state: ShortDeckPokerState, i: int):
    """

    :param state: the game state
    :param i: the player, i = 1 is always first to act and i = 2 is always second to act, but they take turns who
        updates the strategy (only one strategy)
    :return: nothing, updates action count in the strategy of actions chosen according to sigma, this simple choosing of
        actions is what allows the algorithm to build up preference for one action over another in a given spot
    """
    tqdm.write(f"update_strategy: player_i: {i}, state: {state}")
    ph = state.player_i  # this is always the case no matter what i is

    player_not_in_hand = not state.players[i].is_active
    if state.is_terminal or player_not_in_hand or state.betting_round > 0:
        return
    # NOTE(fedden): According to Algorithm 1 in the supplementary material,
    #               we would add in the following bit of logic. However we
    #               already have the game logic embedded in the state class,
    #               and this accounts for the chance samplings. In other words,
    #               it makes sure that chance actions such as dealing cards
    #               happen at the appropriate times.
    # elif h is chance_node:
    #   sample action from strategy for h
    #   update_strategy(rs, h + a, i)
    elif ph == i:
        I = state.info_set
        # calculate regret
        calculate_strategy(regret, sigma, I, state)
        # choose an action based of sigma
        try:
            a = np.random.choice(
                list(sigma[t][I].keys()), 1, p=list(sigma[t][I].values())
            )[0]
        except ValueError:
            p = 1 / len(state.legal_actions)
            probabilities = np.full(len(state.legal_actions), p)
            a = np.random.choice(state.legal_actions, p=probabilities)
            sigma[t][I] = {action: p for action in state.legal_actions}
        # Increment the action counter.
        strategy[I][a] += 1
        # so strategy is counts based on sigma, this takes into account the
        # reach probability so there is no need to pass around that pi guy..
        new_state: ShortDeckPokerState = state.apply_action(a)
        update_strategy(new_state, i)
    else:
        # Traverse each action.
        for a in state.legal_actions:
            # not actually updating the strategy for p_i != i, only one i at a
            # time
            new_state: ShortDeckPokerState = state.apply_action(a)
            update_strategy(new_state, i)


def calculate_strategy(
    regret: Dict[str, Dict[str, float]],
    sigma: Dict[int, Dict[str, Dict[str, float]]],
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
    tqdm.write(f"calculate_strategy: state: {state}")
    rsum = sum([max(x, 0) for x in regret[I].values()])
    for a in state.legal_actions:
        if rsum > 0:
            sigma[t + 1][I][a] = max(regret[I][a], 0) / rsum
        else:
            sigma[t + 1][I][a] = 1 / len(state.legal_actions)


def cfr(state: ShortDeckPokerState, i: int, t: int) -> float:
    """
    regular cfr algo

    :param state: the game state
    :param i: player
    :param t: iteration
    :return: expected value for node for player i
    """
    tqdm.write(f"cfr: player_i: {i}, state: {state}")
    ph = state.player_i

    if state.is_terminal:
        return state.payout[i] * (1 if i == 1 else -1)
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
        calculate_strategy(regret, sigma, I, state)
        # TODO: Does updating sigma here (as opposed to after regret) miss out
        #       on any updates? If so, is there any benefit to having it up
        #       here?
        vo = 0.0
        voa = {}
        for a in state.legal_actions:
            new_state: ShortDeckPokerState = state.apply_action(a)
            voa[a] = cfr(new_state, i, t)
            vo += sigma[t][I][a] * voa[a]
        for a in state.legal_actions:
            regret[I][a] += voa[a] - vo
            # do not need update the strategy based on regret, strategy does
            # that with sigma
        return vo
    else:
        Iph = state.info_set
        calculate_strategy(regret, sigma, Iph, state)
        try:
            a = np.random.choice(
                list(sigma[t][Iph].keys()), 1, p=list(sigma[t][Iph].values())
            )[0]
        except ValueError:
            p = 1 / len(state.legal_actions)
            probabilities = np.full(len(state.legal_actions), p)
            a = np.random.choice(state.legal_actions, p=probabilities)
            sigma[t][Iph] = {action: p for action in state.legal_actions}
        new_state: ShortDeckPokerState = state.apply_action(a)
        return cfr(new_state, i, t)


def cfrp(state: ShortDeckPokerState, i: int, t: int):
    """
    pruning cfr algo, might need to adjust only pruning if not final betting round and if not terminal node

    :param state: the game state
    :param i: player
    :param t: iteration
    :return: expected value for node for player i
    """
    tqdm.write(f"cfrp: player_i: {i}, state: {state}")
    ph = state.player_i

    if state.is_terminal:
        return state.payout[i] * (1 if i == 1 else -1)
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
        calculate_strategy(regret, sigma, I, state)
        # TODO: Does updating sigma here (as opposed to after regret) miss out
        #       on any updates? If so, is there any benefit to having it up
        #       here?
        vo = 0.0
        voa = {}
        explored = {}  # keeps tracked of items that can be skipped
        for a in state.legal_actions:
            if regret[I][a] > C:
                new_state: ShortDeckPokerState = state.apply_action(a)
                voa[a] = cfrp(new_state, i, t)
                explored[a] = True
                vo += sigma[t][I][a] * voa[a]
            else:
                explored[a] = False
        for a in state.legal_actions:
            if explored[a]:
                regret[I][a] += voa[a] - vo
                # do not need update the strategy based on regret, strategy
                # does that with sigma
        return vo
    else:
        Iph = state.info_set
        calculate_strategy(regret, sigma, Iph, state)
        try:
            a = np.random.choice(
                list(sigma[t][Iph].keys()), 1, p=list(sigma[t][Iph].values())
            )[0]
        except ValueError:
            p = 1 / len(state.legal_actions)
            probabilities = np.full(len(state.legal_actions), p)
            a = np.random.choice(state.legal_actions, p=probabilities)
            sigma[t][Iph] = {action: p for action in state.legal_actions}
        new_state: ShortDeckPokerState = state.apply_action(a)
        return cfrp(new_state, i, t)


def new_game(n_players: int, info_set_lut: Dict[str, Any] = {}) -> ShortDeckPokerState:
    """Create a new game of short deck poker."""
    pot = Pot()
    players = [
        ShortDeckPokerPlayer(player_i=player_i, initial_chips=10000, pot=pot)
        for player_i in range(n_players)
    ]
    if info_set_lut:
        # Don't reload massive files, it takes ages.
        state = ShortDeckPokerState(players=players, load_pickle_files=False)
        state.info_set_lut = info_set_lut
    else:
        # Load massive files.
        state = ShortDeckPokerState(players=players)
    return state


def print_strategy(strategy: Dict[str, Dict[str, int]]):
    """Print strategy."""
    for info_set, action_to_probabilities in sorted(strategy.items()):
        norm = sum(list(action_to_probabilities.values()))
        print(f"{info_set}")
        for action, probability in action_to_probabilities.items():
            print(f"  - {action}: {probability / norm:.2f}")


if __name__ == "__main__":
    utils.random.seed(42)
    # TODO(fedden): Note from the supplementary material, the data here will
    #               need to be lower precision: "To save memory, regrets were
    #               stored using 4-byte integers rather than 8-byte doubles.
    #               There was also a ﬂoor on regret at -310,000,000 for every
    #               action. This made it easier to unprune actions that were
    #               initially pruned but later improved. This also prevented
    #               integer overﬂows".
    strategy = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    regret = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    sigma = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: 1 / 3))
    )

    # algorithm constants
    strategy_interval = 200  # it's just to test.
    n_iterations = 20000
    LCFR_threshold = 80
    discount_interval = 10
    prune_threshold = 40
    C = -20000  # somewhat arbitrary
    n_players = 3
    # algorithm presented here, pg.16:
    # https://science.sciencemag.org/content/sci/suppl/2019/07/10/science.aay2400.DC1/aay2400-Brown-SM.pdf
    import ipdb

    ipdb.set_trace()
    logging.info("beginning training")
    info_set_lut = {}
    for t in trange(n_iterations, desc="train iter"):
        sigma[t + 1] = copy.deepcopy(sigma[t])
        for i in range(n_players):  # fixed position i
            # Create a new state.
            state: ShortDeckPokerState = new_game(n_players, info_set_lut)
            info_set_lut = state.info_set_lut
            if t % strategy_interval == 0:
                update_strategy(state, i)
            if t > prune_threshold:
                if random.uniform(0, 1) < 0.05:
                    cfr(state, i, t)
                else:
                    cfrp(state, i, t)
            else:
                cfr(state, i, t)
        if t < LCFR_threshold & t % discount_interval == 0:
            # TODO(fedden): Is discount_interval actually set/managed in
            #               minutes here? In Algorithm 1 this should be managed
            #               in minutes using perhaps the time module, but here
            #               it appears to be being managed by the iterations
            #               count.
            d = (t / discount_interval) / ((t / discount_interval) + 1)
            for I in regret.keys():
                for a in regret[I].keys():
                    regret[I][a] *= d
                    strategy[I][a] *= d
        del sigma[t]
    print_strategy(strategy)
