"""
Notes:
    When state.player_i == i, and i folds, game play continues. I don't think this is necessarily wrong, but
    want to think about it more because the players continue to play after this, even though the other players aren't
    updating regrets. However, they are still calculating strategy, which is helpful, as sigma is only updated when we
    stumble upon a particular infoset, so IF the regret exists already, this is helpful, as now sigma is updated,
    which makes the average strategy for post flop rounds (averaged from the strategy dump every x iterations) better,
    theoretically.

    For my own edification. Rules are same as No Limit, except that small bet and big bet are 2x small blind and
    big blind, respectively. The first two rounds can only use the small bet and the last two rounds can only use
    big bets. There seem to be some different rules over the number of raises in each round with the majority
    of sources saying one bet with three raises (total 4 "raises"). I left off at iteration one, after "skip"
"""
from __future__ import annotations

import copy
import collections
import json
import random
from typing import Any, Dict
import logging
logging.basicConfig(filename='output3.txt', level=logging.DEBUG)

import joblib
import numpy as np
from tqdm import tqdm, trange

from pluribus import utils
from pluribus.games.short_deck.player import ShortDeckPokerPlayer
from pluribus.games.short_deck.state import ShortDeckPokerState
from pluribus.poker.pot import Pot


#log = False


# TODO: In general, wondering how important this function is if we are to use
# the blueprint algo for more than the preflop round? Would using just sigma
# allow for a more complete rendering of strategies for infosets?
# From Future Colin: No, sigma updates wildly
# (see https://github.com/fedden/pluribus-poker-AI/blob/feature/WIP-debug-test-blueprint/research/blueprint_algo/blueprint_kuhn.py)
def update_strategy(state: ShortDeckPokerState, i: int):
    """

    :param state: the game state
    :param i: the player, i = 1 is always first to act and i = 2 is always second to act, but they take turns who
        updates the strategy (only one strategy)
    :return: nothing, updates action count in the strategy of actions chosen according to sigma, this simple choosing of
        actions is what allows the algorithm to build up preference for one action over another in a given spot
    """
    logging.debug("UPDATE STRATEGY")
    logging.debug("########")
    logging.debug("########")
    logging.debug("########")
    logging.debug(f"Iteration: {t}")
    logging.debug(f"Player Set to Update Regret: {i}")
    logging.debug(f"P(h): {state.player_i}")
    logging.debug(f"P(h) Updating Regret? {state.player_i == i}")
    logging.debug(f"Betting Round {state._betting_stage}")

    logging.debug(f"Community Cards {state._table.community_cards}")
    try:
        logging.debug(f"I(h): {state.info_set}")
    except KeyError:
        pass
    logging.debug(f"Betting Action Correct?: {state.players}")
    logging.debug("########")
    logging.debug(f"Regret: {regret}")
    logging.debug("########")
    logging.debug(f"Sigma: {sigma}")
    logging.debug("########")
    logging.debug(f"Strategy: {strategy}")

    logging.debug("########")
    logging.debug("########")
    logging.debug("########")
    ph = state.player_i

    player_not_in_hand = not state.players[i].is_active
    if state.is_terminal or player_not_in_hand or state.betting_round > 0:  # looks good
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
        try:
            logging.debug(f"About to Calculate Strategy, Regret Exists: {regret[I]}")
        except UnboundLocalError:
            logging.debug(f"About to Calculate Strategy, Regret does not exist")
        calculate_strategy(regret, sigma, I, state)
        logging.debug(f"Calculated Strategy for {I}: {sigma[t + 1][I]}")
        # choose an action based of sigma
        try:
            a = np.random.choice(
                list(sigma[t][I].keys()), 1, p=list(sigma[t][I].values())
            )[0]
            logging.debug(f"ACTION SAMPLED: ph {state.player_i} {a}")
        except ValueError:
            p = 1 / len(state.legal_actions) # should be good
            probabilities = np.full(len(state.legal_actions), p)
            a = np.random.choice(state.legal_actions, p=probabilities)
            sigma[t][I] = {action: p for action in state.legal_actions}
            logging.debug(f"ACTION SAMPLED: ph {state.player_i} {a}")
        # Increment the action counter.
        strategy[I][a] += 1
        logging.debug(f"Updated Strategy for {I}: {strategy[I]}")
        # so strategy is counts based on sigma, this takes into account the
        # reach probability so there is no need to pass around that pi guy..
        new_state: ShortDeckPokerState = state.apply_action(a)
        update_strategy(new_state, i)
    else:
        # Traverse each action.
        for a in state.legal_actions:
            logging.debug(f"Going to Traverse {a} for opponent")
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
    logging.debug("CFR")
    logging.debug("########")
    logging.debug("########")
    logging.debug("########")
    logging.debug(f"Iteration: {t}")
    logging.debug(f"Player Set to Update Regret: {i}")
    logging.debug(f"P(h): {state.player_i}")
    logging.debug(f"P(h) Updating Regret? {state.player_i == i}")
    logging.debug(f"Betting Round {state._betting_stage}")


    logging.debug(f"Community Cards {state._table.community_cards}")
    try:
        logging.debug(f"I(h): {state.info_set}")
    except KeyError:
        pass
    logging.debug(f"Betting Action Correct?: {state.players}")
    logging.debug("########")
    logging.debug(f"Regret: {regret}")
    logging.debug("########")
    logging.debug(f"Sigma: {sigma}")
    logging.debug("########")
    logging.debug(f"Strategy: {strategy}")
    logging.debug("########")
    logging.debug("########")
    logging.debug("########")
    ph = state.player_i

    if state.is_terminal:
        assert state.player_at_node is not None or state.player_i
        return state.payout[i] * (1 if i == state.player_at_node else -1) # todo I added this
        # TODO: I think this might need to be different,
        #  but I have not gotten there yet
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
        state.set_player_at_node(ph)  # TODO I added this
        I = state.info_set
        # calculate strategy
        try:
            logging.debug(f"About to Calculate Strategy, Regret Exists: {regret[Iph]}")
        except UnboundLocalError:
            logging.debug(f"About to Calculate Strategy, Regret does not exist")
        calculate_strategy(regret, sigma, I, state)
        logging.debug(f"Calculated Strategy for {I}: {sigma[t+1][I]}")

        # TODO: Does updating sigma here (as opposed to after regret) miss out
        #       on any updates? If so, is there any benefit to having it up
        #       here?
        vo = 0.0
        voa = {}
        for a in state.legal_actions:
            logging.debug(f"ACTION TRAVERSED FOR REGRET:  ph {state.player_i} {a}")
            new_state: ShortDeckPokerState = state.apply_action(a)
            voa[a] = cfr(new_state, i, t)
            logging.debug(f"Got EV for {a}: {voa[a]}")
            vo += sigma[t][I][a] * voa[a]
            if len(state.legal_actions) == 3:
                if a == 'raise':
                    logging.debug(f"Done with EV at {I}: {vo}")
            elif len(state.legal_actions) == 2:
                if a == 'call':
                    logging.debug(f"Done with EV at {I}: {vo}")
            else:
                logging.debug(f"Updated EV at {I}: {vo}")

        for a in state.legal_actions:
            regret[I][a] += voa[a] - vo
        logging.debug(f"Updated Regret at {I}: {regret[I]}")

        # do not need update the strategy based on regret, strategy does
        # that with sigma

        return vo
    else:
        Iph = state.info_set
        print(Iph)
        try:
            logging.debug(f"About to Calculate Strategy, Regret Exists: {regret[Iph]}")
        except UnboundLocalError:
            logging.debug(f"About to Calculate Strategy, Regret does not exist")
        calculate_strategy(regret, sigma, Iph, state)
        logging.debug(f"Calculated Strategy for {Iph}: {sigma[t+1][Iph]}")
        try:
            a = np.random.choice(
                list(sigma[t][Iph].keys()), 1, p=list(sigma[t][Iph].values())
            )[0]
            logging.debug(f"ACTION SAMPLED: ph {state.player_i} {a}")
        except ValueError:
            p = 1 / len(state.legal_actions)
            probabilities = np.full(len(state.legal_actions), p)
            a = np.random.choice(state.legal_actions, p=probabilities)
            sigma[t][Iph] = {action: p for action in state.legal_actions}
            logging.debug(f"ACTION SAMPLED: ph {state.player_i} {a}")
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
    ph = state.player_i

    if state.is_terminal:
        return state.payout[i] * (1 if i == 1 else -1)  # TODO need to check this
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
        # Don't reload massive files, it takes ages. EONS
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
        tqdm.write(f"{info_set}")
        for action, probability in action_to_probabilities.items():
            tqdm.write(f"  - {action}: {probability / norm:.2f}")


def to_dict(**kwargs) -> Dict[str, Any]:
    """Hacky method to convert weird collections dicts to regular dicts."""
    return json.loads(json.dumps(copy.deepcopy(kwargs)))


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
    strategy_interval = 2  # it's just to test.
    n_iterations = 10
    LCFR_threshold = 80
    discount_interval = 1000
    prune_threshold = 4000
    C = -20000  # somewhat arbitrary
    n_players = 3
    print_iteration = 1
    dump_iteration = 10
    update_threshold = 0  # just to test

    # algorithm presented here, pg.16:
    # https://science.sciencemag.org/content/sci/suppl/2019/07/10/science.aay2400.DC1/aay2400-Brown-SM.pdf
    logging.info("beginning training")
    info_set_lut = {}
    for t in trange(1, n_iterations + 1, desc="train iter"):
        if t == 3:
            break
        sigma[t + 1] = copy.deepcopy(sigma[t])
        for i in range(n_players):  # fixed position i
            # Create a new state.
            state: ShortDeckPokerState = new_game(n_players, info_set_lut)
            info_set_lut = state.info_set_lut
            if t > update_threshold and t % strategy_interval == 0:
                # Only start updating after 800 minutes in Pluribus
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
        if (t > update_threshold) & (t % dump_iteration == 0):
            # Only start updating after 800 minutes in Pluribus
            # This is for the post-preflop betting rounds. It seems they
            # dump the current strategy (sigma) throughout
            # training and then take an average.
            # This allows for estimation of expected value in
            # leaf nodes later on using modified versions of the blueprint strategy
            to_persist = to_dict(strategy=strategy, regret=regret, sigma=sigma)
            joblib.dump(to_persist, f"strategy_{t}.gz", compress="gzip")
        del sigma[t]
        if t % print_iteration == 0:
            print_strategy(strategy)

    to_persist = to_dict(strategy=strategy, regret=regret, sigma=sigma)
    joblib.dump(to_persist, "strategy.gz", compress="gzip")
    print_strategy(strategy)
