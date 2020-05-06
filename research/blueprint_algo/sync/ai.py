import logging
from typing import Dict, List

import numpy as np

from agent import Agent
from pluribus.games.short_deck.state import ShortDeckPokerState


log = logging.getLogger("sync.ai")


def update_strategy(agent: Agent, state: ShortDeckPokerState, i: int, t: int, locks):
    """

    :param state: the game state
    :param i: the player, i = 1 is always first to act and i = 2 is always second to act, but they take turns who
        updates the strategy (only one strategy)
    :return: nothing, updates action count in the strategy of actions chosen according to sigma, this simple choosing of
        actions is what allows the algorithm to build up preference for one action over another in a given spot
    """
    log.debug("UPDATE STRATEGY")
    log.debug("########")

    log.debug(f"Iteration: {t}")
    log.debug(f"Player Set to Update Regret: {i}")
    log.debug(f"P(h): {state.player_i}")
    log.debug(f"P(h) Updating Regret? {state.player_i == i}")
    log.debug(f"Betting Round {state._betting_stage}")
    log.debug(f"Community Cards {state._table.community_cards}")
    log.debug(f"Player 0 hole cards: {state.players[0].cards}")
    log.debug(f"Player 1 hole cards: {state.players[1].cards}")
    log.debug(f"Player 2 hole cards: {state.players[2].cards}")
    try:
        log.debug(f"I(h): {state.info_set}")
    except KeyError:
        pass
    log.debug(f"Betting Action Correct?: {state.players}")

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
    #   update_strategy(rs, h + a, i, t)

    elif ph == i:
        I = state.info_set
        # calculate regret
        sigma = calculate_strategy(agent.regret, state)
        log.debug(f"Calculated Strategy for {I}: {sigma}")
        # choose an action based of sigma
        available_actions: List[str] = list(sigma.keys())
        action_probabilities: List[float] = list(sigma.values())
        action: str = np.random.choice(available_actions, p=action_probabilities)
        log.debug(f"ACTION SAMPLED: ph {state.player_i} ACTION: {action}")
        # Increment the action counter.
        locks["strategy"].acquire()
        this_states_strategy = agent.strategy.get(I, state.initial_strategy)
        this_states_strategy[action] += 1
        # Update the master strategy by assigning.
        agent.strategy[I] = this_states_strategy
        locks["strategy"].release()
        new_state: ShortDeckPokerState = state.apply_action(action)
        update_strategy(agent, new_state, i, t, locks)
    else:
        # Traverse each action.
        for action in state.legal_actions:
            log.debug(f"Going to Traverse {action} for opponent")
            new_state: ShortDeckPokerState = state.apply_action(action)
            update_strategy(agent, new_state, i, t, locks)


def calculate_strategy(
    regret: Dict[str, Dict[str, float]], state: ShortDeckPokerState,
) -> Dict[str, float]:
    """

    :param regret: dictionary of regrets, I is key, then each action at I, with values being regret
    :param state: the game state
    :return: doesn't return anything, just updates sigma
    """
    default_probability = 1 / len(state.legal_actions)
    sigma: Dict[str, float] = dict()
    # We don't make use of default dicts anymore, so prepare a dictionary
    # describing uniform regret of zero for all legal actions for this state.
    this_states_regret = regret.get(state.info_set, state.initial_regret)
    regret_sum = sum([max(r, 0) for r in this_states_regret.values()])
    for action in state.legal_actions:
        if regret_sum > 0:
            sigma[action] = max(this_states_regret[action], 0) / regret_sum
        else:
            sigma[action] = default_probability
    return sigma


def cfr(agent: Agent, state: ShortDeckPokerState, i: int, t: int, locks) -> float:
    """
    regular cfr algo

    :param state: the game state
    :param i: player
    :param t: iteration
    :return: expected value for node for player i
    """
    log.debug("CFR")
    log.debug("########")
    log.debug(f"Iteration: {t}")
    log.debug(f"Player Set to Update Regret: {i}")
    log.debug(f"P(h): {state.player_i}")
    log.debug(f"P(h) Updating Regret? {state.player_i == i}")
    log.debug(f"Betting Round {state._betting_stage}")
    log.debug(f"Community Cards {state._table.community_cards}")
    log.debug(f"Player 0 hole cards: {state.players[0].cards}")
    log.debug(f"Player 1 hole cards: {state.players[1].cards}")
    log.debug(f"Player 2 hole cards: {state.players[2].cards}")
    try:
        log.debug(f"I(h): {state.info_set}")
    except KeyError:
        pass
    log.debug(f"Betting Action Correct?: {state.players}")

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
        sigma = calculate_strategy(agent.regret, state)
        log.debug(f"Calculated Strategy for {I}: {sigma}")

        vo = 0.0
        voa: Dict[str, float] = {}
        for action in state.legal_actions:
            log.debug(
                f"ACTION TRAVERSED FOR REGRET: ph {state.player_i} ACTION: {action}"
            )
            new_state: ShortDeckPokerState = state.apply_action(action)
            voa[action] = cfr(agent, new_state, i, t, locks)
            log.debug(f"Got EV for {action}: {voa[action]}")
            vo += sigma[action] * voa[action]
            log.debug(
                f"Added to Node EV for ACTION: {action} INFOSET: {I}\n"
                f"STRATEGY: {sigma[action]}: {sigma[action] * voa[action]}"
            )
        log.debug(f"Updated EV at {I}: {vo}")
        locks["regret"].acquire()
        this_states_regret = agent.regret.get(I, state.initial_regret)
        for action in state.legal_actions:
            this_states_regret[action] += voa[action] - vo
        # Assign regret back to the shared memory.
        agent.regret[I] = this_states_regret
        locks["regret"].release()
        return vo
    else:
        Iph = state.info_set
        sigma = calculate_strategy(agent.regret, state)
        log.debug(f"Calculated Strategy for {Iph}: {sigma}")
        available_actions: List[str] = list(sigma.keys())
        action_probabilities: List[float] = list(sigma.values())
        action: str = np.random.choice(available_actions, p=action_probabilities)
        log.debug(f"ACTION SAMPLED: ph {state.player_i} ACTION: {action}")
        new_state: ShortDeckPokerState = state.apply_action(action)
        return cfr(agent, new_state, i, t, locks)


def cfrp(agent: Agent, state: ShortDeckPokerState, i: int, t: int, c: int, locks):
    """
    pruning cfr algo, might need to adjust only pruning if not final betting round and if not terminal node

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
        sigma = calculate_strategy(agent.regret, state)
        # TODO: Does updating sigma here (as opposed to after regret) miss out
        #       on any updates? If so, is there any benefit to having it up
        #       here?
        vo = 0.0
        voa: Dict[str, float] = dict()
        # Explored dictionary to keep track of regret updates that can be
        # skipped.
        explored: Dict[str, bool] = {action: False for action in state.legal_actions}
        # Get the regret for this state.
        this_states_regret = agent.regret.get(I, state.initial_regret)
        for action in state.legal_actions:
            if this_states_regret[action] > c:
                new_state: ShortDeckPokerState = state.apply_action(action)
                voa[action] = cfrp(agent, new_state, i, t, c, locks)
                explored[action] = True
                vo += sigma[action] * voa[action]
        locks["regret"].acquire()
        # Get the regret for this state again, incase any other process updated
        # it whilst we were doing `cfrp`.
        this_states_regret = agent.regret.get(I, state.initial_regret)
        for action in state.legal_actions:
            if explored[action]:
                this_states_regret[action] += voa[action] - vo
        # Update the master copy of the regret.
        agent.regret[I] = this_states_regret
        locks["regret"].release()
        return vo
    else:
        sigma = calculate_strategy(agent.regret, state)
        available_actions: List[str] = list(sigma.keys())
        action_probabilities: List[float] = list(sigma.values())
        action: str = np.random.choice(available_actions, p=action_probabilities)
        new_state: ShortDeckPokerState = state.apply_action(action)
        return cfrp(agent, new_state, i, t, c, locks)
