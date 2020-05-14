"""
"""
from __future__ import annotations

import logging

logging.basicConfig(filename="test.txt", level=logging.DEBUG)

from tqdm import trange

from pluribus.games.short_deck.state import *


def update_strategy(agent: Agent, state: ShortDeckPokerState, ph_test_node: int):
    """

    :param state: the game state
    :param i: the player, i = 1 is always first to act and i = 2 is always second to act, but they take turns who
        updates the strategy (only one strategy)
    :return: nothing, updates action count in the strategy of actions chosen according to sigma, this simple choosing of
        actions is what allows the algorithm to build up preference for one action over another in a given spot
    """
    logging.debug("UPDATE STRATEGY")
    logging.debug("########")

    logging.debug(f"P(h): {state.player_i}")
    logging.debug(f"Betting Round {state._betting_stage}")
    logging.debug(f"Community Cards {state._table.community_cards}")
    logging.debug(f"Player 0 hole cards: {state.players[0].cards}")
    logging.debug(f"Player 1 hole cards: {state.players[1].cards}")
    logging.debug(f"Player 2 hole cards: {state.players[2].cards}")
    logging.debug(f"Betting Action Correct?: {state.players}")

    ph = state.player_i  # this is always the case no matter what i is

    if ph == ph_test_node:
        try:
            I = state.info_set
        except:
            import ipdb;
            ipdb.set_trace()
        # calculate regret
        logging.debug(f"About to Calculate Strategy, Regret: {agent.regret[I]}")
        logging.debug(f"Current regret: {agent.regret[I]}")
        sigma = calculate_strategy(agent.regret, I, state)
        logging.debug(f"Calculated Strategy for {I}: {sigma[I]}")
        # choose an action based of sigma
        try:
            a = np.random.choice(list(sigma[I].keys()), 1, p=list(sigma[I].values()))[0]
            logging.debug(f"ACTION SAMPLED: ph {state.player_i} ACTION: {a}")
        except ValueError:
            p = 1 / len(state.legal_actions)
            probabilities = np.full(len(state.legal_actions), p)
            a = np.random.choice(state.legal_actions, p=probabilities)
            sigma[I] = {action: p for action in state.legal_actions}
            logging.debug(f"ACTION SAMPLED: ph {state.player_i} ACTION: {a}")
        # Increment the action counter.
        agent.strategy[I][a] += 1
        logging.debug(f"Updated Strategy for {I}: {agent.strategy[I]}")
        return
    else:
        return


def calculate_strategy(
        regret: Dict[str, Dict[str, float]], I: str, state: ShortDeckPokerState,
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
    logging.debug("CFR")
    logging.debug("########")
    logging.debug(f"Iteration: {t}")
    logging.debug(f"Player Set to Update Regret: {i}")
    logging.debug(f"P(h): {state.player_i}")
    logging.debug(f"P(h) Updating Regret? {state.player_i == i}")
    logging.debug(f"Betting Round {state._betting_stage}")
    logging.debug(f"Community Cards {state._table.community_cards}")
    logging.debug(f"Player 0 hole cards: {state.players[0].cards}")
    logging.debug(f"Player 1 hole cards: {state.players[1].cards}")
    logging.debug(f"Player 2 hole cards: {state.players[2].cards}")
    logging.debug(f"Betting Action Correct?: {state.players}")

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
        try:
            I = state.info_set
        except:
            import ipdb;
            ipdb.set_trace()
        # calculate strategy
        logging.debug(f"About to Calculate Strategy, Regret: {agent.regret[I]}")
        logging.debug(f"Current regret: {agent.regret[I]}")
        sigma = calculate_strategy(agent.regret, I, state)
        logging.debug(f"Calculated Strategy for {I}: {sigma[I]}")

        vo = 0.0
        voa = {}
        for a in state.legal_actions:
            logging.debug(
                f"ACTION TRAVERSED FOR REGRET:  ph {state.player_i} ACTION: {a}"
            )
            new_state: ShortDeckPokerState = state.apply_action(a)
            voa[a] = cfr(agent, new_state, i, t)
            logging.debug(f"Got EV for {a}: {voa[a]}")
            vo += sigma[I][a] * voa[a]
            logging.debug(
                f"""Added to Node EV for ACTION: {a} INFOSET: {I} 
                STRATEGY: {sigma[I][a]}: {sigma[I][a] * voa[a]}"""
            )
        logging.debug(f"Updated EV at {I}: {vo}")

        for a in state.legal_actions:
            agent.regret[I][a] += voa[a] - vo
        logging.debug(f"Updated Regret at {I}: {agent.regret[I]}")

        return vo
    else:
        # import ipdb;
        # ipdb.set_trace()
        try:
            Iph = state.info_set
        except:
            import ipdb;
            ipdb.set_trace()
        logging.debug(f"About to Calculate Strategy, Regret: {agent.regret[Iph]}")
        logging.debug(f"Current regret: {agent.regret[Iph]}")
        sigma = calculate_strategy(agent.regret, Iph, state)
        logging.debug(f"Calculated Strategy for {Iph}: {sigma[Iph]}")

        try:
            a = np.random.choice(
                list(sigma[Iph].keys()), 1, p=list(sigma[Iph].values()),
            )[0]
            logging.debug(f"ACTION SAMPLED: ph {state.player_i} ACTION: {a}")

        except ValueError:
            p = 1 / len(state.legal_actions)
            probabilities = np.full(len(state.legal_actions), p)
            a = np.random.choice(state.legal_actions, p=probabilities)
            sigma[Iph] = {action: p for action in state.legal_actions}
            logging.debug(f"ACTION SAMPLED: ph {state.player_i} ACTION: {a}")

        new_state: ShortDeckPokerState = state.apply_action(a)
        return cfr(agent, new_state, i, t)


# this is a decent function for loading up a particular action sequence
def get_game_state(state: ShortDeckPokerState, action_sequence: list):
    """Follow through the action sequence provided to get current node"""
    if not action_sequence:
        return state
    a = action_sequence.pop(0)
    if a == "skip":
        a = action_sequence.pop(0)
    new_state = state.apply_action(a)
    return get_game_state(new_state, action_sequence)


# added some flags for RT
def new_rt_game(
    n_players: int, offline_strategy: Dict, action_sequence, public_cards, real_time_test=True
) -> ShortDeckPokerState:
    """Create a new game of short deck poker."""
    pot = Pot()
    players = [
        ShortDeckPokerPlayer(player_i=player_i, initial_chips=10000, pot=pot)
        for player_i in range(n_players)
    ]
    state = ShortDeckPokerState(
        players=players, offline_strategy=offline_strategy, real_time_test=real_time_test, public_cards=public_cards
    )
    current_game_state = get_game_state(state, action_sequence)
    # decided to make this a one time method rather than something that always updates
    # reason being: we won't need it except for a few choice nodes
    current_game_state.update_hole_cards_bayes()
    return current_game_state


def train(
    offline_strategy: Dict,
    public_cards,
    action_sequence: list,
    n_iterations: int,
    lcfr_threshold: int,
    discount_interval: int,
    n_players: int,
    update_interval: int,
    update_threshold: int,
):
    """Train agent."""
    utils.random.seed(38)
    agent = Agent()

    current_game_state = new_rt_game(3, offline_strategy, action_sequence, public_cards)
    ph_test_node = current_game_state.player_i
    for t in trange(1, n_iterations + 1, desc="train iter"):
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
