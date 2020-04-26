"""
"""
from __future__ import annotations

import copy
import collections
import datetime
import json
import random
from pathlib import Path
from typing import Any, Dict
import logging

logging.basicConfig(filename="after_rewrite_variable_logs_example.txt", level=logging.DEBUG)

import click
import joblib
import numpy as np
import yaml
from tqdm import tqdm, trange

from pluribus import utils
from pluribus.games.short_deck.player import ShortDeckPokerPlayer
from pluribus.games.short_deck.state import ShortDeckPokerState
from pluribus.poker.pot import Pot


class Agent:
    # TODO(fedden): Note from the supplementary material, the data here will
    #               need to be lower precision: "To save memory, regrets were
    #               stored using 4-byte integers rather than 8-byte doubles.
    #               There was also a ﬂoor on regret at -310,000,000 for every
    #               action. This made it easier to unprune actions that were
    #               initially pruned but later improved. This also prevented
    #               integer overﬂows".
    def __init__(self):
        self.strategy = collections.defaultdict(
            lambda: collections.defaultdict(lambda: 0)
        )
        self.regret = collections.defaultdict(
            lambda: collections.defaultdict(lambda: 0)
        )


def update_strategy(agent: Agent, state: ShortDeckPokerState, i: int, t: int):
    """

    :param state: the game state
    :param i: the player, i = 1 is always first to act and i = 2 is always second to act, but they take turns who
        updates the strategy (only one strategy)
    :return: nothing, updates action count in the strategy of actions chosen according to sigma, this simple choosing of
        actions is what allows the algorithm to build up preference for one action over another in a given spot
    """
    logging.debug("UPDATE STRATEGY")
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
    try:
        logging.debug(f"I(h): {state.info_set}")
    except KeyError:
        pass
    logging.debug(f"Betting Action Correct?: {state.players}")

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
        new_state: ShortDeckPokerState = state.apply_action(a)
        update_strategy(agent, new_state, i, t)
    else:
        # Traverse each action.
        for a in state.legal_actions:
            logging.debug(f"Going to Traverse {a} for opponent")

            new_state: ShortDeckPokerState = state.apply_action(a)
            update_strategy(agent, new_state, i, t)


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
    try:
        logging.debug(f"I(h): {state.info_set}")
    except KeyError:
        pass
    logging.debug(f"Betting Action Correct?: {state.players}")

    ph = state.player_i

    if state.is_terminal:
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

        # TODO: checkout caching the regrets that get updated - for the end of the iteration we
        #  could update strategy if that helps..

        return vo
    else:
        Iph = state.info_set
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


def cfrp(agent: Agent, state: ShortDeckPokerState, i: int, t: int, c: int):
    """
    pruning cfr algo, might need to adjust only pruning if not final betting round and if not terminal node

    :param state: the game state
    :param i: player
    :param t: iteration
    :return: expected value for node for player i
    """
    ph = state.player_i

    if state.is_terminal:
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
        sigma = calculate_strategy(agent.regret, I, state)
        # TODO: Does updating sigma here (as opposed to after regret) miss out
        #       on any updates? If so, is there any benefit to having it up
        #       here?
        vo = 0.0
        voa = {}
        explored = {}  # keeps tracked of items that can be skipped
        for a in state.legal_actions:
            if agent.regret[I][a] > c:
                new_state: ShortDeckPokerState = state.apply_action(a)
                voa[a] = cfrp(agent, new_state, i, t, c)
                explored[a] = True
                vo += sigma[I][a] * voa[a]
            else:
                explored[a] = False
        for a in state.legal_actions:
            if explored[a]:
                agent.regret[I][a] += voa[a] - vo
        return vo
    else:
        Iph = state.info_set
        sigma = calculate_strategy(agent.regret, Iph, state)
        try:
            a = np.random.choice(
                list(sigma[Iph].keys()), 1, p=list(sigma[Iph].values()),
            )[0]
        except ValueError:
            p = 1 / len(state.legal_actions)
            probabilities = np.full(len(state.legal_actions), p)
            a = np.random.choice(state.legal_actions, p=probabilities)
            sigma[Iph] = {action: p for action in state.legal_actions}
        new_state: ShortDeckPokerState = state.apply_action(a)
        return cfrp(agent, new_state, i, t, c)


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
        tqdm.write(f"{info_set}")
        for action, probability in action_to_probabilities.items():
            tqdm.write(f"  - {action}: {probability / norm:.2f}")


def to_dict(**kwargs) -> Dict[str, Any]:
    """Hacky method to convert weird collections dicts to regular dicts."""
    return json.loads(json.dumps(copy.deepcopy(kwargs)))


def _create_dir() -> Path:
    """Create and get a unique dir path to save to using a timestamp."""
    time = str(datetime.datetime.now())
    for char in ":- .":
        time = time.replace(char, "_")
    path: Path = Path(f"./results_{time}")
    path.mkdir(parents=True, exist_ok=True)
    return path


@click.command()
@click.option("--strategy_interval", default=2, help=".")
@click.option("--n_iterations", default=10, help=".")
@click.option("--lcfr_threshold", default=80, help=".")
@click.option("--discount_interval", default=1000, help=".")
@click.option("--prune_threshold", default=4000, help=".")
@click.option("--c", default=-20000, help=".")
@click.option("--n_players", default=3, help=".")
@click.option("--print_iteration", default=10, help=".")
@click.option("--dump_iteration", default=10, help=".")
@click.option("--update_threshold", default=0, help=".")
def train(
    strategy_interval: int,
    n_iterations: int,
    lcfr_threshold: int,
    discount_interval: int,
    prune_threshold: int,
    c: int,
    n_players: int,
    print_iteration: int,
    dump_iteration: int,
    update_threshold: int,
):
    """Train agent."""
    # Get the values passed to this method, save this.
    config: Dict[str, int] = {**locals()}
    save_path: Path = _create_dir()
    with open(save_path / "config.yaml", "w") as steam:
        yaml.dump(config, steam)
    utils.random.seed(42)
    agent = Agent()

    info_set_lut = {}
    for t in trange(1, n_iterations + 1, desc="train iter"):
        if t == 2:
            logging.disable(logging.DEBUG)
        for i in range(n_players):  # fixed position i
            # Create a new state.
            state: ShortDeckPokerState = new_game(n_players, info_set_lut)
            info_set_lut = state.info_set_lut
            if t > update_threshold and t % strategy_interval == 0:
                update_strategy(agent, state, i, t)
            if t > prune_threshold:
                if random.uniform(0, 1) < 0.05:
                    cfr(agent, state, i, t)
                else:
                    cfrp(agent, state, i, t, c)
            else:
                cfr(agent, state, i, t)
        if t < lcfr_threshold & t % discount_interval == 0:
            # TODO(fedden): Is discount_interval actually set/managed in
            #               minutes here? In Algorithm 1 this should be managed
            #               in minutes using perhaps the time module, but here
            #               it appears to be being managed by the iterations
            #               count.
            d = (t / discount_interval) / ((t / discount_interval) + 1)
            for I in agent.regret.keys():
                for a in agent.regret[I].keys():
                    agent.regret[I][a] *= d
                    agent.strategy[I][a] *= d
        if (t > update_threshold) & (t % dump_iteration == 0):
            # dump the current
            # strategy (sigma) throughout training and then take an average.
            # This allows for estimation of expected value in leaf nodes later
            # on using modified versions of the blueprint strategy
            to_persist = to_dict(strategy=agent.strategy, regret=agent.regret)
            joblib.dump(to_persist, save_path / f"strategy_{t}.gz", compress="gzip")

        if t % print_iteration == 0:
            print_strategy(agent.strategy)

    to_persist = to_dict(strategy=agent.strategy, regret=agent.regret)
    joblib.dump(to_persist, save_path / "strategy.gz", compress="gzip")
    print_strategy(agent.strategy)


if __name__ == "__main__":
    train()
