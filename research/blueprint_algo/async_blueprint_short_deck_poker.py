"""
"""
from __future__ import annotations

import copy
import datetime
import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List

import click
import joblib
import numpy as np
import yaml
from tqdm import tqdm, trange

from pluribus import utils
from pluribus.games.short_deck.player import ShortDeckPokerPlayer
from pluribus.games.short_deck.state import ShortDeckPokerState
from pluribus.poker.pot import Pot

logging.basicConfig(filename="async_logs.txt", level=logging.DEBUG)


class Agent:
    # TODO(fedden): Note from the supplementary material, the data here will
    #               need to be lower precision: "To save memory, regrets were
    #               stored using 4-byte integers rather than 8-byte doubles.
    #               There was also a ﬂoor on regret at -310,000,000 for every
    #               action. This made it easier to unprune actions that were
    #               initially pruned but later improved. This also prevented
    #               integer overﬂows".
    manager = mp.Manager()

    def __init__(self):
        self.strategy = self.manager.dict()
        self.regret = self.manager.dict()


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
        sigma = calculate_strategy(agent.regret, state)
        logging.debug(f"Calculated Strategy for {I}: {sigma}")
        # choose an action based of sigma
        available_actions: List[str] = list(sigma.keys())
        action_probabilities: List[float] = list(sigma.values())
        action: str = np.random.choice(available_actions, p=action_probabilities)[0]
        logging.debug(f"ACTION SAMPLED: ph {state.player_i} ACTION: {action}")
        # Increment the action counter.
        this_states_strategy = agent.strategy.get(I, state.initial_strategy)
        this_states_strategy[action] += 1
        # Update the master strategy by assigning.
        agent.strategy[I] = this_states_strategy
        new_state: ShortDeckPokerState = state.apply_action(action)
        update_strategy(agent, new_state, i, t)
    else:
        # Traverse each action.
        for action in state.legal_actions:
            logging.debug(f"Going to Traverse {action} for opponent")
            new_state: ShortDeckPokerState = state.apply_action(action)
            update_strategy(agent, new_state, i, t)


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
        logging.debug(f"Calculated Strategy for {I}: {sigma}")

        vo = 0.0
        voa: Dict[str, float] = {}
        for action in state.legal_actions:
            logging.debug(
                f"ACTION TRAVERSED FOR REGRET: ph {state.player_i} ACTION: {action}"
            )
            new_state: ShortDeckPokerState = state.apply_action(action)
            voa[action] = cfr(agent, new_state, i, t)
            logging.debug(f"Got EV for {action}: {voa[action]}")
            vo += sigma[action] * voa[action]
            logging.debug(
                f"Added to Node EV for ACTION: {action} INFOSET: {I}\n"
                f"STRATEGY: {sigma[action]}: {sigma[action] * voa[action]}"
            )
        logging.debug(f"Updated EV at {I}: {vo}")
        this_states_regret = agent.regret.get(I, state.initial_regret)
        for action in state.legal_actions:
            this_states_regret[action] += voa[action] - vo
        # Assign regret back to the shared memory.
        agent.regret[I] = this_states_regret
        return vo
    else:
        Iph = state.info_set
        sigma = calculate_strategy(agent.regret, state)
        logging.debug(f"Calculated Strategy for {Iph}: {sigma}")
        available_actions: List[str] = list(sigma.keys())
        action_probabilities: List[float] = list(sigma.values())
        action: str = np.random.choice(available_actions, p=action_probabilities)[0]
        logging.debug(f"ACTION SAMPLED: ph {state.player_i} ACTION: {action}")
        new_state: ShortDeckPokerState = state.apply_action(action)
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
                voa[action] = cfrp(agent, new_state, i, t, c)
                explored[action] = True
                vo += sigma[action] * voa[action]
        for action in state.legal_actions:
            if explored[action]:
                this_states_regret[action] += voa[action] - vo
        # Update the master copy of the regret.
        agent.regret[I] = this_states_regret
        return vo
    else:
        sigma = calculate_strategy(agent.regret, state)
        available_actions: List[str] = list(sigma.keys())
        action_probabilities: List[float] = list(sigma.values())
        action: str = np.random.choice(available_actions, p=action_probabilities)[0]
        new_state: ShortDeckPokerState = state.apply_action(action)
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


class Sentinel:
    def __init__(self):
        pass


class Worker(mp.Process):
    """"""

    def __init__(
        self,
        queue: mp.Queue,
        n_players: int,
        prune_threshold: int,
        c: int,
        lcfr_threshold: int,
        discount_interval: int,
        update_threshold: int,
        dump_iteration: int,
        save_path: Path,
    ):
        """"""
        super(Worker, self).__init__()
        self._queue: mp.Queue = queue
        self._state: ShortDeckPokerState = None
        self._info_set_lut: Dict[str, str] = {}
        self._n_players = n_players
        self._prune_threshold = prune_threshold
        self._agent = None
        self._c = c
        self._lcfr_threshold = lcfr_threshold
        self._discount_interval = discount_interval
        self._update_threshold = update_threshold
        self._dump_iteration = dump_iteration
        self._save_path = save_path

    def run(self):
        """"""
        while True:
            # TODO(fedden): Do we want to syncronise the reading and writing to
            #               the `self._agent`? We may have to coordinate if the
            #               processes interfere with one another.
            parameters_tuple = self._queue.get(block=True)
            if isinstance(parameters_tuple, tuple):
                t, prune_threshold, i = parameters_tuple
                self._single_iteration(t, prune_threshold, i)
            elif isinstance(parameters_tuple, Sentinel):
                break
            else:
                raise ValueError(f"Unrecognised parameters: {parameters_tuple}")

    def _single_iteration(self, t, prune_threshold, i):
        """"""
        self._setup_new_game()
        use_pruning = np.random.uniform() < 0.95
        if t > prune_threshold and use_pruning:
            cfr(self._agent, self._state, i, t)
        else:
            cfrp(self._agent, self._state, i, t, self._c)
        if t < self._lcfr_threshold & t % self._discount_interval == 0:
            # TODO(fedden): Is discount_interval actually set/managed in
            #               minutes here? In Algorithm 1 this should be managed
            #               in minutes using perhaps the time module, but here
            #               it appears to be being managed by the iterations
            #               count.
            discount_factor = (t / self._discount_interval) / (
                (t / self._discount_interval) + 1
            )
            for info_set in self._agent.regret.keys():
                for action in self._agent.regret[info_set].keys():
                    self._agent.regret[info_set][action] *= discount_factor
                    self._agent.strategy[info_set][action] *= discount_factor
        if t > self._update_threshold and t % self._dump_iteration == 0:
            # dump the current
            # strategy (sigma) throughout training and then take an average.
            # This allows for estimation of expected value in leaf nodes later
            # on using modified versions of the blueprint strategy
            to_persist = to_dict(
                strategy=self._agent.strategy, regret=self._agent.regret
            )
            joblib.dump(
                to_persist, self._save_path / f"strategy_{t}.gz", compress="gzip"
            )

    def _setup_new_game(self):
        """"""
        self._state: ShortDeckPokerState = new_game(self._n_players, self._info_set_lut)
        self._info_set_lut = self._state.info_set_lut


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
    n_processes = 3  # mp.cpu_count() - 1
    config: Dict[str, int] = {**locals()}
    save_path: Path = _create_dir()
    with open(save_path / "config.yaml", "w") as steam:
        yaml.dump(config, steam)
    utils.random.seed(42)
    queue: mp.Queue = mp.Queue(maxsize=100)
    agent = Agent()
    workers = [
        Worker(
            queue=queue,
            n_players=n_players,
            prune_threshold=prune_threshold,
            c=c,
            lcfr_threshold=lcfr_threshold,
            discount_interval=discount_interval,
            update_threshold=update_threshold,
            dump_iteration=dump_iteration,
            save_path=save_path,
        )
        for _ in range(n_processes)
    ]
    for worker in workers:
        worker.start()
    for t in trange(1, n_iterations + 1, desc="train iter"):
        for i in range(n_players):  # fixed position i
            parameters_tuple = (t, prune_threshold, i)
            queue.put(parameters_tuple, block=True)
    for _ in workers:
        queue.put(Sentinel())
    to_persist = to_dict(strategy=agent.strategy, regret=agent.regret)
    joblib.dump(to_persist, save_path / "strategy.gz", compress="gzip")
    print_strategy(agent.strategy)


if __name__ == "__main__":
    train()
