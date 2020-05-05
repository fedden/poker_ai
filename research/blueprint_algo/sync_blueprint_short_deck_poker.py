"""
"""
from __future__ import annotations

import copy
import datetime
import json
import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import click
import joblib
import numpy as np
import yaml
from rich.logging import RichHandler
from tqdm import tqdm, trange

from pluribus import utils
from pluribus.games.short_deck.player import ShortDeckPokerPlayer
from pluribus.games.short_deck.state import ShortDeckPokerState
from pluribus.poker.pot import Pot

FORMAT = "%(message)s"
logging.basicConfig(
    filename="async_logs.txt",
    format=FORMAT,
    datefmt="[%X] ",
    handlers=[RichHandler()],
    level="NOTSET",
)

log = logging.getLogger("rich")
manager = mp.Manager()
InfoSetLookupTable = Dict[str, Dict[Tuple[int, ...], str]]


class Agent:
    # TODO(fedden): Note from the supplementary material, the data here will
    #               need to be lower precision: "To save memory, regrets were
    #               stored using 4-byte integers rather than 8-byte doubles.
    #               There was also a ﬂoor on regret at -310,000,000 for every
    #               action. This made it easier to unprune actions that were
    #               initially pruned but later improved. This also prevented
    #               integer overﬂows".

    def __init__(self):
        self.strategy = manager.dict()
        self.regret = manager.dict()


def update_strategy(agent: Agent, state: ShortDeckPokerState, i: int, t: int):
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
        this_states_strategy = agent.strategy.get(I, state.initial_strategy)
        this_states_strategy[action] += 1
        # Update the master strategy by assigning.
        agent.strategy[I] = this_states_strategy
        new_state: ShortDeckPokerState = state.apply_action(action)
        update_strategy(agent, new_state, i, t)
    else:
        # Traverse each action.
        for action in state.legal_actions:
            log.debug(f"Going to Traverse {action} for opponent")
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
            voa[action] = cfr(agent, new_state, i, t)
            log.debug(f"Got EV for {action}: {voa[action]}")
            vo += sigma[action] * voa[action]
            log.debug(
                f"Added to Node EV for ACTION: {action} INFOSET: {I}\n"
                f"STRATEGY: {sigma[action]}: {sigma[action] * voa[action]}"
            )
        log.debug(f"Updated EV at {I}: {vo}")
        this_states_regret = agent.regret.get(I, state.initial_regret)
        for action in state.legal_actions:
            this_states_regret[action] += voa[action] - vo
        # Assign regret back to the shared memory.
        agent.regret[I] = this_states_regret
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
        action: str = np.random.choice(available_actions, p=action_probabilities)
        new_state: ShortDeckPokerState = state.apply_action(action)
        return cfrp(agent, new_state, i, t, c)


def new_game(
    n_players: int, info_set_lut: InfoSetLookupTable = {}
) -> ShortDeckPokerState:
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


def load_info_set_lut() -> InfoSetLookupTable:
    """"""
    info_set_lut = ShortDeckPokerState.load_pickle_files(".")
    return info_set_lut
    shared_dict = manager.dict()
    for key, value in info_set_lut.items():
        shared_dict[key] = value
    return shared_dict


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
        agent: Agent,
        info_set_lut: InfoSetLookupTable,
        n_players: int,
        prune_threshold: int,
        c: int,
        lcfr_threshold: int,
        discount_interval: int,
        update_threshold: int,
        dump_iteration: int,
        save_path: Path,
        worker_status: Dict[str, bool],
    ):
        """"""
        super(Worker, self).__init__()
        self._queue: mp.Queue = queue
        self._state: ShortDeckPokerState = None
        self._info_set_lut: InfoSetLookupTable = info_set_lut
        self._n_players = n_players
        self._prune_threshold = prune_threshold
        self._agent = agent
        self._c = c
        self._lcfr_threshold = lcfr_threshold
        self._discount_interval = discount_interval
        self._update_threshold = update_threshold
        self._dump_iteration = dump_iteration
        self._save_path = save_path
        self._worker_status = worker_status

    def run(self):
        """"""
        while True:
            # Get the name of the method and the key word arguments needed for
            # the method.
            self._update_status("idle")
            name, kwargs = self._queue.get(block=True)
            if name == "terminate":
                break
            elif name == "cfr":
                function = self._cfr
            elif name == "discount":
                function = self._discount
            elif name == "serialise_agent":
                function = self._serialise_agent
            else:
                raise ValueError(f"Unrecognised function name: {name}")
            self._update_status(name)
            function(**kwargs)

    def _cfr(self, t, i):
        """Search over random game and calculate the strategy."""
        self._state: ShortDeckPokerState = new_game(self._n_players, self._info_set_lut)
        use_pruning = np.random.uniform() < 0.95
        if t > self._prune_threshold and use_pruning:
            cfr(self._agent, self._state, i, t)
        else:
            cfrp(self._agent, self._state, i, t, self._c)

    def _discount(self, t):
        """Discount previous regrets and strategy."""
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

    def _serialise_agent(self, t):
        """Write agent to file."""
        # dump the current
        # strategy (sigma) throughout training and then take an average.
        # This allows for estimation of expected value in leaf nodes later
        # on using modified versions of the blueprint strategy
        to_persist = to_dict(strategy=self._agent.strategy, regret=self._agent.regret)
        joblib.dump(to_persist, self._save_path / f"strategy_{t}.gz", compress="gzip")

    def _update_status(self, status):
        """Update the status of this worker in the shared dictionary."""
        self._worker_status[self.name] = status


import sys
import pdb


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


class Server:
    """"""

    def __init__(
        self,
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
        n_processes: int = mp.cpu_count() - 1,
        seed: Optional[int] = None,
    ):
        """Set up the optimisation server."""
        config: Dict[str, int] = {**locals()}
        self._save_path: Path = _create_dir()
        with open(self._save_path / "config.yaml", "w") as steam:
            yaml.dump(config, steam)
        log.info("saved config")
        if seed is not None:
            utils.random.seed(42)
        self._n_iterations = n_iterations
        self._lcfr_threshold = lcfr_threshold
        self._discount_interval = discount_interval
        self._update_threshold = update_threshold
        self._dump_iteration = dump_iteration
        self._n_players = n_players
        self._info_set_lut: InfoSetLookupTable = load_info_set_lut()
        log.info("Loaded lookup table.")
        self._queue: mp.Queue = mp.Queue(maxsize=n_processes)
        self._worker_status: Dict[str, bool] = manager.dict()
        self._agent: Agent = Agent()
        self._workers: Dict[str, Worker] = dict()
        for _ in range(n_processes):
            worker = Worker(
                queue=self._queue,
                agent=self._agent,
                info_set_lut=self._info_set_lut,
                n_players=n_players,
                prune_threshold=prune_threshold,
                c=c,
                lcfr_threshold=lcfr_threshold,
                discount_interval=discount_interval,
                update_threshold=update_threshold,
                dump_iteration=dump_iteration,
                save_path=self._save_path,
                worker_status=self._worker_status,
            )
            self._workers[worker.name] = worker
            self._workers[worker.name].start()
            log.info(f"started worker [bold green]{worker.name}[/]")

    def search(self):
        """Perform MCCFR and train the agent."""
        import ipdb

        ipdb.set_trace()
        for t in trange(1, self._n_iterations + 1, desc="train iter"):
            for i in range(self._n_players):
                self._send_job("cfr", t=t, i=i)
                if t < self._lcfr_threshold & t % self._discount_interval == 0:
                    self._wait_until_all_workers_are_idle()
                    self._send_job("discount", t=t)
                if t > self._update_threshold and t % self._dump_iteration == 0:
                    self._wait_until_all_workers_are_idle()
                    self._send_job("serialise_agent", t=t)

    def terminate(self):
        """Kill all workers."""
        for _ in self._workers.values():
            name = "terminate"
            kwargs = dict()
            self._queue.put((name, kwargs), block=True)
            log.info("sending sentinel to worker")
        for name, worker in self._workers.items():
            worker.join()
            log.info(f"worker {name} joined.")

    def serialise_agent(self):
        """Write agent to file."""
        to_persist = to_dict(strategy=self._agent.strategy, regret=self._agent.regret)
        joblib.dump(to_persist, self._save_path / "strategy.gz", compress="gzip")
        print_strategy(self._agent.strategy)

    def _send_job(self, name, **kwargs):
        """Send job of type `name` with arguments `kwargs` to worker pool."""
        self._queue.put((name, kwargs), block=True)

    def _wait_until_all_workers_are_idle(self, sleep_secs=0.5):
        """Blocks until all workers have finished their current job."""
        while any(status != "idle" for status in self._worker_status.values()):
            time.sleep(sleep_secs)


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
def search(
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
    server = Server(
        strategy_interval=strategy_interval,
        n_iterations=n_iterations,
        lcfr_threshold=lcfr_threshold,
        discount_interval=discount_interval,
        prune_threshold=prune_threshold,
        c=c,
        n_players=n_players,
        print_iteration=print_iteration,
        dump_iteration=dump_iteration,
        update_threshold=update_threshold,
        # n_processes=1,
        seed=42,
    )
    server.search()
    server.terminate()
    server.serialise_agent()


if __name__ == "__main__":
    search()
