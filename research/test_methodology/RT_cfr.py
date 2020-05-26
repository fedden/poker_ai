from __future__ import annotations

import collections
from typing import Dict
import joblib
from pathlib import Path

from tqdm import trange
import numpy as np
import datetime
import yaml

from pluribus import utils
from pluribus.games.short_deck.state import ShortDeckPokerState, new_game
from pluribus.games.short_deck.agent import Agent


def normalize_strategy(this_info_sets_regret: Dict[str, float]) -> Dict[str, float]:
    """Calculate the strategy based on the current information sets regret."""
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


def calculate_strategy(
    regret: Dict[str, Dict[str, float]],
    I: str,
    state: ShortDeckPokerState,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate strategy based on regret
    """
    sigma = collections.defaultdict(lambda: collections.defaultdict(lambda: 1 / 3))
    rsum = sum([max(x, 0) for x in regret[I].values()])
    for a in state.legal_actions:
        if rsum > 0:
            sigma[I][a] = max(regret[I][a], 0) / rsum
        else:
            sigma[I][a] = 1 / len(state.legal_actions)
    return sigma


def _create_dir(folder_id: str) -> Path:
    """Create and get a unique dir path to save to using a timestamp."""
    time = str(datetime.datetime.now())
    for char in ":- .":
        time = time.replace(char, "_")
    path: Path = Path(f"./{folder_id}_results_{time}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def cfr(agent: Agent, state: ShortDeckPokerState, i: int, t: int) -> float:
    """
    CFR algo with the a temporary regret object for better strategy averaging
    """
    ph = state.player_i

    player_not_in_hand = not state.players[i].is_active
    if state.is_terminal or player_not_in_hand:
        return state.payout[i]

    elif ph == i:
        I = state.info_set
        # Move regret over to temporary object and build off that
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
        # Move regret over to a temporary object and build off that
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


def rts(
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
    """RTS."""
    config: Dict[str, int] = {**locals()}
    save_path: Path = _create_dir('RTS')
    with open(save_path / "config.yaml", "w") as steam:
        yaml.dump(config, steam)
    # TODO: fix the seed
    # utils.random.seed(36)
    agent = Agent(regret_path=regret_path)
    # Load unnormalized strategy to build off
    offline_strategy = joblib.load(offline_strategy_path)
    state: ShortDeckPokerState = new_game(
        3, real_time_test=True, public_cards=public_cards
    )
    # Load current game state
    current_game_state: ShortDeckPokerState = state.load_game_state(
        offline_strategy, action_sequence
    )
    # We don't need the offline strategy for search..
    # del offline_strategy
    for t in trange(1, n_iterations + 1, desc="train iter"):
        for i in range(n_players):  # fixed position i
            # Deal hole cards based on bayesian updating of hole card probs
            state: ShortDeckPokerState = current_game_state.deal_bayes()
            cfr(agent, state, i, t)
        if t < lcfr_threshold & t % discount_interval == 0:
            d = (t / discount_interval) / ((t / discount_interval) + 1)
            for I in agent.tmp_regret.keys():
                for a in agent.tmp_regret[I].keys():
                    agent.tmp_regret[I][a] *= d
        # Add the unnormalized strategy into the original
        # Right now assumes dump_int is a multiple of n_iterations
        if t % dump_int == 0:
            # offline_strategy = joblib.load(offline_strategy_path)
            # Adding the regret back to the regret dict, we'll build off for next RTS
            for I in agent.tmp_regret.keys():
                if agent.tmp_regret != {}:
                    agent.regret[I] = agent.tmp_regret[I].copy()
            for info_set, this_info_sets_regret in sorted(agent.tmp_regret.items()):
                # If this_info_sets_regret == {}, we do nothing
                strategy = normalize_strategy(this_info_sets_regret)
                # Check if info_set exists..
                no_info_set = info_set not in offline_strategy
                if no_info_set or offline_strategy[info_set] == {}:
                    offline_strategy[info_set] = {a: 0 for a in strategy.keys()}
                for action, probability in strategy.items():
                    try:
                        offline_strategy[info_set][action] += probability
                    except:
                        import ipdb;
                        ipdb.set_trace()
            agent.reset_new_regret()

    return agent, offline_strategy
