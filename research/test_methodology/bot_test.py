from typing import List, Dict, DefaultDict
from pathlib import Path
import joblib
import collections

from tqdm import trange
import yaml
import datetime
import numpy as np
from scipy import stats

from pluribus.games.short_deck.state import ShortDeckPokerState, new_game
from pluribus.poker.card import Card


def _calculate_strategy(
        state: ShortDeckPokerState,
        I: str,
        strategy: DefaultDict[str, DefaultDict[str, float]]
) -> str:
    sigma = collections.defaultdict(lambda: collections.defaultdict(lambda: 1 / 3))
    import ipdb;
    ipdb.set_trace()

    try:
        # If strategy is empty, go to other block
        if sigma[I] == {}:
            raise KeyError
        sigma[I] = strategy[I].copy()
        norm = sum(sigma[I].values())
        for a in sigma[I].keys():
            sigma[I][a] /= norm
        a = np.random.choice(
            list(sigma[I].keys()), 1, p=list(sigma[I].values()),
        )[0]
    except KeyError:
        p = 1 / len(state.legal_actions)
        probabilities = np.full(len(state.legal_actions), p)
        a = np.random.choice(state.legal_actions, p=probabilities)
        sigma[I] = {action: p for action in state.legal_actions}
    return a


def _create_dir() -> Path:
    """Create and get a unique dir path to save to using a timestamp."""
    time = str(datetime.datetime.now())
    for char in ":- .":
        time = time.replace(char, "_")
    path: Path = Path(f"./results_{time}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def agent_test(
    hero_strategy_path: str,
    opponent_strategy_path: str,
    real_time_est: bool = False,
    action_sequence: List[str] = None,
    public_cards: List[Card] = [],
    n_outter_iters: int = 30,
    n_inner_iters: int = 100,
    n_players: int = 3,
):
    config: Dict[str, int] = {**locals()}
    save_path: Path = _create_dir()
    with open(save_path / "config.yaml", "w") as steam:
        yaml.dump(config, steam)

    # Load unnormalized strategy for hero
    hero_strategy = joblib.load(hero_strategy_path)
    # Load unnormalized strategy for opponents
    opponent_strategy = joblib.load(opponent_strategy_path)

    # Loading game state we used RTS on
    if real_time_est:
        state: ShortDeckPokerState = new_game(
            n_players, real_time_test=real_time_est, public_cards=public_cards
        )
        current_game_state: ShortDeckPokerState = state.load_game_state(
            opponent_strategy, action_sequence
        )

    # TODO: Right now, this can only be used for loading states if the two strategies
    # are averaged. Even averaging strategies is risky. Loading a game state
    # should be used with caution. It will work only if the probability of reach
    # is identical across strategies. Use the average strategy.

    # Load current game state, either strategy should be identical for this
    EVs = np.array([])
    for _ in trange(1, n_outter_iters):
        EV = np.array([])  # Expected value for player 0 (hero)
        for t in trange(1, n_inner_iters + 1, desc="train iter"):
            for p_i in range(n_players):
                if real_time_est:
                    # Deal hole cards based on bayesian updating of hole card probs
                    state: ShortDeckPokerState = current_game_state.deal_bayes()
                else:
                    state: ShortDeckPokerState = new_game(n_players)
                while True:
                    player_not_in_hand = not state.players[p_i].is_active
                    if state.is_terminal or player_not_in_hand:
                        EV = np.append(EV, state.payout[p_i])
                        break
                    if state.player_i == p_i:
                        random_action: str = _calculate_strategy(
                            state,
                            state.info_set,
                            hero_strategy,
                        )
                    else:
                        random_action: str = _calculate_strategy(
                            state,
                            state.info_set,
                            opponent_strategy,
                        )
                    state = state.apply_action(random_action)
        EVs = np.append(EVs, EV.mean())
    t_stat = (EVs.mean() - 0) / (EVs.std() / np.sqrt(n_outter_iters))
    p_val = stats.t.sf(np.abs(t_stat), n_outter_iters - 1)
    results_dict = {
        'Expected Value': float(EVs.mean()),
        'T Statistic': float(t_stat),
        'P Value': float(p_val)
    }
    with open(save_path / 'results.yaml', "w") as stream:
        yaml.safe_dump(results_dict, stream=stream, default_flow_style=False)


if __name__ == "__main__":
    strat_path = "test_strategy/unnormalized_output/"
    agent_test(
        hero_strategy_path=strat_path + "rts_output.gz",
        opponent_strategy_path=strat_path + "offline_strategy_100.gz",
        real_time_est=True,
        public_cards=[Card("ace", "spades"), Card("queen", "spades"),
                      Card("queen", "hearts")],
        action_sequence=["raise", "call", "call", "call", "call"],
        n_inner_iters=100
    )
