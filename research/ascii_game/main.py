import copy
import random
from collections import deque
from datetime import datetime
from operator import itemgetter
from typing import Any, Dict, List

import joblib
import numpy as np
from blessed import Terminal

from pluribus.games.short_deck.state import new_game, ShortDeckPokerState
from card_collection import CardCollection
from player import Player


class AsciiLogger:
    """"""

    def __init__(self, term: Terminal):
        """"""
        self._log_queue: deque = deque()
        self._term = term
        self.height = None

    def info(self, x: str):
        """"""
        if self.height is None:
            raise ValueError("Logger.height must be set before logging.")
        str_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log_queue.append(f"{self._term.skyblue1(str_time)} {x}")
        if len(self._log_queue) > self.height:
            self._log_queue.popleft()

    def __str__(self) -> str:
        """"""
        if self.height is None:
            raise ValueError("Logger.height must be set before logging.")
        n_logs = len(self._log_queue)
        start = max(n_logs - self.height, 0)
        lines = [self._log_queue[i] for i in range(start, n_logs)]
        return "\n".join(lines)


def _compute_header_str(
    state: ShortDeckPokerState, player_names: List[str], table_rotation: int
) -> str:
    if state.is_terminal:
        player_winnings = []
        for player_i, chips_delta in state.payout.items():
            rotated_i = (player_i + table_rotation) % len(state.players)
            player_winnings.append((player_names[rotated_i], chips_delta))
        player_winnings.sort(key=itemgetter(1), reverse=True)
        player_desc_strings = [
            f"{n} {'wins' if x > 0 else 'loses'} {x} chips" for n, x in player_winnings
        ]
        player_desc: str = ", ".join(player_desc_strings)
        header_str: str = f"{state.betting_stage} - {player_desc}"
    else:
        header_str = state.betting_stage
    return header_str


def print_header(
    state: ShortDeckPokerState, player_names: List[str], table_rotation: int
):
    header_str = _compute_header_str(state, player_names, table_rotation)
    print(term.center(term.yellow(header_str)))
    print(f"\n{term.width * '-'}\n")


def print_footer(selected_action_i: int, legal_actions: List[str]):
    print(f"\n{term.width * '-'}\n")
    actions = []
    for action_i in range(len(legal_actions)):
        action = copy.deepcopy(legal_actions[action_i])
        if action_i == selected_action_i:
            action = term.blink_bold_orangered(action)
        actions.append(action)
    print(term.center("    ".join(actions)))


def print_table(players: List[Player], public_cards: CardCollection):
    for line in players[2].lines:
        print(term.center(line))
    for line_a, line_b in zip(players[1].lines, public_cards.lines):
        print(line_a + " " + line_b)
    for line in players[0].lines:
        print(term.center(line))


def print_log(log: AsciiLogger):
    print(f"\n{term.width * '-'}\n")
    y, _ = term.get_location()
    # Tell the log how far it can print before logging any more.
    log.height = term.height - y - 1
    print(log)


def rotate(l: List[Any], n: int):
    if n > len(l):
        raise ValueError
    return l[-n:] + l[:-n]


term = Terminal()
log = AsciiLogger(term)
debug_quick_start = True
table_position_to_orientation: Dict[int, str] = {0: "top", 1: "right", 2: "bottom"}
n_players: int = 3
pickle_dir: str = "/home/tollie/dev/pluribus/research/blueprint_algo"
if debug_quick_start:
    state: ShortDeckPokerState = new_game(n_players, {}, load_pickle_files=False)
else:
    state: ShortDeckPokerState = new_game(n_players, pickle_dir=pickle_dir)
is_bot: List[bool] = [False, True, True]
player_names: List[str] = ["human", "bot 1", "bot 2"]
selected_action_i: int = 0
table_rotation: int = 0
agent: str = "offline"
strategy_path: str = (
    "/home/tollie/dev/pluribus/research/blueprint_algo/offline_strategy_285800.gz"
)
if not debug_quick_start and agent in {"offline", "online"}:
    offline_strategy = joblib.load(strategy_path)
elif debug_quick_start and agent in {"offline", "online"}:
    offline_strategy = {}
with term.cbreak(), term.hidden_cursor():
    while True:
        # Construct ascii objects to be rendered later.
        players: List[Player] = []
        player_indices = rotate(list(range(n_players)), table_rotation)
        state_players = rotate(state.players, table_rotation)
        for i, (rotated_i, state_player) in enumerate(zip(player_indices, state_players)):
            ascii_player = Player(
                *state_player.cards,
                name=player_names[i],
                term=term,
                info_position=table_position_to_orientation[i],
                hide_cards=is_bot[i] and not state.is_terminal,
                folded=not state.players[i].is_active,
                is_turn=i == state.player_i,
                chips_in_pot=state_player.n_bet_chips,
                chips_in_bank=state_player.n_chips,
                is_small_blind=state_player.is_small_blind,
                is_big_blind=state_player.is_big_blind,
                is_dealer=state_player.is_dealer,
            )
            players.append(ascii_player)
        public_cards = CardCollection(*state.community_cards)
        if state.is_terminal:
            legal_actions = ["quit", "new game"]
        else:
            legal_actions = [] if is_bot[state.player_i] else state.legal_actions
        # Render game.
        print(term.home + term.white + term.clear)
        print_header(state, player_names, table_rotation)
        print_table(players, public_cards)
        print_footer(selected_action_i, legal_actions)
        print_log(log)
        # Make action of some kind.
        if not is_bot[state.player_i] or state.is_terminal:
            key = term.inkey(timeout=None)
            if key.name == "KEY_LEFT":
                selected_action_i -= 1
                if selected_action_i < 0:
                    selected_action_i = len(legal_actions) - 1
            elif key.name == "KEY_RIGHT":
                selected_action_i = (selected_action_i + 1) % len(legal_actions)
            elif key.name == "KEY_ENTER":
                action = legal_actions[selected_action_i]
                selected_action_i = 0
                if action == "quit":
                    log.info(term.pink("quit"))
                    break
                elif action == "new game":
                    log.info(term.green("new game"))
                    if debug_quick_start:
                        state: ShortDeckPokerState = new_game(
                            n_players, state.info_set_lut, load_pickle_files=False,
                        )
                    else:
                        state: ShortDeckPokerState = new_game(
                            n_players, state.info_set_lut,
                        )
                    table_rotation = (table_rotation + 1) % n_players
                else:
                    name = player_names[state.player_i]
                    log.info(term.green(f"{name} chose {action}"))
                    state: ShortDeckPokerState = state.apply_action(action)
        else:
            if agent == "random":
                action = random.choice(state.legal_actions)
            elif agent == "offline":
                default_strategy = {
                    action: 1 / len(state.legal_actions)
                    for action in state.legal_actions
                }
                this_state_strategy = offline_strategy.get(
                    state.info_set, default_strategy
                )
                actions = list(this_state_strategy.keys())
                probabilties = list(this_state_strategy.values())
                action = np.random.choice(actions, p=probabilties)
            name = player_names[state.player_i]
            log.info(f"{agent} {name} chose {action}")
            state: ShortDeckPokerState = state.apply_action(action)
