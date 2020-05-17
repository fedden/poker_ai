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
from card_collection import AsciiCardCollection
from player import AsciiPlayer


class AsciiLogger:
    """"""

    def __init__(self, term: Terminal):
        """"""
        self._log_queue: deque = deque()
        self._term = term
        self.height = None

    def info(self, *args):
        """"""
        if self.height is None:
            raise ValueError("Logger.height must be set before logging.")
        x: str = " ".join(map(str, args))
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


def _compute_header_lines(
    state: ShortDeckPokerState, og_name_to_name: Dict[str, str]
) -> List[str]:
    if state.is_terminal:
        player_winnings = []
        for player_i, chips_delta in state.payout.items():
            p = state.players[player_i]
            player_winnings.append((p, chips_delta))
        player_winnings.sort(key=itemgetter(1), reverse=True)
        winnings_desc_strings = [
            f"{og_name_to_name[p.name]} {'wins' if x > 0 else 'loses'} {x} chips"
            for p, x in player_winnings
        ]
        winnings_desc: str = ", ".join(winnings_desc_strings)
        winning_player = player_winnings[0][0]
        winning_rank: int = state._poker_engine.evaluator.evaluate(
            state.community_cards, winning_player.cards
        )
        winning_hand_class: int = state._poker_engine.evaluator.get_rank_class(
            winning_rank
        )
        winning_hand_desc: str = state._poker_engine.evaluator.class_to_string(
            winning_hand_class
        ).lower()
        return [
            f"{og_name_to_name[winning_player.name]} won with a {winning_hand_desc}",
            winnings_desc,
        ]
    return ["", state.betting_stage]


def print_header(state: ShortDeckPokerState, og_name_to_name: Dict[str, str]):
    for line in _compute_header_lines(state, og_name_to_name):
        print(term.center(term.yellow(line)))
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


def print_table(
    players: Dict[str, AsciiPlayer],
    public_cards: AsciiCardCollection,
    n_table_rotations: int,
    n_spaces_between_cards: int = 4,
    n_chips_in_pot: int = 0,
):
    left_player = players["left"]
    middle_player = players["middle"]
    right_player = players["right"]
    for line in public_cards.lines:
        print(term.center(line))
    print(term.center(f"chips in pot: {n_chips_in_pot}"))
    print("\n\n")
    spacing = " " * n_spaces_between_cards
    for l, m, r in zip(left_player.lines, middle_player.lines, right_player.lines):
        print(term.center(f"{l}{spacing}{m}{spacing}{r}"))


def print_log(log: AsciiLogger):
    print(f"\n{term.width * '-'}\n")
    y, _ = term.get_location()
    # Tell the log how far it can print before logging any more.
    log.height = term.height - y - 1
    print(log)


def rotate_list(l: List[Any], n: int):
    if n > len(l):
        raise ValueError
    return l[n:] + l[:n]


def rotate_int(x, dx, mod):
    x = (x + dx) % mod
    while x < 0:
        x += mod
    return x


term = Terminal()
log = AsciiLogger(term)
debug_quick_start = False
n_players: int = 3
pickle_dir: str = "/home/tollie/dev/pluribus/research/blueprint_algo"
if debug_quick_start:
    state: ShortDeckPokerState = new_game(n_players, {}, load_pickle_files=False)
else:
    state: ShortDeckPokerState = new_game(n_players, pickle_dir=pickle_dir)
n_table_rotations: int = 0
human_i: int = 2
selected_action_i: int = 0
positions = ["left", "middle", "right"]
names = {"left": "BOT 1", "middle": "BOT 2", "right": "HUMAN"}
agent: str = "offline"
strategy_path: str = (
    "/home/tollie/dev/pluribus/research/blueprint_algo/offline_strategy_285800.gz"
)
if not debug_quick_start and agent in {"offline", "online"}:
    offline_strategy = joblib.load(strategy_path)
else:
    offline_strategy = {}
with term.cbreak(), term.hidden_cursor():
    while True:
        # Construct ascii objects to be rendered later.
        ascii_players: Dict[str, AsciiPlayer] = {}
        state_players = rotate_list(state.players[::-1], n_table_rotations)
        og_name_to_position = {}
        og_name_to_name = {}
        for player_i, player in enumerate(state_players):
            position = positions[player_i]
            is_human = names[position].lower() == "human"
            ascii_players[position] = AsciiPlayer(
                *player.cards,
                term=term,
                name=names[position],
                og_name=player.name,
                hide_cards=not is_human and not state.is_terminal,
                folded=not player.is_active,
                is_turn=player.is_turn,
                chips_in_pot=player.n_bet_chips,
                chips_in_bank=player.n_chips,
                is_small_blind=player.is_small_blind,
                is_big_blind=player.is_big_blind,
                is_dealer=player.is_dealer,
            )
            og_name_to_position[player.name] = position
            og_name_to_name[player.name] = names[position]
            if player.is_turn:
                current_player_name = names[position]
        public_cards = AsciiCardCollection(*state.community_cards)
        if state.is_terminal:
            legal_actions = ["quit", "new game"]
            is_human_turn = True
        else:
            og_current_name = state.current_player.name
            is_human_turn = og_name_to_position[og_current_name] == "right"
            if is_human_turn:
                legal_actions = state.legal_actions
            else:
                legal_actions = []
        # Render game.
        print(term.home + term.white + term.clear)
        print_header(state, og_name_to_name)
        print_table(
            ascii_players,
            public_cards,
            n_table_rotations,
            n_chips_in_pot=state._table.pot.total,
        )
        print_footer(selected_action_i, legal_actions)
        print_log(log)
        # import ipdb; ipdb.set_trace()
        # Make action of some kind.
        if is_human_turn:
            # Incase the legal_actions went from length 3 to 2 and we had
            # previously picked the last one.
            selected_action_i %= len(legal_actions)
            key = term.inkey(timeout=None)
            if key.name == "KEY_LEFT":
                selected_action_i -= 1
                if selected_action_i < 0:
                    selected_action_i = len(legal_actions) - 1
            elif key.name == "KEY_RIGHT":
                selected_action_i = (selected_action_i + 1) % len(legal_actions)
            elif key.name == "KEY_ENTER":
                action = legal_actions[selected_action_i]
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
                    n_table_rotations -= 1
                    if n_table_rotations < 0:
                        n_table_rotations = n_players - 1
                else:
                    log.info(term.green(f"{current_player_name} chose {action}"))
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
            log.info(f"{current_player_name} chose {action}")
            state: ShortDeckPokerState = state.apply_action(action)
