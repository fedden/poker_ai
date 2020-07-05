import copy
from operator import itemgetter
from typing import Dict, List

from blessed import Terminal

from poker_ai.games.short_deck.state import ShortDeckPokerState
from poker_ai.terminal.ascii_objects.card_collection import AsciiCardCollection
from poker_ai.terminal.ascii_objects.logger import AsciiLogger
from poker_ai.terminal.ascii_objects.player import AsciiPlayer


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


def print_header(
    term: Terminal,
    state: ShortDeckPokerState,
    og_name_to_name: Dict[str, str]
):
    for line in _compute_header_lines(state, og_name_to_name):
        print(term.center(term.yellow(line)))
    print(f"\n{term.width * '-'}\n")


def print_footer(
    term: Terminal,
    selected_action_i: int,
    legal_actions: List[str]
):
    print(f"\n{term.width * '-'}\n")
    actions = []
    for action_i in range(len(legal_actions)):
        action = copy.deepcopy(legal_actions[action_i])
        if action_i == selected_action_i:
            action = term.blink_bold_orangered(action)
        actions.append(action)
    print(term.center("    ".join(actions)))


def print_table(
    term: Terminal,
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


def print_log(term: Terminal, log: AsciiLogger):
    print(f"\n{term.width * '-'}\n")
    y, _ = term.get_location()
    # Tell the log how far it can print before logging any more.
    log.height = term.height - y - 1
    print(log)
