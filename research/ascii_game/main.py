from blessed import Terminal

from pluribus.poker.card import Card
from card_collection import CardCollection
from player import Player


term = Terminal()

player_1 = Player(
    Card(4, "diamonds"),
    Card("ace", "hearts"),
    name="Player 1",
    term=term,
    info_position="bottom",
    hide_cards=True,
    folded=False,
)
player_2 = Player(
    Card(10, "diamonds"),
    Card(2, "clubs"),
    name="Player 2",
    term=term,
    info_position="right",
    hide_cards=True,
    folded=True,
)
player_3 = Player(
    Card("jack", "spades"),
    Card("ace", "clubs"),
    name="Player 3 (you)",
    term=term,
    info_position="top",
    hide_cards=False,
)
public_cards = CardCollection(
    Card(9, "spades"), Card("ace", "clubs"), Card(3, "hearts"), term=term,
)


with term.cbreak(), term.hidden_cursor():
    print(term.home + term.white + term.clear)
    for line in player_1.lines:
        print(term.center(line))
    for line_a, line_b in zip(player_2.lines, public_cards.lines):
        print(line_a + " " + line_b)
    for line in player_3.lines:
        print(term.center(line))
