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
    hide_cards=True,
)
player_2 = Player(
    Card(10, "diamonds"), Card(2, "clubs"), name="Player 2", term=term, hide_cards=True,
)
player_3 = Player(
    Card("jack", "spades"), Card("ace", "clubs"), name="Player 3", term=term,
)
public_cards = CardCollection(
    Card(9, "spades"), Card("ace", "clubs"), Card(3, "hearts"), term=term,
)


with term.cbreak(), term.hidden_cursor():

    print(term.home + term.white + term.clear)
    print(player_1)
    print(player_2)
    _, cursor_y = term.get_location(timeout=5)
    location = term.move_xy(
        term.width - 1 - public_cards.width, cursor_y - player_2.height
    )
    print(location + f"{public_cards}")
    print(player_3)
