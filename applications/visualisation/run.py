import requests
from typing import Any, Dict

from flask import Flask, render_template, jsonify
from flask_cors import CORS
from pluribus import utils
from pluribus.games.short_deck.player import ShortDeckPokerPlayer
from pluribus.games.short_deck.state import ShortDeckPokerState
from pluribus.poker.card import Card
from pluribus.poker.pot import Pot

utils.random.seed(42)
app = Flask(__name__, static_folder="./dist/static", template_folder="./dist")
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
colours = ["cyan", "lightcoral", "crimson", "#444", "forestgreen", "goldenrod", "gold"]
pot = Pot()
n_players = 3
players = [
    ShortDeckPokerPlayer(player_i=player_i, initial_chips=10000, pot=pot)
    for player_i in range(n_players)
]
state = ShortDeckPokerState(
    players=players, pickle_dir="../../research/blueprint_algo/"
)


def _to_player_dict(
    player_i: int, player: ShortDeckPokerPlayer, pot: Pot,
) -> Dict[str, Any]:
    """Create dictionary to describe player for frontend."""
    return {
        "name": player.name,
        "color": colours[player_i],
        "bank": player.n_chips,
        "onTable": pot[player],
        "hasCards": True,
    }


def _to_card_dict(card: Card) -> Dict[str, str]:
    """Create dictionary to describe card for frontend."""
    return {
        "f": {"spades": "P", "diamonds": "D", "clubs": "C", "hearts": "H",}[card.suit],
        "v": card.rank[0].upper(),
    }


@app.route("/api/state")
def get_state():
    response = {
        "player_playing": state.player_i,
        "players": [
            _to_player_dict(player_i=i, player=p, pot=pot)
            for i, p in enumerate(state.players)
        ],
        "five_cards": [_to_card_dict(c) for c in state._table.community_cards],
    }
    return jsonify(response)


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def catch_all(path):
    if app.debug:
        return requests.get(f"http://localhost:8080/{path}").text
    return render_template("index.html")
