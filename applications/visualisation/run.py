import requests
import threading

from flask import Flask, render_template, jsonify
from flask_cors import CORS
from pluribus import utils
from pluribus.games.short_deck.player import ShortDeckPokerPlayer
from pluribus.games.short_deck.state import ShortDeckPokerState
from pluribus.poker.pot import Pot

from backend import convert


utils.random.seed(42)
_app = Flask(__name__, static_folder="./dist/static", template_folder="./dist")
_cors = CORS(_app, resources={r"/api/*": {"origins": "*"}})
colours = ["cyan", "lightcoral", "crimson", "#444", "forestgreen", "goldenrod", "gold"]
pot = Pot()
n_players = 3
players = [
    ShortDeckPokerPlayer(player_i=player_i, initial_chips=10000, pot=pot)
    for player_i in range(n_players)
]
_state = ShortDeckPokerState(
    players=players, pickle_dir="../../research/blueprint_algo/"
)


def start():
    threading.Thread(target=_app.run).start()


def update_state(state: ShortDeckPokerState):
    global _state
    _state = state


@_app.route("/api/state")
def _get_state():
    response = {
        "player_playing": _state.player_i,
        "players": [
            convert.to_player_dict(player_i=i, player=p, pot=pot)
            for i, p in enumerate(_state.players)
        ],
        "five_cards": [convert.to_card_dict(c) for c in _state._table.community_cards],
    }
    return jsonify(response)


@_app.route("/", defaults={"path": ""})
@_app.route("/<path:path>")
def _catch_all(path):
    if _app.debug:
        return requests.get(f"http://localhost:8080/{path}").text
    return render_template("index.html")
