import logging
import os
import threading

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from poker_ai.games.short_deck.state import ShortDeckPokerState

from backend import convert


logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.getLogger("socketio").setLevel(logging.ERROR)
logging.getLogger("engineio").setLevel(logging.ERROR)
app = Flask(__name__, static_folder="./dist/static", template_folder="./dist")
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, logger=False, engineio_logger=False)


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def _catch_all(path):
    return render_template("index.html")


class PokerPlot:
    """"""

    def __init__(self):
        """Run the visualisation server at port 5000."""
        global app
        self._app = app
        os.environ["WERKZEUG_RUN_MAIN"] = "true"
        args = (app,)
        kwargs = dict(host="localhost", port=5000)
        self._thread = threading.Thread(target=socketio.run, args=args, kwargs=kwargs)
        self._thread.start()

    def update_state(self, state: ShortDeckPokerState):
        """Update the state that should be visualised."""
        state_dict = {
            "player_playing": state.player_i,
            "players": [
                convert.to_player_dict(player_i=i, player=p)
                for i, p in enumerate(state.players)
            ],
            "five_cards": [
                convert.to_card_dict(c) for c in state._table.community_cards
            ],
        }
        with self._app.app_context():
            emit("state", state_dict, namespace="/", broadcast=True)
