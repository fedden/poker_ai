## Visualisation Code

This code is to visualise a given instance of the `ShortDeckPokerState`. [The frontend code is based on this codepen.](https://codepen.io/Rovak/pen/ExYeQar)

It looks like this:
<p align="center">
  <img src="https://github.com/fedden/poker_ai-poker-AI/blob/develop/assets/visualisation.png">
</p>

### How to run

First build the frontend, this will be served a static files by the `PokerPlot` class.
```bash
cd frontend
npm run build
```

Next run the plot in some script, i.e:
```python
from plot import PokerPlot
from poker_ai.games.short_deck.player import ShortDeckPokerPlayer
from poker_ai.games.short_deck.state import ShortDeckPokerState
from poker_ai.poker.pot import Pot


def get_state() -> ShortDeckPokerState:
    """Gets a state to visualise"""
    n_players = 6
    pot = Pot()
    players = [
        ShortDeckPokerPlayer(player_i=player_i, initial_chips=10000, pot=pot)
        for player_i in range(n_players)
    ]
    return ShortDeckPokerState(
        players=players, 
        pickle_dir="../../research/blueprint_algo/"
    )


pp: PokerPlot = PokerPlot()
# If you visit http://localhost:5000/ now you will see an empty table.

# ... later on in the code, as proxy for some code that obtains a new state ...
# Obtain a new state.
state: ShortDeckPokerState = get_state()
# Update the state to be plotted, this is sent via websockets to the frontend.
pp.update_state(state)
# If you visit http://localhost:5000/ now you will see table with 6 players.
```
