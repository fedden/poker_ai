| code-thing      | status        |
| --------------- | ------------- |
| master          | [![Build Status](https://travis-ci.org/fedden/pluribus-poker-AI.svg?branch=master)](https://travis-ci.org/fedden/pluribus-poker-AI)  |
| develop         | [![Build Status](https://travis-ci.org/fedden/pluribus-poker-AI.svg?branch=develop)](https://travis-ci.org/fedden/pluribus-poker-AI) |
| maintainability | [![Maintainability](https://api.codeclimate.com/v1/badges/c5a556dae097b809b4d9/maintainability)](https://codeclimate.com/github/fedden/pluribus-poker-AI/maintainability) |
| coverage        | [![Test Coverage](https://api.codeclimate.com/v1/badges/c5a556dae097b809b4d9/test_coverage)](https://codeclimate.com/github/fedden/pluribus-poker-AI/test_coverage) |
| license         | [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) |

# 🤖 Pluribus Poker AI

This repository will contain a best effort, open source implementation of the key ideas from the [Pluribus poker AI](https://www.cs.cmu.edu/~noamb/papers/19-Science-Superhuman.pdf) that plays [Texas Hold'em Poker](https://en.wikipedia.org/wiki/Texas_hold_'em). This includes the game engine needed to manage a hand of poker, and will implement the ideas from the paper with respect to the AI algorithms.

<p align="center">
  <img src="https://github.com/fedden/pluribus-poker-AI/blob/develop/assets/poker.jpg">
</p>

## Pre-requisites

This repository assumes Python 3.7 or newer is used.

## Installing

There isn't much to do with this repository at the moment but one could install the Python package by cloning this repo and pip installing it:
```bash
git clone https://github.com/fedden/pluribus-poker-AI.git # Though really we should use ssh here!
cd /path/to/pluribus-poker-AI
pip install .
```

## Running tests

I'm working on improving the testing as I progress. You can run the tests by moving to this repositories root directory (i.e `pluribus-poker-AI/`) and call the python test library `pytest`:
```bash
cd /path/to/pluribus-poker-AI
pip install pytest
pytest
```

## Structure

Below is a rough structure of the repository. 

```
├── paper          # Main source of info and documentation :)
├── pluribus       # Main Python library.
│   ├── ai         # Stub functions for ai algorithms.
│   ├── games      # Implementations of poker games as node based objects that
│   │              # can be traversed in a depth-first recursive manner.
│   ├── poker      # WIP general code for managing a hand of poker.
│   └── utils      # Utility code like seed setting.
├── research       # A directory for research/development scripts 
│                  # to help formulate understanding and ideas.
├── scripts        # Scripts to help develop the main library.
└── test           # Python tests.
    ├── functional # Functional tests that test multiple components 
    │              # together.
    └── unit       # Individual tests for functions and objects.
```

## Code Examples

There are two parts to this repository, the code to manage a game of poker, and the code to train an AI algorithm to play the game of poker. The reason the poker engine is being implemented is because it will likely be useful to have a well-integrated poker environment available during the development of the AI algorithm, incase there are tweaks that must be made to accomadate things like the history of state or the replay of a scenario during Monte Carlo Counterfactual Regret Minimisation. The following code is how one might program a round of poker that is deterministic using the engine. This engine is now the first pass that will be used support self play.

```python
from pluribus import utils
from pluribus.ai.dummy import RandomPlayer
from pluribus.poker.table import PokerTable
from pluribus.poker.engine import PokerEngine
from pluribus.poker.pot import Pot

# Seed so things are deterministic.
utils.random.seed(42)

# Some settings for the amount of chips.
initial_chips_amount = 10000
small_blind_amount = 50
big_blind_amount = 100

# Create the pot.
pot = Pot()
# Instanciate six players that will make random moves, make sure 
# they can reference the pot so they can add chips to it.
players = [
    RandomPlayer(
        name=f'player {player_i}',
        initial_chips=initial_chips_amount,
        pot=pot)
    for player_i in range(6)
]
# Create the table with the players on it.
table = PokerTable(players=players, pot=pot)
# Create the engine that will manage the poker game lifecycle.
engine = PokerEngine(
    table=table,
    small_blind=small_blind_amount,
    big_blind=big_blind_amount)
# Play a round of Texas Hold'em Poker!
engine.play_one_round()
```

The Pluribus AI algorithm is the next thing to implement so more coming on that as soon as possible...

## Roadmap

The following todo will change dynamically as my understanding of the algorithms and the pluribus project evolves. 

At first, the goal is to prototype in Python as iteration will be much easier and quicker. Once there is a working prototype, write in a systems level language like C++ and optimise for performance. 

### 1. Game engine iteration.
_Implement a multiplayer working heads up no limit poker game engine to support the self-play._
- [x] Lay down the foundation of game objects (player, card etc).
- [x] Add poker hand evaluation code to the engine.
- [x] Support a player going all in during betting.
- [x] Support a player going all in during payouts.
- [x] Lots of testing for various scenarios to ensure logic is working as expected.

### 2. AI iteration.
_Iterate on the AI algorithms and the integration into the poker engine._
- [ ] Integrate the AI strategy to support self-play in the multiplayer poker game engine.
- [ ] In the game-engine, allow the replay of any round the current hand to support MCCFR. 
- [ ] Implement the creation of the blueprint strategy using Monte Carlo CFR miminisation.
- [ ] Add the real-time search for better strategies during the game.

### 3. Game engine iteration.
_Strengthen the game engine with more tests and allow users to see live visualisation of game state._
- [ ] Add a simple visualisation to allow a game to be visualised as it progresses. 
- [ ] Triple check that the rules are implemented in the poker engine as described in the supplimentary material.
- [ ] Work through the coverage, adding more tests, can never have enough.

<p align="center">
  <img src="https://github.com/fedden/pluribus-poker-AI/blob/develop/assets/regret.jpeg">
</p>

## Contributing

This is an open effort and help, criticisms and ideas are all welcome. 

First of all, please check out the [CONTRIBUTING](/CONTRIBUTING.md) guide.

Feel free to start a discussion on the github issues or to reach out to me at leonfedden at gmail dot com. 

## Useful links and acknowledgements

There have already been a lot of helpful discussions and codebases on the path to building this project, which I'll try to keep updated with links to as I progress.

Naturally the first thing that should be acknowledged is the original paper. Here are the links to the paper that will be referenced to build the AI.
* [Paper](https://www.cs.cmu.edu/~noamb/papers/19-Science-Superhuman.pdf)
* [Supplimentary material](https://science.sciencemag.org/highwire/filestream/728919/field_highwire_adjunct_files/0/aay2400-Brown-SM.pdf)

Following are blogposts and discussions on the paper that served as helpful references.
* [Facebook blog post](https://ai.facebook.com/blog/pluribus-first-ai-to-beat-pros-in-6-player-poker/)
* [HackerNews discussion](https://news.ycombinator.com/item?id=20415379)
* [Other github discussions](https://github.com/whatsdis/pluribus)

Big shout out to the authors of the following repositories! Here are some MIT licensed codebases that I have found, pillaged and refactored to serve as the basis of the poker engine. 
* [Poker game code based on this (dead!?!) python package](https://pypi.org/project/pluribus-python/#data)
* [Pretty darn efficient poker hand evaluation (python 3 fork)](https://github.com/msaindon/deuces)

Useful tools that contributed to the making of the poker engine:
* [Poker hand winner calculator that came in handy for building tests for the engine.](https://www.pokerlistings.com/which-hand-wins-calculator)

Linked Notes 
* [Based off the supplemental materials](https://github.com/fedden/pluribus-poker-AI/blob/develop/paper/linked_notes.md)

MISC:
* Some [original author papers](https://www.cs.cmu.edu/~noamb/research.html)
* [Implementing MCCFR in python](https://www.youtube.com/watch?v=7m4bnmSkjow)
    * [Example Applied to Poker](https://github.com/geohot/ai-notebooks/blob/master/cfr_kuhn_poker.ipynb)

Other useful blog links, papers and resources:
* [Blog post on CFR](https://int8.io/counterfactual-regret-minimization-for-poker-ai/)
* [No regret dynamics tutorial](https://theory.stanford.edu/~tim/f13/l/l17.pdf)
* [Prediction, Learning and Games book.](http://www.ii.uni.wroc.pl/~lukstafi/pmwiki/uploads/AGT/Prediction_Learning_and_Games.pdf)
