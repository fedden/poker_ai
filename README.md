[![Build Status](https://travis-ci.org/fedden/pluribus-poker-AI.svg?branch=develop)](https://travis-ci.org/fedden/pluribus-poker-AI)

# Open Source Pluribus Implementation

This repository will contain a best effort implementation of the key ideas from the Pluribus poker AI bot. 

## Pre-requisites

This repository assumes Python 3.7 or newer is used.

## Installing

There isn't much to do with this repository at the moment but one could install the Python package by cloning this repo and pip installing it:
```bash
git clone https://github.com/fedden/pluribus-poker-AI.git # Though really we should use ssh here!
cd pluribus-poker-AI
pip install .
```

## Rough todo

The following todo will change dynamically as my understanding of the algorithms and the pluribus project evolves. 

At first, the goal is to prototype in Python as iteration will be much easier and quicker. Once there is a working prototype, write in a systems level language like C++ and optimise for performance. 

### 1. Implement a multiplayer working heads up no limit poker game engine to support the self-play.
- [x] Lay down the foundation of game objects (player, card etc).
- [x] Add poker hand evaluation code to the engine.
- [x] Support a player going all in during betting.
- [ ] Support a player going all in during payouts.
- [ ] Add a simple visualisation to allow a game to be visualised as it progresses. 
- [ ] Triple check that the rules are implemented in the poker engine as described in the supplimentary material.

### 2. Iterate on the AI algorithms and the integration into the poker engine. 
- [ ] Integrate the AI strategy to support self-play in the multiplayer poker game engine.
- [ ] In the game-engine, allow the replay of any round the current hand to support MCCFR. 
- [ ] Implement the creation of the blueprint strategy using Monte Carlo CFR miminisation.
- [ ] Add the real-time search for better strategies during the game.

## Structure

```bash
├── paper    # main source of info and documentation
└── pluribus # python code
    ├── ai   # (currently) python stubs for ai algorithms.
    └── game # wip code for managing a hand of poker
```

## Contributing

This is an open effort and help, criticisms and ideas are all welcome. Feel free to start a discussion on the github issues or to reach out to me at leonfedden at gmail dot com. 

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
