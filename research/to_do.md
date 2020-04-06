A Place for Next Steps in Short Deck Implementation

## Abstraction

#### Information Abstraction
- hard code opening hand clusters
- decide how to store these for lookup in blueprint/real time algo
- run for short deck

#### Action Abstraction
- not sure how this fits into blueprint/real time yet

## Blueprint Algo
- apply to contrived short deck game

## Real Time Search Algo
- need isomorphic/lossless handling of cards??  # Non-essential maybe..
- mock up "toy" version 
  - pre-req: stateful version of short deck

### Rules of Contrived Short Deck Game
- 3 players
- 2-9 removed
- no adjustments to hand rankings versus no-limit
- 10000 in stack, 50 small blind, 100 big blind
- limited betting

#### Possible Next Steps
- fix short deck game and roll out to online hosting?
- go right on to full game?

#### Current (Concise) Papers 
- Abstraction
  - https://www.cs.cmu.edu/~sandholm/hierarchical.aamas15.pdf <- this algo
  - http://www.ifaamas.org/Proceedings/aamas2013/docs/p271.pdf <- these features
- Blueprint
  - https://science.sciencemag.org/content/sci/suppl/2019/07/10/science.aay2400.DC1/aay2400-Brown-SM.pdf <- pseudo code
- Real Time Algo
  - https://papers.nips.cc/paper/7993-depth-limited-solving-for-imperfect-information-games.pdf <- build off this
  - make theses changes:
    - [optimized vector-based linear cfr?](https://arxiv.org/pdf/1809.04040.pdf)
    - [only samples chance events?](http://martin.zinkevich.org/publications/ijcai2011_rgbr.pdf)
    
#### TODO: Colin
- Generate abstraction for 20 cards
-- Program to turn that into dictionary and store separately
- Hard code preflop lossless
- Write next steps in docstring of blueprint algo
- Consider getting rid of notebooks before merging into develop..