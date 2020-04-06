# Linked Reading List - Poker

## Primary Readings

1.  [Original Paper](https://www.cs.cmu.edu/~noamb/papers/19-Science-Superhuman.pdf)
2.  [Supplementary Materials](https://science.sciencemag.org/content/sci/suppl/2019/07/10/science.aay2400.DC1/aay2400-Brown-SM.pdf)
3.  [Sandholm](http://www.cs.cmu.edu/~sandholm/)
4.  [Brown](https://www.cs.cmu.edu/~noamb/research.html)

## Notes and links for further reading:

From #2:
- Real Time Search
  - Two improvements upon the [original real time search](https://papers.nips.cc/paper/7993-depth-limited-solving-for-imperfect-information-games.pdf):
    - Searcher chooses from 4 strategies like the other players
    - Uses nested unsafe search that starts at the beginning of the round (but our player's moves are held fixed)
- Equilibrium Finding
  - A few improvements 
    - [Linear MCCFR](https://arxiv.org/pdf/1809.04040.pdf)
      - This is used during the first 400 minutes of the blueprint strategy training (check pseudo-code in #2) 
    - Extreme negative regret pruning
      - Skips entire iteration
      - See pseudo-code in #2
- Notation 
  - (see pg. 8 & 9 of #2)
  - See in most papers introductory paragraphs..
- Abstraction
    - Action Abstraction
      - Blueprint
        - We do keep track of all action sequences, if they are encountered during the blueprint strategy creation,
        however bet sizes are coarse
          - 664,845,654 action sequences
          - See [2nd full paragraph of 274](http://www.ifaamas.org/Proceedings/aamas2013/docs/p271.pdf), this is
          where we get the blue print strategy clusters, but this also is a place I could find explanation of "action sequences"
        - Up to 14 bet sizes
        - Very fine grained pre-flop (as the real time search is almost never used pre-flop)
        - Otherwise, coarsens, for example, 3rd & 4th betting round there are at most 3 bet sizes [.5, 1, all_chips]
      and only two in retaliation to those [1, all_chips]
      - Real time search
        - Uses 6 betting sizes during real time search
        - between 100 and 2000 action sequences???
- Information Abstraction
  - Blueprint
    - a little background:
      - [lossless information](http://www.cs.cmu.edu/~sandholm/extensive.jacm07.pdf)
      - [discusses suit isomorphisms](https://www.cs.cmu.edu/~sandholm/gs3.aaai07.pdf)
    - Uses, I believe, 169-200-200-200, KE-KO as depicted [here](http://www.ifaamas.org/Proceedings/aamas2013/docs/p271.pdf)
    for features
      - This is not stated explicitly, but it makes the most sense
      -  Inherently the algo above is imperfect recall: [pg.273 here](http://www.ifaamas.org/Proceedings/aamas2013/docs/p271.pdf)
      - OHS is a method described in the same paper that clusters the 169 into 8 clusters based on EHS and then runs simulations
    - Current Implementation: [See Clustering Branch](https://github.com/fedden/pluribus-poker-AI/blob/develop/research/clustering/information_abstraction.py)
      - Based on [potential aware](http://www.cs.cmu.edu/afs/cs/Web/People/sandholm/potential-aware_imperfect-recall.aaai14.pdf) 
      algo
      - Potentially should be based on: https://www.cs.cmu.edu/~sandholm/hierarchical.aamas15.pdf
  - Real Time Search
    - Uses 500 buckets here
    -  Definitely takes into account - [potential aware](http://www.cs.cmu.edu/afs/cs/Web/People/sandholm/potential-aware_imperfect-recall.aaai14.pdf)
      - This is exactly [the algorithm I emulated](https://github.com/fedden/pluribus-poker-AI/blob/develop/research/clustering/information_abstraction.py)
    - Additional related material - [48 - potential-aware abstraction](https://www.cs.cmu.edu/~sandholm/hierarchical.aamas15.pdf), [EMD](http://www.cs.cmu.edu/~sandholm/gs3.aaai07.pdf)
    - Potentially should be based on: https://www.cs.cmu.edu/~sandholm/hierarchical.aamas15.pdf    
- Blueprint Computation Algorithm
  - Uses MCCFR with two important improvements
    - https://papers.nips.cc/paper/3306-regret-minimization-in-games-with-incomplete-information.pdf
    - https://papers.nips.cc/paper/3713-monte-carlo-sampling-for-regret-minimization-in-extensive-games.pdf
  - See pseudo-code
  - Also see the draft I created [here](https://github.com/fedden/pluribus-poker-AI/blob/develop/research/blueprint_algo/blueprint_kuhn.py)
  - [Linear MCCFR](https://arxiv.org/pdf/1809.04040.pdf)
    - First 400 minutes of training
  - Also negative regret pruning
    - here are resources this in prior areas for two-player poker:
      - http://www.cs.cmu.edu/~sandholm/BabyTartanian8.ijcai16demo.pdf   
      - https://science.sciencemag.org/content/359/6374/418
  - Blueprint calculated by averaging Snapshots
  
From #2:
  - Real Time Search
    - Pluribus only uses blueprint strategy on first found (of 4), otherwise it is using depth-limited search
      - If, for example, in the first round, there is a slightly off-tree bet (less than $100), Pluribus will 
      use [pseudo harmonic mapping](https://www.ijcai.org/Proceedings/13/Papers/028.pdf)
      - Else if it is a weird bet size, it will search instead
        - Specifically, in the first round, real time search is used if the opponents chooses a bet size over $100 off
        from what size it has in the blueprint action abstraction AND there are no more than four remaining players 
    - Depth-limited search is always used for other rounds
    - The search Pluribus uses is like ones found in perfect information games except:
        - The root note is a chance node with a probability distribution over which possible root nodes can be sampled
          - This is because we do not know which subgame we are in due to imperfect information
          - See pg. 20 [here](https://science.sciencemag.org/content/sci/suppl/2019/07/10/science.aay2400.DC1/aay2400-Brown-SM.pdf)
          for the reach probabilities
    - Expected value depends on probabilities that the searcher assigns the actions
    - Pluribus uses modified version of a previous search for two player games 
      - [original real time search](https://papers.nips.cc/paper/7993-depth-limited-solving-for-imperfect-information-games.pdf)
      - Here's a youtube video for the paper above: https://www.youtube.com/watch?v=S4-g3dPT2gY
    - Differences:
      - It was opponents could choose among different strategies, but now so can the searcher
      - Here are the strategies (regular blueprint, one biased towards folding, one towards calling, one towards raising)
        - Each of these un-normalized is multiplied by 5 and then normalized to get the biases
      - Pluribus starts searching from the beginning of the round, and the searcher's strategy is held fixed
      - Pluribus keeps track of all hole card possibilities for its opponents and updates beliefs with a simple bayes update
    - If sub game is large or search is early in the game (high in tree): use linear CFR like in blue print strategy
    - Else use optimized vector-based CFR that samples only chance events
      - [optimized vector-based linear cfr?](https://arxiv.org/pdf/1809.04040.pdf)
      - [only samples chance events?](http://martin.zinkevich.org/publications/ijcai2011_rgbr.pdf)
    - Pluribus is playing according to the final iteration, not the blueprint strategy
    - Uses lossless abstraction in current round otherwise uses 500 buckets per round for information situations
    - The action abstraction is reduced wildly, usually no more than 5 betting sizes, if a player chooses a size that 
    is different that the model, then the search starts again from the root and the action is included
    - If two searches, the strategy might change, so freeze the decision points so far in the subgame Pluribus is in
    - Anytime leaf node is reached all players remaining in hand choose from 4 strategies or a probability distribution over each
    - See pseudo code [here](https://science.sciencemag.org/content/sci/suppl/2019/07/10/science.aay2400.DC1/aay2400-Brown-SM.pdf)

        
Misc Resources    
  - https://www.youtube.com/watch?v=b7bStIQovcY
  - https://www.youtube.com/watch?v=2dX0lwaQRX0
  - https://www.youtube.com/watch?v=McV4a6umbAY
  - https://www.youtube.com/watch?v=QgCxCeoW5JI&
  - https://www.youtube.com/watch?v=Gz026reyVwc
    - Good history up to 2015 (recommended: ignore unless you understand the above)
    - Abstraction
      - integer programming: http://www.cs.cmu.edu/~sandholm/extensive.jacm07.pdf
      - potential aware
        - https://www.cs.cmu.edu/~sandholm/gs3.aaai07.pdf
        - https://www.cs.cmu.edu/~sandholm/expectation-basedVsPotential-Aware.AAAI08.pdf
      - imperfect recall
        - https://webdocs.cs.ualberta.ca/~games/poker/publications/sara09.pdf
        - https://poker.cs.ualberta.ca/publications/AAMAS13-modelling.pdf
      - most modern algo (as of 2015): https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8459/8487
        - implemented here: https://github.com/fedden/pluribus-poker-AI/blob/develop/research/clustering/information_abstraction.py
      - later algo though: https://www.cs.cmu.edu/~sandholm/hierarchical.aamas15.pdf 

