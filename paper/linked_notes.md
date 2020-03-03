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
      -See pseudo-code in #2
- Notation 
  - (see pg. 8 & 9 of #2)
- Abstraction
    - Action Abstraction
      - Blue Print
        - Newest thought is that we do keep track of all action sequences
          - 664,845,654 action sequences
          - See [2nd full paragraph of 274](http://www.ifaamas.org/Proceedings/aamas2013/docs/p271.pdf), this is
          where we get the blue print strategy clusters, but this also is the only place I could find explanation of "action sequences"
        - Up to 14 bet sizes. 
        - Very fine grained pre-flop (as the real time search is almost never used pre-flop)
        - Otherwise, coarsens, for example, 3rd & 4th betting round there are at most 3 bet sizes [.5, 1, all_chips]
      and only two in retaliation to those [1, all_chips]
      - Real time search
        - Uses 6 betting sizes during real time search
        - between 100 and 2000 action sequences???
- Information Abstraction (my most studied area)
  - Blueprint
    -[lossless information](http://www.cs.cmu.edu/~sandholm/extensive.jacm07.pdf)
    - Uses, I believe, 169-200-200-200, KE-KO as depicted [here](http://www.ifaamas.org/Proceedings/aamas2013/docs/p271.pdf)
      - This is not stated explicitly, but it makes the most sense
      -  Inherently this is imperfect recall: [pg.273 here](http://www.ifaamas.org/Proceedings/aamas2013/docs/p271.pdf)
        - This just means that the clustering doesn't start at the flop and include previous clusters
      - OHS is a method described in the same paper that clusters the 169 into 8 clusters based on EHS and then runs simulations
        - Current thought is that I can do this with non-lossless? Need to figure that out because they group the 169 and sample a fair amount from each
    - Current Implementation: [See Clustering Branch](https://github.com/fedden/pluribus-poker-AI/blob/develop/research/clustering/information_abstraction.py)
  - Real Time Search
    - Uses 500 buckets here
    -  Definitely takes into account (27 - EMD) [potential aware](http://www.cs.cmu.edu/afs/cs/Web/People/sandholm/potential-aware_imperfect-recall.aaai14.pdf)
      - This is exactly [the algorithm I wrote](https://github.com/fedden/pluribus-poker-AI/blob/develop/research/clustering/information_abstraction.py)
    - Additional related material - [48 - potential-aware abstraction](https://www.cs.cmu.edu/~sandholm/hierarchical.aamas15.pdf), [EMD](http://www.cs.cmu.edu/~sandholm/gs3.aaai07.pdf)    
- BluePrint Computation Algorithm
  - Uses MCCFR with two important improvements
    - https://papers.nips.cc/paper/3306-regret-minimization-in-games-with-incomplete-information.pdf
    - https://papers.nips.cc/paper/3713-monte-carlo-sampling-for-regret-minimization-in-extensive-games.pdf
  - See pseudo-code
  - [Linear MCCFR](https://arxiv.org/pdf/1809.04040.pdf)
    - First 400 minutes of training
  -Also negative regret pruning
    - here are resources this in prior areas for two-player poker:
      - http://www.cs.cmu.edu/~sandholm/BabyTartanian8.ijcai16demo.pdf   
      - https://science.sciencemag.org/content/359/6374/418
  - Blueprint gotten by averaging Snapshots
    

