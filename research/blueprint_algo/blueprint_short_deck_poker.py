"""
I think this is right??? This is tough!

call python blueprint_kuhn.py in the blueprint_algo directory

The following code is an attempt at mocking up the pseudo code for the blue print algorithm found in Pluribus.
-- That pseudo code can be found here, pg. 16:
---- https://science.sciencemag.org/content/sci/suppl/2019/07/10/science.aay2400.DC1/aay2400-Brown-SM.pdf

Additionally, this code has been applied to Kuhn poker and left in a functional style that mimics code that has been
pushed to the feature/add-kuhn-poker branch. That code was inspired by the two links below, I believe, pg 12:
-- http://modelai.gettysburg.edu/2013/cfr/cfr.pdf
-- https://github.com/geohot/ai-notebooks/blob/8bdd5ca2d9560f978ea79bb9d0cb8d5acf3dee4b/cfr_kuhn_poker.ipynb

The following code-style choice was made in order to make converting the findings to an OOP style easier and more consistent by
working from a familiar style (meaning, similar variable names, etc.. to the notebook link above),
while not introducing (more) bugs in a pre-mature conversion to work in our Pluribus framework.

The differences between the following code and the notebook above includes:
-- Addition of linear CFR
-- Addition of negative regret pruning
-- A method of estimating reach that is based on counting occurrences rather than carrying around PI for each player
    (PI is the probability of reaching an h)
-- Only updating the strategy every x iterations
-- Based on these changes the algo converges to stratigies for actions, when at nash equilibrium, is at 100% much quicker
    than the previous algorithm, seemingly

The following would still need to be done:
-- Stop using global variables where possible - a lot of potential for subtle bugs
-- Convert to OOP and use the Pluribus game engine
-- Use minutes instead of iterations for the prune_threshold and discount interval
-- Introduce the chance sampling within the algorithm (Kuhn poker can allow for chance to occur before CFR is called,
    where as, No Limit Hold'Em would have chance events that could be reached)
-- Ensure that the algorithm is correct and can be used for No Limit Hold'Em poker
-- Only apply this to the first betting round for Pluribus' blue print strategy
-- Only apply pruning to non-terminal nodes and non-last round of betting as described in the supplementary materials:
---- https://science.sciencemag.org/content/sci/suppl/2019/07/10/science.aay2400.DC1/aay2400-Brown-SM.pdf (pg.14 - 15)
-- Only store the actions as you go (not all will be reached in the blue print strategy creation)

Next Steps:
-- Apply this to a short deck no limit hhold em game in conjuction with the clustering algorithm I mocked up found here:
---- https://github.com/fedden/pluribus-poker-AI/blob/develop/research/clustering/information_abstraction.py
-- Mock up the search algorithm found in the supllemental materials

A note on the blue print strategy: I think there can be some confusion (at least for me) on the blue print strategy.
My current understanding is that the each player (p_i) in the set of players (P) occupies a FIXED position in game play. I_i represents a
set of infosets partitioned from I (set of infosets) that is partitioned to which i (player) it belongs. That player is
defined by what order they go in. You can think of that player as being the expert of their position. They all contribute to the
same blue print strategy (sigma) and normalized blue print startegy (phi). The term sigma_i refers to the strategy that is
contributed to by player i based on the I_i, or infosets that they are responsible for in their position. In any case,
only one strategy profile is updated.

That's my understanding anyway. Here is a good link to that effect (this is referenced from the Pluribus paper with regard to
MCCFR), pg 2, bullet 5 (THIS IS MUCH CLEARER IN THE ACTUAL PAPER IN LINK BELOW):
" For each player i ∈ N a partition Ii of {h ∈ H : P(h) = i} with the property that
A(h) = A(h') whenever h and h' are in the same member of the partition. For Ii ∈ Ii**
we denote by A(Ii) the set A(h) and by P(Ii) the player P(h) for any h ∈ Ii
. Ii**(is the information partition of player i; a set Ii ∈ Ii** is an information set of player i."
**(this one is the set of all infosets for partition i)
-- https://papers.nips.cc/paper/3306-regret-minimization-in-games-with-incomplete-information.pdf

"""
from __future__ import annotations

import copy
import random
from typing import Dict, List, Tuple

import numpy as np
from tqdm import trange

from pluribus import utils
from pluribus.game.actions import Action, Call, Fold, Raise
from pluribus.game.engine import PokerEngine
from pluribus.game.pot import Pot
from pluribus.game.player import Player
from pluribus.game.table import PokerTable


utils.random.seed(42)


ISETS = [
    "1",
    "2",
    "3",  # round 1
    "P1",
    "P2",
    "P3",
    "B1",
    "B2",
    "B3",  # round 2
    "PB1",
    "PB2",
    "PB3",  # round 3
]
HANDS = [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]

# terminal history states
TERMINAL = ["PP", "PBP", "PBB", "BP", "BB"]
ACTIONS = ["P", "B"]


class ShortDeckPokerPlayer(Player):
    """Inherits from Player which will interface easily with the PokerEngine.

    This class should manage the state of the players personal pot, and the
    private cards that are dealt to the player. Also manages whether this
    player has folded or not.
    """

    def __init__(self, player_i: int, initial_chips: int, pot: Pot):
        """Instanciate a player."""
        super().__init__(
            name=f"player_{player_i}",
            initial_chips=initial_chips,
            pot=pot,
        )


class ShortDeckPokerState:
    """The state of a Short Deck Poker game at some given point in time.

    The class is immutable and new state can be instanciated from once an
    action is applied via the `ShortDeckPokerState.new_state` method.
    """
    def __init__(self, players: List[ShortDeckPokerPlayer]):
        """Initialise state."""
        # Get a reference of the pot from the first player.
        self._table = PokerTable(players=players, pot=players[0].pot)
        # TODO(fedden): There are an awful lot of layers of abstraction here,
        #               this could be much neater, maybe refactor and clean
        #               things up a bit here in the future.
        # Shorten the deck.
        self._table.dealer.deck._cards = [
            card for card in self._table.dealer.deck._cards
            if card.rank_int not in {2, 3, 4, 5}
        ]
        small_blind = 50
        big_blind = 100
        self._poker_engine = PokerEngine(
            table=self._table, small_blind=small_blind, big_blind=big_blind
        )
        # Reset the pot, assign betting order to players (might need to remove
        # this), assign blinds to the players.
        self._poker_engine.round_setup()
        # Deal private cards to players.
        self._table.dealer.deal_private_cards(self._table.players)
        # Store the actions as they come in here.
        self._history: List[Action] = []

    def new_state(self, action: Action) -> ShortDeckPokerState:
        """Create a new state after applying an action."""
        if isinstance(action, Call):
            # TODO(fedden): Player called, get money from player if needed.
            pass
        elif isinstance(action, Fold):
            # TODO(fedden): Player folded, update player status.
            pass
        elif isinstance(action, Raise):
            # TODO(fedden): Player raised, get money into pot, and subtract
            #               from player.
            pass
        else:
            raise ValueError(
                f"Expected action to be derived from class Action, but found "
                f"type {type(action)}."
            )
        new_state = ShortDeckPokerState(players=self._table.players)
        # TODO(fedden): Deep copy the parts of state that are needed, and
        #               retain the references needed here too.
        return new_state

    @property
    def h(self):
        """Returns the history."""
        return self._history

    @property
    def rs(self):
        """Returns the players hands."""
        return [player.cards for player in self._table.players]


def payout(rs: Tuple[int, int], h: str) -> int:
    """

    :param rs: realstate, a tuple of two ints, first is card for player one, second player 2
    :param h: the action sequences without the card information
    :return: expected value (at least at that moment in the game)
    """
    if h == "PBP":
        return -100
    elif h == "BP":
        return 100
    m = 100 if (rs[0] > rs[1]) else -100
    if h == "PP":
        return m
    if h in ["BB", "PBB"]:
        return m * 2
    assert False


def get_information_set(rs: Tuple[int, int], h: str) -> str:
    """
    :param rs: realstate, a tuple of two ints, first is card for player one, second player 2
    :param h: the action sequences without the card information
    :return: I: information set, which contains all h that the p_i cannot distinguish
    """
    assert h not in TERMINAL
    if h == "":
        return str(rs[0])
    elif len(h) == 1:
        return h + str(rs[1])
    else:
        return "PB" + str(rs[0])
    assert False


def update_strategy(rs: Tuple[int, int], h: str, i: int):
    """

    :param rs: realstate, a tuple of two ints, first is card for player one, second player 2
    :param h: the action sequences without the card information
    :param i: the player, i = 1 is always first to act and i = 2 is always second to act, but they take turns who
        updates the strategy (only one strategy)
    :return: nothing, updates action count in the strategy of actions chosen according to sigma, this simple choosing of
        actions is what allows the algorithm to build up preference for one action over another in a given spot
    """
    ph = 2 if len(h) == 1 else 1  # this is always the case no matter what i is

    if (
        h in TERMINAL
    ):  # or if p_i is not in the hand or if betting round is > 0, strategy is only
        return  # updated in betting round 1 for Pluribus, but I am doing all rounds in this example
    # elif h is chance_node:  -- we don't care about chance nodes here, but we will for No Limit
    #   sample action from strategy for h
    #   update_strategy(rs, h + a, i)
    elif ph == i:
        I = get_information_set(rs, h)
        # calculate regret
        calculate_strategy(regret, sigma, I)
        # choose an action based of sigma
        a = np.random.choice(list(sigma[t][I].keys()), 1, p=list(sigma[t][I].values()))[
            0
        ]
        strategy[I][a] += 1
        # so strategy is counts based on sigma, this takes into account the reach probability
        # so there is no need to pass around that pi guy..
        update_strategy(rs, h + a, i)
    else:
        for a in ACTIONS:
            # not actually updating the strategy for p_i != i, only one i at a time
            update_strategy(rs, h + a, i)


def calculate_strategy(
    regret: Dict[str, Dict[str, float]],
    sigma: Dict[int, Dict[str, Dict[str, float]]],
    I: str,
):
    """

    :param regret: dictionary of regrets, I is key, then each action at I, with values being regret
    :param sigma: dictionary of strategy updated by regret, iteration is key, then I is key, then each action with prob
    :param I:
    :return: doesn't return anything, just updates sigma
    """
    rsum = sum([max(x, 0) for x in regret[I].values()])
    for a in ACTIONS:
        if rsum > 0:
            sigma[t + 1][I][a] = max(regret[I][a], 0) / rsum
        else:
            sigma[t + 1][I][a] = 1 / len(ACTIONS)


def cfr(rs: Tuple[int, int], h: str, i: int, t: int) -> float:
    """
    regular cfr algo

    :param rs: realstate, a tuple of two ints, first is card for player one, second player 2
    :param h: the action sequences without the card information
    :param i: player
    :param t: iteration
    :return: expected value for node for player i
    """
    ph = 2 if len(h) == 1 else 1  # this is always the case no matter what i is

    if h in TERMINAL:
        return payout(rs, h) * (1 if i == 1 else -1)
    # elif p_i not in hand:
    #   cfr()
    # TODO: this will be needed for No Limit Hold'Em, but in two player the player is always in the hand
    # elif h is chance_node:  -- we don't care about chance nodes here, but we will for No Limit
    #   sample action from strategy for h
    #   cfr()
    elif ph == i:
        I = get_information_set(rs, h)
        # calculate strategy
        calculate_strategy(regret, sigma, I)
        vo = 0.0
        voa = {}
        for a in ACTIONS:
            voa[a] = cfr(rs, h + a, i, t)
            vo += sigma[t][I][a] * voa[a]
        for a in ACTIONS:
            regret[I][a] += voa[a] - vo
            # do not need update the strategy based on regret, strategy does that with sigma
        return vo
    else:
        Iph = get_information_set(rs, h)
        calculate_strategy(regret, sigma, Iph)
        a = np.random.choice(
            list(sigma[t][Iph].keys()), 1, p=list(sigma[t][Iph].values())
        )[0]
        return cfr(rs, h + a, i, t)


def cfrp(rs: Tuple[int, int], h: str, i: int, t: int):
    """
    pruning cfr algo, might need to adjust only pruning if not final betting round and if not terminal node

    :param rs: realstate, a tuple of two ints, first is card for player one, second player 2
    :param h: the action sequences without the card information
    :param i: player
    :param t: iteration
    :return: expected value for node for player i
    """
    ph = 2 if len(h) == 1 else 1

    if h in TERMINAL:
        return payout(rs, h) * (1 if i == 1 else -1)
    # elif p_i not in hand:
    #   cfrp()
    # TODO: this will be needed for No Limit Hold'Em, but in two player the player is always in the hand
    # elif h is chance_node:  -- we don't care about chance nodes here, but we will for No Limit
    #   sample action from strategy for h
    #   cfrp()
    elif ph == i:
        I = get_information_set(rs, h)
        # calculate strategy
        calculate_strategy(regret, sigma, I)
        vo = 0.0
        voa = {}
        explored = {}  # keeps tracked of items that can be skipped
        for a in ACTIONS:
            if regret[I][a] > C:
                voa[a] = cfrp(rs, h + a, i, t)
                explored[a] = True
                vo += sigma[t][I][a] * voa[a]
            else:
                explored[a] = False
        for a in ACTIONS:
            if explored[a]:
                regret[I][a] += voa[a] - vo
                # do not need update the strategy based on regret, strategy does that with sigma
        return vo
    else:
        Iph = get_information_set(rs, h)
        calculate_strategy(regret, sigma, Iph)
        a = np.random.choice(
            list(sigma[t][Iph].keys()), 1, p=list(sigma[t][Iph].values())
        )[0]
        return cfrp(rs, h + a, i, t)


if __name__ == "__main__":
    # init tables
    regret = {}
    strategy = {}
    for I in ISETS:
        regret[I] = {k: 0 for k in ACTIONS}
        strategy[I] = {k: 0 for k in ACTIONS}

    sigma = {1: {}}
    for I in ISETS:
        sigma[1][I] = {k: 1 / len(ACTIONS) for k in ACTIONS}

    # algorithm constants
    strategy_interval = 100
    LCFR_threshold = 4000
    discount_interval = 100
    prune_threshold = 2000
    C = -20000  # somewhat arbitrary
    n_players = 3
    initial_chips = 10000
    pot = Pot()
    players = [
        Player(player_i=i, initial_chips=initial_chips, pot=pot)
        for i in range(n_players)
    ]
    # algorithm presented here, pg.16:
    # https://science.sciencemag.org/content/sci/suppl/2019/07/10/science.aay2400.DC1/aay2400-Brown-SM.pdf
    for t in trange(1, 20000):
        sigma[t + 1] = copy.deepcopy(sigma[t])
        for i in [1, 2]:  # fixed position i
            # Create a new state.
            state = ShortDeckPokerState(players=players)
            h = ""
            rs = random.choice(HANDS)
            if t % strategy_interval == 0:
                update_strategy(rs, h, i)
            if t > prune_threshold:
                if random.uniform(0, 1) < 0.05:
                    cfr(rs, h, i, t)
                else:
                    cfrp(rs, h, i, t)
            else:
                cfr(rs, h, i, t)
        if t < LCFR_threshold & t % discount_interval == 0:
            d = (t / discount_interval) / ((t / discount_interval) + 1)
            for I in ISETS:
                for a in ACTIONS:
                    regret[I][a] *= d
                    strategy[I][a] *= d
        del sigma[t]

    for k, v in strategy.items():
        norm = sum(list(v.values()))
        print("%3s: P:%.4f B:%.4f" % (k, v["P"] / norm, v["B"] / norm))
        # https://en.wikipedia.org/wiki/Kuhn_poker#Optimal_strategy
        # generally close, converges to 1s much faster
        # reducing C should allow for better convergence, but slower..

# code for normalizing strategy
# normalized_strategy = {}
# for I in strategy.keys():
#     d = strategy[I]
#     factor = 1.0/sum(list(d.values()))
#     normalized_d = {k: v*factor for k, v in d.items()}
#     normalized_strategy[I] = normalized_d
#
# print(normalized_strategy)