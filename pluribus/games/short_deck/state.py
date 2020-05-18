from __future__ import annotations

import collections
import copy
import json
import logging
import operator
import os
from typing import Any, Dict, List, Optional, Tuple
from itertools import combinations
import random
import time

import dill as pickle
import numpy as np

from pluribus import utils
from pluribus.poker.card import Card
from pluribus.poker.engine import PokerEngine
from pluribus.games.short_deck.player import ShortDeckPokerPlayer
from pluribus.poker.pot import Pot
from pluribus.poker.table import PokerTable

logger = logging.getLogger("pluribus.games.short_deck.state")
InfoSetLookupTable = Dict[str, Dict[Tuple[int, ...], str]]


def new_game(
    n_players: int, info_set_lut: InfoSetLookupTable = {}, **kwargs
) -> ShortDeckPokerState:
    """Create a new game of short deck poker."""
    pot = Pot()
    players = [
        ShortDeckPokerPlayer(player_i=player_i, initial_chips=10000, pot=pot)
        for player_i in range(n_players)
    ]
    if info_set_lut:
        # Don't reload massive files, it takes ages.
        state = ShortDeckPokerState(players=players,
                                    load_pickle_files=False, **kwargs)
        state.info_set_lut = info_set_lut
    else:
        # Load massive files.
        state = ShortDeckPokerState(players=players, **kwargs)
    return state


class ShortDeckPokerState:
    """The state of a Short Deck Poker game at some given point in time.

    The class is immutable and new state can be instanciated from once an
    action is applied via the `ShortDeckPokerState.new_state` method.
    """

    def __init__(
        self,
        players: List[ShortDeckPokerPlayer],
        small_blind: int = 50,
        big_blind: int = 100,
        pickle_dir: str = ".",
        load_pickle_files: bool = True,
        real_time_test: bool = False,
        public_cards: List[Card] = []
    ):
        """Initialise state."""
        n_players = len(players)
        if n_players <= 1:
            raise ValueError(
                f"At least 2 players must be provided but only {n_players} "
                f"were provided."
            )
        if load_pickle_files:
            self.info_set_lut = self.load_pickle_files(pickle_dir)
        else:
            self.info_set_lut = {}
        # Get a reference of the pot from the first player.
        self._table = PokerTable(
            players=players, pot=players[0].pot, include_ranks=[10, 11, 12, 13, 14]
        )
        # Get a reference of the initial number of chips for the payout.
        self._initial_n_chips = players[0].n_chips
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.real_time_test = real_time_test
        self._poker_engine = PokerEngine(
            table=self._table, small_blind=small_blind, big_blind=big_blind
        )
        # Reset the pot, assign betting order to players (might need to remove
        # this), assign blinds to the players.
        self._poker_engine.round_setup()
        # Deal private cards to players.
        if not self.real_time_test:
            self._poker_engine.table.dealer.deal_private_cards(self._table.players)
        # Store the actions as they come in here.
        self._history: Dict[str, List[str]] = collections.defaultdict(list)
        self._public_information: Dict[str, List[Card]] = collections.defaultdict(list)
        self._betting_stage = "pre_flop"
        self._previous_betting_stage = None
        self._betting_stage_to_round: Dict[str, int] = {
            "pre_flop": 0,
            "flop": 1,
            "turn": 2,
            "river": 3,
            "show_down": 4,
        }
        # Rotate the big and small blind to the final positions for the pre
        # flop round only.
        player_i_order: List[int] = [p_i for p_i in range(n_players)]
        self.players[0].is_small_blind = True
        self.players[1].is_big_blind = True
        self.players[-1].is_dealer = True
        self._player_i_lut: Dict[str, List[int]] = {
            "pre_flop": player_i_order[2:] + player_i_order[:2],
            "flop": player_i_order,
            "turn": player_i_order,
            "river": player_i_order,
            "show_down": player_i_order,
            "terminal": player_i_order,
        }
        self._skip_counter = 0
        # self._first_move_of_current_round = True
        self._reset_betting_round_state()
        for player in self.players:
            player.is_turn = False
        self.current_player.is_turn = True
        # TODO add attribute of public_cards, that can be supplied by
        #   convenience method
        self._public_cards = public_cards
        if public_cards:
            assert len(public_cards) in {3, 4, 5}
        self._public_cards = public_cards
        # only want to do these actions in real game play, as they are slow
        if self.real_time_test:
            # must have offline strategy loaded up
            self._starting_hand_probs = self._initialize_starting_hands()
            # TODO: We might not need this
            cards_in_deck = self._table.dealer.deck._cards_in_deck
            self._evals = [c.eval_card for c in cards_in_deck]
            self._evals_to_cards = {i.eval_card: i for i in cards_in_deck}

    def __repr__(self):
        """Return a helpful description of object in strings and debugger."""
        return f"<ShortDeckPokerState player_i={self.player_i} betting_stage={self._betting_stage}>"

    def apply_action(self, action_str: Optional[str]) -> ShortDeckPokerState:
        """Create a new state after applying an action.

        Parameters
        ----------
        action_str : str or None
            The description of the action the current player is making. Can be
            any of {"fold, "call", "raise"}, the latter two only being possible
            if the agent hasn't folded already.

        Returns
        -------
        new_state : ShortDeckPokerState
            A poker state instance that represents the game in the next
            timestep, after the action has been applied.
        """
        if action_str not in self.legal_actions:
            raise ValueError(
                f"Action '{action_str}' not in legal actions: " f"{self.legal_actions}"
            )
        # Deep copy the parts of state that are needed that must be immutable
        # from state to state.
        lut = self.info_set_lut
        self.info_set_lut = {}
        new_state = copy.deepcopy(self)
        new_state.info_set_lut = self.info_set_lut = lut
        # An action has been made, so alas we are not in the first move of the
        # current betting round.
        # new_state._first_move_of_current_round = False
        if action_str is None:
            # Assert active player has folded already.
            assert (
                not new_state.current_player.is_active
            ), "Active player cannot do nothing!"
        elif action_str == "call":
            action = new_state.current_player.call(players=new_state.players)
            logger.debug("calling")
        elif action_str == "fold":
            action = new_state.current_player.fold()
        elif action_str == "raise":
            bet_n_chips = new_state.big_blind
            if new_state._betting_stage in {"turn", "river"}:
                bet_n_chips *= 2
            biggest_bet = max(p.n_bet_chips for p in new_state.players)
            n_chips_to_call = biggest_bet - new_state.current_player.n_bet_chips
            raise_n_chips = bet_n_chips + n_chips_to_call
            logger.debug(f"betting {raise_n_chips} n chips")
            action = new_state.current_player.raise_to(n_chips=raise_n_chips)
            new_state._n_raises += 1
        else:
            raise ValueError(
                f"Expected action to be derived from class Action, but found "
                f"type {type(action)}."
            )
        # Update the new state.
        skip_actions = ["skip" for _ in range(new_state._skip_counter)]
        new_state._history[new_state.betting_stage] += skip_actions
        new_state._history[new_state.betting_stage].append(str(action))
        # save public information
        # if new_state._first_move_of_current_round:
        #     new_state._public_information[
        #         new_state.betting_stage
        #     ] = new_state._table.community_cards
        new_state._n_actions += 1
        new_state._skip_counter = 0
        # Player has made move, increment the player that is next.
        while True:
            new_state._move_to_next_player()
            # If we have finished betting, (i.e: All players have put the
            # same amount of chips in), then increment the stage of
            # betting.
            finished_betting = not new_state._poker_engine.more_betting_needed
            if finished_betting and new_state.all_players_have_actioned:
                # We have done atleast one full round of betting, increment
                # stage of the game.
                new_state._increment_stage()
                new_state._reset_betting_round_state()
                # new_state._first_move_of_current_round = True
            if not new_state.current_player.is_active:
                new_state._skip_counter += 1
                assert not new_state.current_player.is_active
            elif new_state.current_player.is_active:
                if new_state._poker_engine.n_players_with_moves == 1:
                    # No players left.
                    new_state._betting_stage = "terminal"
                    if not new_state._table.community_cards:
                        new_state._poker_engine.table.dealer.deal_flop(new_state._table)
                # Now check if the game is terminal.
                if new_state._betting_stage in {"terminal", "show_down"}:
                    # Distribute winnings.
                    new_state._poker_engine.compute_winners()
                break
        for player in new_state.players:
            player.is_turn = False
        new_state.current_player.is_turn = True
        return new_state

    @staticmethod
    def load_pickle_files(pickle_dir: str) -> Dict[str, Dict[Tuple[int, ...], str]]:
        """Load pickle files into memory."""
        file_names = [
            "preflop_lossless.pkl",
            "flop_lossy_2.pkl",
            "turn_lossy_2.pkl",
            "river_lossy_2.pkl",
        ]
        betting_stages = ["pre_flop", "flop", "turn", "river"]
        info_set_lut: Dict[str, Dict[Tuple[int, ...], str]] = {}
        for file_name, betting_stage in zip(file_names, betting_stages):
            file_path = os.path.join(pickle_dir, file_name)
            if not os.path.isfile(file_path):
                raise ValueError(
                    f"File path not found {file_path}. Ensure pickle_dir is "
                    f"set to directory containing pickle files"
                )
            with open(file_path, "rb") as fp:
                info_set_lut[betting_stage] = pickle.load(fp)
        return info_set_lut

    def _move_to_next_player(self):
        """Ensure state points to next valid active player."""
        self._player_i_index += 1
        if self._player_i_index >= len(self.players):
            self._player_i_index = 0

    def _reset_betting_round_state(self):
        """Reset the state related to counting types of actions."""
        self._all_players_have_made_action = False
        self._n_actions = 0
        self._n_raises = 0
        self._player_i_index = 0
        self._n_players_started_round = self._poker_engine.n_active_players
        while not self.current_player.is_active:
            self._skip_counter += 1
            self._player_i_index += 1

    def _increment_stage(self):
        """Once betting has finished, increment the stage of the poker game."""
        # Progress the stage of the game.
        if self._betting_stage == "pre_flop":
            # Progress from private cards to the flop.
            self._betting_stage = "flop"
            self._previous_betting_stage = "pre_flop"
            if len(self._public_cards) >= 3:
                community_cards = self._public_cards[:3]
                self._poker_engine.table.community_cards += community_cards
            else:
                self._poker_engine.table.dealer.deal_flop(self._table)
            self._public_information[
                self.betting_stage
            ] = self._table.community_cards.copy()
        elif self._betting_stage == "flop":
            # Progress from flop to turn.
            self._betting_stage = "turn"
            self._previous_betting_stage = "flop"
            if len(self._public_cards) >= 4:
                community_cards = self._public_cards[3:4]
                self._poker_engine.table.community_cards += community_cards
            else:
                self._poker_engine.table.dealer.deal_turn(self._table)
            self._public_information[
                self.betting_stage
            ] = self._table.community_cards.copy()
        elif self._betting_stage == "turn":
            # Progress from turn to river.
            self._betting_stage = "river"
            self._previous_betting_stage = "turn"
            if len(self._public_cards) == 5:
                community_cards = self._public_cards[4:]
                self._poker_engine.table.community_cards += community_cards
            else:
                self._poker_engine.table.dealer.deal_river(self._table)
            self._public_information[
                self.betting_stage
            ] = self._table.community_cards.copy()
        elif self._betting_stage == "river":
            # Progress to the showdown.
            self._betting_stage = "show_down"
            self._previous_betting_stage = "river"
        elif self._betting_stage in {"show_down", "terminal"}:
            pass
        else:
            raise ValueError(f"Unknown betting_stage: {self._betting_stage}")

    def _normalize_bayes(self):
        """Normalize probability of reach for each player"""
        n_players = len(self.players)
        for player in range(n_players):
            total_prob = sum(self._starting_hand_probs[player].values())
            for starting_hand, prob in self._starting_hand_probs[player].items():
                self._starting_hand_probs[player][starting_hand] = prob / total_prob

    def _update_hole_cards_bayes(self, offline_strategy: Dict):
        """Get probability of reach for each pair of hole cards for each player"""
        n_players = len(self._table.players)
        player_indices: List[int] = [p_i for p_i in range(n_players)]
        for p_i in player_indices:
            for starting_hand in self._starting_hand_probs[p_i].keys():
                # TODO: is this bad?
                if "p_reach" in locals():
                    del p_reach
                action_sequence: Dict[str, List[str]] = collections.defaultdict(list)
                previous_betting_stage = "pre_flop"
                first_action_round = False
                for idx, betting_stage in enumerate(self._history.keys()):
                    # import ipdb;
                    # ipdb.set_trace()
                    n_actions_round = len(self._history[betting_stage])
                    for i in range(n_actions_round):
                        # if i == 0:
                        #     betting_stage = previous_betting_stage
                        # elif i == n_actions_round - 1:
                        #     previous_betting_stage = betting_stage
                        # else:
                        #     betting_stage = betting_round
                        action = self._history[betting_stage][i]
                        while action == 'skip':
                            i += 1  # action sequences don't end in skip
                            action = self._history[betting_stage][i]
                        # TODO: maybe a method already exists for this?
                        if betting_stage == "pre_flop":
                            ph = (i + 2) % n_players
                        else:
                            ph = i % n_players
                        if p_i != ph:
                            prob_reach_all_hands = []
                            num_hands = 0
                            for opp_starting_hand in self._starting_hand_probs[
                                p_i
                            ].keys():
                                # TODO: clean this up
                                public_evals = [
                                    c.eval_card
                                    for c in self._public_information[betting_stage]
                                ]
                                if len(set(opp_starting_hand).union(set(public_evals)).union(set(starting_hand))) < \
                                        len(opp_starting_hand) + len(starting_hand) + len(public_evals):
                                    prob = 0
                                    num_hands += 1
                                else:
                                    num_hands += 1

                                    public_cards = self._public_information[
                                        betting_stage
                                    ]
                                    public_cards_evals = [c.eval_card for c in public_cards]
                                    infoset = self._info_set_helper(
                                        opp_starting_hand,
                                        public_cards_evals,
                                        action_sequence,
                                        betting_stage,
                                    )
                                    # check to see if the strategy exists, if not equal probability
                                    # TODO: is this hacky? problem with defaulting to 1 / 3, is that it
                                    #  doesn't work for calculations that need to be made with the object's values

                                    try:  # TODO: with or without keys
                                        prob = offline_strategy[infoset][action]
                                    except KeyError:
                                        prob = 1 / len(self.legal_actions)
                                prob_reach_all_hands.append(prob)
                            # import ipdb;
                            # ipdb.set_trace()
                            prob2 = sum(prob_reach_all_hands) / num_hands
                            if "p_reach" not in locals():
                                p_reach = prob2
                            else:
                                p_reach *= prob2
                        elif p_i == ph:
                            public_evals = [
                                c.eval_card
                                for c in self._public_information[betting_stage]
                            ]
                            if len(set(starting_hand).union(set(public_evals))) < (
                                len(public_evals) + 2
                            ):
                                prob = 0
                            else:
                                public_cards = self._public_information[betting_stage]
                                public_cards_evals = [c.eval_card for c in public_cards]
                                infoset = self._info_set_helper(
                                    starting_hand,
                                    public_cards_evals,
                                    action_sequence,
                                    betting_stage,
                                )
                                #  TODO: Check this
                                try:
                                    prob = offline_strategy[infoset][action]
                                except KeyError:
                                    prob = 1 / len(self.legal_actions)
                            if "p_reach" not in locals():
                                p_reach = prob
                            else:
                                p_reach *= prob
                        action_sequence[betting_stage].append(action)
                self._starting_hand_probs[p_i][tuple(starting_hand)] = p_reach
        self._normalize_bayes()
        # TODO: delete this? at least for our purposes we don't need it again

    def deal_bayes(self):
        start = time.time()
        lut = self.info_set_lut
        self.info_set_lut = {}
        new_state = copy.deepcopy(self)
        new_state.info_set_lut = self.info_set_lut = lut
        end = time.time()
        print(f"Took {start - end} to load")

        players = list(range(len(self.players)))
        random.shuffle(players)

        # TODO should contain the current public cards/heros real hand, if exists
        card_evals_selected = []

        for player in players:
            # does this maintain order?
            starting_hand_eval = new_state._get_starting_hand(player)
            len_union = len(set(starting_hand_eval).union(set(card_evals_selected)))
            len_individual = len(starting_hand_eval) + len(card_evals_selected)
            while len_union < len_individual:
                starting_hand_eval = new_state._get_starting_hand(player)
                len_union = len(set(starting_hand_eval).union(set(card_evals_selected)))
                len_individual = len(starting_hand_eval) + len(card_evals_selected)
            for card_eval in starting_hand_eval:
                card = new_state._evals_to_cards[card_eval]
                new_state.players[player].add_private_card(card)
            card_evals_selected += starting_hand_eval
        cards_selected = [new_state._evals_to_cards[c] for c in card_evals_selected]
        cards_selected += new_state._public_cards
        for card in cards_selected:
            new_state._table.dealer.deck.remove(card)
        return new_state
    # TODO add convenience method to supply public cards

    def load_game_state(self, offline_strategy: Dict, action_sequence: list):
        """
        Follow through the action sequence provided to get current node.
        :param action_sequence: List of actions without 'skip'
        """
        if not action_sequence:
            # TODO: not 100 percent sure I need to deep copy
            lut = self.info_set_lut
            self.info_set_lut = {}
            new_state = copy.deepcopy(self)
            new_state.info_set_lut = self.info_set_lut = lut
            new_state._update_hole_cards_bayes(offline_strategy)
            return new_state
        a = action_sequence.pop(0)
        if a == "skip":
            a = action_sequence.pop(0)
        new_state = self.apply_action(a)
        return new_state.load_game_state(offline_strategy, action_sequence)

    def _get_starting_hand(self, player_idx: int):
        """Get starting hand based on probability of reach"""
        starting_hand_idxs = list(range(len(self._starting_hand_probs[player_idx].keys())))
        starting_hands_probs = list(self._starting_hand_probs[player_idx].values())
        starting_hand_idx = np.random.choice(starting_hand_idxs, 1, p=starting_hands_probs)[0]
        starting_hand = list(self._starting_hand_probs[player_idx].keys())[starting_hand_idx]
        return starting_hand

    def _initialize_starting_hands(self):
        """Dictionary of starting hands to store probabilities in"""
        assert self.betting_stage == "pre_flop"
        # TODO: make this abstracted for n_players
        starting_hand_probs = {0: {}, 1: {}, 2: {}}
        n_players = len(self.players)
        starting_hands = self._get_card_combos(2)
        for p_i in range(n_players):
            for starting_hand in starting_hands:
                starting_hand_probs[p_i][
                    tuple([c.eval_card for c in starting_hand])
                ] = 1
        return starting_hand_probs

    def _info_set_helper(
        self, hole_cards, public_cards, action_sequence, betting_stage
    ):
        # didn't want to combine this with the other, as we may want to modularize soon
        """Get the information set for the current player."""
        cards = sorted(hole_cards, reverse=True,)
        cards += sorted(public_cards, reverse=True,)
        eval_cards = tuple(cards)
        try:
            cards_cluster = self.info_set_lut[betting_stage][eval_cards]
        except KeyError:
            if not self.info_set_lut:
                raise ValueError("Pickle luts must be loaded for info set.")
            elif eval_cards not in self.info_set_lut[self._betting_stage]:
                raise ValueError("Cards {cards} not in pickle files.")
            else:
                raise ValueError("Unrecognised betting stage in pickle files.")
        info_set_dict = {
            "cards_cluster": cards_cluster,
            "history": [
                {betting_stage: [str(action) for action in actions]}
                for betting_stage, actions in action_sequence.items()
            ],
        }
        return json.dumps(
            info_set_dict, separators=(",", ":"), cls=utils.io.NumpyJSONEncoder
        )

    def _get_card_combos(self, num_cards):
        """Get combinations of cards"""
        return list(combinations(self._poker_engine.table.dealer.deck._cards_in_deck, num_cards))

    @property
    def community_cards(self) -> List[Card]:
        """Return all shared/public cards."""
        return self._table.community_cards

    @property
    def private_hands(self) -> Dict[ShortDeckPokerPlayer, List[Card]]:
        """Return all private hands."""
        return {p: p.cards for p in self.players}

    @property
    def initial_regret(self) -> Dict[str, float]:
        """Returns the default regret for this state."""
        return {action: 0 for action in self.legal_actions}

    @property
    def initial_strategy(self) -> Dict[str, float]:
        """Returns the default strategy for this state."""
        return {action: 0 for action in self.legal_actions}

    @property
    def betting_stage(self) -> str:
        """Return betting stage."""
        return self._betting_stage

    @property
    def previous_betting_stage(self) -> str:
        """Return previous betting stage."""
        return self._previous_betting_stage

    @property
    def all_players_have_actioned(self) -> bool:
        """Return whether all players have made atleast one action."""
        return self._n_actions >= self._n_players_started_round

    @property
    def n_players_started_round(self) -> bool:
        """Return n_players that started the round."""
        return self._n_players_started_round

    # @property
    # def first_move_of_current_round(self) -> bool:
    #     """Return boolfor first move of current round."""
    #     return self._first_move_of_current_round

    @property
    def player_i(self) -> int:
        """Get the index of the players turn it is."""
        return self._player_i_lut[self._betting_stage][self._player_i_index]

    @player_i.setter
    def player_i(self, _: Any):
        """Raise an error if player_i is set."""
        raise ValueError(f"The player_i property should not be set.")

    @property
    def betting_round(self) -> int:
        """Algorithm 1 of pluribus supp. material references betting_round."""
        try:
            betting_round = self._betting_stage_to_round[self._betting_stage]
        except KeyError:
            raise ValueError(
                f"Attemped to get betting round for stage "
                f"{self._betting_stage} but was not supported in the lut with "
                f"keys: {list(self._betting_stage_to_round.keys())}"
            )
        return betting_round

    @property
    def info_set(self) -> str:
        """Get the information set for the current player."""
        cards = sorted(
            self.current_player.cards,
            key=operator.attrgetter("eval_card"),
            reverse=True,
        )
        cards += sorted(
            self._table.community_cards,
            key=operator.attrgetter("eval_card"),
            reverse=True,
        )
        eval_cards = tuple([card.eval_card for card in cards])
        try:
            cards_cluster = self.info_set_lut[self._betting_stage][eval_cards]
        except KeyError:
            return "default info set, please ensure you load it correctly"
        # Convert history from a dict of lists to a list of dicts as I'm
        # paranoid about JSON's lack of care with insertion order.
        info_set_dict = {
            "cards_cluster": cards_cluster,
            "history": [
                {betting_stage: [str(action) for action in actions]}
                for betting_stage, actions in self._history.items()
            ],
        }
        return json.dumps(
            info_set_dict, separators=(",", ":"), cls=utils.io.NumpyJSONEncoder
        )

    @property
    def payout(self) -> Dict[int, int]:
        """Return player index to payout number of chips dictionary."""
        n_chips_delta = dict()
        for player_i, player in enumerate(self.players):
            n_chips_delta[player_i] = player.n_chips - self._initial_n_chips
        return n_chips_delta

    @property
    def is_terminal(self) -> bool:
        """Returns whether this state is terminal or not.

        The state is terminal once all rounds of betting are complete and we
        are at the show down stage of the game or if all players have folded.
        """
        return self._betting_stage in {"show_down", "terminal"}

    @property
    def players(self) -> List[ShortDeckPokerPlayer]:
        """Returns players in table."""
        return self._table.players

    @property
    def current_player(self) -> ShortDeckPokerPlayer:
        """Returns a reference to player that makes a move for this state."""
        return self._table.players[self.player_i]

    @property
    def legal_actions(self) -> List[Optional[str]]:
        """Return the actions that are legal for this game state."""
        actions: List[Optional[str]] = []
        if self.current_player.is_active:
            actions += ["fold", "call"]
            if self._n_raises < 3:
                # In limit hold'em we can only bet/raise if there have been
                # less than three raises in this round of betting, or if there
                # are two players playing.
                actions += ["raise"]
        else:
            actions += [None]
        return actions
