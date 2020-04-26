from __future__ import annotations

import operator
import copy
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import dill as pickle

from pluribus.poker.actions import Action
from pluribus.poker.card import Card
from pluribus.poker.engine import PokerEngine
from pluribus.games.short_deck.player import ShortDeckPokerPlayer
from pluribus.poker.table import PokerTable

logger = logging.getLogger(__name__)


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
    ):
        """Initialise state."""
        n_players = len(players)
        if n_players <= 1:
            raise ValueError(
                f"At least 2 players must be provided but only {n_players} "
                f"were provided."
            )
        if load_pickle_files:
            self.info_set_lut = self._load_pickle_files(pickle_dir)
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
        self._poker_engine = PokerEngine(
            table=self._table, small_blind=small_blind, big_blind=big_blind
        )
        # Reset the pot, assign betting order to players (might need to remove
        # this), assign blinds to the players.
        self._poker_engine.round_setup()
        # Deal private cards to players.
        self._table.dealer.deal_private_cards(self._table.players)
        # Store the actions as they come in here.
        self._history: List[str] = []
        self._betting_stage = "pre_flop"
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
        self._player_i_lut: Dict[str, List[int]] = {
            "pre_flop": player_i_order[2:] + player_i_order[:2],
            "flop": player_i_order,
            "turn": player_i_order,
            "river": player_i_order,
            "show_down": player_i_order,
            "terminal": player_i_order,
        }
        self._reset_betting_round_state()

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
        new_state._history.append(str(action))
        new_state._n_actions += 1
        # Player has made move, increment the player that is next.
        while True:
            new_state._move_to_next_player()
            if new_state.current_player.is_active:
                # If we have finished betting, (i.e: All players have put the
                # same amount of chips in), then increment the stage of
                # betting.
                finished_betting = not new_state._poker_engine.more_betting_needed
                if finished_betting and new_state.all_players_have_actioned:
                    # We have done atleast one full round of betting, increment
                    # stage of the game.
                    new_state._increment_stage()
                    new_state._reset_betting_round_state()
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
            else:
                # The current player isn't active, and we are not terminal.
                # We'll move to the next player in the next iteration of this
                # while loop, but append a null action to the history to
                # signify the notation h · 0 in algorithm 1 of the
                # supplementary material of the Pluribus paper.
                new_state._history.append("skip")
            assert not new_state.current_player.is_active
        return new_state

    def _move_to_next_player(self):
        """Ensure state points to next valid active player."""
        self._player_i_index += 1
        if self._player_i_index >= len(self.players):
            self._player_i_index = 0

    def _load_pickle_files(
        self, pickle_dir: str
    ) -> Dict[str, Dict[Tuple[int, ...], str]]:
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

    def _reset_betting_round_state(self):
        """Reset the state related to counting types of actions."""
        self._all_players_have_made_action = False
        self._n_actions = 0
        self._n_raises = 0
        self._player_i_index = 0
        while not self.current_player.is_active:
            self._player_i_index += 1

    def _increment_stage(self):
        """Once betting has finished, increment the stage of the poker game."""
        # Progress the stage of the game.
        if self._betting_stage == "pre_flop":
            # Progress from private cards to the flop.
            self._betting_stage = "flop"
            self._poker_engine.table.dealer.deal_flop(self._table)
        elif self._betting_stage == "flop":
            # Progress from flop to turn.
            self._betting_stage = "turn"
            self._poker_engine.table.dealer.deal_turn(self._table)
        elif self._betting_stage == "turn":
            # Progress from turn to river.
            self._betting_stage = "river"
            self._poker_engine.table.dealer.deal_river(self._table)
        elif self._betting_stage == "river":
            # Progress to the showdown.
            self._betting_stage = "show_down"
        elif self._betting_stage in {"show_down", "terminal"}:
            pass
        else:
            raise ValueError(f"Unknown betting_stage: {self._betting_stage}")

    @property
    def betting_stage(self) -> str:
        """Return betting stage."""
        return self._betting_stage

    @property
    def all_players_have_actioned(self) -> bool:
        """Return whether all players have made atleast one action."""
        return self._n_actions >= self._poker_engine.n_active_players

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
            if not self.info_set_lut:
                raise ValueError("Pickle luts must be loaded for info set.")
            elif eval_cards not in self.info_set_lut[self._betting_stage]:
                raise ValueError("Cards {cards} not in pickle files.")
            else:
                raise ValueError("Unrecognised betting stage in pickle files.")
        action_history = [str(action) for action in self._history]
        return f"cards_cluster={cards_cluster}, history={action_history}"

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
            if self._n_raises < 3 or self._poker_engine.n_active_players == 2:
                # In limit hold'em we can only bet/raise if there have been
                # less than three raises in this round of betting, or if there
                # are two players playing.
                actions += ["raise"]
        else:
            actions += [None]
        return actions
