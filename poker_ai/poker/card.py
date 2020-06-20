from typing import Dict, List, Set, Union

from poker_ai.poker.evaluation.eval_card import EvaluationCard


def get_all_suits() -> Set[str]:
    """Get set of suits that the card can take on."""
    return {"spades", "diamonds", "clubs", "hearts"}


def get_all_ranks() -> List[str]:
    """Get the list of ranks the card could be."""
    return [
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "jack",
        "queen",
        "king",
        "ace",
    ]


class Card:
    """Card to represent a poker card."""

    def __init__(self, rank: Union[str, int], suit: str):
        """Instanciate the card."""
        if not isinstance(rank, (int, str)):
            raise ValueError(f"rank should be str/int but was: {type(rank)}.")
        elif isinstance(rank, str):
            rank = self._str_to_rank(rank)
        if rank < 2 or rank > 14:
            raise ValueError(
                f"rank should be between 2 and 14 (inclusive) but was {rank}"
            )
        if suit not in get_all_suits():
            raise ValueError(f"suit {suit} must be in {get_all_suits()}")
        self._rank = rank
        self._suit = suit
        rank_char = self._rank_to_char(rank)
        suit_char = self.suit.lower()[0]
        self._eval_card = EvaluationCard.new(f"{rank_char}{suit_char}")

    def __repr__(self):
        """Pretty printing the object."""
        icon = self._suit_to_icon(self.suit)
        return f"<Card card=[{self.rank} of {self.suit} {icon}]>"

    def __int__(self):
        return self._eval_card

    def __lt__(self, other):
        return self.rank < other.rank
        # raise NotImplementedError("Boolean operations not supported")

    def __le__(self, other):
        return self.rank <= other.rank
        # raise NotImplementedError("Boolean operations not supported")

    def __gt__(self, other):
        return self.rank > other.rank
        # raise NotImplementedError("Boolean operations not supported")

    def __ge__(self, other):
        return self.rank >= other.rank
        # raise NotImplementedError("Boolean operations not supported")

    def __eq__(self, other):
        return int(self) == int(other)

    def __ne__(self, other):
        return int(self) != int(other)

    def __hash__(self):
        return hash(int(self))

    @property
    def eval_card(self) -> EvaluationCard:
        """Return an `EvaluationCard` for use in the `Evaluator`."""
        return self._eval_card

    @property
    def rank_int(self) -> int:
        """Get the rank as an int"""
        return self._rank

    @property
    def rank(self) -> str:
        """Get the rank as a string."""
        return self._rank_to_str(self._rank)

    @property
    def suit(self) -> str:
        """Get the suit."""
        return self._suit

    def _str_to_rank(self, string: str) -> int:
        """Convert the string rank to the integer rank."""
        return {
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "10": 10,
            "jack": 11,
            "queen": 12,
            "king": 13,
            "ace": 14,
            "t": 10,
            "j": 11,
            "q": 12,
            "k": 13,
            "a": 14,
        }[string.lower()]

    def _rank_to_str(self, rank: int) -> str:
        """Convert the integer rank to the string rank."""
        return {
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "10",
            11: "jack",
            12: "queen",
            13: "king",
            14: "ace",
        }[rank]

    def _rank_to_char(self, rank: int) -> str:
        """Convert the int rank to char used by the `EvaluationCard` object."""
        return {
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "T",
            11: "J",
            12: "Q",
            13: "K",
            14: "A",
        }[rank]

    def _suit_to_icon(self, suit: str) -> str:
        """Icons for pretty printing."""
        return {"hearts": "♥", "diamonds": "♦", "clubs": "♣", "spades": "♠"}[suit]

    def to_dict(self) -> Dict[str, Union[int, str]]:
        """Turn into dict."""
        return dict(rank=self._rank, suit=self._suit)

    @staticmethod
    def from_dict(x: Dict[str, Union[int, str]]):
        """From dict turn into class."""
        if set(x) != {"rank", "suit"}:
            raise NotImplementedError(f"Unrecognised dict {x}")
        return Card(rank=x["rank"], suit=x["suit"])

    def to_dict(self) -> Dict[str, Union[int, str]]:
        """Turn into dict."""
        return dict(rank=self._rank, suit=self._suit)

    @staticmethod
    def from_dict(x: Dict[str, Union[int, str]]):
        """From dict turn into class."""
        if set(x) != {"rank", "suit"}:
            raise NotImplementedError(f"Unrecognised dict {x}")
        return Card(rank=x["rank"], suit=x["suit"])

