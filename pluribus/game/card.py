from pluribus.game.evaluation.eval_card import EvaluationCard


def get_all_suits():
    """"""
    return {"spades", "diamonds", "clubs", "hearts"}


def get_all_ranks():
    """"""
    return [
        "2", "3", "4", "5", "6", "7", "8", "9", "10",
        "jack", "queen", "king", "ace"
    ]


class Card:
    """"""

    def __init__(self, rank, suit):
        """"""
        if not isinstance(rank, int):
            raise ValueError(
                f'rank should be int but was type: {type(rank)}.')
        if rank < 2 or rank > 14:
            raise ValueError(
                f'rank should be between 2 and 14 (inclusive) but was {rank}')
        if suit not in get_all_suits():
            raise ValueError(f'suit {suit} must be in {get_all_suits()}')
        self._rank = rank
        self._suit = suit
        rank_char = self._rank_to_char(rank)
        suit_char = self.suit.lower()[0]
        self._eval_card = EvaluationCard.new(f'{rank_char}{suit_char}')

    @property
    def eval_card(self):
        return self._eval_card

    @property
    def rank(self):
        return self._rank_to_str(self._rank)

    @property
    def suit(self):
        return self._suit

    def _rank_to_str(self, rank):
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
            14: "ace"
        }[rank]

    def _rank_to_char(self, rank):
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
            14: "A"
        }[rank]

    def _suit_to_icon(self, suit):
        return {
            "hearts": "♥", "diamonds": "♦", "clubs": "♣", "spades": "♠"
        }[suit]

    def __repr__(self):
        icon = self._suit_to_icon(self.suit)
        return f'<Card card=[{self.rank} of {self.suit} {icon}]>'
