from typing import Tuple

from blessed import Terminal


class AsciiCardCollection:
    def __init__(
        self,
        *cards,
        hide_cards: bool = False,
        term: Terminal = None,
    ):
        """"""
        self.term = term
        self.cards = cards
        self.update(hide_cards)

    def __str__(self):
        return "\n".join(self.lines)

    def update(self, hide_cards: bool):
        """"""
        if hide_cards:
            cards = [None for _ in self.cards]
        else:
            cards = self.cards
        self.lines, self.width, self.height = self._create_card_collection_str(
            *cards, return_string=False
        )

    def _create_card_collection_str(
        self, *cards, return_string=True
    ) -> Tuple[str, int, int]:
        """
        Essentially the dealers method of print ascii cards. This method hides the first card, shows it flipped over
        :param cards: A list of card objects, the first will be hidden
        :return: A string, the nice ascii version of cards
        """
        hide_lines = ["┌─────────┐"] + ["│░░░░░░░░░│"] * 7 + ["└─────────┘"]
        all_lines = ["" for _ in range(9)]
        for card in cards:
            if card is None:
                card_lines = hide_lines
            else:
                card_lines = self._ascii_card(card, return_string=False)
            all_lines = [x + y for x, y in zip(all_lines, card_lines)]
        if return_string:
            card_str = "\n".join(all_lines)
        else:
            card_str = all_lines
        width = len(all_lines[0])
        height = len(all_lines)
        return card_str, width, height

    @staticmethod
    def _ascii_card(*cards, return_string=True):
        """
        Instead of a boring text version of the card we render an ASCII image of the card.
        :param cards: One or more card objects
        :param return_string: By default we return the string version of the card, but the dealer hide the 1st card and we
        keep it as a list so that the dealer can add a hidden card in front of the list
        """
        # we will use this to prints the appropriate icons for each card
        name_to_symbol = {
            "spades": "♠",
            "diamonds": "♦",
            "hearts": "♥",
            "clubs": "♣",
        }
        # create an empty list of list, each sublist is a line
        lines = [[] for _ in range(9)]
        for index, card in enumerate(cards):
            # "King" should be "K" and "10" should still be "10"
            if card.rank == "10":  # ten is the only one who's rank is 2 char long
                rank = card.rank
                space = (
                    ""  # if we write "10" on the card that line will be 1 char to long
                )
            else:
                rank = card.rank[
                    0
                ]  # some have a rank of 'King' this changes that to a simple 'K' ("King" doesn't fit)
                space = " "  # no "10", we use a blank space to will the void
            # get the cards suit in two steps
            suit = name_to_symbol[card.suit.lower()]
            rank = rank.upper()
            # add the individual card on a line by line basis
            lines[0].append("┌─────────┐")
            lines[1].append(
                "│{}{}       │".format(rank, space)
            )  # use two {} one for char, one for space or char
            lines[2].append("│         │")
            lines[3].append("│         │")
            lines[4].append("│    {}    │".format(suit))
            lines[5].append("│         │")
            lines[6].append("│         │")
            lines[7].append("│       {}{}│".format(space, rank))
            lines[8].append("└─────────┘")
        result = ["".join(line) for line in lines]
        # hidden cards do not use string
        if return_string:
            return "\n".join(result)
        else:
            return result
