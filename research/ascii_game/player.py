from blessed import Terminal

from card_collection import CardCollection


class Player:
    def __init__(
        self,
        *cards,
        name: str = "",
        chips_in_pot: int = 0,
        chips_in_bank: int = 0,
        term: Terminal = None,
        **card_collection_kwargs,
    ):
        self.cards = cards
        self.card_collection_kwargs = card_collection_kwargs
        self.chips_in_pot = chips_in_pot
        self.chips_in_bank = chips_in_bank
        self.name = name
        self.term = term
        self.update()

    def update(self):
        card_collection = CardCollection(
            *self.cards, term=self.term, **self.card_collection_kwargs
        )
        self.lines = card_collection.lines
        self.lines[1] += f" {self.name}"
        self.lines[2] += f" bet chips: {self.chips_in_pot}"
        self.lines[3] += f" bank roll: {self.chips_in_bank}"
        self.width = max(len(line) for line in self.lines)
        self.height = len(self.lines)

    def __str__(self):
        self.update()
        return "\n".join(self.lines)
