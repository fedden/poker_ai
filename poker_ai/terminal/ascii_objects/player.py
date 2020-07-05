from blessed import Terminal

from poker_ai.terminal.ascii_objects.card_collection import AsciiCardCollection


class AsciiPlayer:
    def __init__(
        self,
        *cards,
        term: Terminal,
        name: str = "",
        og_name: str = "",
        chips_in_pot: int = 0,
        chips_in_bank: int = 0,
        folded: bool = False,
        is_turn: bool = False,
        is_small_blind: bool = False,
        is_big_blind: bool = False,
        is_dealer: bool = False,
        **card_collection_kwargs,
    ):
        self.cards = cards
        self.card_collection_kwargs = card_collection_kwargs
        self.chips_in_pot = chips_in_pot
        self.chips_in_bank = chips_in_bank
        self.name = name
        self.og_name = og_name
        self.term = term
        self.folded = folded
        self.is_turn = is_turn
        self.is_small_blind = is_small_blind
        self.is_big_blind = is_big_blind
        self.is_dealer = is_dealer
        self.update()

    def stylise_name(self, name: str, extra: str) -> str:
        if self.folded:
            name = f"{name} (folded)"
        if self.is_turn:
            name = f"**{name}**"
        if extra:
            name = f"{name} ({extra})"
        return name

    def update(self):
        card_collection = AsciiCardCollection(
            *self.cards, term=self.term, **self.card_collection_kwargs
        )
        self.lines = card_collection.lines
        if self.is_small_blind:
            extra = "SB"
        elif self.is_big_blind:
            extra = "BB"
        elif self.is_dealer:
            extra = "D"
        else:
            extra = ""
        info = [
            self.stylise_name(self.name, extra),
            f"bet chips: {self.chips_in_pot}",
            f"bank roll: {self.chips_in_bank}",
        ]
        card_width = len(self.lines[0])
        self.lines = [i + (card_width - len(i)) * " " for i in info] + self.lines
        self.width = max(len(line) for line in self.lines)
        self.height = len(self.lines)

    def __str__(self):
        self.update()
        return "\n".join(self.lines)
