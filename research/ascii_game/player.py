from blessed import Terminal

from card_collection import AsciiCardCollection


class AsciiPlayer:
    def __init__(
        self,
        *cards,
        term: Terminal,
        name: str = "",
        chips_in_pot: int = 0,
        chips_in_bank: int = 0,
        info_position: str = "right",
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
        self.term = term
        self.folded = folded
        self.is_turn = is_turn
        self.info_position = info_position
        self.is_small_blind = is_small_blind
        self.is_big_blind = is_big_blind
        self.is_dealer = is_dealer
        self.update()

    def stylise_name(self, name: str, extra: str) -> str:
        if self.folded:
            name = f"{name} (folded)"
        if self.is_turn:
            name = self.term.orangered(f"{name} {self.term.blink_bold('turn')}")
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
        if self.info_position == "right":
            max_len = max(len(i) for i in info)
            for line_i, line in enumerate(info):
                self.lines[1 + line_i] += f" {line}"
            max_len = max(len(l) for l in self.lines)
            for line_i, line in enumerate(self.lines):
                n_spaces = max_len - len(line)
                self.lines[line_i] += f" {n_spaces * ' '}"
        elif self.info_position == "top":
            self.lines = info + self.lines
        elif self.info_position == "bottom":
            self.lines = self.lines + info
        else:
            raise NotImplementedError(
                f"info position {self.info_position} not supported")
        self.width = max(len(line) for line in self.lines)
        self.height = len(self.lines)

    def __str__(self):
        self.update()
        return "\n".join(self.lines)
