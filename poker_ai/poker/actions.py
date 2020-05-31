from __future__ import annotations


__all__ = ["Call", "Fold", "Raise", "AbstractedRaise"]

DUMMY_AMOUNTS = [10, 100, 500, 1000, 5000, 10000]


class Action:
    def __init__(self):
        pass


class Call(Action):
    def __repr__(self):
        return "call"


class Fold(Action):
    def __repr__(self):
        return "fold"


class Raise(Action):
    def __call__(self, amount):
        self.amount = amount

    def __repr__(self):
        return f"raise"


class AbstractedRaise(Action):
    def __init__(self, allowed_amounts):
        self.amounts = allowed_amounts

    def __call__(self, amount):
        if amount not in self.amounts:
            raise Exception(
                f"Specified amount '{amount}' is not valid for this action "
                f"abstraction, check 'allowed_amounts()' for more information"
            )
        self.amount = amount

    def __repr__(self):
        return f"raise {self.amount}"

    @property
    def allowed_amounts(self):
        return self.amounts
