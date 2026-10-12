"""What a question will take before anything is spent: what's held, what must be fetched, and what
the model calls will cost."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Final, Literal

from ..domain.money import aud
from ..output.progress import usual_range

__all__ = ["PATHS", "Cost", "Estimate"]

Path = Literal["saved", "pending", "fetch", "new_bank"]


@dataclass(frozen=True, slots=True)
class Cost:
    calls: tuple[int, int]
    usd: tuple[Decimal, Decimal]
    ceiling_calls: int
    ceiling_usd: Decimal
    seconds: tuple[int, int]
    """How long a run on this path usually takes, from the recorded live runs."""


#: Measured cost by path. A walk-led metric reads the bank's own walk from its pages, a
#: chart or two, which a question answered by the identity alone doesn't need. The saved
#: paths allow for the memo's one rewrite: Westpac's 1H26 net profit needed it, and cost
#: A$1.15 in two minutes. A result that may not be out yet costs nothing if the ASX's list
#: doesn't have it, and a fetch's price if it does.
PATHS: Final[dict[str, Cost]] = {
    "saved": Cost((2, 6), (Decimal("0.15"), Decimal("0.90")), 10, Decimal("2.00"), (25, 150)),
    "saved_walk": Cost((3, 10), (Decimal("0.25"), Decimal("1.20")), 14, Decimal("3.00"), (25, 150)),
    "pending": Cost((0, 16), (Decimal("0.00"), Decimal("2.50")), 25, Decimal("5.00"), (5, 120)),
    "fetch": Cost((5, 16), (Decimal("0.40"), Decimal("2.50")), 25, Decimal("5.00"), (40, 120)),
    "new_bank": Cost((8, 18), (Decimal("0.70"), Decimal("3.00")), 30, Decimal("5.00"), (60, 180)),
}


@dataclass
class Estimate:
    """What one question will take, said before it runs."""

    asked: str
    """The question as typed: bank, period and metric."""
    heading: str = ""
    """The question as understood: "National Australia Bank · net interest margin · 1H26
    against 1H25"."""
    path: Path | None = None
    walk_led: bool = False
    held: list[str] = field(default_factory=list)
    """What the saved data already holds for it."""
    steps: list[str] = field(default_factory=list)
    """What the agent will do, in order, in plain words."""
    notes: list[str] = field(default_factory=list)
    """Anything a reader should know first, such as a brand answered by its owner."""
    stop: str | None = None
    """Why it can't go ahead at all, when it can't: no model call could change that."""

    @property
    def cost(self) -> Cost | None:
        if self.path is None:
            return None
        return PATHS["saved_walk" if self.path == "saved" and self.walk_led else self.path]

    @property
    def outside_saved_data(self) -> bool:
        return self.path in {"pending", "fetch", "new_bank"}

    def lines(self) -> list[str]:
        """The estimate as a person reads it."""
        out = [self.heading or self.asked]
        out += self.notes
        if self.stop is not None:
            return [*out, self.stop]
        if self.path == "saved":
            out.append("In the saved data: " + "; ".join(self.held) + ".")
        elif self.path == "pending":
            out.append(
                "Not in the saved data, and it may not have been released yet. The ASX's own "
                "list of the bank's announcements is checked first, for nothing; only if the "
                "result is there is it downloaded, read and explained."
            )
        elif self.path == "fetch":
            out.append(
                "Outside the saved data. This needs external calls: a search of the bank's "
                "site and the ASX, a download, and model calls to read and explain it."
            )
        elif self.path == "new_bank":
            out.append(
                "Outside the saved data, for a bank not read before. This needs external "
                "calls: a search of the ASX, a download, and model calls to read and explain it."
            )
        out += [f"  {number}. {step}" for number, step in enumerate(self.steps, start=1)]
        cost = self.cost
        if cost is not None:
            low, high = cost.usd
            out.append(
                f"Estimated cost: {aud(low)} to {aud(high)}, about {cost.calls[0]} to "
                f"{cost.calls[1]} model calls. It won't spend more than "
                f"{aud(cost.ceiling_usd)} or make more than {cost.ceiling_calls} calls."
            )
            out.append(f"It usually takes {usual_range(*cost.seconds)}.")
        return out
