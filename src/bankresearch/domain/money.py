"""Money in Australian dollars. Anthropic prices and bills model calls in US dollars, and every
cost the tool shows or accepts is converted at the one rate in config/models.yaml."""

from __future__ import annotations

import re
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal

from .registry import get_registry

__all__ = ["aud", "aud_amounts", "to_aud", "to_usd", "usd_per_aud"]

#: The RBA's rate struck at 4pm on 9 October 2026, for a config that doesn't give one.
_RBA_9_OCTOBER_2026 = Decimal("0.6979")

#: A "$" with no letters before it, ahead of a figure: "$300 million", "($27m)".
_BARE_DOLLAR = re.compile(r"(?<![A-Za-z$])\$(?=\s?\d)")
_QUOTED = re.compile(r'"[^"]*"|“[^”]*”')


def usd_per_aud() -> Decimal:
    """US dollars to the Australian dollar, from ``currency.usd_per_aud``."""
    try:
        configured = (get_registry().models.get("currency") or {}).get("usd_per_aud")
    except Exception:
        configured = None
    return Decimal(str(configured)) if configured else _RBA_9_OCTOBER_2026


def to_aud(usd: Decimal | str | int | float) -> Decimal:
    return Decimal(str(usd)) / usd_per_aud()


def to_usd(aud_amount: Decimal | str | int | float) -> Decimal:
    return Decimal(str(aud_amount)) * usd_per_aud()


def aud(usd: Decimal | str | int | float, places: int = 2, *, up: bool = False) -> str:
    """A cost billed in US dollars, said in Australian dollars: "A$0.34". A ceiling rounds up,
    so the figure shown is never below the most the call can cost."""
    step = Decimal(1).scaleb(-places)
    value = to_aud(usd).quantize(step, rounding=ROUND_CEILING if up else ROUND_HALF_UP)
    return f"A${value:,.{places}f}"


def aud_amounts(text: str) -> str:
    """Each amount written with a bare "$" written as "A$". The banks report in Australian
    dollars, so a memo's "$300 million" is A$300 million; a quotation keeps the bank's words."""
    out: list[str] = []
    last = 0
    for quoted in _QUOTED.finditer(text):
        out.append(_BARE_DOLLAR.sub("A$", text[last : quoted.start()]))
        out.append(quoted.group(0))
        last = quoted.end()
    out.append(_BARE_DOLLAR.sub("A$", text[last:]))
    return "".join(out)
