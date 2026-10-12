"""Units, tolerances and number normalisation. Uses Decimal throughout, never float."""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Final, Literal

__all__ = [
    "AUD_UNITS",
    "RATIO_UNITS",
    "Unit",
    "normalise_aud",
    "normalise_numbers",
    "same_published_figure",
    "to_aud_m",
    "to_bps",
    "to_pct",
    "value_tolerance",
    "walk_tolerance",
]

Unit = Literal["pct", "bps", "aud_k", "aud_m", "aud_bn", "ratio", "days"]

RATIO_UNITS: Final[frozenset[str]] = frozenset({"pct", "bps", "ratio"})
AUD_UNITS: Final[frozenset[str]] = frozenset({"aud_k", "aud_m", "aud_bn"})

_BPS_PER_PCT: Final = Decimal(100)
_PCT_PER_RATIO: Final = Decimal(100)
_M_PER_BN: Final = Decimal(1000)
_K_PER_M: Final = Decimal(1000)


def to_pct(value: Decimal, unit: Unit) -> Decimal:
    """Express a ratio-like value in percent."""
    if unit == "pct":
        return value
    if unit == "bps":
        return value / _BPS_PER_PCT
    if unit == "ratio":
        return value * _PCT_PER_RATIO
    raise ValueError(f"{unit} is not a ratio unit")


def to_bps(value: Decimal, unit: Unit) -> Decimal:
    """Express a ratio-like value in basis points."""
    if unit == "bps":
        return value
    return to_pct(value, unit) * _BPS_PER_PCT


def to_aud_m(value: Decimal, unit: Unit) -> Decimal:
    """Express an amount in AUD millions."""
    if unit == "aud_m":
        return value
    if unit == "aud_bn":
        return value * _M_PER_BN
    if unit == "aud_k":
        return value / _K_PER_M
    raise ValueError(f"{unit} is not an AUD unit")


def normalise_aud(value: Decimal, unit: Unit) -> tuple[Decimal, Unit]:
    """An AUD figure expressed in millions, leaving other units untouched.

    Real packs mix units in one document: the income statement in $M and the average balance
    sheet in $B. Comparing them without converting puts a margin out by a factor of a
    thousand, so this runs before every identity rather than being left to each caller. A
    smaller bank prints its statements in thousands, which is converted the same way.
    """
    if unit in {"aud_bn", "aud_k"}:
        return to_aud_m(value, unit), "aud_m"
    return value, unit


def _decimal_places(value: Decimal) -> int:
    """Decimal places as the figure was published, trailing zeros included.

    ``Decimal("1.80").normalize()`` is ``1.8``, which would treat a margin published to two
    decimal places as precise to one and make its tolerance ten times wider. So the exponent
    is used as given.
    """
    exponent = value.as_tuple().exponent
    return -exponent if isinstance(exponent, int) and exponent < 0 else 0


def value_tolerance(unit: Unit, *, reported: Decimal | None = None) -> Decimal:
    """Half the last reported digit.

    When ``reported`` is given, the tolerance follows that figure's own precision: a ratio
    given to two decimal places tolerates half a basis point, one given to one decimal place
    tolerates five. Without it, the default is the two-decimal case.
    """
    if unit in RATIO_UNITS:
        if reported is None:
            return Decimal("0.5")  # bps
        return (
            to_bps(_half_of_last_digit(reported), "pct")
            if unit == "pct"
            else to_bps(_half_of_last_digit(reported), unit)
        )
    if unit in AUD_UNITS:
        if reported is None:
            return Decimal("0.5")  # AUD m
        return to_aud_m(_half_of_last_digit(reported), unit)
    raise ValueError(f"no tolerance defined for unit {unit}")


def _half_of_last_digit(reported: Decimal) -> Decimal:
    """Half the last digit the issuer printed, in the unit it printed it in.

    Measured before any unit conversion. Risk-weighted assets of "$426.4bn" are only known
    to within $50m. Converting first turns that into 426,400.0 with an apparent precision of
    $0.05m, and a capital ratio that's right to the published digit then fails its own
    identity by a fraction of a basis point.
    """
    return Decimal("0.5") * Decimal(10) ** -_decimal_places(reported)


def same_published_figure(value: Decimal, unit: Unit, other: Decimal, other_unit: Unit) -> bool:
    """Whether two readings are the same published figure, to the coarser one's precision.

    The comparison is done in the unit ``value_tolerance`` uses: basis points for a ratio,
    millions for an amount. Margins of 12.42% and 12.75% differ by 0.33 of a percentage
    point, or 33 basis points. Against a tolerance of half a basis point, comparing in
    percentage points would wrongly call them the same figure.
    """
    if unit in RATIO_UNITS and other_unit in RATIO_UNITS:
        left, right = to_bps(value, unit), to_bps(other, other_unit)
    elif unit in AUD_UNITS and other_unit in AUD_UNITS:
        left, right = to_aud_m(value, unit), to_aud_m(other, other_unit)
    else:
        return False
    slack = max(value_tolerance(unit, reported=value), value_tolerance(other_unit, reported=other))
    return abs(left - right) <= slack


def walk_tolerance(unit: Unit, steps: int) -> Decimal:
    """Reconciliation tolerance for a walk of ``steps`` rounded components.

    The larger of a floor and a per-step allowance: 1bp or 0.5bp per step for
    ratios, AUD 2m or AUD 0.5m per step for amounts. Each step is rounded separately, so the
    allowance grows with the number of steps.
    """
    if steps < 0:
        raise ValueError("steps must not be negative")
    if unit in RATIO_UNITS:
        return max(Decimal("1"), Decimal("0.5") * steps)
    if unit in AUD_UNITS:
        return max(Decimal("2"), Decimal("0.5") * steps)
    raise ValueError(f"no walk tolerance defined for unit {unit}")


# Matches a number with optional thousands separators, decimals, sign and a
# trailing unit marker.
_NUMBER = re.compile(
    r"(?P<sign>[-+(]?)\s*(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
    r"(?P<unit>%|bps?|basis points?|bn|billion|m|million)?",
    re.IGNORECASE,
)


def normalise_numbers(text: str) -> set[Decimal]:
    """Every number in ``text``, as the set of values it could stand for.

    A citation check needs ``1.95%`` on the page to match a claim of ``195bp``, and
    ``1,234`` to match ``1234``. Rather than guess which reading was meant, each number
    contributes every plausible normalisation and the check passes if any of them matches.
    That gives up a little precision to avoid failing correct citations.
    """
    found: set[Decimal] = set()
    for match in _NUMBER.finditer(text):
        raw = match.group("num").replace(",", "")
        try:
            value = Decimal(raw)
        except Exception:  # pragma: no cover - regex already constrains this
            continue
        if match.group("sign") in {"-", "("}:
            value = -value
        unit = (match.group("unit") or "").lower()

        found.add(value)
        if unit in {"%"}:
            found.add(value * _BPS_PER_PCT)  # also as bps
        elif unit.startswith("b") and unit not in {"bn", "billion"}:
            found.add(value / _BPS_PER_PCT)  # bps, also as pct
        elif unit in {"bn", "billion"}:
            found.add(value * _M_PER_BN)
        elif unit in {"m", "million"}:
            found.add(value / _M_PER_BN)
        else:
            # Bare number: it may be a percentage quoted without its sign.
            found.add(value * _BPS_PER_PCT)
    return found
