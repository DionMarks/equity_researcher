"""Exact attribution by Shapley value over the factors of an identity."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import permutations
from math import factorial

__all__ = [
    "MAX_FACTORS",
    "TooManyFactorsError",
    "as_decimal",
    "ratio_split",
    "ratio_split_components",
    "shapley",
    "shapley_exact",
    "weighted_average_split",
]

# Identities are capped at 5 factors, so at most 120 orderings. Past that the
# enumeration gets expensive, and an identity with six factors should probably be nested
# instead.
MAX_FACTORS = 5

Number = Decimal | Fraction | int
Factors = Mapping[str, Number]
ExactFn = Callable[[dict[str, Fraction]], Fraction]

# Enough digits that the conversion is never the limiting factor.
_OUTPUT_PRECISION = Decimal("1e-28")
_WORKING_PRECISION = 60


class TooManyFactorsError(ValueError):
    """Raised when an identity declares more factors than the cap allows."""


def _to_fraction(value: Number) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def _check(x0: Factors, x1: Factors) -> tuple[str, ...]:
    if set(x0) != set(x1):
        missing = sorted(set(x0) ^ set(x1))
        raise ValueError(f"factor sets differ between periods: {missing}")
    if len(x0) > MAX_FACTORS:
        raise TooManyFactorsError(
            f"{len(x0)} factors exceeds the cap of {MAX_FACTORS}; nest the decomposition instead"
        )
    if not x0:
        raise ValueError("no factors given")
    # Sorted so the enumeration, and therefore every output, is deterministic.
    return tuple(sorted(x0))


def shapley_exact(f: ExactFn, x0: Factors, x1: Factors) -> dict[str, Fraction]:
    """Exact Shapley contributions, as rationals.

    Averages each factor's marginal contribution over all orderings in which the
    factors could be moved from their period-0 to their period-1 value.
    """
    keys = _check(x0, x1)
    start = {k: _to_fraction(x0[k]) for k in keys}
    finish = {k: _to_fraction(x1[k]) for k in keys}

    totals = dict.fromkeys(keys, Fraction(0))
    for order in permutations(keys):
        state = dict(start)
        previous = f(dict(state))
        for key in order:
            state[key] = finish[key]
            current = f(dict(state))
            totals[key] += current - previous
            previous = current

    divisor = factorial(len(keys))
    return {k: totals[k] / divisor for k in keys}


def as_decimal(value: Fraction) -> Decimal:
    """Convert an exact rational to Decimal at full output precision.

    Needs a local context: 28 decimal places plus the integer part of a bank-sized figure
    exceeds the default 28-digit precision, and the default context would raise instead of
    rounding.
    """
    with localcontext() as context:
        context.prec = _WORKING_PRECISION
        return (Decimal(value.numerator) / Decimal(value.denominator)).quantize(_OUTPUT_PRECISION)


def shapley(f: ExactFn, x0: Factors, x1: Factors) -> dict[str, Decimal]:
    """Shapley contributions as ``Decimal``, for display and storage."""
    return {k: as_decimal(v) for k, v in shapley_exact(f, x0, x1).items()}


def ratio_split(a0: Number, b0: Number, a1: Number, b1: Number) -> tuple[Fraction, Fraction]:
    """Numerator and denominator contributions to a change in ``a / b``.

    This is the closed form of the two-factor Shapley value.
    ``test_ratio_split_matches_the_generic_enumeration`` checks it against the generic
    enumeration::

        d(a/b) = da * (1/b0 + 1/b1)/2  -  db * (a0 + a1)/(2*b0*b1)

    Returns signed contributions that sum to the change, so callers don't have to remember
    which term carries the minus sign.
    """
    fa0, fb0, fa1, fb1 = (_to_fraction(v) for v in (a0, b0, a1, b1))
    if fb0 == 0 or fb1 == 0:
        raise ZeroDivisionError("a ratio cannot be decomposed across a zero denominator")

    numerator = (fa1 - fa0) * (Fraction(1) / fb0 + Fraction(1) / fb1) / 2
    denominator = -(fb1 - fb0) * (fa0 + fa1) / (2 * fb0 * fb1)
    return numerator, denominator


def ratio_split_components(
    a0_parts: Mapping[str, Number],
    b0: Number,
    a1_parts: Mapping[str, Number],
    b1: Number,
) -> tuple[dict[str, Fraction], Fraction]:
    """As :func:`ratio_split`, with the numerator effect split by component.

    The numerator effect is linear in the numerator, so each income line in a bridge
    contributes its own share of the ratio movement, and the shares add up to the whole
    numerator effect. That's what lets a cost-to-income movement be reported by expense
    category without a plug.
    """
    if set(a0_parts) != set(a1_parts):
        raise ValueError("numerator components differ between periods")
    per_component: dict[str, Fraction] = {}
    for key in sorted(a0_parts):
        per_component[key], _ = ratio_split(a0_parts[key], b0, a1_parts[key], b1)
    _, denominator = ratio_split(
        sum((_to_fraction(v) for v in a0_parts.values()), Fraction(0)),
        b0,
        sum((_to_fraction(v) for v in a1_parts.values()), Fraction(0)),
        b1,
    )
    return per_component, denominator


def weighted_average_split(
    w0: Mapping[str, Number],
    m0: Mapping[str, Number],
    w1: Mapping[str, Number],
    m1: Mapping[str, Number],
) -> tuple[dict[str, Fraction], dict[str, Fraction]]:
    """Mix and rate effects for a weighted average ``sum(w_i * m_i)``.

    Exact, by the same two-factor argument as :func:`ratio_split`::

        d(sum w_i m_i) = sum dw_i * (m_i0 + m_i1)/2  +  sum dm_i * (w_i0 + w_i1)/2

    Used for the divisional mix-versus-rate cross-check on NIM. Whether the margin fell
    because pricing fell or because lending shifted toward thinner-margin products can only
    be answered if the split is exact.
    """
    keys = sorted(set(w0) | set(m0) | set(w1) | set(m1))
    for name, mapping in (("w0", w0), ("m0", m0), ("w1", w1), ("m1", m1)):
        missing = [k for k in keys if k not in mapping]
        if missing:
            raise ValueError(f"{name} is missing components {missing}")

    mix: dict[str, Fraction] = {}
    rate: dict[str, Fraction] = {}
    for key in keys:
        fw0, fw1 = _to_fraction(w0[key]), _to_fraction(w1[key])
        fm0, fm1 = _to_fraction(m0[key]), _to_fraction(m1[key])
        mix[key] = (fw1 - fw0) * (fm0 + fm1) / 2
        rate[key] = (fm1 - fm0) * (fw0 + fw1) / 2
    return mix, rate
