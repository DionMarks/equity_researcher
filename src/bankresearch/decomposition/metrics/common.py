"""Shared machinery for the per-metric Path B decompositions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Any, Literal

from ...domain.models import DriverContribution
from ..shapley import as_decimal, ratio_split

__all__ = [
    "Ctx",
    "bps",
    "driver",
    "flow_effects",
    "ratio_effects",
    "stock_effects",
    "to_change_unit",
]

_BPS_PER_UNIT_RATIO = Fraction(10_000)


@dataclass(frozen=True, slots=True)
class Ctx:
    """Everything one metric's decomposition is allowed to look at."""

    metric_id: str
    prior: dict[str, Any]
    current: dict[str, Any]
    reported_prior: Decimal
    reported_current: Decimal
    days: int | None = None
    pcp_days: int | None = None
    year_days: int | None = None
    """The length of the financial year each period belongs to, for annualising."""
    pcp_year_days: int | None = None

    def pair(self, component: str) -> tuple[Fraction, Fraction]:
        return Fraction(self.prior[component]), Fraction(self.current[component])

    def has(self, *components: str) -> bool:
        return all(c in self.prior and c in self.current for c in components)

    def has_current(self, *components: str) -> bool:
        return all(c in self.current for c in components)

    def delta(self, component: str) -> Fraction:
        before, after = self.pair(component)
        return after - before


def to_change_unit(value: Decimal, change_unit: Literal["bps", "aud_m"]) -> Decimal:
    """A reported-unit movement expressed in the metric's change unit.

    Ratios are published in percent and their movements quoted in basis points,
    so a 0.18pp move becomes 18bp. Amounts are already in AUD millions.
    """
    if change_unit == "bps":
        return value * Decimal(100)
    return value


def bps(value: Fraction) -> Decimal:
    """A contribution to a pure ratio, expressed in basis points."""
    return as_decimal(value * _BPS_PER_UNIT_RATIO)


def ratio_effects(
    num_prior: Fraction, den_prior: Fraction, num_current: Fraction, den_current: Fraction
) -> tuple[Fraction, Fraction]:
    """Numerator and denominator contributions to a change in ``num / den``."""
    return ratio_split(num_prior, den_prior, num_current, den_current)


def stock_effects(
    components: Mapping[str, str],
    ctx: Ctx,
    *,
    other_prior: Fraction,
    other_current: Fraction,
    side: Literal["numerator", "denominator"],
    den_prior: Fraction,
    den_current: Fraction,
) -> dict[str, Fraction]:
    """Split a ratio effect across components measured as stocks in both periods.

    Both splits are linear in the component movements, so the parts sum to the aggregate
    effect exactly:

    - numerator: ``d(a_j) * (1/b0 + 1/b1)/2``
    - denominator: ``-d(b_j) * (a0 + a1)/(2*b0*b1)``

    The denominator coefficient uses the total denominator, not the component's own, which
    is why this can't be done by calling :func:`ratio_split` on each component.
    """
    out: dict[str, Fraction] = {}
    for component, driver_id in components.items():
        if not ctx.has(component):
            continue
        change = ctx.delta(component)
        if side == "numerator":
            out[driver_id] = change * (Fraction(1) / den_prior + Fraction(1) / den_current) / 2
        else:
            out[driver_id] = -change * (other_prior + other_current) / (2 * den_prior * den_current)
    return out


def flow_effects(
    components: Mapping[str, str],
    ctx: Ctx,
    *,
    den_prior: Fraction,
    den_current: Fraction,
) -> dict[str, Fraction]:
    """Split a numerator effect across components already stated as movements.

    A capital walk discloses flows (organic generation, dividends net of DRP, buybacks)
    rather than opening and closing stocks, so each value is already the movement and needs
    no differencing.
    """
    out: dict[str, Fraction] = {}
    for component, driver_id in components.items():
        if component not in ctx.current:
            continue
        change = Fraction(ctx.current[component])
        out[driver_id] = change * (Fraction(1) / den_prior + Fraction(1) / den_current) / 2
    return out


def driver(
    driver_id: str,
    contribution: Decimal | None,
    *,
    parent_id: str | None = None,
    label: str = "",
    note: str = "",
    evidence_type: str = "computed",
    from_components: tuple[str, ...] = (),
) -> DriverContribution:
    if contribution is None:
        direction: Literal["+", "-", "0"] = "0"
    elif contribution > 0:
        direction = "+"
    elif contribution < 0:
        direction = "-"
    else:
        direction = "0"
    return DriverContribution(
        driver_id=driver_id,
        parent_id=parent_id,
        label=label,
        contribution=contribution,
        direction=direction,
        method="computed",
        evidence_type=evidence_type,  # type: ignore[arg-type]
        from_components=list(from_components),
        note=note,
    )
