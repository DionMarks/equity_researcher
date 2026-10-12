"""Path A: the attribution the issuer published itself."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
from itertools import pairwise
from typing import Literal

from ..domain.models import SourceRef
from ..domain.units import RATIO_UNITS, Unit, to_bps, walk_tolerance

__all__ = [
    "MidpointMismatch",
    "Walk",
    "WalkStep",
    "chain_walks",
    "choose_path_a",
    "find_pcp_walk",
]

WalkMethod = Literal["management_walk", "chained_walk"]


class MidpointMismatch(ValueError):
    """Raised when two walk legs do not meet, which signals a restatement."""


@dataclass(frozen=True, slots=True)
class WalkStep:
    """One bar of a published waterfall."""

    label: str
    """The issuer's own wording, preserved for the report."""
    driver_id: str | None
    """Canonical id, or ``None`` when no taxon claimed the row confidently.

    Left unmapped rather than forced, because a wrong mapping attributes a real number to
    the wrong cause.
    """
    value: Decimal
    evidence_id: str | None = None
    quote: str | None = None
    """The issuer's own words for this step, where the walk was read from its text."""
    source: SourceRef | None = None
    """Where this step is printed, when that isn't the walk's own source. Each leg of a
    chained walk keeps its own page, so every step cites a figure that page prints."""


@dataclass(frozen=True, slots=True)
class Walk:
    """A movement in a level, attributed to steps by the issuer."""

    opening_period: str
    closing_period: str
    opening_value: Decimal
    closing_value: Decimal
    steps: list[WalkStep]
    unit: Unit
    source: SourceRef
    method: WalkMethod = "management_walk"
    sources: tuple[SourceRef, ...] = field(default=())
    read_from: Literal["table", "chart", "text"] = "table"
    """Where the steps were read: a table's cells, a chart's printed bar labels, or the
    issuer's prose."""

    def __post_init__(self) -> None:
        if not self.sources:
            object.__setattr__(self, "sources", (self.source,))

    @property
    def declared_change(self) -> Decimal:
        return self.closing_value - self.opening_value

    @property
    def sum_of_steps(self) -> Decimal:
        return sum((s.value for s in self.steps), Decimal(0))

    def checksum_residual(self) -> Decimal:
        """Declared change minus the steps. Zero when the walk ties."""
        return self.declared_change - self.sum_of_steps

    def reconciles(self, tolerance: Decimal) -> bool:
        return abs(self.checksum_residual()) <= tolerance

    def bridges(self, closing: str, opening: str) -> bool:
        return (
            self.closing_period.upper() == closing.upper()
            and self.opening_period.upper() == opening.upper()
        )

    def unmapped_steps(self) -> int:
        return sum(1 for s in self.steps if s.driver_id is None)

    def mapped_steps(self) -> list[WalkStep]:
        return [s for s in self.steps if s.driver_id is not None]


def find_pcp_walk(walks: list[Walk], period: str, pcp: str) -> Walk | None:
    """The walk that itself declares it bridges ``pcp`` to ``period``."""
    for candidate in walks:
        if candidate.bridges(period, pcp):
            return candidate
    return None


def chain_walks(
    legs: list[Walk],
    midpoint_tolerance: Decimal = Decimal("1"),
) -> Walk:
    """Compose consecutive walks into one, summing driver by driver.

    Requires at least two contiguous legs whose values meet at each join.
    """
    if len(legs) < 2:
        raise ValueError("chaining needs at least two legs")

    ordered = list(legs)
    for earlier, later in pairwise(ordered):
        if earlier.closing_period.upper() != later.opening_period.upper():
            raise ValueError(
                f"legs are not contiguous: {earlier.opening_period}->"
                f"{earlier.closing_period} then {later.opening_period}->"
                f"{later.closing_period}"
            )
        # Compared in the tolerance's own unit. A ratio walk holds its values in per cent
        # and its tolerance in basis points, so comparing them unconverted divides the gap
        # by a hundred. A 12bp restatement between two halves was read as 0.12 against a
        # tolerance of 2 and chained anyway, which is exactly what this check is meant to
        # stop.
        gap = later.opening_value - earlier.closing_value
        if earlier.unit in RATIO_UNITS:
            gap = to_bps(gap, earlier.unit)
        if abs(gap) > midpoint_tolerance:
            raise MidpointMismatch(
                f"leg closing at {earlier.closing_value} does not meet the next leg "
                f"opening at {later.opening_value} ({earlier.closing_period}); the "
                "issuer appears to have restated between the two, so the legs cannot "
                "be chained"
            )
        if earlier.unit != later.unit:
            raise ValueError(f"legs disagree on unit: {earlier.unit} vs {later.unit}")

    # Each leg's steps are kept as printed, with the page they're printed on. Summing them
    # driver by driver here gave figures no page prints (ANZ's cash profit added +71bps in
    # one half and +81bps in the next, and +152bps is on neither page), so a cited step
    # couldn't be checked against its source. The decomposition sums by driver anyway.
    steps = [
        replace(item, source=item.source or leg.source) for leg in ordered for item in leg.steps
    ]
    kinds = {leg.read_from for leg in ordered}
    read_from: Literal["table", "chart", "text"] = (
        "chart" if "chart" in kinds else "text" if "text" in kinds else "table"
    )

    return Walk(
        opening_period=ordered[0].opening_period,
        closing_period=ordered[-1].closing_period,
        opening_value=ordered[0].opening_value,
        closing_value=ordered[-1].closing_value,
        steps=steps,
        unit=ordered[0].unit,
        source=ordered[-1].source,
        method="chained_walk",
        sources=tuple(s for leg in ordered for s in leg.sources),
        read_from=read_from,
    )


def _allowance(walk: Walk, floor: Decimal) -> Decimal:
    """The reconciliation tolerance this walk is entitled to, never below ``floor``."""
    return max(floor, walk_tolerance(walk.unit, max(len(walk.steps), 1)))


def choose_path_a(
    walks: list[Walk],
    period: str,
    pcp: str,
    tolerance: Decimal,
    intervening: str | None = None,
    endpoint_rounding: Decimal = Decimal(0),
) -> Walk | None:
    """The strongest walk available for this comparison, or nothing.

    Prefers a walk the issuer published for exactly this span, then falls back to chaining
    consecutive legs. Returns ``None`` rather than stretching a six-month walk across a year
    or using one that fails its own checksum.

    ``tolerance`` is a floor. Each candidate is judged against its own step count, since
    every bar is rounded separately and a six-bar walk carries six bars' worth of rounding.
    Judging against a single step's allowance used to reject ANZ's published margin walk
    (six bars, a real 2bp residual, against 1bp), and the run then reported no drivers while
    the caveats said a walk had been read and reconciled.

    ``endpoint_rounding`` is the rounding in the two published figures the walk runs
    between, which its bars owe nothing to. CBA prints its CET1 ratio to one decimal place,
    so its two ends alone can be 10bps from the sum of its bars. Leaving it out turned down
    CBA's FY25 chain (bars summing to -5bps against a printed 12.3% at both ends) after each
    of its legs had been read and reconciled against those same printed figures.
    """
    direct = find_pcp_walk(walks, period, pcp)
    if direct is not None and direct.reconciles(_allowance(direct, tolerance) + endpoint_rounding):
        return direct

    # Chain: find legs that form a contiguous path from pcp to period.
    by_opening: dict[str, list[Walk]] = {}
    for candidate in walks:
        by_opening.setdefault(candidate.opening_period.upper(), []).append(candidate)

    chain: list[Walk] = []
    cursor = pcp.upper()
    seen: set[str] = set()
    while cursor != period.upper():
        if cursor in seen:
            return None  # a cycle; refuse rather than loop
        seen.add(cursor)
        options = [w for w in by_opening.get(cursor, ()) if w.closing_period.upper() != cursor]
        if not options:
            return None
        if intervening is not None and len(chain) == 0:
            preferred = [w for w in options if w.closing_period.upper() == intervening.upper()]
            options = preferred or options
        leg = options[0]
        chain.append(leg)
        cursor = leg.closing_period.upper()
        if len(chain) > 4:  # a pcp span is at most two halves; anything longer is wrong
            return None

    if len(chain) < 2:
        return None
    try:
        chained = chain_walks(chain, midpoint_tolerance=tolerance)
    except (MidpointMismatch, ValueError):
        return None
    # A chain carries every leg's rounding, so the floor scales with the legs and the
    # allowance is still taken from the chained walk's own steps.
    allowance = _allowance(chained, tolerance * len(chain)) + endpoint_rounding
    return chained if chained.reconciles(allowance) else None
