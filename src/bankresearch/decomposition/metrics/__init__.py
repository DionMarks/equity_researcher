"""Path B: the exact accounting decomposition, one module per metric."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from ...domain.models import Decomposition
from ...domain.registry import component_name
from ...domain.units import Unit, value_tolerance, walk_tolerance
from ..identities import IDENTITIES
from . import cet1, cost_to_income, credit_impairment, headline_earnings, nim, roe
from .common import Ctx, to_change_unit

__all__ = ["decompose_path_b", "reconciliation_tolerance"]


def reconciliation_tolerance(
    change_unit: str,
    steps: int,
    reported_prior: Decimal,
    reported_current: Decimal,
    reported_unit: str,
) -> Decimal:
    """How far the drivers may miss the movement before it counts as a finding.

    Two sources of slack, both worked out rather than picked: the rounding of each driver,
    and the rounding of the two published figures the drivers reconcile to. Leaving out the
    second reports a one-basis-point residual as unexplained on a ratio published to one
    decimal place.
    """
    from_steps = walk_tolerance(change_unit, max(steps, 1))  # type: ignore[arg-type]
    if change_unit == "aud_m":
        # Each line's rounding is half the last digit the bank printed, so the allowance per
        # line scales with it. Sized for whole millions, it let BNK's bridge, printed to the
        # thousand, miss by A$3m on a movement of A$5m. Whole millions are unchanged.
        places = min(
            max(
                (
                    -int(v.as_tuple().exponent)
                    for v in (reported_prior, reported_current)
                    if int(v.as_tuple().exponent) < 0
                ),
                default=0,
            ),
            3,
        )
        scale = Decimal(10) ** -places
        from_steps = max(2 * scale, Decimal("0.5") * scale * max(steps, 1))
    unit: Unit = "pct" if reported_unit in {"pct", "bps", "ratio"} else "aud_m"
    from_reported = value_tolerance(unit, reported=reported_prior) + value_tolerance(
        unit, reported=reported_current
    )
    return from_steps + from_reported


_DISPATCH = {
    "nim": nim.decompose,
    "headline_earnings": headline_earnings.decompose,
    # The bridge reads whichever of its lines it's given, so stopped before tax it is the
    # same module with tax and what follows it left out of the inputs.
    "profit_before_tax": headline_earnings.decompose,
    "roe": roe.decompose,
    "cet1_ratio": cet1.decompose,
    "credit_impairment_charge": credit_impairment.decompose,
    "cost_to_income": cost_to_income.decompose,
}


def decompose_path_b(
    metric_id: str,
    prior: dict[str, Any],
    current: dict[str, Any],
    reported_prior: Decimal,
    reported_current: Decimal,
    days: int | None = None,
    pcp_days: int | None = None,
    year_days: int | None = None,
    pcp_year_days: int | None = None,
    identity: str | None = None,
) -> Decomposition:
    """Decompose a movement from the identity's components.

    ``reported_prior`` and ``reported_current`` are the issuer's own published figures, and
    the movement between them is always the target. It's never a change the identity happens
    to produce, or one the bank states against some other column.

    ``identity`` names a different identity for the same metric: ``profit_before_tax`` for
    a headline struck before tax.
    """
    spec = IDENTITIES[identity or metric_id]  # KeyError for an unknown metric, deliberately
    change_unit = spec.change_unit
    reported_unit = "pct" if change_unit == "bps" else "aud_m"
    headline_change = to_change_unit(reported_current - reported_prior, change_unit)

    usable = spec.usable_components(set(prior) & set(current))
    missing = (
        []
        if usable is not None
        else [
            component
            for component in spec.components
            if component not in prior or component not in current
        ]
    )
    if missing:
        return Decomposition(
            metric_id=metric_id,  # type: ignore[arg-type]
            change_unit=change_unit,
            headline_change=headline_change,
            drivers=[],
            residual=headline_change,
            tolerance=reconciliation_tolerance(
                change_unit, 0, reported_prior, reported_current, reported_unit
            ),
            path="none",
            missing_inputs=missing,
            notes=[
                "The accounting identity couldn't split the movement: it needs "
                + ", ".join(component_name(name) for name in sorted(missing))
                + ", and those weren't read."
            ],
        )

    ctx = Ctx(
        metric_id=metric_id,
        prior=prior,
        current=current,
        reported_prior=reported_prior,
        reported_current=reported_current,
        days=days,
        pcp_days=pcp_days,
        year_days=year_days,
        pcp_year_days=pcp_year_days,
    )
    drivers, notes = _DISPATCH[identity or metric_id](ctx)

    # Only parentless drivers tie to the movement; a nested driver decomposes a
    # different identity and would be double counted.
    contributions = [
        d.contribution for d in drivers if d.parent_id is None and d.contribution is not None
    ]
    explained = sum(contributions, Decimal(0))
    residual = headline_change - explained
    tolerance = reconciliation_tolerance(
        change_unit,
        len(contributions),
        reported_prior,
        reported_current,
        reported_unit,
    )

    return Decomposition(
        metric_id=metric_id,  # type: ignore[arg-type]
        change_unit=change_unit,
        headline_change=headline_change,
        drivers=drivers,
        residual=residual,
        tolerance=tolerance,
        path="B",
        path_b_residual=residual,
        notes=notes,
    )
