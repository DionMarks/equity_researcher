"""Credit impairment charge: individually assessed plus collectively assessed."""

from __future__ import annotations

from fractions import Fraction

from ...domain.models import DriverContribution
from ..identities import IMPAIRMENT_DIVISIONS
from ..shapley import as_decimal
from .common import Ctx, driver

COLLECTIVE_SPLIT = {
    "collective_volume": "cic.collective_volume",
    "collective_quality": "cic.collective_quality",
    "collective_outlook": "cic.collective_outlook",
    "overlays": "cic.overlays",
}


def decompose(ctx: Ctx) -> tuple[list[DriverContribution], list[str]]:
    split = ("impairment_individual", "impairment_collective")
    held = set(ctx.prior) & set(ctx.current)
    if not set(split) <= held and set(IMPAIRMENT_DIVISIONS) <= held:
        return _by_division(ctx)

    # The individually assessed driver is defined net of write-backs and recoveries, so an
    # issuer that prints recoveries on their own line has them counted here.
    individual: tuple[str, ...] = ("impairment_individual",)
    if "impairment_recoveries" in held:
        individual = (*individual, "impairment_recoveries")
    drivers = [
        driver(
            "cic.individual",
            as_decimal(sum((ctx.delta(component) for component in individual), Fraction(0))),
            from_components=individual,
        )
    ]
    notes: list[str] = []
    if "impairment_recoveries" in held:
        notes.append(
            "Recoveries are printed on their own line and counted with the individually "
            "assessed charge, which is defined net of them."
        )

    if set(COLLECTIVE_SPLIT) <= set(ctx.prior) and set(COLLECTIVE_SPLIT) <= set(ctx.current):
        drivers.extend(
            driver(
                driver_id,
                as_decimal(ctx.delta(component)),
                from_components=(component,),
            )
            for component, driver_id in sorted(COLLECTIVE_SPLIT.items(), key=lambda kv: kv[1])
        )
    else:
        drivers.append(
            driver(
                "cic.collective",
                as_decimal(ctx.delta("impairment_collective")),
                from_components=("impairment_collective",),
            )
        )
        notes.append(
            "The collective charge is attributed in aggregate: no split of it into volume, "
            "quality, outlook and overlays was read."
        )
    return drivers, notes


def _by_division(ctx: Ctx) -> tuple[list[DriverContribution], list[str]]:
    """Each division's movement is its contribution, as the lines of a sum always are."""
    drivers = [
        driver(
            "cic." + component.removeprefix("impairment_"),
            as_decimal(ctx.delta(component)),
            from_components=(component,),
        )
        for component in IMPAIRMENT_DIVISIONS
    ]
    return drivers, [
        "No split of the charge into individually and collectively assessed was read, so it's "
        "split by division: that says where the charge rose, not why."
    ]
