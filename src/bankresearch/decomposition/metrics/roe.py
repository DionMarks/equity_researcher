"""Return on equity: annualised headline earnings / average equity."""

from __future__ import annotations

from ...domain.models import DriverContribution
from ..identities import annualise
from .common import Ctx, bps, driver

EQUITY_COMPONENTS = ("buybacks", "retained_earnings", "drp", "equity_other")

EARNINGS_LINES: tuple[tuple[str, str, int], ...] = (
    ("nii", "earn.nii", 1),
    ("other_income", "earn.other_income", 1),
    ("operating_expenses", "earn.operating_expenses", -1),
    ("credit_impairment", "earn.credit_impairment", -1),
    ("tax", "earn.tax", -1),
)


def decompose(ctx: Ctx) -> tuple[list[DriverContribution], list[str]]:
    earnings0, earnings1 = ctx.pair("headline_earnings")
    equity0, equity1 = ctx.pair("average_equity")

    days = ctx.days or 365
    pcp_days = ctx.pcp_days or 365
    annual0 = annualise(earnings0, pcp_days, ctx.pcp_year_days or 365)
    annual1 = annualise(earnings1, days, ctx.year_days or 365)

    from .common import ratio_effects

    earnings_effect, equity_effect = ratio_effects(annual0, equity0, annual1, equity1)
    sides = ("headline_earnings", "average_equity")
    drivers = [
        driver("roe.earnings", bps(earnings_effect), from_components=sides),
        driver("roe.equity", bps(equity_effect), from_components=sides),
    ]
    notes: list[str] = []

    # The earnings effect, split across the lines of the earnings bridge as detail. The
    # effect is linear in annualised earnings, so each line's part is its own annualised
    # movement at the same rate, in basis points of return like its parent. They were once
    # given in A$m under a parent in basis points, and a memo said ANZ's lower costs "added
    # 208.0 bps" to its return, for a A$208m saving that added about 59.
    rate = (1 / equity0 + 1 / equity1) / 2
    for component, driver_id, sign in EARNINGS_LINES:
        if ctx.has(component):
            before, after = ctx.pair(component)
            moved = annualise(after, days, ctx.year_days or 365) - annualise(
                before, pcp_days, ctx.pcp_year_days or 365
            )
            drivers.append(
                driver(
                    driver_id,
                    bps(moved * sign * rate),
                    parent_id="roe.earnings",
                    from_components=(component,),
                )
            )

    disclosed = [c for c in EQUITY_COMPONENTS if c in ctx.current]
    if disclosed:
        notes.append(
            "Equity movement components disclosed ("
            + ", ".join(sorted(disclosed))
            + ") but the driver taxonomy defines no ids for them, so the equity effect "
            "is reported in aggregate."
        )
    return drivers, notes
