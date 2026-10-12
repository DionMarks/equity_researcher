"""CET1 ratio: CET1 capital / RWA. A stock at the balance date, so no day count."""

from __future__ import annotations

from fractions import Fraction

from ...domain.models import DriverContribution
from .common import Ctx, bps, driver, flow_effects, ratio_effects, stock_effects

# The issuer discloses its capital movement as flows, not as opening and closing
# stocks, so each value is already the period's movement.
CAPITAL_FLOWS = {
    "organic_capital": "cet1.organic_capital",
    "dividends_net_drp": "cet1.dividends",
    "buybacks": "cet1.buyback",
    "deductions": "cet1.deductions_other",
    "m_and_a": "cet1.m_and_a",
    "regulatory": "cet1.regulatory",
}

RWA_STOCKS = {
    "rwa_credit": "cet1.rwa_credit",
    "rwa_irrbb": "cet1.rwa_irrbb",
    "rwa_market": "cet1.rwa_market",
    "rwa_operational": "cet1.rwa_operational",
}

_TINY = Fraction(1, 10**12)


def decompose(ctx: Ctx) -> tuple[list[DriverContribution], list[str]]:
    cap0, cap1 = ctx.pair("cet1_capital")
    rwa0, rwa1 = ctx.pair("rwa")
    capital_effect, rwa_effect = ratio_effects(cap0, rwa0, cap1, rwa1)

    drivers: list[DriverContribution] = []
    notes: list[str] = []

    flows = (
        flow_effects(CAPITAL_FLOWS, ctx, den_prior=rwa0, den_current=rwa1)
        if ctx.has_current("organic_capital", "dividends_net_drp")
        else {}
    )
    if flows:
        flow_components = {v: k for k, v in CAPITAL_FLOWS.items()}
        drivers.extend(
            driver(
                driver_id,
                bps(value),
                from_components=(flow_components[driver_id], "rwa"),
            )
            for driver_id, value in sorted(flows.items())
        )
        gap = sum(flows.values(), Fraction(0)) - capital_effect
        if abs(gap) > _TINY:
            notes.append(
                "Disclosed capital movements do not sum to the change in CET1 capital "
                f"({bps(gap)}bp apart); the difference is reported in the residual."
            )
    else:
        drivers.append(
            driver(
                "cet1.capital",
                bps(capital_effect),
                from_components=("cet1_capital", "rwa"),
            )
        )
        # Said about what was read, not about the bank: CBA does break its capital movement
        # down, in a chart this run didn't read, and the checker rightly removed a memo
        # sentence that said CBA reports it only in aggregate.
        notes.append(
            "The capital movement is attributed in aggregate: no breakdown of it by "
            "component was read."
        )

    # Partial RWA detail cannot tie, so the split is used only when complete.
    if set(RWA_STOCKS) <= set(ctx.prior) and set(RWA_STOCKS) <= set(ctx.current):
        parts = stock_effects(
            RWA_STOCKS,
            ctx,
            other_prior=cap0,
            other_current=cap1,
            side="denominator",
            den_prior=rwa0,
            den_current=rwa1,
        )
        rwa_components = {v: k for k, v in RWA_STOCKS.items()}
        drivers.extend(
            driver(
                driver_id,
                bps(value),
                from_components=(rwa_components[driver_id], "rwa", "cet1_capital"),
            )
            for driver_id, value in sorted(parts.items())
        )
    else:
        drivers.append(driver("cet1.rwa", bps(rwa_effect), from_components=("rwa", "cet1_capital")))
        if any(c in ctx.current for c in RWA_STOCKS):
            notes.append(
                "Risk-weighted assets are attributed in aggregate: only part of a split by "
                "risk type was read, and a partial split could not reconcile."
            )

    return drivers, notes
