"""Net interest margin: annualised NII / average interest-earning assets."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

from ...domain.models import DriverContribution
from ..identities import annualise
from ..shapley import as_decimal, weighted_average_split
from .common import Ctx, bps, driver, ratio_effects

_PCT_TO_BPS = Decimal(100)


def decompose(ctx: Ctx) -> tuple[list[DriverContribution], list[str]]:
    notes: list[str] = []
    weights_current = ctx.current.get("divisional_weights")
    margins_current = ctx.current.get("divisional_margins")
    weights_prior = ctx.prior.get("divisional_weights")
    margins_prior = ctx.prior.get("divisional_margins")

    if weights_prior and margins_prior and weights_current and margins_current:
        mix, rate = weighted_average_split(
            weights_prior, margins_prior, weights_current, margins_current
        )
        mix_total = sum(mix.values(), Fraction(0))
        rate_total = sum(rate.values(), Fraction(0))
        drivers = [
            driver(
                "nim.asset_mix",
                as_decimal(mix_total) * _PCT_TO_BPS,
                label="Divisional mix",
                from_components=("divisional_weights", "divisional_margins"),
                note="Exact mix effect of a shift in balances between divisions.",
            ),
            driver(
                "nim.other",
                as_decimal(rate_total) * _PCT_TO_BPS,
                label="Divisional rate effect",
                from_components=("divisional_weights", "divisional_margins"),
                note=(
                    "Aggregate margin movement within divisions. The accounting identity "
                    "can't separate asset from deposit pricing; the bank's own walk does."
                ),
            ),
        ]
        return drivers, notes

    # Without divisional detail, the identity splits into net interest income and balance
    # growth. Neither is an economic cause, and this used to return no drivers at all
    # instead.
    #
    # That made NIM the only ratio that behaved this way. CET1 reports an aggregate capital
    # effect with a note that the split wasn't disclosed, return on equity reports earnings
    # against equity, and cost-to-income reports expense against income, all with the same
    # kind of arithmetic. Returning nothing here left the reader with a movement and a note,
    # which tells them less than a labelled pair of effects that sums to the movement.
    #
    # These drivers carry ``inputs_aggregated``, which caps them at Medium, and a note
    # saying what the identity can't separate.
    nii0, nii1 = ctx.pair("nii")
    aiea0, aiea1 = ctx.pair("aiea")
    annual0 = annualise(nii0, ctx.pcp_days or 365, ctx.pcp_year_days or 365)
    annual1 = annualise(nii1, ctx.days or 365, ctx.year_days or 365)
    income_effect, balance_effect = ratio_effects(annual0, aiea0, annual1, aiea1)
    notes.append(
        "Divisional weights and margins were unavailable, so the margin is decomposed "
        "only into the two halves of its own identity: annualised net interest income "
        "and average interest-earning assets. That is arithmetic, not cause: it does "
        "not separate asset pricing from deposit pricing, mix from rate, or Markets from "
        "the banking book. The issuer's own margin walk does, and is used where one is "
        "published; reported in aggregate here."
    )
    return [
        driver(
            "nim.income",
            bps(income_effect),
            label="Net interest income",
            from_components=("nii", "aiea"),
            note=(
                "The margin effect of annualised net interest income, holding average "
                "interest-earning assets at the prior period."
            ),
        ),
        driver(
            "nim.balance_growth",
            bps(balance_effect),
            label="Average interest-earning assets",
            from_components=("nii", "aiea"),
            note=(
                "The margin effect of balance-sheet growth, holding annualised net "
                "interest income at the prior period. Growth dilutes the margin even "
                "where income rises."
            ),
        ),
    ], notes
