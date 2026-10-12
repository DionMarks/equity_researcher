"""Cost-to-income: operating expenses / operating income."""

from __future__ import annotations

from ...domain.models import DriverContribution
from .common import Ctx, bps, driver, ratio_effects, stock_effects

EXPENSE_CATEGORIES = {
    "exp_wages_inflation": "cti.exp_wages_inflation",
    "exp_investment": "cti.exp_investment",
    "exp_productivity": "cti.exp_productivity",
    "exp_notable_items": "cti.exp_notable_items",
}

INCOME_COMPONENTS = {"nii": "cti.inc_nii", "other_income": "cti.inc_other_income"}


def decompose(ctx: Ctx) -> tuple[list[DriverContribution], list[str]]:
    exp0, exp1 = ctx.pair("operating_expenses")
    inc0, inc1 = ctx.pair("operating_income")
    expense_effect, income_effect = ratio_effects(exp0, inc0, exp1, inc1)

    sides = ("operating_expenses", "operating_income")
    drivers = [
        driver("cti.expense_growth", bps(expense_effect), from_components=sides),
        driver("cti.income_growth", bps(income_effect), from_components=sides),
    ]
    notes: list[str] = []

    # The taxonomy declares these as children of the two aggregate lines, so they
    # are detail beneath them rather than drivers in their own right.
    categories = stock_effects(
        EXPENSE_CATEGORIES,
        ctx,
        other_prior=inc0,
        other_current=inc1,
        side="numerator",
        den_prior=inc0,
        den_current=inc1,
    )
    expense_components = {v: k for k, v in EXPENSE_CATEGORIES.items()}
    drivers.extend(
        driver(
            driver_id,
            bps(value),
            parent_id="cti.expense_growth",
            from_components=(expense_components[driver_id], *sides),
        )
        for driver_id, value in sorted(categories.items())
    )

    components = stock_effects(
        INCOME_COMPONENTS,
        ctx,
        other_prior=exp0,
        other_current=exp1,
        side="denominator",
        den_prior=inc0,
        den_current=inc1,
    )
    income_components = {v: k for k, v in INCOME_COMPONENTS.items()}
    drivers.extend(
        driver(
            driver_id,
            bps(value),
            parent_id="cti.income_growth",
            from_components=(income_components[driver_id], *sides),
        )
        for driver_id, value in sorted(components.items())
    )

    if not categories:
        notes.append(
            "No breakdown of expenses by category was read, so the expense effect is one "
            "figure, not split by category."
        )
    return drivers, notes
