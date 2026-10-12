"""The headline earnings bridge, and the three-way split of net interest income."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

from ...domain.models import DriverContribution
from ..identities import nii_from_share
from ..shapley import as_decimal, shapley_exact
from .common import Ctx, driver

# (component, driver id, sign). Expenses, impairment, tax and non-controlling
# interests are stated as positive magnitudes and reduce earnings.
BRIDGE: tuple[tuple[str, str, int], ...] = (
    ("nii", "earn.nii", 1),
    ("other_income", "earn.other_income", 1),
    ("operating_expenses", "earn.operating_expenses", -1),
    ("credit_impairment", "earn.credit_impairment", -1),
    ("tax", "earn.tax", -1),
    ("minority_interests", "earn.minority_interests", -1),
    ("notable_items", "earn.notable_items", 1),
    ("discontinued_ops", "earn.discontinued_ops", 1),
)

NII_FACTORS = {
    "days": "earn.nii_day_count",
    "nim": "earn.nii_margin",
    "aiea": "earn.nii_volume",
}


def _nii(x: dict[str, Fraction]) -> Fraction:
    """Calls the shared identity instead of repeating it.

    ``identities.py`` is meant to be the one place this arithmetic lives. This used to be a
    local copy of :func:`nii_from_margin`. The two were identical, but nothing would have
    caught it if one had been fixed and the other not.
    """
    return nii_from_share(x["nim"], x["aiea"], x["days"])


def decompose(ctx: Ctx) -> tuple[list[DriverContribution], list[str]]:
    drivers: list[DriverContribution] = []
    notes: list[str] = []

    for component, driver_id, sign in BRIDGE:
        if not ctx.has(component):
            continue
        drivers.append(
            driver(
                driver_id,
                as_decimal(ctx.delta(component) * sign),
                from_components=(component,),
            )
        )

    if ctx.has("nim", "aiea") and ctx.days and ctx.pcp_days:
        nim0, nim1 = ctx.pair("nim")
        aiea0, aiea1 = ctx.pair("aiea")
        # "days" is the share of its year each period covers, so the day-count effect also
        # carries a 29 February in either year.
        share0 = Fraction(ctx.pcp_days, ctx.pcp_year_days or 365)
        share1 = Fraction(ctx.days, ctx.year_days or 365)
        parts = shapley_exact(
            _nii,
            {"days": share0, "nim": nim0, "aiea": aiea0},
            {"days": share1, "nim": nim1, "aiea": aiea1},
        )
        drivers.extend(
            driver(
                NII_FACTORS[factor],
                as_decimal(value),
                parent_id="earn.nii",
                from_components=("nim", "aiea"),
            )
            for factor, value in sorted(parts.items(), key=lambda kv: NII_FACTORS[kv[0]])
        )
        reconstructed = _nii({"days": share1, "nim": nim1, "aiea": aiea1})
        gap = reconstructed - Fraction(ctx.current["nii"])
        if abs(as_decimal(gap)) > Decimal("1"):
            size = abs(as_decimal(gap)).quantize(Decimal("0.1"))
            side = "above" if gap > 0 else "below"
            notes.append(
                f"Net interest income rebuilt from margin and balances is A${size}m {side} "
                "the reported figure, so its split into volume, margin and day count is "
                "indicative."
            )
    else:
        # Named for what's missing: BNK's margin is held and its average balances aren't, and
        # "margin and average interest-earning assets unavailable" was half wrong.
        missing = [
            name
            for name, component in (("margin", "nim"), ("average interest-earning assets", "aiea"))
            if not ctx.has(component)
        ] or ["day count"]
        notes.append(
            f"{' and '.join(missing).capitalize()} not held for both periods, so net interest "
            "income is not split into volume, margin and day count."
        )

    return drivers, notes
