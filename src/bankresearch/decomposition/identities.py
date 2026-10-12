"""The accounting identity behind each metric, and what its residual means."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from fractions import Fraction
from typing import Final, Literal

__all__ = [
    "DAYS_IN_YEAR",
    "IDENTITIES",
    "IMPAIRMENT_DIVISIONS",
    "IdentitySpec",
    "ResidualClass",
    "annualise",
    "cet1_pct",
    "classify_residual",
    "cost_to_income_pct",
    "credit_impairment_charge",
    "headline_earnings",
    "identity_for",
    "nii_from_share",
    "nim_pct",
    "roe_pct",
]

DAYS_IN_YEAR: Final = 365
_HUNDRED: Final = Fraction(100)

Number = Decimal | Fraction | int


def _f(value: Number) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def annualise(flow: Number, days: int, year_days: int = DAYS_IN_YEAR) -> Fraction:
    """Scale a period flow to an annual rate.

    ``days`` is the period's own inclusive day count and ``year_days`` the length of the
    financial year it belongs to, 366 in a year that takes in a 29 February. A full year
    therefore annualises by exactly one, leap year or not.
    """
    if days <= 0 or year_days <= 0:
        raise ValueError("days must be positive")
    return _f(flow) * year_days / days


# --- ratios ----------------------------------------------------------------


def nim_pct(nii: Number, aiea: Number, days: int, year_days: int = DAYS_IN_YEAR) -> Fraction:
    """Net interest margin, in percent: annualised NII over average interest-earning assets."""
    if _f(aiea) == 0:
        raise ZeroDivisionError("average interest-earning assets must not be zero")
    return annualise(nii, days, year_days) / _f(aiea) * _HUNDRED


def cet1_pct(capital: Number, rwa: Number) -> Fraction:
    """CET1 ratio, in percent. A stock at the balance date: no day count."""
    if _f(rwa) == 0:
        raise ZeroDivisionError("risk-weighted assets must not be zero")
    return _f(capital) / _f(rwa) * _HUNDRED


def roe_pct(
    earnings: Number, average_equity: Number, days: int, year_days: int = DAYS_IN_YEAR
) -> Fraction:
    """Return on equity, in percent: annualised earnings over average equity."""
    if _f(average_equity) == 0:
        raise ZeroDivisionError("average equity must not be zero")
    return annualise(earnings, days, year_days) / _f(average_equity) * _HUNDRED


def cost_to_income_pct(expenses: Number, operating_income: Number) -> Fraction:
    """Cost-to-income ratio, in percent. Both legs span the same period, so no day count."""
    if _f(operating_income) == 0:
        raise ZeroDivisionError("operating income must not be zero")
    return _f(expenses) / _f(operating_income) * _HUNDRED


# --- signed sums -----------------------------------------------------------


def headline_earnings(
    nii: Number,
    other_income: Number,
    operating_expenses: Number,
    credit_impairment: Number,
    tax: Number,
    minority_interests: Number = 0,
    notable_items: Number = 0,
    discontinued_ops: Number = 0,
) -> Fraction:
    """The earnings bridge, in AUD m.

    Sign convention: expenses, impairment, tax and non-controlling interests
    are positive magnitudes that reduce earnings, so a bigger charge is a negative
    contribution. Notable items are signed as the issuer reports them, because when the
    headline includes them, the change in notable items is itself a driver line.
    """
    return (
        _f(nii)
        + _f(other_income)
        - _f(operating_expenses)
        - _f(credit_impairment)
        - _f(tax)
        - _f(minority_interests)
        + _f(notable_items)
        + _f(discontinued_ops)
    )


def credit_impairment_charge(
    individual: Number, collective: Number, recoveries: Number = 0
) -> Fraction:
    """Total P&L charge, in AUD m. Additive; a net write-back is negative.

    ``recoveries`` is for an issuer that prints them on their own line rather than inside
    the individually assessed charge, signed as a reduction of the charge (Westpac:
    173 + 388 - 118 = 443).
    """
    return _f(individual) + _f(collective) + _f(recoveries)


def nii_from_margin(
    nim_percent: Number, aiea: Number, days: int, year_days: int = DAYS_IN_YEAR
) -> Fraction:
    """Net interest income implied by margin, balances and day count.

    The inverse of :func:`nim_pct`. Shapley is applied to this identity to split a change in
    NII into volume, margin and day-count effects.
    """
    return nii_from_share(nim_percent, aiea, Fraction(days, year_days))


def nii_from_share(nim_percent: Number, aiea: Number, share_of_year: Number) -> Fraction:
    """Net interest income implied by margin, balances and the share of a year the period
    covers. The day-count effect is a change in that share, which a leap year moves too."""
    return _f(nim_percent) / _HUNDRED * _f(aiea) * _f(share_of_year)


# --- identity registry -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class IdentitySpec:
    """What Path B needs in order to run for one metric."""

    metric_id: str
    kind: Literal["ratio", "sum"]
    change_unit: Literal["bps", "aud_m"]
    components: tuple[str, ...]
    """Components that MUST be resolved before the identity can be evaluated."""
    optional_components: tuple[str, ...] = ()
    needs_days: bool = False
    numerator: tuple[str, ...] = ()
    denominator: tuple[str, ...] = ()
    alternatives: tuple[tuple[str, ...], ...] = ()
    """Other complete sets of components the identity can be evaluated over, tried in order
    when ``components`` aren't all there. Each must be present whole: a split with a part
    missing would put the missing part in the residual and still look like an answer."""

    def usable_components(self, available: set[str]) -> tuple[str, ...] | None:
        """The first complete set of components among those available, or ``None``."""
        for candidate in (self.components, *self.alternatives):
            if set(candidate) <= available:
                return candidate
        return None


#: The five segments a credit impairment charge can be split by when the issuer doesn't
#: disclose it as individually and collectively assessed (DESIGN.md ADR-038). CBA reports
#: its loan impairment expense this way and no other.
IMPAIRMENT_DIVISIONS: Final[tuple[str, ...]] = (
    "impairment_div_retail",
    "impairment_div_business",
    "impairment_div_institutional",
    "impairment_div_new_zealand",
    "impairment_div_other",
)


IDENTITIES: Final[dict[str, IdentitySpec]] = {
    "nim": IdentitySpec(
        metric_id="nim",
        kind="ratio",
        change_unit="bps",
        components=("nii", "aiea"),
        optional_components=("divisional_weights", "divisional_margins"),
        needs_days=True,
        numerator=("nii",),
        denominator=("aiea",),
    ),
    "headline_earnings": IdentitySpec(
        metric_id="headline_earnings",
        kind="sum",
        change_unit="aud_m",
        components=("nii", "other_income", "operating_expenses", "credit_impairment", "tax"),
        optional_components=(
            "minority_interests",
            "notable_items",
            "discontinued_ops",
            "nim",
            "aiea",
        ),
        needs_days=True,
    ),
    # The same bridge, stopping at profit before tax, for a bank that leads with it (Judo,
    # from its 2026 half year). Tax, the non-controlling interests and any discontinued
    # result come after that line, so none of them is a driver of it (DESIGN.md ADR-062).
    "profit_before_tax": IdentitySpec(
        metric_id="profit_before_tax",
        kind="sum",
        change_unit="aud_m",
        components=("nii", "other_income", "operating_expenses", "credit_impairment"),
        optional_components=("notable_items", "nim", "aiea"),
        needs_days=True,
    ),
    "roe": IdentitySpec(
        metric_id="roe",
        kind="ratio",
        change_unit="bps",
        components=("headline_earnings", "average_equity"),
        optional_components=("buybacks", "retained_earnings", "drp", "equity_other"),
        needs_days=True,
        numerator=("headline_earnings",),
        denominator=("average_equity",),
    ),
    "cet1_ratio": IdentitySpec(
        metric_id="cet1_ratio",
        kind="ratio",
        change_unit="bps",
        components=("cet1_capital", "rwa"),
        optional_components=(
            "rwa_credit",
            "rwa_irrbb",
            "rwa_market",
            "rwa_operational",
            "organic_capital",
            "dividends_net_drp",
            "buybacks",
            "deductions",
        ),
        numerator=("cet1_capital",),
        denominator=("rwa",),
    ),
    "credit_impairment_charge": IdentitySpec(
        metric_id="credit_impairment_charge",
        kind="sum",
        change_unit="aud_m",
        components=("impairment_individual", "impairment_collective"),
        optional_components=(
            "impairment_recoveries",
            "collective_volume",
            "collective_quality",
            "collective_outlook",
            "overlays",
            "gross_loans",
        ),
        alternatives=(IMPAIRMENT_DIVISIONS,),
    ),
    "cost_to_income": IdentitySpec(
        metric_id="cost_to_income",
        kind="ratio",
        change_unit="bps",
        components=("operating_expenses", "operating_income"),
        optional_components=(
            "nii",
            "other_income",
            "exp_wages_inflation",
            "exp_investment",
            "exp_productivity",
            "exp_notable_items",
        ),
        numerator=("operating_expenses",),
        denominator=("operating_income",),
    ),
}


def identity_for(metric_id: str, *, pre_tax: bool = False) -> IdentitySpec | None:
    """The identity a request's movement is split by: its metric's own, or for a headline
    struck before tax, the bridge that stops there."""
    if pre_tax and metric_id == "headline_earnings":
        return IDENTITIES["profit_before_tax"]
    return IDENTITIES.get(metric_id)


class ResidualClass(StrEnum):
    """What an unexplained gap between reported and recomputed means."""

    DEFINITION_OR_ROUNDING = "definition or rounding"
    UNEXPLAINED = "unexplained"


def classify_residual(residual: Number, tolerance: Number) -> ResidualClass:
    """Label a residual. At or within tolerance is rounding; beyond it is a finding."""
    if abs(_f(residual)) <= _f(tolerance):
        return ResidualClass.DEFINITION_OR_ROUNDING
    return ResidualClass.UNEXPLAINED
