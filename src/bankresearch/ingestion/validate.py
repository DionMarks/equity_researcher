"""Checks extracted figures against the accounting identities they must satisfy."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
from fractions import Fraction
from typing import Final

from ..decomposition.identities import (
    IMPAIRMENT_DIVISIONS,
    cet1_pct,
    cost_to_income_pct,
    credit_impairment_charge,
    headline_earnings,
    nim_pct,
    roe_pct,
)
from ..domain.calendar import fiscal_period
from ..domain.models import Basis, Fact
from ..domain.registry import get_registry
from ..domain.units import Unit, normalise_aud, to_bps, value_tolerance

__all__ = ["IdentityCheck", "ValidationOutcome", "validate_facts"]

# Identities whose failure quarantines the components but keeps the headline.
_HEADLINE_OF: dict[str, str] = {
    "cet1_ratio": "cet1_ratio",
    "nim": "nim",
    "roe": "roe",
    "cost_to_income": "cost_to_income",
    "headline_earnings": "headline_earnings",
    "profit_before_tax": "profit_before_tax",
    "credit_impairment_charge": "credit_impairment_charge",
    "rwa_sum": "rwa",
    "impairment_by_division": "credit_impairment_charge",
}


@dataclass(frozen=True, slots=True)
class IdentityCheck:
    """One identity, evaluated for one period."""

    name: str
    period: str
    reported: Decimal | None
    recomputed: Decimal | None
    gap: Decimal | None
    tolerance: Decimal
    unit: str
    passed: bool
    inputs: tuple[str, ...] = ()
    detail: str = ""

    @property
    def skipped(self) -> bool:
        return self.reported is None or self.recomputed is None


@dataclass
class ValidationOutcome:
    """Which figures may be stored, which may not, and why."""

    validated: list[Fact] = field(default_factory=list)
    quarantined: list[tuple[Fact, str]] = field(default_factory=list)
    checks: list[IdentityCheck] = field(default_factory=list)

    @property
    def failed_checks(self) -> list[IdentityCheck]:
        return [c for c in self.checks if not c.skipped and not c.passed]

    @property
    def any_input_failed(self) -> bool:
        return bool(self.failed_checks)

    def summary(self) -> str:
        ran = [c for c in self.checks if not c.skipped]
        return (
            f"{len(self.validated)} validated, {len(self.quarantined)} quarantined, "
            f"{sum(1 for c in ran if c.passed)}/{len(ran)} identity checks passed"
        )


def _ratio_tolerance(
    reported: Decimal,
    numerator: Decimal,
    denominator: Decimal,
    unit_of_parts: Unit,
    *,
    own: Decimal | None = None,
    numerator_tolerance: Decimal | None = None,
    denominator_tolerance: Decimal | None = None,
) -> Decimal:
    """Reported rounding plus the carried-through component rounding, in basis points.

    Each term defaults to the precision of the value as it arrives here, and is overridden
    with the precision the issuer actually published, which normalisation may have changed.
    """
    own = value_tolerance("pct", reported=reported) if own is None else own
    part_tolerance = (
        value_tolerance(unit_of_parts, reported=numerator)
        if numerator_tolerance is None
        else numerator_tolerance
    )
    bottom = (
        value_tolerance(unit_of_parts, reported=denominator)
        if denominator_tolerance is None
        else denominator_tolerance
    )
    propagated = (
        to_bps(reported, "pct") * (part_tolerance / abs(numerator) + bottom / abs(denominator))
        if numerator and denominator
        else Decimal(0)
    )
    return own + abs(propagated)


def _sum_tolerance(reported: Decimal, parts: list[Decimal]) -> Decimal:
    own = value_tolerance("aud_m", reported=reported)
    return own + sum((value_tolerance("aud_m", reported=part) for part in parts), Decimal(0))


def _comparable(bases: list[Basis]) -> bool:
    """Whether these figures are on one basis, so they can be checked against each other.

    Notable items must agree exactly: a ratio that excludes them and a numerator that
    includes them are different quantities. The earnings basis must agree where both figures
    state one. ``n/a`` means the figure has no earnings basis to disagree about (a balance
    or a capital ratio, say), so it's comparable with either.
    """
    if len({basis.ex_notable_items for basis in bases}) > 1:
        return False
    stated = {basis.earnings_basis for basis in bases} - {"n/a"}
    return len(stated) <= 1


def _as_decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def validate_facts(facts: list[Fact], bank_id: str, periods: tuple[str, ...]) -> ValidationOutcome:
    """Run every applicable identity, then decide what may be stored.

    A component can be held twice, once as the issuer's headline and once excluding notable
    items. The identities are run over each reading on its own: the headline set first,
    using a variant only where the headline has no reading of that component, which is how
    every figure was checked before variants were kept; then the variant set the same way
    round, keeping only the checks that involve a variant. A check that fails can only
    quarantine the readings it was run on, so a variant that doesn't add up never costs
    the headline its figure, or the other way round.
    """
    profile = get_registry().bank(bank_id)
    readings: dict[tuple[str, str, bool], Fact] = {}
    precision: dict[tuple[str, str, bool], Decimal] = {}
    for fact in facts:
        key = (fact.period, fact.component, fact.basis.ex_notable_items)
        # Normalised to millions, because a pack states its income statement in $M and
        # its average balance sheet in $B and the identities compare the two.
        value, unit = normalise_aud(fact.value, fact.unit)
        readings[key] = fact.model_copy(update={"value": value, "unit": unit})
        # Each figure's precision is taken before normalisation, because converting first
        # destroys it: NAB publishes risk-weighted assets as "$426.4bn", known to $50m,
        # and as 426,400.0 that reads as $0.05m.
        precision[key] = value_tolerance(fact.unit, reported=fact.value)

    outcome = ValidationOutcome()
    quarantine_reasons: dict[tuple[str, str, bool], str] = {}

    for period in periods:
        resolved = fiscal_period(period, profile.fy_end_month)
        days, year_days = resolved.days, resolved.year_days
        headline_set = _reading_set(readings, period, prefer_variant=False)
        variant_set = _reading_set(readings, period, prefer_variant=True)
        for values, is_variant in ((headline_set, False), (variant_set, True)):
            if not values:
                continue
            checks = _checks_for(
                values,
                days,
                period,
                year_days=year_days,
                precision={
                    (period, component): precision[(period, component, fact.basis.ex_notable_items)]
                    for component, fact in values.items()
                },
            )
            for check in checks:
                if is_variant and not any(
                    values[name] is not headline_set.get(name)
                    for name in check.inputs
                    if name in values
                ):
                    continue  # the same readings the headline set already checked
                if is_variant:
                    check = replace(
                        check,
                        detail=f"{_VARIANT_DETAIL}; {check.detail}"
                        if check.detail
                        else _VARIANT_DETAIL,
                    )
                outcome.checks.append(check)
                if check.skipped or check.passed:
                    continue
                # A bank nobody configured has had none of its pages checked, so a ratio read
                # from another table than its parts may be a division's: Macquarie's deck
                # gives its Banking and Financial Services cost-to-income of 51% on a page the
                # reader took for the group's, against group costs and income on p.7 that make
                # 65.5%. Then nothing in the failed identity is trusted. A sum that misses is
                # different: it's far more often a line the bridge doesn't know (BNK's
                # goodwill impairment) than a division's total, so the printed total stands.
                headline = _HEADLINE_OF.get(check.name)
                if (
                    profile.discovered
                    and headline in values
                    and _read_apart(check, values, headline)
                ):
                    headline = None
                for component in check.inputs:
                    if component == headline or component not in values:
                        continue  # the published figure stands on its own warrant
                    reading = values[component]
                    quarantine_reasons[(period, component, reading.basis.ex_notable_items)] = (
                        f"{check.name} identity for {period}"
                        + (" excluding notable items" if is_variant else "")
                        + f" is off by {check.gap} {check.unit} against a tolerance of "
                        f"{check.tolerance}"
                    )

    for fact in facts:
        reason = quarantine_reasons.get((fact.period, fact.component, fact.basis.ex_notable_items))
        if reason is None:
            outcome.validated.append(fact.model_copy(update={"validated": True}))
        else:
            outcome.quarantined.append((fact, reason))
    return outcome


_VARIANT_DETAIL = "excluding notable items"
#: The identities that check a published ratio against its parts.
_RATIOS: Final = frozenset({"cet1_ratio", "nim", "roe", "cost_to_income"})


def _read_apart(check: IdentityCheck, values: dict[str, Fact], headline: str) -> bool:
    """Whether a failed ratio was read from another table than every one of its parts."""
    if check.name not in _RATIOS:
        return False
    tables = {values[name].table_id for name in check.inputs if name in values and name != headline}
    return bool(tables) and values[headline].table_id not in tables


def _reading_set(
    readings: dict[tuple[str, str, bool], Fact], period: str, *, prefer_variant: bool
) -> dict[str, Fact]:
    """One reading per component for a period: the preferred treatment where there is
    one, otherwise the other."""
    chosen: dict[str, Fact] = {}
    for (fact_period, component, ex_notable_items), fact in readings.items():
        if fact_period != period:
            continue
        if component not in chosen or ex_notable_items is prefer_variant:
            chosen[component] = fact
    return chosen


#: Basis points a return on equity may differ from its identity by for the numerator
#: adjustments issuers make and don't print beside it.
_ROE_DEFINITION_ALLOWANCE: Final = Decimal("1")


#: The lines of the earnings bridge down to profit after tax, tax last so the bridge to
#: profit before tax is the same lines without it.
_BRIDGE_AFTER_TAX: Final = ("nii", "other_income", "operating_expenses", "credit_impairment", "tax")


def _notable_items_in_bridge(values: dict[str, Fact]) -> Decimal:
    """Notable items, where they're a line of the bridge's own table, else nothing.

    Judo prints its FY24 non-recurring costs between underlying and statutory profit (2025
    annual report p.44); Westpac's notable items run through its other lines, and adding
    them again would count them twice.
    """
    notable = values.get("notable_items")
    in_bridge = (
        notable is not None
        and "nii" in values
        and notable.table_id is not None
        and notable.table_id == values["nii"].table_id
    )
    return notable.value if notable is not None and in_bridge else Decimal(0)


def _checks_for(
    values: dict[str, Fact],
    days: int,
    period: str,
    precision: dict[tuple[str, str], Decimal] | None = None,
    *,
    year_days: int = 365,
) -> list[IdentityCheck]:
    """Every identity that has all its inputs, evaluated for one period.

    An identity missing an input is skipped, not failed, because having nothing to check is
    a different finding from a check that didn't hold. An identity whose inputs aren't all
    on the same basis is skipped for the same reason: the figures aren't comparable, so
    their disagreement isn't evidence of an error. Westpac publishes its expense-to-income
    ratio excluding notable items and its operating income including them, and checking one
    against the other reports a 90bp error in two figures that are both exactly as the bank
    printed them.
    """
    checks: list[IdentityCheck] = []
    tolerances = precision or {}

    def tolerance_of(component: str, fallback: Decimal) -> Decimal:
        return tolerances.get((period, component), fallback)

    def need(*components: str) -> dict[str, Decimal] | None:
        """The components' values, or ``None`` if any is missing, zero-valued where it
        must divide, or stated on a basis the others are not."""
        found: dict[str, Decimal] = {}
        bases: list[Basis] = []
        for component in components:
            fact = values.get(component)
            if fact is None:
                return None
            found[component] = fact.value
            bases.append(fact.basis)
        if not _comparable(bases):
            return None
        return found

    def ratio_check(
        name: str,
        reported_component: str,
        numerator: str,
        denominator: str,
        recomputed: Decimal,
        parts: dict[str, Decimal],
        allowance: Decimal = Decimal(0),
    ) -> None:
        reported = parts[reported_component]
        gap = to_bps(recomputed, "pct") - to_bps(reported, "pct")
        tolerance = allowance + _ratio_tolerance(
            reported,
            parts[numerator],
            parts[denominator],
            values[numerator].unit,
            own=tolerance_of(reported_component, value_tolerance("pct", reported=reported)),
            numerator_tolerance=tolerance_of(
                numerator, value_tolerance(values[numerator].unit, reported=parts[numerator])
            ),
            denominator_tolerance=tolerance_of(
                denominator, value_tolerance(values[denominator].unit, reported=parts[denominator])
            ),
        )
        checks.append(
            IdentityCheck(
                name=name,
                period=period,
                reported=reported,
                recomputed=recomputed,
                gap=gap.quantize(Decimal("0.001")),
                tolerance=tolerance.quantize(Decimal("0.001")),
                unit="bps",
                passed=abs(gap) <= tolerance,
                inputs=(reported_component, numerator, denominator),
            )
        )

    # CET1 = capital / RWA. A stock at the balance date, so no day count.
    parts = need("cet1_ratio", "cet1_capital", "rwa")
    if parts is not None and parts["rwa"]:
        ratio_check(
            "cet1_ratio",
            "cet1_ratio",
            "cet1_capital",
            "rwa",
            _as_decimal(cet1_pct(parts["cet1_capital"], parts["rwa"])),
            parts,
        )

    # NIM = annualised net interest income / average interest-earning assets.
    parts = need("nim", "nii", "aiea")
    if parts is not None and parts["aiea"]:
        ratio_check(
            "nim",
            "nim",
            "nii",
            "aiea",
            _as_decimal(nim_pct(parts["nii"], parts["aiea"], days, year_days)),
            parts,
        )

    # ROE = annualised headline earnings / average equity.
    parts = need("roe", "headline_earnings", "average_equity")
    if parts is not None and parts["average_equity"]:
        ratio_check(
            "roe",
            "roe",
            "headline_earnings",
            "average_equity",
            _as_decimal(
                roe_pct(parts["headline_earnings"], parts["average_equity"], days, year_days)
            ),
            parts,
            # Issuers take small items off the numerator before dividing: Westpac's
            # return on equity uses net profit less distributions it doesn't name in the
            # same table, and its FY24 and FY25 figures recompute 0.7bp high, against a
            # rounding tolerance of 0.6bp. A wrong figure misses by far more: CBA's
            # FY25 average equity read from the wrong table missed by 19bp.
            allowance=_ROE_DEFINITION_ALLOWANCE,
        )

    # Cost-to-income = operating expenses / operating income.
    parts = need("cost_to_income", "operating_expenses", "operating_income")
    if parts is not None and parts["operating_income"]:
        ratio_check(
            "cost_to_income",
            "cost_to_income",
            "operating_expenses",
            "operating_income",
            _as_decimal(cost_to_income_pct(parts["operating_expenses"], parts["operating_income"])),
            parts,
        )

    # The earnings bridge, after tax and, where the issuer prints one, before it.
    for total, bridge in (
        ("headline_earnings", _BRIDGE_AFTER_TAX),
        ("profit_before_tax", _BRIDGE_AFTER_TAX[:-1]),
    ):
        parts = need(total, *bridge)
        if parts is None:
            continue
        optional = values.get("minority_interests") if total == "headline_earnings" else None
        recomputed = _as_decimal(
            headline_earnings(
                nii=parts["nii"],
                other_income=parts["other_income"],
                operating_expenses=parts["operating_expenses"],
                credit_impairment=parts["credit_impairment"],
                tax=parts.get("tax", Decimal(0)),
                minority_interests=optional.value if optional else Decimal(0),
                notable_items=_notable_items_in_bridge(values),
            )
        )
        reported = parts[total]
        gap = recomputed - reported
        tolerance = _sum_tolerance(reported, [parts[name] for name in bridge])
        checks.append(
            IdentityCheck(
                total,
                period,
                reported,
                recomputed,
                gap.quantize(Decimal("0.01")),
                tolerance,
                "aud_m",
                abs(gap) <= tolerance,
                (total, *bridge),
            )
        )

    # Impairment = individually assessed + collectively assessed, less any recoveries the
    # issuer prints on a line of their own.
    split: tuple[str, ...] = ("impairment_individual", "impairment_collective")
    if "impairment_recoveries" in values:
        split = (*split, "impairment_recoveries")
    parts = need("credit_impairment_charge", *split)
    if parts is not None:
        recomputed = _as_decimal(
            credit_impairment_charge(
                parts[split[0]],
                parts[split[1]],
                parts.get("impairment_recoveries", Decimal(0)),
            )
        )
        reported = parts["credit_impairment_charge"]
        gap = recomputed - reported
        tolerance = _sum_tolerance(reported, [parts[name] for name in split])
        checks.append(
            IdentityCheck(
                "credit_impairment_charge",
                period,
                reported,
                recomputed,
                gap.quantize(Decimal("0.01")),
                tolerance,
                "aud_m",
                abs(gap) <= tolerance,
                ("credit_impairment_charge", *split),
            )
        )

    # The charge by division must sum to the group's charge (ADR-038). Only a complete
    # split is ever stored, so a missing division skips the check rather than failing it.
    parts = need("credit_impairment_charge", *IMPAIRMENT_DIVISIONS)
    if parts is not None:
        recomputed = sum((parts[name] for name in IMPAIRMENT_DIVISIONS), Decimal(0))
        reported = parts["credit_impairment_charge"]
        gap = recomputed - reported
        tolerance = _sum_tolerance(reported, [parts[name] for name in IMPAIRMENT_DIVISIONS])
        checks.append(
            IdentityCheck(
                "impairment_by_division",
                period,
                reported,
                recomputed,
                gap.quantize(Decimal("0.01")),
                tolerance,
                "aud_m",
                abs(gap) <= tolerance,
                ("credit_impairment_charge", *IMPAIRMENT_DIVISIONS),
            )
        )

    # RWA by risk type must sum to total RWA, or the split cannot be used.
    rwa_parts = ("rwa_credit", "rwa_irrbb", "rwa_market", "rwa_operational")
    parts = need("rwa", *rwa_parts)
    if parts is not None:
        recomputed = sum((parts[name] for name in rwa_parts), Decimal(0))
        reported = parts["rwa"]
        gap = recomputed - reported
        tolerance = _sum_tolerance(reported, [parts[name] for name in rwa_parts])
        checks.append(
            IdentityCheck(
                "rwa_sum",
                period,
                reported,
                recomputed,
                gap.quantize(Decimal("0.01")),
                tolerance,
                "aud_m",
                abs(gap) <= tolerance,
                ("rwa", *rwa_parts),
            )
        )

    return checks
