"""Step 1: resolve the request against the issuer's calendar and labels."""

from __future__ import annotations

import uuid
from calendar import monthrange
from dataclasses import dataclass
from datetime import date

from ...domain.calendar import AmbiguousPeriodError, PeriodKind, fiscal_period, parse_period
from ...domain.models import AbstainReason, RequestSpec
from ...domain.registry import (
    BankProfile,
    MetricDefinition,
    Registry,
    UnknownBankError,
    UnknownMetricError,
    get_registry,
)

__all__ = ["AbstainSignal", "default_as_of", "expected_release_date", "resolve_request", "today"]


@dataclass(frozen=True, slots=True)
class AbstainSignal(Exception):
    """Raised to end a run with a reason code and an explanation."""

    reason: AbstainReason
    detail: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.reason}: {self.detail}"


def expected_release_date(bank: BankProfile, period_label: str) -> date:
    """The last day of the month this issuer usually releases this period's result.

    A month-end rather than a specific day, so a release published anywhere in
    that month is visible to the as-of filter. Half-year figures are released in
    the issuer's half-year month; second-half figures appear inside the full-year
    result, so they share the full-year release month.
    """
    period = fiscal_period(period_label, bank.fy_end_month)
    key = "H1" if period.kind is PeriodKind.H1 else "FY"
    month = bank.results_release_months.get(key)
    if month is None:
        # No configured release month: fall back to two months after the close,
        # which is the shortest reporting lag any of these issuers achieves.
        month = (period.end.month + 1) % 12 + 1
    year = period.end.year + (1 if month < period.end.month else 0)
    return date(year, month, monthrange(year, month)[1])


def today() -> date:
    """Today's date, in one place so a test can set it."""
    return date.today()


def default_as_of(bank: BankProfile, period_label: str) -> date:
    """The date a question is answered as at when it doesn't give one: the end of the month
    the result is usually released, or today if that month hasn't ended. A later date would
    claim to know what's published after today: Bank of Queensland's FY26 question was dated
    31 October on 12 October, and paid to search for a result that wasn't out."""
    return min(expected_release_date(bank, period_label), today())


def resolve_request(
    bank: str,
    period: str,
    metric: str,
    as_of: date | None = None,
    registry: Registry | None = None,
    run_id: str | None = None,
) -> RequestSpec:
    """Turn free-form arguments into a ``RequestSpec``, or abstain."""
    reg = registry or get_registry()

    try:
        profile: BankProfile = reg.bank(bank)
    except UnknownBankError as exc:
        raise AbstainSignal("BANK_UNKNOWN", str(exc)) from exc

    try:
        period_label = parse_period(period)
    except AmbiguousPeriodError as exc:
        raise AbstainSignal("AMBIGUOUS_INPUT", str(exc)) from exc

    try:
        definition: MetricDefinition = reg.metric(metric)
    except UnknownMetricError as exc:
        raise AbstainSignal("METRIC_OUT_OF_SCOPE", str(exc)) from exc

    if definition.metric_id in profile.not_reported:
        raise AbstainSignal(
            "METRIC_NOT_REPORTED",
            f"{profile.bank_id} does not report {definition.display_name}. "
            f"{profile.not_reported[definition.metric_id]} This is a different finding "
            "from missing evidence: it will still be true after more documents are "
            "ingested.",
        )

    current = fiscal_period(period_label, profile.fy_end_month)
    prior = current.pcp()

    rule = definition.label_for(profile, period_label)
    headline_label = rule.label if rule is not None else None

    return RequestSpec(
        run_id=run_id or uuid.uuid4().hex[:12],
        bank_id=profile.bank_id,
        metric_id=definition.metric_id,
        period=current.label,
        comparison_period=prior.label,
        period_start=current.start,
        period_end=current.end,
        comparison_start=prior.start,
        comparison_end=prior.end,
        headline_label=headline_label,
        pre_tax=rule is not None and rule.pre_tax and definition.metric_id == "headline_earnings",
        as_of=as_of or default_as_of(profile, current.label),
    )


def caveats_for(spec: RequestSpec, registry: Registry | None = None) -> list[str]:
    """Caveats the resolved request already implies.

    A label change and a definition break between the two periods are both known before any
    document is read, and both change how the answer should be read, so they're attached
    here rather than discovered later.
    """
    reg = registry or get_registry()
    profile = reg.bank(spec.bank_id)
    definition = reg.metric(spec.metric_id)
    out: list[str] = []

    rule = definition.label_for(profile, spec.period)
    prior_rule = definition.label_for(profile, spec.comparison_period)
    if rule is not None and rule.caveat:
        out.append(rule.caveat)
    # A move to a pre-tax headline isn't a change of measure within the comparison: both
    # periods are read before tax, and the rule's own caveat says so.
    if (
        rule is not None
        and prior_rule is not None
        and rule.label != prior_rule.label
        and rule.pre_tax == prior_rule.pre_tax
    ):
        out.append(
            f"{profile.bank_id} labelled this measure {prior_rule.label!r} in "
            f"{spec.comparison_period} and {rule.label!r} in {spec.period}; the "
            "comparison spans a change in the headline measure."
        )

    for change in reg.regime_changes_for(
        spec.bank_id, spec.metric_id, spec.comparison_start, spec.period_end
    ):
        out.append(str(change.get("caveat", change.get("name", ""))).strip())
    return out
