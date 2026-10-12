"""The result contract."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from ..domain.models import (
    AbstainReason,
    AlternativeBasis,
    Basis,
    DataConflict,
    Diagnostics,
    EvidenceCard,
    RequestSpec,
    RunStatus,
)
from ..domain.units import Unit

__all__ = [
    "Change",
    "Confidence",
    "Driver",
    "HeadlineValue",
    "Reconciliation",
    "ResearchResult",
    "build_change",
    "describe_basis",
    "request_names",
    "share_of_change",
    "tidy",
]

_STRICT = ConfigDict(extra="forbid")

# Suppress the percentage when |prior| is below this share of |current|.
_PRIOR_SHARE_FLOOR = Decimal("0.10")


def tidy(value: Decimal | None) -> Decimal | None:
    """A computed figure as ``result.json`` carries it: 420, 50.659, 0. Exact rational
    arithmetic leaves 28 digits behind ("420.0000000000000000000000000000", "0E-28") that no
    reader needs, so the JSON is rounded to a thousandth; the run itself keeps every digit."""
    if value is None:
        return None
    rounded = value.quantize(Decimal("0.001"))
    if rounded == rounded.to_integral_value():
        return rounded.quantize(Decimal(1)) + Decimal(0)  # adding zero turns -0 into 0
    return rounded.normalize()


class Confidence(BaseModel):
    """A level, the score behind it, and every rubric rule that applied."""

    model_config = _STRICT

    level: Literal["High", "Medium", "Low"]
    score: float
    reasons: list[str] = Field(default_factory=list)


class HeadlineValue(BaseModel):
    """One of the two figures the comparison is struck between."""

    model_config = _STRICT

    value: Decimal
    unit: Unit
    label_at_bank: str
    """The issuer's own wording, e.g. "Cash profit" or "Net profit"."""
    basis: Basis = Field(default_factory=Basis)
    evidence_ids: list[str] = Field(default_factory=list)
    restated: bool = False
    original_value: Decimal | None = None
    """The pre-restatement figure, where the comparative was restated."""


class Change(BaseModel):
    """The movement, with the percentage withheld where it would mislead."""

    model_config = _STRICT

    absolute: Decimal
    change_unit: Literal["bps", "aud_m"]
    pct: Decimal | None = None
    pct_suppressed_reason: str | None = None

    @field_serializer("absolute", when_used="json")
    def _tidy(self, value: Decimal | None) -> Decimal | None:
        return tidy(value)


class Driver(BaseModel):
    """One attributed driver, as it appears in the report."""

    model_config = _STRICT

    driver_id: str
    parent_id: str | None = None
    label: str
    contribution: Decimal | None = None
    direction: Literal["+", "-", "0"]
    share_of_change: Decimal | None = None
    method: Literal["management_walk", "chained_walk", "computed", "qualitative"]
    evidence_type: Literal[
        "computed",
        "disclosed_quantified",
        "disclosed_quantified_chained",
        "disclosed_qualitative",
        "inferred",
    ]
    explanation: str = ""
    """How the figure was arrived at, where that needs saying: "AUD m contribution to
    earnings, not to the ratio"."""
    cause: str = ""
    """Why it moved, as the issuer states it, in one sentence that cites the issuer's own
    words. Checked against those words, and left empty rather than guessed."""
    cause_evidence_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: Confidence

    @field_serializer("contribution", "share_of_change", when_used="json")
    def _tidy(self, value: Decimal | None) -> Decimal | None:
        return tidy(value)


class Reconciliation(BaseModel):
    """Whether the drivers account for the movement."""

    model_config = _STRICT

    headline_change: Decimal
    sum_of_drivers: Decimal
    residual: Decimal
    tolerance: Decimal
    within_tolerance: bool

    @field_serializer(
        "headline_change", "sum_of_drivers", "residual", "tolerance", when_used="json"
    )
    def _tidy(self, value: Decimal | None) -> Decimal | None:
        return tidy(value)


class ModelCall(BaseModel):
    """One model call the answer made: what it was for, which model, and what it cost."""

    model_config = _STRICT

    step: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    web_searches: int = 0
    tool_calls: list[str] = Field(default_factory=list)
    """The tools the model asked to use in this call, by name."""
    cost_usd: Decimal | None = None
    cached: bool = False
    """Answered from the response cache: the same request was made before, so it cost
    nothing this time."""


class Trail(BaseModel):
    """How the answer was reached, beyond the figures it cites: every model call in order,
    and every filing found and read during the run."""

    model_config = _STRICT

    model_calls: list[ModelCall] = Field(default_factory=list)
    fetched: list[str] = Field(default_factory=list)
    """A filing the run had to find, in the words the run recorded: which document, from
    where, and how many of its figures were stored."""


class ResearchResult(BaseModel):
    """Everything one run produced. Written to ``runs/<run_id>/result.json``."""

    model_config = _STRICT

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    created_at: datetime
    as_of: date
    request: RequestSpec
    status: RunStatus
    abstain_reason: AbstainReason | None = None
    current: HeadlineValue | None = None
    prior: HeadlineValue | None = None
    change: Change | None = None
    drivers: list[Driver] = Field(default_factory=list)
    residual: Driver | None = None
    reconciliation: Reconciliation | None = None
    headline_confidence: Confidence | None = None
    """The confidence in the two figures themselves, before attribution is considered.

    It was always computed but then discarded, so only the combined ``overall_confidence``
    reached the result. The two answer different questions ("is this figure right" versus
    "how much of this movement is explained"), and the eval calibrates its accuracy measure
    against the first, so it needs to be able to see it.
    """
    overall_confidence: Confidence | None = None
    narrative: str = ""
    issuer_commentary: list[str] = Field(default_factory=list)
    """Evidence ids of the issuer's own sentences explaining this movement, found by code
    and against the same period a year earlier. Shown as they stand when there's no
    narrative, and the only source a driver's stated cause may cite."""
    caveats: list[str] = Field(default_factory=list)
    data_conflicts: list[DataConflict] = Field(default_factory=list)
    alternative_bases: list[AlternativeBasis] = Field(default_factory=list)
    evidence: list[EvidenceCard] = Field(default_factory=list)
    trail: Trail = Field(default_factory=Trail)
    diagnostics: Diagnostics = Field(default_factory=Diagnostics)

    @field_validator("narrative")
    @classmethod
    def _narrative_is_short(cls, value: str) -> str:
        # 150 words or fewer.
        if len(value.split()) > 150:
            raise ValueError("narrative must be 150 words or fewer")
        return value

    @property
    def top_level_drivers(self) -> list[Driver]:
        return [d for d in self.drivers if d.parent_id is None]

    @property
    def shares_mean_something(self) -> bool:
        """Whether every top-level driver moved the same way as the metric.

        With drivers that offset, shares run past 100% and below zero: CBA's margin split
        into net interest income at -319.3% and balances at 420.9%. Both are true and
        neither tells a reader anything, so they're shown as not meaningful. The figures
        stay in result.json.
        """
        shares = [d.share_of_change for d in self.top_level_drivers]
        return all(share is not None and 0 <= share <= 100 for share in shares)

    def children_of(self, driver_id: str) -> list[Driver]:
        return [d for d in self.drivers if d.parent_id == driver_id]

    def evidence_by_id(self) -> dict[str, EvidenceCard]:
        return {card.evidence_id: card for card in self.evidence}

    def cited_evidence_ids(self) -> set[str]:
        ids: set[str] = set()
        for value in (self.current, self.prior):
            if value is not None:
                ids |= set(value.evidence_ids)
        for driver in self.drivers:
            ids |= set(driver.evidence_ids) | set(driver.cause_evidence_ids)
        # Without a narrative, the issuer's own sentences are shown as they stand, so they
        # are cited. With one, only the passages the memo actually uses are.
        if not self.narrative:
            ids |= set(self.issuer_commentary)
        return ids


_EARNINGS_WORDS = {"cash": "cash basis", "statutory": "statutory basis", "underlying": "underlying"}
_SCOPE_WORDS = {"level2": "APRA Level 2", "level1": "APRA Level 1", "division": "divisional"}


def request_names(result: ResearchResult) -> tuple[str, str]:
    """The bank's and the metric's names for a heading: the configured names, or the
    request as typed where it never resolved to a bank and metric this tool knows.

    "UP" as a bank used to crash the renderer looking up a bank that wasn't there, and an
    out-of-scope metric was headed with the placeholder "Net interest margin".
    """
    from ..domain.registry import get_registry

    registry = get_registry()
    request = result.request
    bank = registry.banks.get(request.bank_id)
    if request.asked_as is not None:
        bank_asked, _, metric_asked = request.asked_as.partition(" / ")
        return (bank.name if bank else bank_asked), metric_asked or request.metric_id
    metric = registry.metrics.get(request.metric_id)
    name = metric.display_name if metric else request.metric_id
    # Earnings are each bank's own measure, so the heading names it: Judo's answer is its
    # profit before tax, and was headed "Cash earnings" above a row saying otherwise.
    if request.metric_id == "headline_earnings":
        own = result.current.label_at_bank if result.current else request.headline_label
        name = own or name
    return (bank.name if bank else request.bank_id, name)


def describe_basis(basis: Basis) -> str:
    """A basis in the words a reader would use, or "" where there's nothing to say.

    Only what distinguishes the figure is named: a group figure with no earnings basis and
    nothing excluded is just the figure.
    """
    words = [
        _EARNINGS_WORDS.get(basis.earnings_basis, ""),
        "excluding notable items" if basis.ex_notable_items else "",
        _SCOPE_WORDS.get(basis.scope, ""),
        "continuing operations" if basis.continuing_ops is True else "",
        "including discontinued operations" if basis.continuing_ops is False else "",
        "pro forma" if basis.pro_forma else "",
    ]
    return ", ".join(word for word in words if word)


def build_change(
    prior: Decimal,
    current: Decimal,
    change_unit: Literal["bps", "aud_m"],
    *,
    show_pct: bool,
) -> Change:
    """The movement, deciding whether a percentage change is safe to quote."""
    absolute = current - prior if change_unit == "aud_m" else (current - prior) * Decimal(100)

    if not show_pct:
        return Change(absolute=absolute, change_unit=change_unit)

    if prior == 0:
        return Change(
            absolute=absolute,
            change_unit=change_unit,
            pct_suppressed_reason="the prior period was zero, so a percentage change is undefined",
        )
    if (prior > 0) != (current > 0):
        return Change(
            absolute=absolute,
            change_unit=change_unit,
            pct_suppressed_reason=(
                "the figure changed sign between the two periods, so a percentage "
                "change would not describe the movement"
            ),
        )
    if current != 0 and abs(prior) < _PRIOR_SHARE_FLOOR * abs(current):
        return Change(
            absolute=absolute,
            change_unit=change_unit,
            pct_suppressed_reason=(
                "the prior period is less than 10% of the current one, so a percentage "
                "change would be arithmetically true and materially misleading"
            ),
        )

    return Change(
        absolute=absolute,
        change_unit=change_unit,
        pct=((current - prior) / abs(prior) * Decimal(100)).quantize(Decimal("0.1")),
    )


def share_of_change(
    contribution: Decimal | None, headline_change: Decimal, tolerance: Decimal
) -> Decimal | None:
    """A driver's share of the movement, or ``None`` when that would mislead.

    Withheld when the movement itself is inside tolerance: dividing by rounding
    produces shares of several hundred percent that mean nothing.
    """
    if contribution is None:
        return None
    if abs(headline_change) <= tolerance:
        return None
    return (contribution / headline_change * Decimal(100)).quantize(Decimal("0.1"))
