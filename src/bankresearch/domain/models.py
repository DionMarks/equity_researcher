"""Shared data types."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .units import Unit

__all__ = [
    "AbstainReason",
    "AlternativeBasis",
    "Availability",
    "Basis",
    "CandidateValue",
    "ConflictClass",
    "DataConflict",
    "Decomposition",
    "Diagnostics",
    "DocMeta",
    "DocType",
    "DriverContribution",
    "EvidenceCard",
    "EvidenceType",
    "Fact",
    "HeadlineLabel",
    "MetricId",
    "ReadingItem",
    "ReadingList",
    "RequestSpec",
    "ResolvedInputs",
    "Restatement",
    "RunStatus",
    "SourceRef",
    "ValidationReport",
]

MetricId = Literal[
    "nim",
    "headline_earnings",
    "roe",
    "cet1_ratio",
    "credit_impairment_charge",
    "cost_to_income",
]

DocType = Literal[
    "RESULTS_DOCUMENT",
    "RESTATEMENT_NOTICE",
    "RESULTS_RELEASE",
    "INVESTOR_PRESENTATION",
    "PILLAR3",
    "ANNUAL_REPORT",
]

RunStatus = Literal["OK", "PARTIAL", "ABSTAIN"]

# Every abstention says what was found and what is missing.
AbstainReason = Literal[
    "AMBIGUOUS_INPUT",
    "METRIC_OUT_OF_SCOPE",
    "BANK_UNKNOWN",
    "PERIOD_NOT_YET_REPORTED",
    "DOCS_UNAVAILABLE",
    "METRIC_NOT_REPORTED",
    "INSUFFICIENT_EVIDENCE",
]

ConflictClass = Literal[
    "UNIT",
    "ROUNDING",
    "BASIS",
    "SCOPE",
    "PRO_FORMA",
    "RESTATEMENT",
    "TIMING",
    "UNEXPLAINED",
]

EvidenceType = Literal[
    "computed",
    "disclosed_quantified",
    "disclosed_quantified_chained",
    "disclosed_qualitative",
    "inferred",
]

Method = Literal["management_walk", "chained_walk", "computed", "qualitative"]

_STRICT = ConfigDict(extra="forbid", frozen=True)


class SourceRef(BaseModel):
    """Where a figure was read. Required for anything that claims to be evidence."""

    model_config = _STRICT

    doc_id: str
    doc_type: DocType
    page: int = Field(ge=0)
    table_id: str | None = None
    publish_date: date

    def cite(self) -> str:
        return f"{self.doc_id} p.{self.page}" + (f" table {self.table_id}" if self.table_id else "")


class Basis(BaseModel):
    """The dimension along which two correct figures can differ."""

    model_config = ConfigDict(extra="forbid")

    earnings_basis: Literal["cash", "statutory", "underlying", "n/a"] = "n/a"
    ex_notable_items: bool = False
    scope: Literal["group", "level2", "level1", "division"] = "group"
    pro_forma: bool = False
    continuing_ops: bool | None = None

    def differs_from(self, other: Basis) -> bool:
        return (
            self.earnings_basis != other.earnings_basis
            or self.ex_notable_items != other.ex_notable_items
            or self.scope != other.scope
            or self.pro_forma != other.pro_forma
            or self.continuing_ops != other.continuing_ops
        )


class RequestSpec(BaseModel):
    """One resolved question. The comparison period is always the pcp."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    bank_id: str
    metric_id: MetricId
    period: str
    comparison_period: str
    period_start: date
    period_end: date
    comparison_start: date
    comparison_end: date
    headline_label: str | None = None
    asked_as: str | None = None
    """The request as typed, kept where it never resolved to a known bank and metric, so
    an abstention can say what was asked rather than show a placeholder."""
    pre_tax: bool = False
    """The bank leads with a measure struck before tax, so both periods are compared before
    tax and the earnings bridge stops there (DESIGN.md ADR-062)."""
    as_of: date

    @property
    def measure(self) -> str:
        """The component that holds the figure asked about: the metric itself, or for a
        pre-tax headline, profit before tax."""
        return "profit_before_tax" if self.pre_tax else self.metric_id

    @property
    def period_days(self) -> int:
        return (self.period_end - self.period_start).days + 1

    @property
    def comparison_days(self) -> int:
        return (self.comparison_end - self.comparison_start).days + 1

    @property
    def periods(self) -> tuple[str, str]:
        """The only two periods any tool may read. Scope is enforced in code."""
        return (self.period, self.comparison_period)


class DocMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    bank_id: str
    period: str
    doc_type: DocType
    publish_date: date
    url: str
    sha256: str | None = None
    page_count: int | None = None


class Fact(BaseModel):
    """A validated figure with its vintage. Never overwritten."""

    model_config = ConfigDict(extra="forbid")

    fact_id: str
    bank_id: str
    period: str
    component: str
    value: Decimal
    """The figure as the identities use it: deductions as positive magnitudes, so the
    bridge can subtract them."""
    published_value: Decimal | None = None
    """The figure as the page prints it, brackets and all.

    Kept because a citation is a claim about what the page says, and after sign
    normalisation the value above no longer matches it. Without this, every citation of a
    bracketed deduction would fail the page check even though it's correct.
    """
    unit: Unit
    basis: Basis = Field(default_factory=Basis)
    doc_id: str
    doc_type: DocType
    page_no: int
    table_id: str | None = None
    column_header: str | None = None
    """Where in the table this came from. Needed to quote the cell in a citation,
    so it is a field rather than free text in a note."""
    publish_date: date
    validated: bool = False
    is_restated: bool = False


class Restatement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_fact_id: str
    restated_fact_id: str
    note: str | None = None
    footnoted: bool = False
    """Whether the issuer explained the restatement. Unfootnoted costs confidence."""


class HeadlineLabel(BaseModel):
    """Which label the issuer led with, read from its key-metrics table."""

    model_config = ConfigDict(extra="forbid")

    bank_id: str
    period: str
    metric_id: MetricId
    label: str
    doc_id: str
    page_no: int


class EvidenceCard(BaseModel):
    """Something this run read, with the page it came from."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    component: str
    period: str
    value: Decimal | None = None
    published_value: Decimal | None = None
    """What the page prints, which is what a citation check compares against."""
    unit: Unit
    direction: Literal["+", "-", "0"] | None = None
    basis: Basis = Field(default_factory=Basis)
    statement: str
    quote: str | None = None
    passage: str | None = None
    """The issuer's whole sentence, verbatim, where the card is its commentary or a step of a
    walk it wrote out. The quote stays within its 25 words for the memo; the writer and the
    checker need the whole sentence, because the cause is usually in its second half."""
    source: SourceRef
    is_restated: bool = False
    method: Literal["table", "text", "chart", "computed"]
    checks_passed: list[str] = Field(default_factory=list)
    checks_failed: list[str] = Field(default_factory=list)

    @field_validator("statement")
    @classmethod
    def _statement_is_short(cls, v: str) -> str:
        # 40 words or fewer, paraphrased.
        if len(v.split()) > 40:
            raise ValueError("statement must be 40 words or fewer")
        return v

    @field_validator("quote")
    @classmethod
    def _quote_is_short(cls, v: str | None) -> str | None:
        # 25 words or fewer, verbatim.
        if v is not None and len(v.split()) > 25:
            raise ValueError("quote must be 25 words or fewer")
        return v

    @field_validator("passage")
    @classmethod
    def _passage_is_one_sentence(cls, v: str | None) -> str | None:
        # One sentence of commentary, not a page. Longer than this and it's two sentences
        # the PDF ran together, or a table read as prose.
        if v is not None and len(v.split()) > 120:
            raise ValueError("passage must be 120 words or fewer")
        return v

    @property
    def is_quantitative(self) -> bool:
        return self.value is not None


class ReadingItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component: str
    doc_id: str
    pages: list[int] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)
    rationale: str = ""


class ReadingList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ReadingItem] = Field(default_factory=list)

    def pages_for(self, component: str) -> list[tuple[str, int]]:
        return [
            (item.doc_id, page)
            for item in self.items
            if item.component == component
            for page in item.pages
        ]


class Availability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_docs: list[DocMeta] = Field(default_factory=list)
    comparison_docs: list[DocMeta] = Field(default_factory=list)
    reportable: bool = True
    reason: AbstainReason | None = None
    detail: str = ""
    release_pending: bool = False
    """Nothing is held, and the as-of date falls in the month the result is usually
    released. ``PERIOD_NOT_YET_REPORTED`` holds unless the ASX's own list has it."""

    @property
    def has_documents(self) -> bool:
        return bool(self.current_docs) and bool(self.comparison_docs)


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)
    missing_required: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed and not self.missing_required


class CandidateValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: Decimal
    unit: Unit
    basis: Basis = Field(default_factory=Basis)
    source: SourceRef
    vintage: date
    is_restated: bool = False
    footnoted: bool = False
    evidence_id: str | None = None


class DataConflict(BaseModel):
    """Two sources disagreeing, classified before it is resolved."""

    model_config = ConfigDict(extra="forbid")

    component: str
    candidates: list[CandidateValue]
    classification: ConflictClass
    chosen: CandidateValue
    rule_applied: str
    impact: str = ""


class AlternativeBasis(BaseModel):
    """A figure on a basis the issuer did not lead with. Context, never the answer."""

    model_config = ConfigDict(extra="forbid")

    label: str
    basis: Basis
    current: Decimal | None = None
    prior: Decimal | None = None
    change: Decimal | None = None
    unit: Unit
    evidence_ids: list[str] = Field(default_factory=list)
    note: str = ""


class ResolvedInputs(BaseModel):
    """What reconciliation settled on, and what decomposition may use.

    The two periods are kept in separate maps rather than one keyed by period. Every
    identity uses exactly two periods, and a single map makes it easy to pull a component
    from the wrong one.
    """

    model_config = ConfigDict(extra="forbid")

    current: dict[str, Decimal] = Field(default_factory=dict)
    prior: dict[str, Decimal] = Field(default_factory=dict)
    units: dict[str, Unit] = Field(default_factory=dict)
    bases: dict[str, Basis] = Field(default_factory=dict)
    """``"{period}:{component}"`` -> the basis the chosen figure was published on.

    Passed through so decomposition can refuse a walk whose components aren't on the ratio's
    own basis. Westpac publishes its expense-to-income ratio excluding Notable Items and its
    operating income including them, and a walk built across the two is out by 94 basis
    points even with every figure read correctly.
    """
    evidence_ids: dict[str, list[str]] = Field(default_factory=dict)
    """``"{period}:{component}"`` -> the evidence ids behind that value."""
    conflicts: list[DataConflict] = Field(default_factory=list)
    alternative_bases: list[AlternativeBasis] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    failed_validation: list[str] = Field(default_factory=list)
    restated_components: list[str] = Field(default_factory=list)
    restatement_footnoted: bool = True

    def get(
        self, component: str, period: Literal["current", "prior"] = "current"
    ) -> Decimal | None:
        return (self.current if period == "current" else self.prior).get(component)

    def pair(self, component: str) -> tuple[Decimal, Decimal] | None:
        """Prior and current values, or ``None`` if either is missing."""
        before, after = self.prior.get(component), self.current.get(component)
        if before is None or after is None:
            return None
        return before, after

    def have_both(self, *components: str) -> bool:
        return all(self.pair(component) is not None for component in components)

    def citations_for(self, component: str, period_label: str) -> list[str]:
        return self.evidence_ids.get(f"{period_label}:{component}", [])

    def comparable(self, *components: str) -> bool:
        """Whether every stated basis among these components agrees.

        Notable items must match exactly, and the earnings basis must match where both
        figures state one. ``n/a`` means the figure has no earnings basis to disagree about,
        such as a balance or a capital ratio.
        """
        found = [
            basis for key, basis in self.bases.items() if key.split(":", 1)[-1] in set(components)
        ]
        if not found:
            return True
        if len({basis.ex_notable_items for basis in found}) > 1:
            return False
        return len({basis.earnings_basis for basis in found} - {"n/a"}) <= 1


class DriverContribution(BaseModel):
    """One attributed driver, before narrative and confidence are attached."""

    model_config = ConfigDict(extra="forbid")

    driver_id: str
    parent_id: str | None = None
    label: str = ""
    contribution: Decimal | None = None
    direction: Literal["+", "-", "0"] = "0"
    method: Method = "computed"
    evidence_type: EvidenceType = "computed"
    evidence_ids: list[str] = Field(default_factory=list)
    from_components: list[str] = Field(default_factory=list)
    """The resolved components this contribution was computed from.

    Every number in the output has to trace to an evidence card or to a calculation over
    cards. A computed driver is the second kind, so it records its inputs and the
    orchestrator turns them into citations.
    """
    note: str = ""

    @field_validator("direction")
    @classmethod
    def _direction_set(cls, v: str) -> str:
        return v


class Decomposition(BaseModel):
    """The output of the arithmetic. No model has touched any number in here."""

    model_config = ConfigDict(extra="forbid")

    metric_id: MetricId
    change_unit: Literal["bps", "aud_m"]
    headline_change: Decimal
    drivers: list[DriverContribution] = Field(default_factory=list)
    residual: Decimal = Decimal(0)
    tolerance: Decimal = Decimal(0)
    path: Literal["A", "B", "A+B", "none"] = "none"
    """Which path produced the drivers: A is the issuer's walk, B the identity."""
    path_a_residual: Decimal | None = None
    path_b_residual: Decimal | None = None
    disagreements: list[str] = Field(default_factory=list)
    """Driver ids where paths A and B differ by more than twice the tolerance."""
    notes: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)

    @property
    def within_tolerance(self) -> bool:
        return abs(self.residual) <= self.tolerance

    @property
    def sum_of_drivers(self) -> Decimal:
        """What the drivers explain, summed over the top level only.

        A nested driver breaks down its parent, not the movement, so including it double
        counts. NAB's return on equity splits an earnings effect of -306.5bp into four
        income statement lines, and adding those in made the memo report -1,637bp explained
        against a -320bp movement with a +1.4bp residual. The residual uses the same rule,
        so the reconciliation now adds up.
        """
        return sum(
            (
                d.contribution
                for d in self.drivers
                if d.parent_id is None and d.contribution is not None
            ),
            Decimal(0),
        )


class Diagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: Decimal = Decimal(0)
    latency_ms: int = 0
    model_ids: dict[str, str] = Field(default_factory=dict)
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    config_hash: str = ""
    git_sha: str = ""
    warnings: list[str] = Field(default_factory=list)
    declined_steps: list[str] = Field(default_factory=list)
    """Steps whose paid call the consent gate turned down. On the command line that only
    happens when a run has used up the budget approved for it, and the run is then reported
    as stopped short rather than shown."""
    budget_breached: bool = False
    created_at: datetime | None = None
