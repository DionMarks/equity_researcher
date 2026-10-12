"""Ingests one document into the stores."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..domain.calendar import fiscal_period
from ..domain.models import DocMeta, Fact, HeadlineLabel
from ..domain.registry import get_registry
from ..memory.fact_store import FactStore
from .artifacts import write_processed
from .facts import extract_facts
from .parse_pdf import ParsedDocument, PdfParser, PyMuPdfParser
from .validate import IdentityCheck, validate_facts

__all__ = [
    "IngestReport",
    "PreparedDocument",
    "ingest_document",
    "prepare_document",
    "store_prepared",
]


@dataclass
class IngestReport:
    """What ingesting one document achieved."""

    doc_id: str
    page_count: int
    tables: int
    facts_stored: int = 0
    facts_quarantined: int = 0
    checks: list[IdentityCheck] = field(default_factory=list)
    quarantine_reasons: list[str] = field(default_factory=list)
    headline_labels: int = 0
    walk_tables: int = 0

    @property
    def checks_run(self) -> int:
        return sum(1 for check in self.checks if not check.skipped)

    @property
    def checks_passed(self) -> int:
        return sum(1 for check in self.checks if not check.skipped and check.passed)

    def summary(self) -> str:
        return (
            f"{self.doc_id}: {self.page_count} pages, {self.tables} tables, "
            f"{self.facts_stored} facts stored, {self.facts_quarantined} quarantined, "
            f"{self.checks_passed}/{self.checks_run} identity checks passed"
        )


@dataclass
class PreparedDocument:
    """A document parsed, its figures extracted and checked, and not yet stored.

    The split is what lets a corpus be ingested in parallel: parsing and extraction are
    CPU-bound and independent per document, while the store is one SQLite file that must be
    written from one place, in a fixed order, so the stored corpus is the same whichever
    process finished first.
    """

    meta: DocMeta
    page_count: int
    tables: int
    walk_tables: int
    checks: list[IdentityCheck]
    validated: list[Fact]
    quarantined: list[tuple[Fact, str]]


def prepare_document(
    path: Path,
    meta: DocMeta,
    parser: PdfParser | None = None,
    periods: tuple[str, ...] | None = None,
    processed_root: Path | None = None,
) -> PreparedDocument:
    """Parse, extract and validate one filing, without touching the store.

    ``periods`` defaults to the document's own period and its prior corresponding
    period, which is what its comparative columns carry. ``processed_root`` is
    where the parsed artefacts are written so a query need never parse this PDF
    again.
    """
    registry = get_registry()
    profile = registry.bank(meta.bank_id)
    engine = parser or PyMuPdfParser()
    document: ParsedDocument = engine.parse(path, meta.doc_id)

    wanted = periods or (
        meta.period,
        fiscal_period(meta.period, profile.fy_end_month).pcp().label,
    )
    extracted = extract_facts(document, meta, wanted)
    outcome = validate_facts(extracted, meta.bank_id, wanted)
    if processed_root is not None:
        write_processed(document, processed_root)
    return PreparedDocument(
        meta=meta.model_copy(update={"page_count": document.page_count, "sha256": document.sha256}),
        page_count=document.page_count,
        tables=len(document.tables()),
        walk_tables=len(document.walk_pages()),
        checks=outcome.checks,
        validated=outcome.validated,
        quarantined=outcome.quarantined,
    )


def store_prepared(prepared: PreparedDocument, store: FactStore) -> IngestReport:
    """Write a prepared document's figures to the store, replacing what it said before."""
    meta = prepared.meta
    # Re-reading a document replaces what it said rather than adding to it.
    store.forget_document(meta.doc_id)
    store.put_document(meta)
    report = IngestReport(
        doc_id=meta.doc_id,
        page_count=prepared.page_count,
        tables=prepared.tables,
        checks=prepared.checks,
        walk_tables=prepared.walk_tables,
    )
    for fact in prepared.validated:
        store.put_fact(fact)
        report.facts_stored += 1
    for fact, reason in prepared.quarantined:
        store.quarantine(fact, reason)
        report.facts_quarantined += 1
        report.quarantine_reasons.append(f"{fact.component} {fact.period}: {reason}")
    report.headline_labels = _record_headline_labels(prepared.validated, meta, store)
    return report


def ingest_document(
    path: Path,
    meta: DocMeta,
    store: FactStore,
    parser: PdfParser | None = None,
    periods: tuple[str, ...] | None = None,
    processed_root: Path | None = None,
) -> IngestReport:
    """Parse, extract, validate and store one filing."""
    prepared = prepare_document(
        path, meta, parser=parser, periods=periods, processed_root=processed_root
    )
    return store_prepared(prepared, store)


def _record_headline_labels(facts: list[Fact], meta: DocMeta, store: FactStore) -> int:
    """Record which label the issuer led with, based on the metrics actually found.

    The headline rule is data-driven so it works for a bank we haven't seen:
    whatever label the key-metrics table used is the headline for that period.
    """
    registry = get_registry()
    recorded = 0
    for metric_id in registry.metrics:
        rule = registry.metrics[metric_id].label_for(registry.bank(meta.bank_id), meta.period)
        # A headline struck before tax is recorded where its own figure is printed.
        held_as = "profit_before_tax" if rule is not None and rule.pre_tax else metric_id
        fact = next((f for f in facts if f.component == held_as and f.period == meta.period), None)
        if fact is None:
            continue
        label = registry.metrics[metric_id].display_name
        if rule is not None:
            label = rule.label
        store.put_headline_label(
            HeadlineLabel(
                bank_id=meta.bank_id,
                period=meta.period,
                metric_id=metric_id,  # type: ignore[arg-type]
                label=label,
                doc_id=meta.doc_id,
                page_no=fact.page_no,
            )
        )
        recorded += 1
    return recorded
