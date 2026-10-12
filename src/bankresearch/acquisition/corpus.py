"""Builds the corpus from the manifest: download, then ingest, then report."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..domain.models import DocMeta
from ..ingestion.pipeline import IngestReport, PreparedDocument, prepare_document, store_prepared
from ..memory.fact_store import FactStore
from ..settings import Settings
from .fetch import Fetcher
from .manifest import ManifestEntry, load_manifest, write_manifest

__all__ = ["CorpusReport", "build_corpus"]


@dataclass
class CorpusReport:
    attempted: int = 0
    fetched: int = 0
    from_cache: int = 0
    ingested: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)
    reports: list[IngestReport] = field(default_factory=list)

    @property
    def facts_stored(self) -> int:
        return sum(report.facts_stored for report in self.reports)

    @property
    def facts_quarantined(self) -> int:
        return sum(report.facts_quarantined for report in self.reports)

    @property
    def checks_run(self) -> int:
        return sum(report.checks_run for report in self.reports)

    @property
    def checks_passed(self) -> int:
        return sum(report.checks_passed for report in self.reports)

    def summary(self) -> str:
        return (
            f"{self.ingested}/{self.attempted} documents ingested "
            f"({self.from_cache} already cached), {self.facts_stored} facts stored, "
            f"{self.facts_quarantined} quarantined, "
            f"{self.checks_passed}/{self.checks_run} identity checks passed, "
            f"{len(self.failures)} failure(s)"
        )


def build_corpus(
    store: FactStore,
    settings: Settings,
    registry_banks: dict[str, tuple[str, ...]],
    entries: list[ManifestEntry] | None = None,
    only_bank: str | None = None,
    only_period: str | None = None,
    update_manifest: bool = True,
    workers: int | None = None,
) -> CorpusReport:
    """Fetch and ingest every manifest document, or a filtered subset.

    ``workers`` is how many documents are parsed at once: by default one per processor, up
    to eight, and one when there is only one document.
    """
    all_entries = entries if entries is not None else load_manifest()
    selected = [
        entry
        for entry in all_entries
        if (only_bank is None or entry.bank_id == only_bank.upper())
        and (only_period is None or entry.period == only_period.upper())
    ]

    fetcher = Fetcher(raw_root=settings.raw_dir)
    report = CorpusReport(attempted=len(selected))
    updated: dict[str, str] = {}

    to_prepare: list[tuple[str, Path, DocMeta]] = []
    for entry in selected:
        try:
            allowed = registry_banks.get(entry.bank_id, ())
            result = fetcher.fetch(entry.url, allowed)
        except Exception as exc:  # a 404 or a moved document must not stop the rest
            report.failures.append((entry.doc_id, f"fetch: {type(exc).__name__}: {exc}"))
            continue
        report.fetched += 1
        if result.from_cache:
            report.from_cache += 1
        updated[entry.doc_id] = result.sha256
        to_prepare.append(
            (
                entry.doc_id,
                result.path,
                entry.to_meta().model_copy(update={"sha256": result.sha256}),
            )
        )

    prepared = _prepare_all(to_prepare, settings.processed_dir, workers)
    for doc_id, _, _ in to_prepare:
        outcome = prepared[doc_id]
        if isinstance(outcome, str):
            report.failures.append((doc_id, f"ingest: {outcome}"))
            continue
        try:
            ingested = store_prepared(outcome, store)
        except Exception as exc:
            report.failures.append((doc_id, f"ingest: {type(exc).__name__}: {exc}"))
            continue
        report.ingested += 1
        report.reports.append(ingested)

    if update_manifest and updated:
        write_manifest(
            [
                entry
                if entry.doc_id not in updated
                else replace(entry, sha256=updated[entry.doc_id])
                for entry in all_entries
            ]
        )
    return report


def _prepare_one(job: tuple[str, Path, DocMeta, Path]) -> PreparedDocument | str:
    """Parse, extract and validate one document; a failure comes back as its reason."""
    _, path, meta, processed_root = job
    try:
        return prepare_document(path, meta, processed_root=processed_root)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _prepare_all(
    jobs: list[tuple[str, Path, DocMeta]], processed_root: Path, workers: int | None
) -> dict[str, PreparedDocument | str]:
    """Every document prepared, in parallel processes where that helps."""
    count = workers or min(8, os.cpu_count() or 1)
    # Largest first: the run takes as long as its longest document, so that one starts
    # straight away rather than after the small ones.
    full = sorted(
        ((doc_id, path, meta, processed_root) for doc_id, path, meta in jobs),
        key=lambda job: -(job[1].stat().st_size if job[1].exists() else 0),
    )
    if count <= 1 or len(full) <= 1:
        return {job[0]: _prepare_one(job) for job in full}
    with ProcessPoolExecutor(max_workers=count) as pool:
        return dict(zip((job[0] for job in full), pool.map(_prepare_one, full), strict=True))
