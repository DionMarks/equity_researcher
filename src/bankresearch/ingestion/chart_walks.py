"""Walks read from charts, turned into the same ``Walk`` a table produces."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from itertools import pairwise
from pathlib import Path
from typing import Final

from ..decomposition.taxonomy import map_label
from ..decomposition.walks import Walk, WalkStep
from ..domain.models import DocMeta, SourceRef
from ..domain.registry import get_registry
from ..domain.units import RATIO_UNITS, Unit, to_aud_m, to_bps
from ..llm.budget import Budget
from ..llm.client import LLMClient
from ..llm.consent import ConsentDenied
from ..llm.tracing import PromptLibrary, Tracer
from .charts import ChartReading, read_walk_chart
from .parse_pdf import ParsedDocument, ParsedPage, render_page

__all__ = [
    "ChartWalkOutcome",
    "WalkCandidate",
    "chart_walks_across",
    "chart_walks_in",
    "rank_walk_candidates",
]


@dataclass
class ChartWalkOutcome:
    """What the vision pass produced, and what it refused."""

    walks: list[Walk]
    readings: list[ChartReading]
    notes: list[str]
    refused: str = ""
    """Why reading stopped early, when a paid call was declined."""

    @property
    def quarantined(self) -> list[ChartReading]:
        return [reading for reading in self.readings if not reading.usable]


#: Words that name each metric on a page. A page that names none of them isn't a candidate
#: at any score. Westpac's liquidity-coverage waterfall is a real waterfall with real
#: labelled bars, and reading it as a margin walk wastes a vision call on a reading the
#: checksum will just throw away.
_METRIC_WORDS: Final[dict[str, tuple[str, ...]]] = {
    "nim": ("net interest margin", "nim"),
    "cet1_ratio": ("cet1", "common equity tier 1"),
    "headline_earnings": ("cash earnings", "cash profit", "net profit", "npat"),
    "roe": ("return on equity", "roe"),
    "cost_to_income": ("cost to income", "expense to income"),
    "credit_impairment_charge": ("impairment",),
}

#: A unit on a line of its own, i.e. a chart's axis label.
_BARE_UNIT = re.compile(r"^\s*(?:bps|basis points|\$m|\$bn|%)\s*$", re.IGNORECASE | re.MULTILINE)


def _is_label(line: str) -> bool:
    """Whether a line reads as a chart label rather than a sentence.

    CBA's divisional margin commentary is full of driver words ("higher deposit margins",
    "unfavourable deposit mix") inside prose. A bar label is short and unpunctuated; a
    clause fragment is neither.
    """
    return 3 <= len(line) <= 40 and not line.rstrip().endswith((".", ",", ";", ":"))


def _label_candidates(lines: list[str]) -> list[str]:
    """Each line, and each adjacent pair joined.

    A chart spreads a label over as many lines as it needs. ANZ draws "Assets pricing" as
    "Assets" above "pricing", so neither line maps to a driver, and the page that has the
    answer scored below one that only mentions margins. Pairs are tried as well as single
    lines, the same way the table reader handles a wrapped row label.
    """
    out = [line for line in lines if _is_label(line)]
    out += [joined for first, second in pairwise(lines) if _is_label(joined := f"{first} {second}")]
    return out


def rank_walk_pages(
    document: ParsedDocument,
    metric_id: str,
    doc_type: str,
    endpoints: tuple[Decimal, Decimal, Unit] | None = None,
) -> list[ParsedPage]:
    """Candidate walk pages for this metric in one document, best first."""
    scored = [
        (score, -page.page_no, page)
        for page in document.pages
        if (score := score_walk_page(page, metric_id, doc_type, endpoints)) is not None
    ]
    scored.sort(key=lambda item: (-item[0], -item[1]))
    return [page for _, _, page in scored]


def score_walk_page(
    page: ParsedPage,
    metric_id: str,
    doc_type: str,
    endpoints: tuple[Decimal, Decimal, Unit] | None = None,
) -> float | None:
    """How strongly a page reads as a walk of this metric, or ``None`` if it can't be one.

    Only pages that name the metric are considered, and those are then scored. The signals
    are what separate a group walk from the dozens of other pages that also say "bps":

    - the published opening and closing values printed on the page. A walk of this figure
      starts and ends at them, and other charts don't. This is the strongest signal there
      is, and it's why the reported endpoints are passed in;
    - labels the driver taxonomy recognises, which is what a row of named bars looks like.
      The aggregate nodes (net interest income, average balances) don't count: they have no
      synonyms because a walk bar is never labelled with them, and divisional pages are
      full of them;
    - the metric in the page's own title, not just somewhere in the text;
    - a bare unit line, which is an axis label;
    - no extractable table. A vector-drawn waterfall has no raster image, so
      ``classify_page`` sees sparse text and calls it "text": ANZ's page 25 is
      ``kind="text"`` with no tables. Page kind is almost useless here, and the absence of
      a table is the better signal.
    """
    if not page.likely_walk_page or page.kind == "boilerplate":
        return None
    words = _METRIC_WORDS.get(metric_id, ())
    text = page.text or ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not any(word in text.lower() for word in words):
        return None

    score = 0.0
    if endpoints is not None:
        shown = sum(1 for value in endpoints[:2] if _printed(value, endpoints[2], text))
        score += {2: 4.0, 1: 1.5}.get(shown, 0.0)
    drivers = {map_label(metric_id, line) for line in _label_candidates(lines)}
    aggregate = _aggregate_ids(metric_id)
    named = {d for d in drivers if d and not d.endswith(".other") and d not in aggregate}
    score += 2.0 * min(len(named), 4)
    if any(word in " ".join(lines[:3]).lower() for word in words):
        score += 3.0
    if _BARE_UNIT.search(text):
        score += 1.5
    score += 2.0 if not page.tables else -1.0 * min(len(page.tables), 2)
    if lines and sum(1 for line in lines if len(line) <= 28) / len(lines) > 0.6:
        score += 1.0
    if len(text) > 2500:
        score -= 2.0
    if doc_type == "INVESTOR_PRESENTATION":
        score += 0.5
    if page.page_no <= 2:
        score -= 2.0
    return score


@dataclass(frozen=True)
class WalkCandidate:
    """One page that might hold the walk, with the document it's in."""

    score: float
    document: ParsedDocument
    meta: DocMeta
    page: ParsedPage
    pdf_path: Path


def rank_walk_candidates(
    documents: list[tuple[ParsedDocument, DocMeta, Path]],
    metric_id: str,
    endpoints: tuple[Decimal, Decimal, Unit] | None = None,
) -> list[WalkCandidate]:
    """Candidate pages across every document, best first.

    Ranked together rather than one document at a time. Ranking within each document and
    reading them in turn read three wrong pages of Westpac's investor pack (an income walk
    and two divisional profit walks) before page 13 of its results document, which was the
    best candidate in its own document and the actual margin walk.
    """
    found = [
        WalkCandidate(score, document, meta, page, pdf_path)
        for document, meta, pdf_path in documents
        for page in document.pages
        if (score := score_walk_page(page, metric_id, meta.doc_type, endpoints)) is not None
    ]
    found.sort(key=lambda c: (-c.score, c.meta.doc_id, c.page.page_no))
    return found


def _printed(value: Decimal, unit: Unit, text: str) -> bool:
    """Whether a published figure appears on the page as a figure, not inside another one."""
    shown = f"{value:,}" if unit in ("aud_m", "aud_bn") else f"{value}"
    forms = {shown, shown.replace(",", "")}
    return any(re.search(rf"(?<![\d.,]){re.escape(form)}(?![\d])", text) for form in forms if form)


def _aggregate_ids(metric_id: str) -> frozenset[str]:
    """Driver ids the taxonomy declares with no synonyms: aggregate nodes, never walk bars."""
    group = get_registry().drivers.get(metric_id, {})
    return frozenset(
        driver_id for driver_id, spec in group.items() if not (spec.get("synonyms") or ())
    )


def chart_walks_in(
    document: ParsedDocument,
    metric_id: str,
    meta: DocMeta,
    *,
    pdf_path: Path,
    reported_opening: Decimal,
    reported_closing: Decimal,
    unit: Unit,
    opening_period: str,
    closing_period: str,
    client: LLMClient,
    budget: Budget | None = None,
    model: str = "claude-opus-5",
    effort: str = "high",
    max_pages: int = 3,
    tracer: Tracer | None = None,
    prompts: PromptLibrary | None = None,
) -> ChartWalkOutcome:
    """Read the pages of one document that look like a walk for this metric."""
    return chart_walks_across(
        [(document, meta, pdf_path)],
        metric_id,
        reported_opening=reported_opening,
        reported_closing=reported_closing,
        unit=unit,
        opening_period=opening_period,
        closing_period=closing_period,
        client=client,
        budget=budget,
        model=model,
        effort=effort,
        max_pages=max_pages,
        tracer=tracer,
        prompts=prompts,
    )


def chart_walks_across(
    documents: list[tuple[ParsedDocument, DocMeta, Path]],
    metric_id: str,
    *,
    reported_opening: Decimal,
    reported_closing: Decimal,
    unit: Unit,
    opening_period: str,
    closing_period: str,
    client: LLMClient,
    budget: Budget | None = None,
    model: str = "claude-opus-5",
    effort: str = "high",
    max_pages: int = 4,
    tracer: Tracer | None = None,
    prompts: PromptLibrary | None = None,
    first: list[tuple[str, int]] | None = None,
    period_ends: tuple[date, date] | None = None,
) -> ChartWalkOutcome:
    """Read the best candidate pages across all documents, and keep the first that balances.

    ``first`` is pages to read before the ranked ones: the pages page search pointed at,
    when the ranking found nothing to read.

    Candidates are ranked together and read in order until one reconciles, at most
    ``max_pages`` of them. ``likely_walk_page`` matches any page that says "bps", which is
    every KPI table in a results pack (54 of NAB's 108 pages, 76 of ANZ's 131), so the
    ranking is what decides which few pages are worth a vision call.
    """
    outcome = ChartWalkOutcome(walks=[], readings=[], notes=[])
    if not client.available:
        outcome.notes.append(
            "a charted walk needs a vision call: no model configured, so the issuer's own "
            "attribution is unavailable and only the accounting identity remains"
        )
        return outcome

    candidates = rank_walk_candidates(
        documents, metric_id, (reported_opening, reported_closing, unit)
    )
    if first:
        by_id = {meta.doc_id: (document, meta, path) for document, meta, path in documents}
        chosen = [
            WalkCandidate(float("inf"), document, meta, page, path)
            for doc_id, page_no in first
            if doc_id in by_id
            for document, meta, path in [by_id[doc_id]]
            if (page := document.page(page_no)) is not None
        ]
        seen = {(c.meta.doc_id, c.page.page_no) for c in chosen}
        candidates = chosen + [c for c in candidates if (c.meta.doc_id, c.page.page_no) not in seen]
    if not candidates:
        outcome.notes.append(
            f"no page in the documents read charts a {metric_id} walk: nothing names the "
            "metric and reads as a chart rather than a table"
        )
        return outcome

    to_read = candidates[:max_pages]
    for candidate in to_read:
        meta, page = candidate.meta, candidate.page
        image, media_type = render_page(candidate.pdf_path, page.page_no)
        source = SourceRef(
            doc_id=meta.doc_id,
            doc_type=meta.doc_type,
            page=page.page_no,
            publish_date=meta.publish_date,
        )
        try:
            reading = read_walk_chart(
                page_image=image,
                media_type=media_type,
                source=source,
                reported_opening=reported_opening,
                reported_closing=reported_closing,
                unit=unit,
                client=client,
                budget=budget,
                metric_id=metric_id,
                opening_period=opening_period,
                closing_period=closing_period,
                period_ends=period_ends,
                model=model,
                effort=effort,
                tracer=tracer,
                prompts=prompts,
            )
        except ConsentDenied as refusal:
            # Caught per page, so a grant that runs out part-way keeps the readings it
            # already paid for, along with the notes saying what they were.
            outcome.refused = str(refusal)
            outcome.notes.append(
                f"chart reading stopped after {len(outcome.readings)} page(s): {refusal}"
            )
            break
        outcome.readings.append(reading)
        where = f"{meta.doc_id} p.{page.page_no}"
        if not reading.usable:
            if reading.note:
                outcome.notes.append(f"{where}: {reading.note}")
            continue

        outcome.walks.append(
            _as_walk(
                reading,
                metric_id=metric_id,
                source=source,
                unit=unit,
                opening_period=opening_period,
                closing_period=closing_period,
                reported_opening=reported_opening,
                reported_closing=reported_closing,
            )
        )
        # A walk that reconciled against the issuer's own published endpoints is the answer.
        # Reading more candidates costs a vision call each and can only give a second
        # opinion that nothing would decide between.
        outcome.notes.append(
            f"{where}: a {metric_id} walk read and reconciled against the published endpoints"
        )
        break
    return outcome


def _as_walk(
    reading: ChartReading,
    *,
    metric_id: str,
    source: SourceRef,
    unit: Unit,
    opening_period: str,
    closing_period: str,
    reported_opening: Decimal,
    reported_closing: Decimal,
) -> Walk:
    """A balanced reading as a ``Walk``.

    The endpoints are the issuer's published figures, never the ones read off the chart: the
    chart gives the attribution and the text gives the levels, which is what makes the
    checksum a check against independent evidence.

    Everything is converted into one unit, because ``Walk`` assumes that:
    ``checksum_residual`` is ``declared_change - sum_of_steps``, and the endpoints and steps
    rarely arrive in the same unit. ANZ states its margin as 1.53% and draws the walk in
    basis points, so the Walk held percentage endpoints next to basis-point steps under one
    ``unit`` field that was true of neither. The residual came out -0.97 against a real -2,
    and the evidence card read "Capital & replicating portfolio | pct | 2" for a figure
    that's two basis points.

    Ratios use basis points and amounts use millions, which is the convention the
    decomposition already reads contributions in.

    Labels are mapped through the same taxonomy a tabled walk uses, and a label it doesn't
    recognise stays unmapped and becomes "other", which reconciles and names no cause. A
    wrong mapping would reconcile too, and name the wrong one.
    """
    step_unit = reading.unit or unit
    if unit in RATIO_UNITS:
        canonical: Unit = "bps"
        opening, closing = to_bps(reported_opening, unit), to_bps(reported_closing, unit)
        steps = [(label, to_bps(value, step_unit)) for label, value in reading.labelled]
    else:
        canonical = "aud_m"
        opening, closing = to_aud_m(reported_opening, unit), to_aud_m(reported_closing, unit)
        steps = [(label, to_aud_m(value, step_unit)) for label, value in reading.labelled]

    return Walk(
        opening_period=opening_period,
        closing_period=closing_period,
        opening_value=opening,
        closing_value=closing,
        steps=[
            WalkStep(
                label=label,
                driver_id=map_label(metric_id, label),
                value=value,
                evidence_id=None,
            )
            for label, value in steps
        ],
        unit=canonical,
        source=source,
        method="management_walk",
        sources=(source,),
        read_from="chart",
    )
