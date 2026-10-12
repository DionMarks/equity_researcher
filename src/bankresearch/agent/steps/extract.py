"""Step 5: turning stored figures into evidence cards."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from ...domain.models import EvidenceCard, Fact, RequestSpec, SourceRef
from ...ingestion.facts import cell_under, line_item_index
from ...ingestion.parse_pdf import ParsedDocument

__all__ = ["cards_from_facts", "cards_from_walk", "quote_for", "short_quote"]

#: A cell holding one figure: digits with thousands separators, an optional sign or
#: brackets, a currency sign and a percent sign.
_CELL_FIGURE = re.compile(r"^\(?-?\$?\s?\d[\d,]*(?:\.\d+)?\)?%?$")


def _cell_figure(cell: str) -> Decimal | None:
    """The figure a table cell prints, or ``None`` if it prints none: "(1,234)" is -1234,
    "2.08%" is 2.08, and a dash or a word is nothing."""
    text = (cell or "").strip().replace("\u2212", "-")
    if not _CELL_FIGURE.match(text):
        return None
    try:
        value = Decimal(re.sub(r"[^\d.]", "", text))
    except InvalidOperation:
        return None
    return -value if text.startswith(("(", "-")) else value


def quote_for(fact: Fact, documents: dict[str, ParsedDocument]) -> str | None:
    """The cell this figure came from, as ``label | column | value``.

    A table cell never appears as running prose on the page, so a citation to one can't be a
    verbatim sentence. Quoting the cell's own coordinates can be checked against the stored
    table and says exactly what was read.
    """
    document = documents.get(fact.doc_id)
    if document is None or fact.table_id is None:
        return None
    table = document.table(fact.table_id)
    if table is None:
        return None
    printed = fact.published_value if fact.published_value is not None else fact.value
    found: list[tuple[str, str]] = []
    headed = [position for position, header in enumerate(table.header) if header.strip()]
    for row in table.rows:
        if not row:
            continue
        for position, header in enumerate(table.header):
            if fact.column_header and header != fact.column_header:
                continue
            # The cell the figure was read from, which can sit a column off its heading.
            cell = (
                cell_under(row, position, headed)
                if fact.column_header
                else (row[position] if position < len(row) else "")
            )
            # Compared as figures, not as text: a cell prints "10,252" or "(726)" for a
            # figure stored as 10252 or 726, and most of CBA's table figures went unquoted.
            figure = _cell_figure(cell)
            if figure is not None and abs(figure) == abs(printed):
                found.append((row[0], f"{row[0]} | {header} | {cell.strip()}"))
    if not found:
        return None
    # Where two rows print the same figure, the row that names the component: Judo's
    # underlying and statutory profit before tax were both $125.6m in FY25, and its
    # statutory figure was quoted under the underlying row's name.
    index = line_item_index(fact.bank_id)
    named = [quote for label, quote in found if fact.component in index.components_for(label)]
    return (named or [quote for _, quote in found])[0]


def cards_from_facts(
    facts: list[Fact],
    spec: RequestSpec,
    documents: dict[str, ParsedDocument] | None = None,
    start_at: int = 1,
) -> list[EvidenceCard]:
    """One evidence card per validated figure, in a stable order."""
    pages = documents or {}
    ordered = sorted(
        facts, key=lambda fact: (fact.period != spec.period, fact.component, fact.doc_id)
    )
    cards: list[EvidenceCard] = []
    for offset, fact in enumerate(ordered, start=start_at):
        cards.append(
            EvidenceCard(
                evidence_id=f"ev_{offset:03d}",
                component=fact.component,
                period=fact.period,
                value=fact.value,
                published_value=fact.published_value,
                unit=fact.unit,
                basis=fact.basis,
                statement=(
                    f"{fact.component} for {fact.period} is {fact.value} ({fact.unit}), "
                    f"read from a table on page {fact.page_no}."
                ),
                quote=quote_for(fact, pages),
                source=SourceRef(
                    doc_id=fact.doc_id,
                    doc_type=fact.doc_type,
                    page=fact.page_no,
                    table_id=fact.table_id,
                    publish_date=fact.publish_date,
                ),
                is_restated=fact.is_restated,
                method="table",
                checks_passed=["identity"] if fact.validated else [],
            )
        )
    return cards


def cards_from_walk(walk, spec: RequestSpec, start_at: int) -> list[EvidenceCard]:  # type: ignore[no-untyped-def]
    """One card per walk step, so a disclosed driver can cite the row it came from.

    A chained walk cites every leg that fed it, which is what makes corroboration across two
    documents real rather than just claimed.
    """
    cards: list[EvidenceCard] = []
    # A bar read off a chart is cited as a chart. Calling it a table claimed a cell the page
    # doesn't have, and sent a text-layer check looking for a number that's only in an image.
    # A step the issuer wrote out is cited as text, with its own sentence.
    read_from = getattr(walk, "read_from", "table")
    method = {"chart": "chart", "text": "text"}.get(read_from, "table")
    where = {"chart": "chart", "text": "written walk"}.get(read_from, "walk")
    for offset, step in enumerate(walk.steps, start=start_at):
        source = getattr(step, "source", None) or (
            walk.sources[0] if len(walk.sources) == 1 else walk.source
        )
        passage = getattr(step, "quote", None)
        cards.append(
            EvidenceCard(
                evidence_id=f"ev_{offset:03d}",
                component=f"{spec.metric_id}.walk.{step.driver_id or 'unmapped'}",
                period=spec.period,
                value=step.value,
                unit=walk.unit,
                statement=(
                    f"The issuer's own {where} attributes {step.value}{walk.unit} to {step.label}."
                ),
                quote=short_quote(passage)
                if passage
                else f"{step.label} | {walk.unit} | {step.value}",
                passage=passage,
                source=source,
                method="text" if passage else method,  # type: ignore[arg-type]
                checks_passed=["walk_checksum", *(["quote_on_page"] if passage else [])],
            )
        )
    return cards


#: A quote is 25 words or fewer.
_QUOTE_WORDS = 25


def short_quote(passage: str) -> str:
    """The first 25 words of a sentence, marked as cut where it was."""
    words = passage.split()
    if len(words) <= _QUOTE_WORDS:
        return passage
    return " ".join(words[: _QUOTE_WORDS - 1]) + " ..."
