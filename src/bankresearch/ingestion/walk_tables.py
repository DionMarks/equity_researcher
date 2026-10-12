"""Reads a management walk out of a table."""

from __future__ import annotations

import re

from ..decomposition.taxonomy import map_label
from ..decomposition.walks import Walk, WalkStep
from ..domain.models import DocMeta, SourceRef
from ..domain.registry import get_registry
from ..domain.units import Unit
from .facts import parse_cell
from .parse_pdf import ParsedDocument, ParsedTable

__all__ = ["names_metric", "walk_from_table", "walks_in"]

_PERIOD_IN_LABEL = re.compile(r"\b([12]H\d{2}|FY\d{2})\b", re.IGNORECASE)
_PUNCTUATION = re.compile(r"[^a-z0-9]+")


def _normalise(text: str) -> str:
    return _PUNCTUATION.sub(" ", (text or "").lower()).strip()


def _period_in(label: str) -> str | None:
    match = _PERIOD_IN_LABEL.search(label or "")
    return match.group(1).upper() if match else None


def names_metric(text: str, metric_id: str, bank_id: str | None = None) -> bool:
    """Whether this text names the metric, and not a different one in scope."""
    registry = get_registry()
    haystack = _normalise(text)
    if not haystack:
        return False

    matched: set[str] = set()
    for candidate in registry.metrics:
        for name in registry.metric_names(candidate, bank_id):
            normalised = _normalise(name)
            if len(normalised) < 4:
                continue
            if normalised in haystack:
                matched.add(candidate)
                break
    return matched == {metric_id}


def walk_from_table(
    table: ParsedTable, metric_id: str, meta: DocMeta, value_column: str | None = None
) -> Walk | None:
    """A walk for this metric, if this table is one. ``None`` otherwise."""
    if len(table.rows) < 3:
        return None

    labelled = [row for row in table.rows if row and row[0].strip()]
    opening_index = next((i for i, row in enumerate(labelled) if _period_in(row[0])), None)
    closing_index = next(
        (
            i
            for i in range(len(labelled) - 1, -1, -1)
            if _period_in(labelled[i][0]) and i != opening_index
        ),
        None,
    )
    if opening_index is None or closing_index is None or closing_index <= opening_index:
        return None

    opening_period = _period_in(labelled[opening_index][0])
    closing_period = _period_in(labelled[closing_index][0])
    if opening_period is None or closing_period is None or opening_period == closing_period:
        return None

    # Bind the walk to its metric. Checked against the caption and the two period
    # rows, which is where an issuer says what the waterfall is of.
    context = " ".join((table.caption, labelled[opening_index][0], labelled[closing_index][0]))
    if not names_metric(context, metric_id, meta.bank_id):
        return None

    position = _value_position(table, value_column)
    if position is None:
        return None

    opening_value = parse_cell(_at(labelled[opening_index], position))
    closing_value = parse_cell(_at(labelled[closing_index], position))
    if opening_value is None or closing_value is None:
        return None

    steps: list[WalkStep] = []
    for row in labelled[opening_index + 1 : closing_index]:
        value = parse_cell(_at(row, position))
        if value is None:
            continue
        label = row[0].strip()
        steps.append(WalkStep(label=label, driver_id=map_label(metric_id, label), value=value))
    if not steps:
        return None

    unit: Unit = table.units or ("bps" if _looks_like_bps(table) else "aud_m")
    return Walk(
        opening_period=opening_period,
        closing_period=closing_period,
        opening_value=opening_value,
        closing_value=closing_value,
        steps=steps,
        unit=unit,
        source=SourceRef(
            doc_id=meta.doc_id,
            doc_type=meta.doc_type,
            page=table.page_no,
            table_id=table.table_id,
            publish_date=meta.publish_date,
        ),
    )


def _at(row: tuple[str, ...], position: int) -> str:
    return row[position] if position < len(row) else ""


def _value_position(table: ParsedTable, value_column: str | None) -> int | None:
    """The column holding the step values.

    A walk table usually has one value column. Where it has several, the caller names it;
    otherwise the last non-empty header wins, since that's where a single value column sits.
    """
    headers = list(table.header)
    if value_column is not None:
        lowered = [h.strip().lower() for h in headers]
        try:
            return lowered.index(value_column.strip().lower())
        except ValueError:
            return None
    candidates = [i for i, header in enumerate(headers) if header and header.strip()]
    if not candidates:
        return 1 if len(headers) > 1 else None
    return candidates[-1]


def _looks_like_bps(table: ParsedTable) -> bool:
    text = (table.caption + " " + " ".join(table.header)).lower()
    return "bps" in text or "basis point" in text


def walks_in(document: ParsedDocument, metric_id: str, meta: DocMeta) -> list[Walk]:
    """Every table in this document that reads as a walk for this metric."""
    found: list[Walk] = []
    for table in document.tables():
        walk = walk_from_table(table, metric_id, meta)
        if walk is not None:
            found.append(walk)
    return found
