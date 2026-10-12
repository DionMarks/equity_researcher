"""Which rows hold the figures a question needs, for a bank whose wording isn't configured."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import yaml

from ...domain.models import DocMeta
from ...ingestion.facts import LineItemIndex, cell_under, clean_label, parse_cell, without_stamp
from ...ingestion.parse_pdf import ParsedDocument
from ...ingestion.tables import map_columns
from ...llm.budget import Budget
from ...llm.client import LLMClient, LLMRequest, LLMUnavailable, ModelUnreachable
from ...llm.tracing import PromptLibrary, Tracer, default_library
from ...settings import get_settings

__all__ = ["OfferedRow", "RowMapping", "map_rows", "offer_rows"]

#: Rows shown for each missing figure, closest first. A results document has thousands of
#: rows; the figure is almost always among the few whose words are closest to its name.
ROWS_PER_FIGURE: Final = 12
_WORD: Final = re.compile(r"[a-z0-9]+")
_COMMON: Final = frozenset({"the", "of", "and", "to", "for", "on", "in", "a", "m", "s"})


@dataclass(frozen=True, slots=True)
class OfferedRow:
    """One row nothing matched, as the model is shown it."""

    number: int
    label: str
    page_no: int
    caption: str
    cells: tuple[tuple[str, str], ...]
    """Period and cell, for the columns that map to the periods asked about."""

    def line(self) -> str:
        cells = " | ".join(f"{period}: {cell}" for period, cell in self.cells)
        caption = f" | table: {self.caption}" if self.caption else ""
        return f"{self.number}. p.{self.page_no}{caption} | {self.label} | {cells}"


@dataclass
class RowMapping:
    """The labels a model matched, by component, and what was refused."""

    labels: dict[str, str] = field(default_factory=dict)
    """Label as printed (cleaned of footnotes and units) to the component it holds."""
    refused: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    settled: bool = False
    """The question was answered: the model replied, or no row was close enough to ask
    about. Not when the call was refused or failed, so it's asked again next time."""


@lru_cache(maxsize=4)
def _vocabulary(config_dir: str) -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(
        (Path(config_dir) / "line_items.yaml").read_text(encoding="utf-8")
    )
    return raw


def meanings(config_dir: Path | None = None) -> dict[str, str]:
    """What each figure is, in plain words."""
    raw = _vocabulary(str(config_dir or get_settings().config_dir))
    return {str(k): str(v) for k, v in (raw.get("meanings") or {}).items()}


def _words(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.lower()) if word not in _COMMON}


#: The one word a figure can be named by on its own.
_OWN_WORDS: Final = {"tax": "tax", "nim": "nim", "roe": "roe", "rwa": "rwa"}


def _figure_words(component: str, config_dir: Path | None = None) -> set[str]:
    """The words a row holding ``component`` is likely to use: its labels and meaning."""
    raw = _vocabulary(str(config_dir or get_settings().config_dir))
    spec = {**(raw.get("headline") or {}), **(raw.get("components") or {})}.get(component) or {}
    labels = [*(spec.get("synonyms") or ()), meanings(config_dir).get(component, "")]
    words: set[str] = set()
    for label in labels:
        words |= _words(label)
    return words - {"millions", "per", "cent", "whole", "group", "its", "bank"}


def offer_rows(
    document: ParsedDocument,
    meta: DocMeta,
    fy_end_month: int,
    periods: Sequence[str],
    index: LineItemIndex,
    missing: Sequence[str],
) -> list[OfferedRow]:
    """The rows nothing matched that are most likely to hold a missing figure.

    Only group pages, only rows with a figure in a column for one of ``periods``, and only
    rows whose words share something with the figure's name. For each missing figure, the
    closest rows by words in common, earliest page first among equals.
    """
    wanted = {p.upper() for p in periods}
    found: dict[str, tuple[str, int, str, tuple[tuple[str, str], ...]]] = {}
    for page in document.pages:
        if index.is_divisional_page(page.text, page.heading_path):
            continue
        for table in page.tables:
            headers = [h for h in table.header if h and h.strip()]
            if not headers:
                continue
            usable: dict[int, str] = {}
            for point_in_time in (False, True):
                try:
                    mapping = map_columns(
                        headers,
                        fy_end_month=fy_end_month,
                        doc_period=meta.period,
                        point_in_time=point_in_time,
                    )
                except ValueError:
                    continue
                for position, header in enumerate(table.header):
                    period = mapping.get(header) if header else None
                    if period is not None and period.upper() in wanted:
                        usable.setdefault(position, period.upper())
            if not usable:
                continue
            headed = [position for position, header in enumerate(table.header) if header.strip()]
            for row in table.rows:
                if not row or not row[0].strip():
                    continue
                # Named as the bank names it, without the ASX's stamp in front, so a name kept
                # with the bank matches the row in a copy that isn't stamped.
                label = without_stamp(clean_label(row[0]))
                key = " ".join(_WORD.findall(label.lower()))
                if not key or key in found:
                    continue
                if index.components_for(row[0]) or index.is_excluded(row[0]):
                    continue
                under = {
                    period: cell_under(row, position, headed).strip()
                    for position, period in sorted(usable.items())
                }
                cells = tuple(
                    (period, cell) for period, cell in under.items() if parse_cell(cell) is not None
                )
                if cells:
                    found[key] = (label, page.page_no, table.caption.strip()[:80], cells)

    chosen: dict[str, tuple[str, int, str, tuple[tuple[str, str], ...]]] = {}
    order = list(found)
    for component in missing:
        vocabulary = _figure_words(component)
        scored = [
            (len(_words(found[key][0]) & vocabulary), -position, key)
            for position, key in enumerate(order)
        ]
        best = sorted((s for s in scored if s[0] > 0), reverse=True)[:ROWS_PER_FIGURE]
        for _, _, key in best:
            chosen.setdefault(key, found[key])
    ranked = sorted(chosen.values(), key=lambda item: item[1])
    return [
        OfferedRow(number, label, page_no, caption, cells)
        for number, (label, page_no, caption, cells) in enumerate(ranked, start=1)
    ]


def map_rows(
    missing: Sequence[str],
    rows: Sequence[OfferedRow],
    client: LLMClient,
    *,
    model: str,
    effort: str,
    bank_name: str,
    budget: Budget | None = None,
    tracer: Tracer | None = None,
    prompts: PromptLibrary | None = None,
) -> RowMapping:
    """Ask which offered row, if any, holds each missing figure, and check the answer."""
    outcome = RowMapping()
    if not rows or not missing:
        outcome.settled = True
        return outcome
    if not client.available:
        outcome.notes.append("matching the bank's own row names needs a model, and none is set")
        return outcome
    if budget is not None and not budget.may_call("map_rows"):
        outcome.notes.append("matching the bank's own row names stopped on its budget")
        return outcome

    described = meanings()
    prompt = (prompts or default_library()).get("map_rows")
    request = LLMRequest(
        step="map_rows",
        model=model,
        effort=effort,
        system=prompt.text,
        messages=(
            {
                "role": "user",
                "content": (
                    f"The bank: {bank_name}.\n\nMeasures wanted:\n"
                    + "\n".join(f"- {c}: {described.get(c, c)}" for c in missing)
                    + "\n\nRows (number, page, table, label, then each cell by period):\n"
                    + "\n".join(row.line() for row in rows)
                ),
            },
        ),
        output_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["mappings"],
            "properties": {
                "mappings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["row", "component"],
                        "properties": {
                            "row": {"type": "integer"},
                            "component": {"type": "string"},
                        },
                    },
                }
            },
        },
    )
    try:
        response = client.complete(request)
    except LLMUnavailable as error:
        if isinstance(error, ModelUnreachable):  # the run stops; see llm.client
            raise
        outcome.notes.append(f"matching the bank's own row names stopped: {error}")
        return outcome
    if budget is not None:
        budget.record("map_rows", model, response.usage, cached=response.cached)
    if tracer is not None:
        tracer.model_call(request, response, prompt=prompt)
    outcome.settled = True

    by_number = {row.number: row for row in rows}
    proposed: dict[str, list[OfferedRow]] = {}
    answers = (response.structured or {}).get("mappings") or []
    for answer in answers if isinstance(answers, list) else ():
        if not isinstance(answer, dict):
            continue
        number = answer.get("row")
        row = by_number.get(number) if isinstance(number, int) else None
        component = str(answer.get("component") or "")
        if row is None:
            outcome.refused[str(answer.get("row"))] = "not one of the rows offered"
            continue
        if component not in missing:
            outcome.refused[row.label] = f"{component!r} wasn't one of the figures asked for"
            continue
        if not _specific(row.label, component):
            outcome.refused[row.label] = "one ordinary word, which would match too much"
            continue
        proposed.setdefault(component, []).append(row)

    for component, matched in proposed.items():
        if len({row.label for row in matched}) > 1:
            for row in matched:
                outcome.refused[row.label] = f"one of several rows proposed for {component}"
            continue
        outcome.labels[matched[0].label] = component
    if outcome.labels:
        outcome.notes.append(
            "rows named in the bank's own words, matched by a model: "
            + "; ".join(f"{label!r} as {component}" for label, component in outcome.labels.items())
        )
    if outcome.refused:
        outcome.notes.append(
            "proposed matches refused: "
            + "; ".join(f"{label!r} ({why})" for label, why in outcome.refused.items())
        )
    return outcome


def _specific(label: str, component: str) -> bool:
    """Whether a label says enough to be kept as a name: two words, an abbreviation, or the
    figure's own word. BNK labels its tax line just "Tax", which names nothing else; "Total"
    or "Interest" would match rows all over every later filing."""
    words = label.split()
    if len(words) >= 2:
        return True
    if len(words) != 1:
        return False
    word = words[0]
    return (word.isupper() and len(word) >= 3) or word.lower() == _OWN_WORDS.get(component)
