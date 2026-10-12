"""Turns a filing into pages, tables and an outline."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, Protocol

from ..domain.units import Unit
from .layout_tables import extract_layout_tables
from .outline import outline_from_spans, spans_from_pdf
from .tables import detect_restated, detect_units, names_a_period

__all__ = [
    "PageKind",
    "ParsedDocument",
    "ParsedPage",
    "ParsedTable",
    "PdfParser",
    "PyMuPdfParser",
    "classify_page",
    "has_row_labels",
    "looks_like_a_table",
    "page_features",
]

PageKind = Literal["text", "table", "chart", "mixed", "boilerplate"]

_BOILERPLATE = re.compile(
    r"disclaimer|glossary|important notice|forward[- ]looking statement|"
    r"this (?:document|presentation) (?:is|has been) prepared|"
    r"no representation or warranty|definitions? of (?:non-)?statutory",
    re.IGNORECASE,
)

# Keywords that mark a page as likely to carry a walk, so chart vision is run
# only where it could pay for itself.
#: What a page with a walk on it says. A smaller bank can title its walk with nothing but the
#: two periods: BNK Banking Corporation's FY26 margin walk is headed "NIM FY25 to FY26" (deck
#: p.19), and without the last pattern it was never a candidate.
_WALK_HINTS = re.compile(
    r"movement|walk|waterfall|\bbps\b|basis points|bridge|reconciliation"
    r"|\b(?:FY|1H|2H)\s?\d{2}\s+(?:to|vs\.?)\s+(?:FY|1H|2H)\s?\d{2}\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ParsedTable:
    """One extracted table, with everything a citation and a fact need."""

    table_id: str
    page_no: int
    caption: str
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    units: Unit | None = None
    restated: bool = False

    def to_markdown(self) -> str:
        if not self.header:
            return ""
        lines = [
            "| " + " | ".join(self.header) + " |",
            "| " + " | ".join("---" for _ in self.header) + " |",
        ]
        for row in self.rows:
            padded = list(row) + [""] * (len(self.header) - len(row))
            lines.append("| " + " | ".join(padded[: len(self.header)]) + " |")
        return "\n".join(lines)

    def cell(self, row_label: str, column: str) -> str | None:
        """The cell at a row label and column header, matched case-insensitively."""
        try:
            index = [h.strip().lower() for h in self.header].index(column.strip().lower())
        except ValueError:
            return None
        wanted = row_label.strip().lower()
        for row in self.rows:
            if row and row[0].strip().lower() == wanted and index < len(row):
                return row[index]
        return None


@dataclass(frozen=True, slots=True)
class ParsedPage:
    page_no: int
    text: str
    kind: PageKind
    heading_path: tuple[str, ...] = ()
    tables: tuple[ParsedTable, ...] = ()

    @property
    def likely_walk_page(self) -> bool:
        return bool(_WALK_HINTS.search(self.text))


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    doc_id: str
    sha256: str
    page_count: int
    pages: tuple[ParsedPage, ...] = ()
    outline: tuple[tuple[str, int], ...] = field(default=())
    """(heading, first page) pairs, from the PDF's own table of contents."""

    def page(self, page_no: int) -> ParsedPage | None:
        return next((p for p in self.pages if p.page_no == page_no), None)

    def tables(self) -> tuple[ParsedTable, ...]:
        return tuple(table for page in self.pages for table in page.tables)

    def table(self, table_id: str) -> ParsedTable | None:
        return next((t for t in self.tables() if t.table_id == table_id), None)

    def walk_pages(self) -> tuple[int, ...]:
        return tuple(p.page_no for p in self.pages if p.likely_walk_page)


def page_features(text: str, table_count: int, image_area_share: float) -> dict[str, float]:
    """The measurable properties page classification is based on."""
    characters = len(text)
    digits = sum(1 for character in text if character.isdigit())
    return {
        "characters": float(characters),
        "numeric_density": (digits / characters) if characters else 0.0,
        "table_count": float(table_count),
        "image_area_share": image_area_share,
    }


def classify_page(text: str, table_count: int, image_area_share: float) -> PageKind:
    """Label a page from its features, before any model is consulted.

    Boilerplate is checked first and wins outright: a disclaimer page is dense
    with text and numbers and would otherwise look like content.
    """
    if _BOILERPLATE.search(text):
        return "boilerplate"

    features = page_features(text, table_count, image_area_share)
    has_tables = features["table_count"] >= 1
    chart_like = features["image_area_share"] >= 0.25
    wordy = features["characters"] >= 400

    if has_tables and chart_like:
        return "mixed"
    if chart_like:
        return "chart"
    if has_tables:
        return "mixed" if wordy else "table"
    return "text"


class PdfParser(Protocol):
    """Parsing is swappable, so a better table engine needs no caller change."""

    def parse(self, path: Path, doc_id: str) -> ParsedDocument: ...


RENDER_DPI = 150
"""Resolution for a page rendered for vision.

At 150dpi the small print on these charts (a "(3)bps" set in 6pt above a bar) is just
legible, and below that it isn't. Higher resolutions cost more tokens without making
anything more readable, and the bars themselves aren't measured, since an unlabelled bar has
no value.
"""


def render_page(path: Path, page_no: int, dpi: int = RENDER_DPI) -> tuple[bytes, str]:
    """One page as a PNG, for the chart reader.

    Returned as bytes with its media type rather than written to disk. A rendered page is
    derived data: the citation belongs to the PDF and the page number, not to an image file
    someone could later replace.
    """
    import pymupdf

    with pymupdf.open(path) as document:  # type: ignore[no-untyped-call]
        if not 1 <= page_no <= document.page_count:
            raise ValueError(f"{path.name} has no page {page_no}")
        pixmap = document[page_no - 1].get_pixmap(dpi=dpi)
        return bytes(pixmap.tobytes("png")), "image/png"


class PyMuPdfParser:
    """PyMuPDF for text, outline and page geometry; pdfplumber for tables."""

    def __init__(self, max_pages: int | None = None) -> None:
        self.max_pages = max_pages

    def parse(self, path: Path, doc_id: str) -> ParsedDocument:
        import pymupdf

        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        ruled_by_page = self._extract_tables(path, doc_id)

        # pymupdf ships py.typed but leaves Document.__init__ unannotated.
        with pymupdf.open(path) as document:  # type: ignore[no-untyped-call]
            outline = tuple(
                (entry[1].strip(), int(entry[2]))
                for entry in (document.get_toc() or [])
                if len(entry) >= 3 and entry[2] >= 1
            )
            if not outline:
                # Results releases and investor packs often have no outline, and without
                # headings every page looks the same to retrieval and the divisional guard
                # has nothing to go on. Headings are only recovered from typography here; a
                # document's own outline is never overridden by a heuristic.
                outline = outline_from_spans(spans_from_pdf(path, self.max_pages))
            limit = self.max_pages or document.page_count
            pages: list[ParsedPage] = []
            for index in range(min(limit, document.page_count)):
                page = document[index]
                page_no = index + 1
                text = page.get_text("text") or ""
                page_area = abs(page.rect.width * page.rect.height) or 1.0
                image_area = sum(
                    abs(block[2] - block[0]) * abs(block[3] - block[1])
                    for block in page.get_text("blocks")
                    if len(block) > 6 and block[6] == 1
                )
                # Use ruled tables where they exist, and also rebuild from word positions
                # where none of them names a period. A ruled table whose header doesn't
                # survive line detection is worse than no table: it can't be read, and
                # keeping it alone stopped the rebuild that could have read it. CBA rules
                # the page with its return on equity, and pdfplumber reduces that header to
                # ``['', '', 'Half Year Ended', '', '', '', '']``. Walk tables aren't keyed
                # by period at all, so they're added rather than replaced.
                #
                # A ruled table that kept its periods but lost its row labels is no better.
                # ANZ's 2025 full-year summary comes out of line detection as
                # ``('9,092', '', '17,961')`` under ``Sep 25 | | Sep 25``: the label column
                # and the middle of the table are gone, and the margin, return on equity
                # and cost-to-income rows can't be told apart.
                ruled = ruled_by_page.get(page_no, ())
                page_tables = ruled
                if not any(
                    names_a_period(cell)
                    for table in ruled
                    if has_row_labels(table.rows)
                    for cell in table.header
                ):
                    page_tables = (
                        *ruled,
                        *extract_layout_tables(
                            [(w[0], w[1], w[2], w[3], w[4]) for w in page.get_text("words")],
                            page_no,
                            doc_id,
                            text,
                        ),
                    )
                pages.append(
                    ParsedPage(
                        page_no=page_no,
                        text=text,
                        kind=classify_page(text, len(page_tables), image_area / page_area),
                        heading_path=self._heading_for(outline, page_no),
                        tables=page_tables,
                    )
                )
            page_count = document.page_count

        return ParsedDocument(
            doc_id=doc_id,
            sha256=digest,
            page_count=page_count,
            pages=tuple(pages),
            outline=outline,
        )

    @staticmethod
    def _heading_for(outline: tuple[tuple[str, int], ...], page_no: int) -> tuple[str, ...]:
        """The nearest outline entry at or before this page."""
        best: str | None = None
        for heading, start in outline:
            if start <= page_no:
                best = heading
            else:
                break
        return (best,) if best else ()

    @staticmethod
    def _extract_tables(path: Path, doc_id: str) -> dict[int, tuple[ParsedTable, ...]]:
        import pdfplumber

        found: dict[int, list[ParsedTable]] = {}
        with pdfplumber.open(path) as document:
            for index, page in enumerate(document.pages):
                page_no = index + 1
                # Ruled extraction is the slow step, and most of a real pack has no rules at
                # all. Pages with no lines or rectangles are skipped before trying; checking
                # the drawing primitives costs almost nothing compared with an extraction
                # that will find nothing. On the four majors' 2026 filings this saves most
                # of the ingestion time.
                if not (page.lines or page.rects):
                    continue
                page_text = page.extract_text() or ""
                extracted = page.extract_tables()
                for position, raw in enumerate(extracted or []):
                    cleaned = [
                        [(cell or "").strip().replace("\n", " ") for cell in row]
                        for row in raw
                        if row and any((cell or "").strip() for cell in row)
                    ]
                    if not looks_like_a_table(cleaned):
                        continue
                    header = tuple(cleaned[0])
                    rows = tuple(tuple(row) for row in cleaned[1:])
                    caption = _caption_above(page_text, header)
                    found.setdefault(page_no, []).append(
                        ParsedTable(
                            table_id=f"{doc_id}_p{page_no}_t{position}",
                            page_no=page_no,
                            caption=caption,
                            header=header,
                            rows=rows,
                            units=(
                                detect_units(caption + " " + " ".join(header))
                                or _thousands_below(rows)
                            ),
                            restated=detect_restated(caption + " " + page_text),
                        )
                    )
        return {page: tuple(items) for page, items in found.items()}


def _thousands_below(rows: tuple[tuple[str, ...], ...]) -> Unit | None:
    """Thousands, when the first rows under the header say so and hold no figure.

    BNK Banking Corporation prints "Amounts in thousands of AUD" and "($'000s)" on the line
    under its key-metrics header (FY26 report, p.9). Only thousands is read from there: a
    "%" or "$m" under one column is that column's unit, not the table's.
    """
    for row in rows[:2]:
        if any(_FIGURE.search(cell) for cell in row):
            continue
        if detect_units(" ".join(row)) == "aud_k":
            return "aud_k"
    return None


#: A cell that holds a figure rather than a word or a unit.
_FIGURE: Final = re.compile(r"^\(?-?\$?\s?\d[\d,]*(?:\.\d+)?\)?%?$")


def looks_like_a_table(rows: list[list[str]]) -> bool:
    """Whether an extraction is a table rather than text that happened to line up.

    The text-alignment fallback is needed for packs typeset without rules, but it will also
    carve a heading or a disclaimer paragraph into columns. A real financial table has at
    least two columns, at least two data rows, and figures somewhere other than the label
    column, so that's what's required.
    """
    if len(rows) < 3:
        return False
    width = max(len(row) for row in rows)
    if width < 2:
        return False
    data_rows = rows[1:]
    numeric = sum(1 for row in data_rows if any(_has_number(cell) for cell in row[1:]))
    return numeric >= max(2, len(data_rows) // 2)


def has_row_labels(rows: tuple[tuple[str, ...], ...] | list[list[str]]) -> bool:
    """Whether most rows carrying a figure also say what the figure is.

    A table is read by its row labels, so one whose first column is all figures has lost
    them, usually to line detection merging the label column into its neighbours.
    """
    with_figures = [row for row in rows if any(_has_number(cell) for cell in row)]
    if not with_figures:
        return False
    labelled = sum(1 for row in with_figures if _LETTER.search(row[0] or ""))
    return labelled * 2 >= len(with_figures)


_NUMBER_CELL = re.compile(r"\d")
_LETTER = re.compile(r"[A-Za-z]")


def _has_number(cell: str) -> bool:
    return bool(_NUMBER_CELL.search(cell or ""))


def _caption_above(page_text: str, header: tuple[str, ...]) -> str:
    """The line immediately above the header row, which is usually the caption."""
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    first_header_cell = next((cell for cell in header if cell), "")
    if not first_header_cell:
        return lines[0] if lines else ""
    for index, line in enumerate(lines):
        if first_header_cell.lower() in line.lower():
            return lines[index - 1] if index else ""
    return lines[0] if lines else ""
