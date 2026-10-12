"""Rebuilds tables from word positions, which is how real packs are typeset."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..domain.units import Unit
from .tables import detect_restated, detect_units, names_a_period

if TYPE_CHECKING:  # pragma: no cover - import cycle at type-check time only
    from .parse_pdf import ParsedTable

__all__ = ["Column", "Word", "extract_layout_tables", "group_lines", "header_columns"]

_MONTHS = (
    "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
    "january|february|march|april|june|july|august|september|october|november|december"
)
# "Mar 26", "Mar 2026" as two tokens, or a fiscal label as one.
_MONTH_TOKEN = re.compile(rf"^({_MONTHS})$", re.IGNORECASE)
_YEAR_TOKEN = re.compile(r"^(\d{2}|\d{4})$")
_FISCAL_TOKEN = re.compile(r"^(?:[12]H|H[12]|HY|FY)\s?(?:FY)?\d{2,4}$", re.IGNORECASE)
# A month and year set as one word: Judo heads its analyst pack "Jun-25 | Jun-24 | YoY".
_MONTH_YEAR_TOKEN = re.compile(rf"^(?:{_MONTHS})[-'/](?:\d{{2}}|\d{{4}})$", re.IGNORECASE)
# A movement column: "v.", "v", "vs" immediately before the period.
_VERSUS = re.compile(r"^(?:v|v\.|vs|vs\.|versus)$", re.IGNORECASE)
# A dash between two periods means the same: Westpac heads a movement column "Mar 25" over
# "- Sept 24". It counts only when a period follows it, since a dash alone is a nil figure.
_DASH = re.compile("^[-\u2013]$")
# A movement column named from the left: "Jun 26 vs", "Jun 25 %", "Dec 25 %". The
# marker follows the period rather than preceding it, which is how a comparison written
# across two header lines reads on each of them.
_TRAILING_MOVEMENT = re.compile(r"^(?:%|v|v\.|vs|vs\.|versus|mov|mov't|movement)$", re.IGNORECASE)

_NUMERIC = re.compile(r"^\(?-?\$?\s?[\d,]+(?:\.\d+)?\)?%?$")
# A bare dash holds a column position without carrying a value.
_PLACEHOLDER = re.compile("^[-\u2013\u2014]$")

_LINE_TOLERANCE = 3.0
_COLUMN_TOLERANCE = 30.0
_MIN_PERIOD_COLUMNS = 2
_MAX_BLANK_LINES = 2
_MIN_ROWS = 2
# A trailing movement marker only belongs to the period when it's set right up against it.
# CBA writes "Jun 26 vs" with a 2pt gap; a period column followed by a separate movement
# column leaves 80pt or more.
_ADJACENT = 12.0


@dataclass(frozen=True, slots=True)
class Word:
    x0: float
    x1: float
    text: str

    @property
    def centre(self) -> float:
        return (self.x0 + self.x1) / 2


@dataclass(frozen=True, slots=True)
class Column:
    """One period column, identified by the x span of its header tokens."""

    label: str
    x0: float
    x1: float

    @property
    def centre(self) -> float:
        return (self.x0 + self.x1) / 2

    def distance_to(self, word: Word) -> float:
        return abs(word.centre - self.centre)


def group_lines(words: list[tuple[float, float, float, float, str]]) -> list[list[Word]]:
    """Group positioned words into lines, each sorted left to right.

    Words are grouped by how close their baselines are, not by rounding each one to a fixed
    band. Rounding looks the same but isn't: two words 1pt apart land either side of a band
    edge maybe one time in three, and when they do the row gets torn in half. On NAB's
    divisional pages that tore "Other operating income 526 517 515" into a lone ``526`` and
    a three-cell row missing its current period. Nothing later can notice a dropped figure
    like that, because what's left is a well-formed row.
    """
    entries = sorted((y0, x0, x1, text.strip()) for x0, y0, x1, _y1, text in words if text.strip())
    lines: list[list[Word]] = []
    current: list[Word] = []
    baseline: float | None = None
    for y0, x0, x1, text in entries:
        if baseline is not None and y0 - baseline > _LINE_TOLERANCE:
            lines.append(sorted(current, key=lambda w: w.x0))
            current, baseline = [], None
        if baseline is None:
            baseline = y0
        current.append(Word(x0, x1, text))
    if current:
        lines.append(sorted(current, key=lambda w: w.x0))
    return lines


def header_columns(
    line: list[Word], below: list[Word] | None = None, merge_repeats: bool = True
) -> list[Column]:
    """Period columns named by this line, with movement columns left out.

    A movement column is recognised in three ways. Usually "v." sits right before the period
    on the same line. It can also come after it (CBA heads its comparison columns "Jun 26
    vs" over "Jun 25 %"), so a period followed by "vs" or "%" counts too. And these packs
    also split the marker over two lines, "Mar 26" on one and "v. Sep 25" below, so the line
    below is checked for a "v." at the same x position.

    The trailing form matters as much as the leading one. CBA's full-year summary prints
    three blocks side by side (statutory full year, cash full year, cash half year), each
    with its own comparison column, so "Jun 25" shows up both as a period and as the
    right-hand half of "Jun 26 vs Jun 25 %". Without this the margin column won on position
    and every row's figure was read as its percentage movement: net interest income of
    $24,023m became 7.

    Finally, since a period can't legitimately head two columns in one table, a repeated
    label keeps only its leftmost occurrence.
    """
    columns: list[Column] = []
    index = 0
    while index < len(line):
        word = line[index]
        # Only a marker set against this period belongs to it. NAB heads its full-year
        # summary "Sep 25 v | Sep 25 | Mar 25", and the "v" closing the movement header
        # took the half-year "Sep 25" 31pt to its right for a movement too. With that
        # column gone, the two blocks read as one table of halves.
        preceded_by_versus = (
            index > 0
            and (_VERSUS.match(line[index - 1].text) or _DASH.match(line[index - 1].text))
            and word.x0 - line[index - 1].x1 <= _ADJACENT
        )

        def movement_follows(after: int) -> bool:
            """Whether a movement marker is set right up against the period before it.

            Adjacency is what separates "Jun 26 vs" (one comparison header) from "Mar 26" as
            its own column with "v. Sep 25" heading the next one.
            """
            if after >= len(line) or not _TRAILING_MOVEMENT.match(line[after].text):
                return False
            return line[after].x0 - line[after - 1].x1 <= _ADJACENT

        if _FISCAL_TOKEN.match(word.text) or _MONTH_YEAR_TOKEN.match(word.text):
            if not preceded_by_versus and not movement_follows(index + 1):
                columns.append(Column(word.text, word.x0, word.x1))
            index += 1
            continue

        if (
            _MONTH_TOKEN.match(word.text)
            and index + 1 < len(line)
            and _YEAR_TOKEN.match(line[index + 1].text)
            and not _day_of_the_next_column(line, index + 1, word, below)
        ):
            year = line[index + 1]
            if not preceded_by_versus and not movement_follows(index + 2):
                columns.append(Column(f"{word.text} {year.text}", word.x0, year.x1))
            index += 2
            continue

        # A month with its year set on the line below, as Westpac's 1H25 pack heads its
        # columns: "March | Sept | March" over "2025 | 2024 | 2024".
        if _MONTH_TOKEN.match(word.text) and below is not None:
            year_below = _year_under(word, below)
            if year_below is not None:
                if not preceded_by_versus and not movement_follows(index + 1):
                    columns.append(
                        Column(
                            f"{word.text} {year_below.text}",
                            min(word.x0, year_below.x0),
                            max(word.x1, year_below.x1),
                        )
                    )
                index += 1
                continue

        index += 1

    if below:
        versus_positions = _versus_positions(below)
        columns = [
            column
            for column in columns
            if not any(abs(x - column.x0) <= _COLUMN_TOLERANCE for x in versus_positions)
        ]

    if not merge_repeats:
        return columns

    deduplicated: list[Column] = []
    seen: set[str] = set()
    for column in columns:
        key = column.label.lower()
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(column)
    return deduplicated


def _day_of_the_next_column(
    line: list[Word], at: int, month: Word, below: list[Word] | None
) -> bool:
    """Whether the number after a month is the day of the next column's date, not this
    month's year.

    Westpac's 1H25 capital table heads its columns "31 March | 30 Sept | 31 March" over
    "2025 | 2024 | 2024" (p.34). Read as a month and its year, "March 30" and "Sept 31" took
    the columns to 2030 and 2031, and the CET1 ratio was never stored. A number is a day
    when it's set against the month after it and a four-digit year sits under this month.
    """
    number = line[at]
    following = line[at + 1] if at + 1 < len(line) else None
    return (
        len(number.text) <= 2
        and int(number.text) <= 31
        and following is not None
        and _MONTH_TOKEN.match(following.text) is not None
        and following.x0 - number.x1 <= _ADJACENT
        and below is not None
        and _year_under(month, below) is not None
    )


def _year_under(word: Word, below: list[Word]) -> Word | None:
    """A year printed directly under this word on the next line, if one is."""
    for candidate in below:
        if (
            _YEAR_TOKEN.match(candidate.text)
            and len(candidate.text) == 4
            and candidate.x0 <= word.x1 + 4
            and candidate.x1 >= word.x0 - 4
        ):
            return candidate
    return None


def _years_under_header(below: list[Word] | None, columns: list[Column]) -> bool:
    """Whether the next line only finishes the header: a year under each of two or more
    of its columns, and nothing that reads as a figure elsewhere."""
    if not below:
        return False
    years = [
        word
        for word in below
        if _YEAR_TOKEN.match(word.text)
        and len(word.text) == 4
        and any(word.x0 <= column.x1 + 4 and word.x1 >= column.x0 - 4 for column in columns)
    ]
    return len(years) >= 2


def _versus_positions(below: list[Word]) -> list[float]:
    """Where the line below starts the second half of a movement header: a "v." anywhere,
    or a dash set right up against the period that follows it."""
    positions: list[float] = []
    for index, word in enumerate(below):
        if _VERSUS.match(word.text):
            positions.append(word.x0)
        elif _DASH.match(word.text) and index + 1 < len(below):
            following = below[index + 1]
            names_a_period = (
                _MONTH_TOKEN.match(following.text)
                or _MONTH_YEAR_TOKEN.match(following.text)
                or _FISCAL_TOKEN.match(following.text)
            )
            if names_a_period and following.x0 - word.x1 <= _ADJACENT:
                positions.append(word.x0)
    return positions


def split_into_blocks(columns: list[Column]) -> list[list[Column]]:
    """One table's columns, cut into the blocks its header actually describes.

    A results summary is often three tables typeset as one. CBA's full-year summary sets the
    statutory full year, the cash full year and the cash half year side by side, each with
    its own comparison column, so "30 Jun 26" heads three columns that mean three different
    things::

        Full Year Ended   Full Year Ended   Half Year Ended
        (statutory basis) (cash basis)      (cash basis)
        30 Jun 26  vs     30 Jun 26 30 Jun 25  vs   30 Jun 26 31 Dec 25  vs

    Keeping only the leftmost of each repeated label reads the statutory current year
    against the cash prior year: two correctly extracted figures, a movement that never
    happened, and a cost-to-income ratio 22bp off the one the bank published. So a repeat
    starts a new block instead, and each block is offered as its own table. A block naming
    fewer than two periods is dropped, which is what discards the statutory column here,
    since on its own it has nothing to compare against.
    """
    blocks: list[list[Column]] = []
    current: list[Column] = []
    seen: set[str] = set()
    for column in columns:
        key = column.label.lower()
        if key in seen:
            blocks.append(current)
            current, seen = [], set()
        seen.add(key)
        current.append(column)
    if current:
        blocks.append(current)
    return blocks


def _is_value(text: str) -> bool:
    return bool(_NUMERIC.match(text)) or bool(_PLACEHOLDER.match(text))


def _assign(word: Word, columns: list[Column]) -> int | None:
    """The column this figure belongs to, by overlap first and then proximity."""
    for position, column in enumerate(columns):
        if word.x0 <= column.x1 and word.x1 >= column.x0:
            return position
    nearest = min(range(len(columns)), key=lambda i: columns[i].distance_to(word))
    return nearest if columns[nearest].distance_to(word) <= _COLUMN_TOLERANCE else None


def _row_from(
    line: list[Word], columns: list[Column], label_edge: float | None = None
) -> tuple[str, list[str]] | None:
    """Split a line into its label and one cell per column, or ``None``.

    A line with a label and no figures is returned too, with empty cells. These are section
    headings ("Retail Banking Services", "Level 1 regulatory capital structure", "Group
    performance - cash earnings basis"), and dropping them for having no figures is what let
    a division's margin be read as the group's. What they mean is decided in
    :mod:`bankresearch.ingestion.sections`; here they're just kept.

    ``label_edge`` is where the table's label column ends, when these columns are a block
    further right. Words between it and the block belong to the blocks to the left, and are
    neither label nor figure here. Without it, ANZ's full-year block was labelled "Net
    interest margin 1.54% 1.56%", with the half-year figures glued on, and no row matched.
    """
    left_edge = min(column.x0 for column in columns) - 6.0
    label_end = left_edge if label_edge is None else min(label_edge, left_edge)
    label_words = [w.text for w in line if w.x1 <= label_end]
    cells = [""] * len(columns)
    for word in line:
        if word.x1 <= left_edge or not _is_value(word.text):
            continue
        position = _assign(word, columns)
        if position is None:
            continue  # a figure outside every period column, e.g. a page reference
        if not cells[position]:
            cells[position] = word.text
    if not any(cells) and not label_words:
        return None
    return " ".join(label_words).strip(), cells


def extract_layout_tables(
    page_words: list[tuple[float, float, float, float, str]],
    page_no: int,
    doc_id: str,
    page_text: str = "",
) -> list[ParsedTable]:
    """Every period-keyed table on a page, reconstructed from word positions."""
    from .parse_pdf import ParsedTable

    lines = group_lines(page_words)
    tables: list[ParsedTable] = []
    index = 0
    position = 0

    while index < len(lines):
        columns = header_columns(lines[index], _next_line(lines, index))
        if len(columns) < _MIN_PERIOD_COLUMNS:
            index += 1
            continue

        # A header is written over two or three lines as often as one, so the lines that
        # follow are merged in rather than compared. Picking the richest single line was
        # tried and is wrong: NAB sets its divisional headers as ``Mar 26 | Sep 25 | Mar 25
        # | Mar 26 v | Mar 26 v`` over ``$m | $m | $m | Sep 25 % | Mar 25 %``, and the
        # second line names two periods that are the right-hand halves of the two movement
        # columns. Picking it threw away the real columns and read every row's movement as
        # its figure.
        cursor = index
        # Years set on the line under their months belong to the header, not the rows.
        if _years_under_header(_next_line(lines, index), columns):
            cursor = index + 1
        spans = header_columns(lines[index], _next_line(lines, index), merge_repeats=False)
        for lookahead in range(cursor + 1, min(cursor + 3, len(lines))):
            extra = header_columns(lines[lookahead], _next_line(lines, lookahead))
            if not extra:
                break
            columns = _merge_columns(columns, extra)
            spans = sorted(
                [
                    *spans,
                    *header_columns(
                        lines[lookahead], _next_line(lines, lookahead), merge_repeats=False
                    ),
                ],
                key=lambda column: column.x0,
            )
            cursor = lookahead

        # A header that names a period twice describes several blocks typeset as one table,
        # and each block is read separately. Where each period is named once, this is just
        # the merged column list.
        blocks = [block for block in split_into_blocks(spans) if len(block) >= _MIN_PERIOD_COLUMNS]
        if not blocks:
            blocks = [columns]

        # The rows are located once, against the widest block so the label edge is the
        # same for all of them, and then read again per block.
        widest = max(blocks, key=len)
        row_lines: list[list[Word]] = []
        blanks = 0
        content = 0
        scan = cursor + 1
        while scan < len(lines):
            if len(header_columns(lines[scan], _next_line(lines, scan))) >= _MIN_PERIOD_COLUMNS:
                break  # the next table's header
            row = _row_from(lines[scan], widest)
            if row is None:
                # A line with no label and no figure. Before the first figure it's the rest of
                # the header (the "$m $m $m" units line) or furniture in the margin, and
                # counting it as blank ended NAB's credit impairment table before its first
                # row: the units line, a section heading and a margin tab ("4D") made three.
                # From the first figure on, it counts as before.
                if content == 0:
                    scan += 1
                    continue
                blanks += 1
                if blanks > _MAX_BLANK_LINES and row_lines:
                    break
                scan += 1
                continue
            label, cells = row
            if not any(cells):
                # A heading, or prose on a mixed page. Kept because it may start a section,
                # but it isn't content: it counts toward the run of blank lines that ends
                # the table, so prose below a table can't extend it forever.
                blanks += 1
                if label:
                    row_lines.append(lines[scan])
                if blanks > _MAX_BLANK_LINES and content:
                    break
                scan += 1
                continue
            blanks = 0
            content += 1
            if label or row_lines:
                row_lines.append(lines[scan])
            scan += 1

        if content >= _MIN_ROWS:
            caption = _caption_before(lines, cursor)
            units = detect_units(caption) or _units_near(lines, cursor)
            restated = detect_restated(page_text)
            label_header = _label_header(lines[index : cursor + 1], widest)
            label_edge = min(column.x0 for block in blocks for column in block) - 6.0
            for block in blocks:
                rows: list[tuple[str, ...]] = []
                for line in row_lines:
                    read = _row_from(line, block, label_edge)
                    if read is None:
                        continue
                    label, cells = read
                    if label or any(cells):
                        rows.append((label, *cells))
                if not rows:
                    continue
                tables.append(
                    ParsedTable(
                        table_id=f"{doc_id}_p{page_no}_L{position}",
                        page_no=page_no,
                        caption=caption,
                        header=(label_header, *(column.label for column in block)),
                        rows=tuple(rows),
                        units=units,
                        restated=restated,
                    )
                )
                position += 1

        index = max(scan, cursor + 1)

    return tables


def _label_header(header_lines: list[list[Word]], columns: list[Column]) -> str:
    """The words set over the label column, on the header's own lines.

    They describe every row beneath them, and nothing else records them: the caption is the
    line above and the columns start to their right. Westpac heads its 1H26 summary
    "Excluding Notable Items, $m" on the same line as "March 2026 Sept 2025 March 2025", and
    with those words dropped every figure in the table was stored as the headline when it
    is the variant, which left its expense-to-income ratio with no expenses on its own
    basis.

    Text that names a period is left out. In the header row it would claim that period's
    column, and the leftmost claim wins, so the label column would take the real column's
    place.
    """
    left_edge = min(column.x0 for column in columns) - 6.0
    words = [
        word.text
        for line in header_lines
        for word in line
        if word.x1 <= left_edge and not _is_value(word.text)
    ]
    text = " ".join(words).strip()
    if not text or len(text) > 90 or names_a_period(text):
        return ""
    return text


def _merge_columns(first: list[Column], second: list[Column]) -> list[Column]:
    """Both header lines' columns, left to right, each period kept only once.

    A period can't legitimately head two columns of one table, so where a label repeats, the
    leftmost wins. In every pack seen so far that's the period's own column, and the repeat
    is the right-hand half of a movement header.
    """
    merged: list[Column] = []
    seen: set[str] = set()
    for column in sorted([*first, *second], key=lambda c: c.x0):
        key = column.label.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(column)
    return merged


def _next_line(lines: list[list[Word]], index: int) -> list[Word] | None:
    return lines[index + 1] if index + 1 < len(lines) else None


def _caption_before(lines: list[list[Word]], header_index: int) -> str:
    """The nearest preceding line that is text rather than figures."""
    for position in range(header_index - 1, max(-1, header_index - 5), -1):
        text = " ".join(word.text for word in lines[position]).strip()
        if not text:
            continue
        if any(_is_value(word.text) for word in lines[position]):
            continue
        if len(text) <= 80:
            return text
    return ""


def _units_near(lines: list[list[Word]], header_index: int) -> Unit | None:
    """A unit marker printed just under the header, as ``$M`` usually is."""
    for position in range(header_index, min(len(lines), header_index + 3)):
        text = " ".join(word.text for word in lines[position])
        unit = detect_units(text)
        if unit is not None:
            return unit
    return None
