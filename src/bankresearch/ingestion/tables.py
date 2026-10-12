"""Reads a table's header row and works out which period each column reports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from ..domain.calendar import AmbiguousPeriodError, PeriodKind, fiscal_period, parse_period
from ..domain.units import Unit

__all__ = [
    "ColumnPeriod",
    "detect_restated",
    "detect_units",
    "map_columns",
    "parse_column_period",
]

_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_MONTH_NAMES = "|".join(sorted(_MONTHS, key=len, reverse=True))

# "31 March 2026", "31 Mar 26"
_DAY_MONTH_YEAR = re.compile(
    rf"\b(?P<day>\d{{1,2}})\s+(?P<month>{_MONTH_NAMES})\b[\s,'-]*(?P<year>\d{{2,4}})",
    re.IGNORECASE,
)
# "Mar 26", "Mar-26", "Mar'26", "March 2026"
_MONTH_YEAR = re.compile(
    rf"\b(?P<month>{_MONTH_NAMES})\b[\s,'\-/]*(?P<year>\d{{2,4}})\b", re.IGNORECASE
)
# An explicit fiscal label anywhere in the header.
_FISCAL_LABEL = re.compile(r"\b((?:[12]H|H[12]|HY)\s?(?:FY)?\s?\d{2,4}|FY\s?\d{2,4})\b", re.I)

_HALF_PHRASES = re.compile(r"half\s*year|\b6\s*months?\b|\bsix\s*months?\b|\binterim\b", re.I)
_FULL_PHRASES = re.compile(r"full\s*year|\b12\s*months?\b|\btwelve\s*months?\b|\bannual\b", re.I)

# ``re-?present`` without a verb ending also matches "representation", which appears on
# every disclaimer page and isn't a restatement, so the verb forms are required.
_RESTATED = re.compile(
    r"restat|re-?present(?:ed|ing)\b|re-?classif|comparatives?\s+have\s+been|"
    r"prior\s+period.{0,20}adjust",
    re.IGNORECASE,
)

# "% Mov't", "Mov't %", "% change": a movement column's heading, never the table's unit.
_MOVEMENT_HEADING = re.compile(
    r"%\s*(?:mov(?:'?t|ement)?|chg|change|growth)\b|"
    r"\b(?:mov(?:'?t|ement)?|chg|change|growth)\s*%",
    re.IGNORECASE,
)

_UNIT_PATTERNS: tuple[tuple[re.Pattern[str], Unit], ...] = (
    # A smaller bank prints its statements in thousands: "$'000", "($'000s)", "In
    # thousands of AUD" (BNK Banking Corporation's FY26 report, p.9 and p.24). Read as
    # millions, every balance was a thousand times too large and still added up.
    (
        re.compile(
            r"\$\s?['\u2019]?\s?000s?\b|\$\s?thousands?\b|"
            r"\bthousands?\s+of\s+(?:aud|australian\s+dollars|dollars)\b",
            re.I,
        ),
        "aud_k",
    ),
    # "$B" is how the majors write billions in a balance-sheet table. Missing it leaves
    # average balances a thousand times too small, which the NIM identity catches as a
    # 15,000bp gap.
    (re.compile(r"\$\s?bn\b|\$\s?b\b|\$\s?billion|a\$\s?bn?\b", re.I), "aud_bn"),
    (re.compile(r"\$\s?m\b|\$\s?million|a\$m\b|\$m\b", re.I), "aud_m"),
    (re.compile(r"\bbps\b|basis\s*points?", re.I), "bps"),
    (re.compile(r"%|per\s*cent", re.I), "pct"),
)


def detect_units(text: str) -> Unit | None:
    """The unit a table reports in, or ``None`` rather than a guess.

    A per-cent sign that belongs to a movement heading is removed first. Westpac captions
    its statutory income statement "Half Year Half Year Half Year % Mov't", and reading that
    "%" as the table's unit made every figure on the page a percentage: an income tax
    expense of $1,491m became 1,491% and was thrown out as implausible, taking the earnings
    bridge with it.
    """
    cleaned = _MOVEMENT_HEADING.sub(" ", text or "")
    for pattern, unit in _UNIT_PATTERNS:
        if pattern.search(cleaned):
            return unit
    return None


def detect_restated(text: str) -> bool:
    """Whether a caption or footnote says the comparatives were restated."""
    return bool(_RESTATED.search(text))


def _two_digit_year(raw: str) -> int:
    value = int(raw)
    return value if value >= 1000 else 2000 + value


@dataclass(frozen=True, slots=True)
class ColumnPeriod:
    """One column header, and the period or periods it could report."""

    header: str
    end_date: date | None
    candidates: tuple[str, ...]
    explicit_kind: PeriodKind | None = None
    unmatched_reason: str = ""

    @property
    def is_ambiguous(self) -> bool:
        return len(self.candidates) > 1

    def resolve(
        self, doc_period: str, point_in_time: bool = False, among_halves: bool = False
    ) -> str | None:
        """Pick one label, using the document's own period to break a tie.

        A half-year document's comparison column is a half; a full-year document's is a
        year. For a point-in-time metric both readings name the same balance date, so either
        is correct, and the document type still decides which label is reported.

        ``among_halves`` says the column sits in a table of halves, so a flow read from it
        is the second half whatever the document covers. See :func:`map_columns`.
        """
        if not self.candidates:
            return None
        if len(self.candidates) == 1:
            return self.candidates[0]

        try:
            doc_kind = parse_period(doc_period)[:2]
        except AmbiguousPeriodError:
            return None
        wanted = "FY" if doc_kind == "FY" and (point_in_time or not among_halves) else "2H"
        for candidate in self.candidates:
            if candidate.startswith(wanted):
                return candidate
        return self.candidates[0]


def _labels_ending_on(end: date, fy_end_month: int) -> tuple[str, ...]:
    """Every period label for this issuer that ends on exactly this date.

    Generated and then checked against the calendar module, so the two can't disagree about
    when a period ends.
    """
    found: list[str] = []
    for fy in (end.year - 1, end.year, end.year + 1):
        suffix = f"{fy % 100:02d}"
        for prefix in ("1H", "2H", "FY"):
            label = f"{prefix}{suffix}"
            try:
                if fiscal_period(label, fy_end_month).end == end:
                    found.append(label)
            except (AmbiguousPeriodError, ValueError):  # pragma: no cover - defensive
                continue
    # Halves before full years, so a flow reading is offered first.
    return tuple(sorted(found, key=lambda label: (label.startswith("FY"), label)))


def names_a_period(header: str) -> bool:
    """Whether a header cell names a period at all, without needing the issuer.

    Used to decide whether a ruled table is worth keeping. Resolving the label to a fiscal
    period needs the issuer's year end, which isn't known while the PDF is being parsed, but
    recognising that a cell says "31 Dec 25" or "FY26" doesn't.
    """
    text = header or ""
    return bool(
        _DAY_MONTH_YEAR.search(text) or _MONTH_YEAR.search(text) or _FISCAL_LABEL.search(text)
    )


def parse_column_period(header: str, fy_end_month: int) -> ColumnPeriod:
    """Map one column header onto the period it reports."""
    text = (header or "").strip()
    if not text:
        return ColumnPeriod(header, None, (), unmatched_reason="empty header")

    explicit_kind: PeriodKind | None = None
    if _FULL_PHRASES.search(text):
        explicit_kind = PeriodKind.FY
    elif _HALF_PHRASES.search(text):
        explicit_kind = PeriodKind.H1  # refined below once the date is known

    label_match = _FISCAL_LABEL.search(text)
    if label_match is not None:
        try:
            label = parse_period(label_match.group(1))
        except AmbiguousPeriodError:
            label = None
        if label is not None:
            return ColumnPeriod(
                header=header,
                end_date=fiscal_period(label, fy_end_month).end,
                candidates=(label,),
                explicit_kind=PeriodKind(label[:2]) if label.startswith("FY") else None,
            )

    end: date | None = None
    match = _DAY_MONTH_YEAR.search(text)
    if match is not None:
        month = _MONTHS[match.group("month").lower()]
        end = fiscal_period(f"FY{_two_digit_year(match.group('year')) % 100:02d}", month).end
    else:
        match = _MONTH_YEAR.search(text)
        if match is not None:
            month = _MONTHS[match.group("month").lower()]
            end = fiscal_period(f"FY{_two_digit_year(match.group('year')) % 100:02d}", month).end

    if end is None:
        return ColumnPeriod(
            header,
            None,
            (),
            explicit_kind=explicit_kind,
            unmatched_reason="no fiscal label or period end date in the header",
        )

    candidates = _labels_ending_on(end, fy_end_month)
    if not candidates:
        return ColumnPeriod(
            header,
            end,
            (),
            explicit_kind=explicit_kind,
            unmatched_reason=(
                f"{end.isoformat()} is not the end of any half or full year for an "
                f"issuer whose year ends in month {fy_end_month}"
            ),
        )

    if explicit_kind is PeriodKind.FY:
        narrowed = tuple(c for c in candidates if c.startswith("FY"))
        return ColumnPeriod(header, end, narrowed or candidates, PeriodKind.FY)
    if explicit_kind is not None:
        halves = tuple(c for c in candidates if not c.startswith("FY"))
        if halves:
            kind = PeriodKind.H1 if halves[0].startswith("1H") else PeriodKind.H2
            return ColumnPeriod(header, end, halves, kind)

    return ColumnPeriod(header, end, candidates)


def map_columns(
    headers: list[str],
    fy_end_month: int,
    doc_period: str,
    point_in_time: bool = False,
) -> dict[str, str]:
    """Map a header row onto period labels, leaving non-period columns out.

    When a period resolves twice, its leftmost column wins. Real packs label their movement
    columns with periods too: Westpac's key-metrics table reads ``March 2026 | Sept 2025 |
    March 2025 | Mar 26 | Sept 25 | Mar 25``, where the last three head a "% Mov't" block,
    and in every pack seen so far the period columns come first.

    Raising on the duplicate was tried and is worse. It threw away Westpac's whole
    key-metrics table, and with it the group margin, return on equity and capital ratio.
    Rejecting the whole document isn't safer than reading the part it understands.

    A column that can only be a half makes the table a table of halves. ANZ's 2025
    full-year pack heads its half-year block ``Sep 25 | Mar 25``, with "Half Year" on the
    line above where no column reads it, and "Sep 25" was taken as the full year because the
    document is one. Its second-half average assets of $1,177bn were stored as the full
    year's $1,160bn. Set beside March, September is the second half. A balance is the same
    figure on either reading, so ``point_in_time`` keeps the document's own label for it.
    """
    parsed = [parse_column_period(header, fy_end_month) for header in headers]
    among_halves = any(
        column.candidates and not any(label.startswith("FY") for label in column.candidates)
        for column in parsed
    )
    mapping: dict[str, str] = {}
    claimed: dict[str, str] = {}
    for header, column in zip(headers, parsed, strict=True):
        label = column.resolve(doc_period, point_in_time=point_in_time, among_halves=among_halves)
        if label is None or label in claimed:
            continue
        claimed[label] = header
        mapping[header] = label
    return mapping
