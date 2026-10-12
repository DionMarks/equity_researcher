"""The issuer's own explanation of a movement, found by code."""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import yaml

from ..domain.calendar import PeriodKind, fiscal_period
from ..domain.models import DocMeta, RequestSpec
from ..ingestion.parse_pdf import ParsedDocument, ParsedPage
from ..settings import get_settings

__all__ = [
    "Passage",
    "Placed",
    "comparison_of",
    "division_pages",
    "find_commentary",
    "find_division_commentary",
    "page_sentences",
    "placed_sentences",
    "reads_as_prose",
]

#: How much of the brief commentary may take. About 2,500 tokens.
MAX_PASSAGES: Final = 20
#: A passage longer than this is usually two sentences the PDF ran together, or a table
#: flattened into prose, and is left out rather than cut.
MAX_WORDS: Final = 90
_MIN_WORDS: Final = 6

#: Where commentary is read from. An investor presentation's text layer is slide fragments
#: and chart labels run together ("Revised economic outlook in our base case provision
#: scenario Higher Markets income..."), which reads as a sentence and isn't one. The annual
#: report is used only when nothing else for the period is held.
_DOC_WEIGHT: Final[dict[str, float]] = {
    "RESULTS_DOCUMENT": 1.5,
    "RESULTS_RELEASE": 1.0,
    "ANNUAL_REPORT": 0.0,
}

_MONTHS: Final[dict[str, int]] = {
    name: number
    for number, names in enumerate(
        (
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ),
        start=1,
    )
    for name in names
}
_MONTH = "|".join(sorted(_MONTHS, key=len, reverse=True))
#: "March 2026", "Mar 26", "31 December 2025".
_MONTH_YEAR = re.compile(rf"\b({_MONTH})\.?\s+(\d{{4}}|\d{{2}})\b", re.IGNORECASE)
#: Westpac's "First Half 2026", "Second Half 2025".
_NAMED_HALF = re.compile(r"\b(first|second)\s+half\s+(?:of\s+)?(\d{4}|\d{2})\b", re.IGNORECASE)
#: "1H26", "2H25", "FY25".
_LABEL = re.compile(r"\b([12]H|FY)\s?(\d{4}|\d{2})\b", re.IGNORECASE)
#: Words that put two dates against each other.
_COMPARATOR = re.compile(
    r"\s(?:v\.?|vs\.?|versus|against|compared (?:with|to)|on|in line with|from|since)\s|"
    "\\s[\u2013\u2014-]\\s",
    re.IGNORECASE,
)
_BULLET = re.compile("^[\u2022\u25aa\u25cf\u2023\u2043\u2219\u00b7o*-]\\s*")
_SENTENCE_END = re.compile(r"(?<=[.;!?])\s+(?=[A-Z(\"'\u2018\u201c])")
_DEFAULT_STATEMENT = re.compile(
    r"(?:comments?|commentary|results?|comparisons?|figures)\b[^.]{0,60}?\b"
    r"(?:are|is)\s+(?:versus|compared\s+(?:with|to)|against|relative\s+to)\b[^.]{0,120}",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Passage:
    """One sentence of the issuer's own commentary, with where it was printed."""

    text: str
    topic: str
    doc_id: str
    doc_type: str
    page_no: int
    publish_date: date
    score: float


@dataclass(frozen=True, slots=True)
class _Ends:
    """The period-end dates a comparison can be placed by."""

    current: date
    pcp: date
    previous: date | None
    """The end of the period immediately before, where it isn't the pcp (a half's prior
    half). A full year has none."""
    half_on_half: bool = False
    """The comparison is the half immediately before, as it is for one leg of a chained
    walk. "On the prior half" then places a sentence against it, and "on the prior
    corresponding period" places it elsewhere."""


@lru_cache(maxsize=4)
def _vocabulary(config_dir: str) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(
        (Path(config_dir) / "commentary.yaml").read_text(encoding="utf-8")
    )
    items = yaml.safe_load((Path(config_dir) / "line_items.yaml").read_text(encoding="utf-8"))
    loaded["divisions"] = tuple(items.get("sections", {}).get("divisions", ()))
    loaded["group_sections"] = tuple(
        (*items.get("sections", {}).get("group", ()), *loaded.get("group_commentary", ()))
    )
    return loaded


@lru_cache(maxsize=1)
def _default_config_dir() -> str:
    return str(get_settings().config_dir)


def _words(config_dir: str | None = None) -> dict[str, Any]:
    return _vocabulary(config_dir or _default_config_dir())


def _ends(spec: RequestSpec, fy_end_month: int) -> _Ends:
    period = fiscal_period(spec.period, fy_end_month)
    previous = period.hoh().end if period.kind is not PeriodKind.FY else None
    if previous is not None and previous == spec.comparison_end:
        return _Ends(
            current=spec.period_end, pcp=spec.comparison_end, previous=None, half_on_half=True
        )
    return _Ends(current=spec.period_end, pcp=spec.comparison_end, previous=previous)


def _year(raw: str) -> int:
    value = int(raw)
    return value if value >= 1000 else 2000 + value


def _month_end(year: int, month: int) -> date:
    import calendar

    return date(year, month, calendar.monthrange(year, month)[1])


def _dates_in(text: str, fy_end_month: int) -> list[tuple[int, date]]:
    """Every period end the text names, with where it names it, in reading order."""
    found: list[tuple[int, date]] = []
    for match in _MONTH_YEAR.finditer(text):
        month = _MONTHS[match.group(1).lower()]
        found.append((match.start(), _month_end(_year(match.group(2)), month)))
    for match in _NAMED_HALF.finditer(text):
        half = "1H" if match.group(1).lower() == "first" else "2H"
        label = f"{half}{_year(match.group(2)) % 100:02d}"
        found.append((match.start(), fiscal_period(label, fy_end_month).end))
    for match in _LABEL.finditer(text):
        label = f"{match.group(1).upper()}{_year(match.group(2)) % 100:02d}"
        try:
            found.append((match.start(), fiscal_period(label, fy_end_month).end))
        except ValueError:
            continue
    found.sort()
    return found


def comparison_of(text: str, ends: _Ends, fy_end_month: int, words: dict[str, Any]) -> str | None:
    """``"pcp"``, ``"other"`` or ``None``: what this text compares the period against.

    Its own wording decides first, then any two period ends it names side by side, then a
    "since" date. ``None`` means it doesn't say.
    """
    lowered = f" {text.lower()} "
    last_half, last_year = words["prior_period"], words["same_period_last_year"]
    if ends.half_on_half:
        last_half, last_year = last_year, last_half
    if any(_has(lowered, phrase) for phrase in last_half):
        return "other"
    if any(_has(lowered, phrase) for phrase in last_year):
        return "pcp"
    dates = [when for _, when in _dates_in(text, fy_end_month)]
    if ends.previous is not None and ends.previous in dates:
        # A highlights panel that says "6% on 1H25, 8% on 2H25" in one breath. It does
        # state the year-on-year movement, but nothing in it says which cause goes with
        # which comparison.
        return "other"
    if len(dates) >= 2 and _COMPARATOR.search(lowered):
        first, second = dates[0], dates[1]
        if {first, second} == {ends.current, ends.pcp}:
            return "pcp"
        if ends.current in (first, second) and ends.pcp not in (first, second):
            return "other"
    since = re.search(r"\bsince\b", lowered)
    if since and dates:
        after = [when for at, when in _dates_in(text, fy_end_month) if at >= since.start() - 1]
        if after:
            return "pcp" if after[0] == ends.pcp else "other"
    return None


_QUOTES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"'})


def _plain(text: str) -> str:
    """Lowercase with typographic quotes straightened, for matching only."""
    return text.translate(_QUOTES).lower()


def _about_a_division(lowered: str, divisions: Sequence[str]) -> bool:
    """Whether a division is the subject: named within the sentence's first dozen words.

    "The statutory net profit after tax from Retail Banking Services was $2,727 million" is
    a division's result in the group's vocabulary. A division named later is usually a
    cause the group gives ("...competitive pressures impacting deposits in New Zealand
    Banking"), and stays.
    """
    opening = " ".join(lowered.split()[:12])
    return any(_has(f" {opening} ", division) for division in divisions)


@lru_cache(maxsize=1024)
def _phrase(phrase: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])")


@lru_cache(maxsize=256)
def _any_of(phrases: tuple[str, ...]) -> re.Pattern[str]:
    """One pattern for a topic's phrases, so a sentence that names none of them is ruled out
    in one search instead of one per phrase."""
    ordered = sorted(set(phrases), key=len, reverse=True)
    return re.compile(r"(?<![a-z])(?:" + "|".join(re.escape(p) for p in ordered) + r")(?![a-z])")


def _has(lowered: str, phrase: str) -> bool:
    return _phrase(phrase).search(lowered) is not None


def _shape(line: str) -> str:
    """A line with its digits blanked, so "12  Commonwealth Bank..." and "13  Commonwealth
    Bank..." count as the same running header."""
    return re.sub(r"\d+", "#", " ".join(line.split())).lower()


#: A document's repeated lines, by document and content hash. Counting them reads every
#: line of every page, and a run asks for them several times over the same documents.
_COUNTS: dict[tuple[str, str], dict[str, int]] = {}


def _line_counts(document: ParsedDocument) -> dict[str, int]:
    key = (document.doc_id, str(getattr(document, "sha256", "") or len(document.pages)))
    cached = _COUNTS.get(key)
    if cached is None:
        cached = _COUNTS[key] = _count_lines(document)
    return cached


def _count_lines(document: ParsedDocument) -> dict[str, int]:
    counts: dict[str, int] = {}
    for page in document.pages:
        # A comparison heading repeats on every page that has one ("March 2026 v March
        # 2025" heads each of NAB's line-item pages) and is the one repeated line that must
        # stay: it says which movement the sentences under it explain.
        seen = {
            _shape(line)
            for line in (page.text or "").splitlines()
            if line.strip() and not _is_comparison_heading(line.strip())
        }
        for shape in seen:
            counts[shape] = counts.get(shape, 0) + 1
    return counts


def _ubiquitous(document: ParsedDocument) -> frozenset[str]:
    """Lines on most pages: the pack's own navigation bar, not a section's banner.

    Westpac prints "PERFORMANCE REVIEW / DIRECTORS' REPORT / 2026 INTERIM FINANCIAL REPORT"
    at the top of 53 pages whatever the section, so reading a page's section off its first
    lines would put all of them in the directors' report. CBA's "Divisional Performance
    (continued)" banner is on 11 pages of 174 and does mark its section.
    """
    pages = max(len(document.pages), 1)
    return frozenset(
        shape for shape, count in _line_counts(document).items() if count / pages >= 0.4
    )


def _head_without_navigation(text: str, everywhere: frozenset[str]) -> list[str]:
    """A page's lines, with its navigation bar taken off the top.

    A navigation bar is several of the pack's ubiquitous lines together: Westpac's odd pages
    open with all five of its section names. One ubiquitous line on its own is a banner and
    stays, because it names the page's own section: Westpac's even pages open with the
    pack's title and then "DIRECTORS' REPORT", the same words its navigation bar uses, and
    that page really is the directors' report.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    head = lines[:6]
    navigation = [
        line
        for line in head
        if _shape(line) in everywhere and len(re.findall(r"[A-Za-z]{3,}", line)) >= 1
    ]
    if len(navigation) >= 3:
        return [line for line in lines if _shape(line) not in everywhere]
    return lines


def running_lines(document: ParsedDocument) -> frozenset[str]:
    """Lines printed on several pages: running headers, footers and section banners.

    Left in, they join onto the first sentence of every page, and a sentence that starts
    "Group Performance Analysis (continued) Commonwealth Bank of Australia - Profit
    Announcement..." can't be quoted, or found again on its page as a sentence.
    """
    counts = _line_counts(document)
    return frozenset(shape for shape, count in counts.items() if count >= 4 and len(shape) <= 140)


def _is_comparison_heading(line: str) -> bool:
    """A line that is nothing but a comparison: "March 2026 v March 2025", "First Half 2026
    - First Half 2025", "NIM movement since December 2024".

    Recognised wherever it falls, because NAB prints it straight after a chart's axis labels
    and it would otherwise be read as the start of the chart's caption.
    """
    mentions = len(_MONTH_YEAR.findall(line)) + len(_NAMED_HALF.findall(line))
    mentions += len(_LABEL.findall(line))
    lowered = f" {line.lower()} "
    if mentions >= 2 and _COMPARATOR.search(lowered):
        return len(re.findall(r"[a-z]{4,}", lowered)) <= 8
    return mentions == 1 and " since " in lowered and len(line) <= 60


def _is_title(line: str) -> bool:
    """A short line set as a title: "Net Interest Income (continued)", "Average Interest
    Earning Assets". Mostly capitalised words, no closing punctuation."""
    bare = re.sub(r"\([^)]*\)", " ", line)
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z'&-]*", bare) if len(word) > 3]
    if not words or len(line) > 70 or line.endswith((".", ",", ";", ":")):
        return False
    return sum(1 for word in words if word[0].isupper()) / len(words) >= 0.75


def page_sentences(text: str, skip: frozenset[str] = frozenset()) -> list[tuple[str, int]]:
    """The page's sentences, each with the index of the heading block it sits under.

    PDF text arrives a line at a time, broken wherever the column was narrow. Lines are
    joined back into sentences: a bullet, a blank line or a title starts a new block, a word
    split by a hyphen at a line end is rejoined, and each block is split at sentence ends.
    A short line that names a period, between blocks, is returned as a heading (a negative
    index) so the caller can read comparisons off it. ``skip`` is the document's running
    headers, which are dropped.

    Worked out once per page text: every run reads the same pages, and splitting them was
    most of what a run with no model spent its time on.
    """
    return list(_page_sentences(text, skip))


@lru_cache(maxsize=8192)
def _page_sentences(text: str, skip: frozenset[str]) -> tuple[tuple[str, int], ...]:
    return tuple(_split_page(text, skip))


def _split_page(text: str, skip: frozenset[str]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    block: list[str] = []
    heading_index = 0

    def flush() -> None:
        if not block:
            return
        joined = ""
        for line in block:
            if joined.endswith("-") and line[:1].islower():
                joined = joined[:-1] + line
            else:
                joined = f"{joined} {line}" if joined else line
        block.clear()
        for sentence in _SENTENCE_END.split(joined):
            cleaned = " ".join(sentence.split())
            if cleaned:
                out.append((cleaned, heading_index))

    raw = [line.strip() for line in (text or "").splitlines()]
    filled = [index for index, line in enumerate(raw) if line]
    # A repeated line is a running header or footer only where headers and footers are:
    # the first or last few lines of the page. CBA repeats "expires on 12 August 2026 and
    # will not be extended." on several pages, mid-paragraph, and dropping it there joined
    # the sentence before it to the one after, a passage the page never prints.
    edges = set(filled[:_EDGE_LINES]) | set(filled[-_EDGE_LINES:])
    lines = [
        line
        for index, line in enumerate(raw)
        if not (index in edges and _shape(line) in skip) and not re.fullmatch(r"\d{1,3}", line)
    ]
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line:
            flush()
            continue
        # A bullet is its glyph alone or followed by a space, or a symbol set against its
        # first word. "o" is a bullet glyph and a word: CBA's FY25 pack prints one word to a
        # line (p.30), and "or" read as a bullet broke "Trading income increased by $22
        # million or 2%" in two and left a passage starting "r 2% to $1,199 million".
        glyph = line[:1]
        if _BULLET.match(line) and (
            len(line) == 1 or line[1:2].isspace() or (not glyph.isalpha() and line[1:2].isalpha())
        ):
            flush()
            line = _BULLET.sub("", line)
            if not line:
                continue
        previous = lines[index - 2] if index >= 2 else ""
        starts_block = not block or previous.endswith((".", ":", ";"))
        names_a_period = bool(
            _MONTH_YEAR.search(line) or _NAMED_HALF.search(line) or _LABEL.search(line)
        )
        if (
            len(line) <= 100
            and not line.endswith((".", ",", ";"))
            and names_a_period
            and (starts_block or _is_comparison_heading(line))
        ):
            # CBA breaks "...December 2025 versus December" / "2024" over two lines.
            while index < len(lines) and re.fullmatch(r"(?:19|20)\d{2}", lines[index]):
                line = f"{line} {lines[index]}"
                index += 1
            flush()
            heading_index += 1
            out.append((line, -heading_index))
            continue
        following = lines[index] if index < len(lines) else ""
        repeated_title = (
            0 < len(line.split()) <= 4
            and not line.endswith((".", ",", ";", ":"))
            and following.lower().startswith(line.lower())
        )
        if repeated_title or (
            _is_title(line) and (index >= len(lines) or not lines[index][:1].islower())
        ):
            flush()
            heading_index += 1
            out.append((line, -heading_index))
            continue
        block.append(line)
    flush()
    return out


def _page_section(lines: list[str], words: dict[str, Any]) -> bool | None:
    """Whether a page's own banner puts it in the group's commentary: True, False, or None
    when the page has no banner and continues the section before it.

    Westpac's odd pages carry only the navigation bar, so page 51 of its directors' report
    says nothing about its section; it is in the directors' report because page 50 is.
    """
    head = _plain(" ".join(lines[:4]))
    if any(marker in head for marker in words["not_group_commentary"]):
        return False
    first = _plain(" ".join(lines[:2]))
    if any(first.startswith(division) for division in words["divisions"]):
        return False
    if any(marker in head for marker in words["group_sections"]):
        return True
    return None


def _document_default(
    document: ParsedDocument, ends: _Ends, fy_end_month: int, words: dict[str, Any]
) -> str | None:
    """What the document says its commentary compares against unless stated otherwise.

    CBA: "Comments are versus prior comparative period unless stated otherwise". NAB: "The
    March 2026 half year results are compared with the September 2025 half year results",
    the other way round. Read from the early pages, where packs state it.
    """
    for page in document.pages[:40]:
        for match in _DEFAULT_STATEMENT.finditer(page.text or ""):
            placed = comparison_of(match.group(0), ends, fy_end_month, words)
            if placed is not None:
                return placed
    return None


def _figure_forms(values: Iterable[Decimal]) -> list[str]:
    """How a published figure might be written in prose: 2.04%, $5,445 million, 5,445."""
    forms: list[str] = []
    for value in values:
        magnitude = abs(value)
        if magnitude == magnitude.to_integral_value():
            whole = f"{int(magnitude):,}"
            forms += [whole, whole.replace(",", "")]
        else:
            forms.append(f"{magnitude:f}".rstrip("0").rstrip("."))
    return [form for form in forms if len(form.replace(",", "").replace(".", "")) >= 3]


#: A sentence about a notable or significant item.
_NOTABLE = re.compile(r"\b(?:large )?(?:notable|significant) items?\b")

#: How many lines at the top and at the bottom of a page can be a running header or footer.
_EDGE_LINES: Final = 4

#: How a sentence that continues the one before it opens.
_CONTINUES = re.compile(r"^(?:This|These|The (?:increase|decrease|rise|fall|result|movement)|It)\b")


@dataclass(frozen=True, slots=True)
class Placed:
    """One sentence of a document, with the comparison it was placed against."""

    page: ParsedPage
    sentence: str
    placed: str | None
    """``"pcp"``, ``"other"``, or ``None`` where nothing says."""
    section: tuple[str, ...]
    """The topics named by the heading the sentence sits under."""


def placed_sentences(
    document: ParsedDocument,
    spec: RequestSpec,
    *,
    fy_end_month: int,
    topics: Sequence[str] = (),
    config_dir: str | None = None,
    pages: Collection[int] | None = None,
) -> Iterator[Placed]:
    """Every sentence on the document's group commentary pages, in order, placed.

    ``pages`` replaces the group's pages with these, for a division's own commentary.

    A sentence is placed by its own wording; failing that, by the sentence before it in the
    same block ("Net interest income increased $606 million or 5% on the prior half. The
    result was driven by..." continues the first one's comparison without restating it);
    then by the heading it sits under, which runs on to the next page until another heading
    replaces it (CBA's half-on-half margin commentary starts on p.28 and finishes on p.29);
    and last by what the document says it compares against unless stated otherwise.
    """
    words = _words(config_dir)
    ends = _ends(spec, fy_end_month)
    default = _document_default(document, ends, fy_end_month, words)
    skip = running_lines(document)
    everywhere = _ubiquitous(document)
    carried: str | None = None
    in_group = True
    for page in document.pages:
        if page.kind == "boilerplate":
            carried = None
            continue
        lines = _head_without_navigation(page.text or "", everywhere)
        section = _page_section(lines, words) if lines else None
        in_group = in_group if section is None else section
        in_scope = in_group if pages is None else page.page_no in pages
        if not lines or not in_scope:
            carried = None
            continue
        headings: dict[int, str | None] = {0: carried}
        # The topic a heading names carries to the lines under it: ANZ lists its margin
        # walk as bullets ("Assets pricing (-4 bps): driven by ongoing competition") under
        # "Net interest margin - March 2026 Half Year v March 2025 Half Year".
        sections: dict[int, tuple[str, ...]] = {0: ()}
        last: tuple[int, str | None] = (0, None)
        for sentence, where in page_sentences(page.text or "", skip):
            if where < 0:
                placed = comparison_of(sentence, ends, fy_end_month, words)
                headings[-where] = placed if placed is not None else headings.get(-where - 1)
                carried = headings[-where]
                named = _topics_named(_plain(sentence), topics, words)
                sections[-where] = named or sections.get(-where - 1, ())
                continue
            own = comparison_of(sentence, ends, fy_end_month, words)
            placed = own if own is not None else (last[1] if last[0] == where else None)
            if own is not None:
                last = (where, own)
            if placed is None:
                placed = headings.get(where)
            if placed is None:
                placed = default
            yield Placed(page, sentence, placed, sections.get(where, ()))


def find_commentary(
    spec: RequestSpec,
    documents: Iterable[tuple[ParsedDocument, DocMeta]],
    topics: Sequence[str],
    *,
    fy_end_month: int,
    figures: dict[str, Sequence[Decimal]] | None = None,
    per_topic: int = 4,
    metric_quota: int = 4,
    max_passages: int = MAX_PASSAGES,
    min_score: float = 6.0,
    config_dir: str | None = None,
) -> list[Passage]:
    """The issuer's sentences explaining this movement, best first within each topic.

    ``topics`` are the metric and the components its drivers were computed from, the metric
    first. ``figures`` are the published values of each, so a sentence that states the
    group's own figure ranks above one that might be about a division.
    """
    words = _words(config_dir)
    figures = figures or {}
    movement = [re.compile(rf"(?<![a-z]){re.escape(w)}(?![a-z])") for w in words["movement"]]
    causal = [re.compile(rf"(?<![a-z]){re.escape(w)}(?![a-z])") for w in words["causal"]]

    scored: dict[str, list[Passage]] = {topic: [] for topic in topics}
    held = [
        (document, meta)
        for document, meta in documents
        if meta.period.upper() == spec.period.upper()
        and meta.bank_id == spec.bank_id
        and meta.doc_type in _DOC_WEIGHT
    ]
    if any(meta.doc_type != "ANNUAL_REPORT" for _, meta in held):
        held = [(document, meta) for document, meta in held if meta.doc_type != "ANNUAL_REPORT"]
    for document, meta in held:
        # The topics the sentence before named, for one that continues it: NAB gives its
        # expense rise, then "This was primarily driven by salary expense inflation and...",
        # which names no topic and is the only sentence with the cause in it.
        before: tuple[int, tuple[str, ...]] = (-1, ())
        for item in placed_sentences(
            document, spec, fy_end_month=fy_end_month, topics=topics, config_dir=config_dir
        ):
            sentence, placed = item.sentence, item.placed
            # The promise is that a passage is the bank's own words, so it is checked
            # against its page here rather than assumed from how it was assembled.
            if not _printed_on(sentence, item.page.text or ""):
                continue
            lowered_all = _plain(sentence)
            named_here = _topics_named(lowered_all, topics, words)
            # A sentence the PDF split from its subject starts in lower case. NAB's collective
            # write-back "was driven by a net release of forward-looking provisions, combined
            # with the release of provisions held for customers that transferred to
            # individually assessed" (FY25 p.26): it continues the sentence before, and the
            # words it mentions in passing aren't its subject, or the sentence after it would
            # be filed under the individual charge.
            fragment = sentence[:1].islower() and before[0] == item.page.page_no
            if fragment and before[1]:
                named_here, continued = (), before[1]
            else:
                continued = (
                    before[1]
                    if not named_here
                    and before[0] == item.page.page_no
                    and _CONTINUES.match(sentence)
                    else ()
                )
            before = (item.page.page_no, named_here or continued)
            if placed == "other":
                continue
            lowered = _plain(sentence)
            count = len(sentence.split())
            if not _MIN_WORDS <= count <= MAX_WORDS or not reads_as_prose(sentence):
                continue
            # A sentence that continues the one before needn't restate its movement: "was
            # driven by a net release of forward-looking provisions" is all cause.
            if not continued and not any(pattern.search(lowered) for pattern in movement):
                continue
            if _about_a_division(lowered, words["divisions"]):
                continue
            for topic in topics:
                if fragment and topic not in continued:
                    continue
                phrases = tuple(words["topics"].get(topic, ()))
                named = bool(phrases) and _any_of(phrases).search(f" {lowered} ") is not None
                hits = [p for p in phrases if _has(f" {lowered} ", p)] if named else []
                if not hits and topic in continued:
                    hits = []
                    inherited = False
                    score_base = 4.0  # it explains the sentence before, which named the topic
                else:
                    score_base = None
                    inherited = not hits and topic in item.section
                if not hits and not inherited and score_base is None:
                    continue
                score = (
                    score_base
                    if score_base is not None
                    else 2.0
                    if inherited
                    else 3.0 + 0.5 * (len(hits) - 1)
                )
                # The subject of the sentence is what it explains. A topic first named
                # a dozen words in is usually a cause of something else: "Markets income
                # increased $35 million with a $56 million increase in net interest income"
                # is about Markets income.
                position = _first_position(lowered, hits)
                if position is not None and position > 12:
                    score -= 1.5
                # So is a sentence whose subject is another metric the request didn't ask
                # about: "The collective provision to credit risk-weighted assets ratio
                # decreased by 7 basis points" is about provisioning, though it names risk-
                # weighted assets, and NAB's CET1 answer quoted it as the bank's words.
                elsewhere = _first_position(lowered, _other_topics(words, topics))
                if position is not None and elsewhere is not None and elsewhere < position:
                    continue
                if any(pattern.search(lowered) for pattern in causal):
                    score += 2.5
                score += 2.0 if placed == "pcp" else -0.5
                score += _DOC_WEIGHT.get(meta.doc_type, 0.0)
                if position is not None and position <= 6:
                    score += 1.5
                if any(form in sentence for form in _figure_forms(figures.get(topic, ()))):
                    score += 2.0
                # A notable item changes how the whole movement reads. NAB's expenses rose
                # $1,560m, of which $1,347m was large notable items; the sentence giving the
                # $243m underlying rise has no causal word and used to lose to sentences about
                # sub-lines, so the memo put salary inflation down as the cause of all of it.
                if _NOTABLE.search(lowered):
                    score += 2.0
                if count > 45:
                    score -= 1.0
                scored[topic].append(
                    Passage(
                        text=sentence,
                        topic=topic,
                        doc_id=meta.doc_id,
                        doc_type=meta.doc_type,
                        page_no=item.page.page_no,
                        publish_date=meta.publish_date,
                        score=score,
                    )
                )

    chosen: list[Passage] = []
    for index, topic in enumerate(topics):
        quota = metric_quota if index == 0 else per_topic
        ranked = sorted(scored[topic], key=lambda p: (-p.score, p.doc_id, p.page_no))
        taken = 0
        for passage in ranked:
            if taken >= quota or len(chosen) >= max_passages or passage.score < min_score:
                break
            # A pack says the same thing in its summary and again in its detail, a word or
            # two apart. One copy is evidence; two are a wasted slot.
            if any(_same_sentence(passage.text, kept.text) for kept in chosen):
                continue
            chosen.append(passage)
            taken += 1
    return chosen


_CONTINUED_TITLE = re.compile(r"\s*\(continued\)\s*$", re.IGNORECASE)
_LOSS_RATE = re.compile(r"\bas a percentage of\b|\bbasis points?\b|\bbps\b")
#: Pages a division's commentary may run to after the page that names it. CBA's longest,
#: New Zealand, is six pages in all.
_DIVISION_PAGES: Final = 6


def division_pages(document: ParsedDocument, divisions: dict[str, str]) -> dict[int, str]:
    """The pages each division's commentary is on, as page number to division key.

    A division's pages name it just under the section title, "(continued)" after the first,
    as the fact extractor reads them: CBA heads every divisional page "Divisional
    Performance (continued)" and puts "Retail Banking Services" a line or two below. A page
    that names no division continues the one before it, until a page names another.
    """
    names = {_plain(" ".join(name.split())): key for name, key in divisions.items()}
    found: dict[int, str] = {}
    current: str | None = None
    since = 0
    for page in document.pages:
        lines = [line.strip() for line in (page.text or "").splitlines() if line.strip()][:6]
        titles = {
            names[title]
            for line in lines
            if (title := _plain(" ".join(_CONTINUED_TITLE.sub("", line).split()))) in names
        }
        # A page whose top lines name several divisions is showing a navigation bar, not
        # starting a division: a presentation's divisional slides list every division and
        # highlight one, which text extraction can't see.
        named = next(iter(titles)) if len(titles) == 1 else None
        if named is not None:
            current, since = named, 0
        else:
            since += 1
            # The last division runs on into whatever follows it, so a division is held to
            # the few pages after the one that names it.
            if page.kind == "boilerplate" or not lines or since > _DIVISION_PAGES:
                current = None
        if current is not None:
            found[page.page_no] = current
    return found


def find_division_commentary(
    spec: RequestSpec,
    documents: Iterable[tuple[ParsedDocument, DocMeta]],
    divisions: dict[str, str],
    wanted: dict[str, str],
    *,
    fy_end_month: int,
    figures: dict[str, Sequence[Decimal]] | None = None,
    per_topic: int = 2,
    min_score: float = 6.5,
    config_dir: str | None = None,
) -> list[Passage]:
    """What each division says about its own impairment charge, for its driver.

    ``divisions`` is the bank's division titles to keys, ``wanted`` each key to the topic its
    driver was computed from (``retail`` to ``impairment_div_retail``). A division's pages are
    left out of the group's commentary on purpose, so a division's driver had no cause to
    cite: CBA's Business Banking charge fell $45m "primarily driven by lower individually
    assessed provision charges, including an increase in write-backs" (FY26 p.69). A
    sentence about the division's profit that mentions its charge in passing ("as well as a
    13% decrease in loan impairment expense") is not about the charge, and ranks below it.
    """
    words = _words(config_dir)
    figures = figures or {}
    phrases = tuple(words["topics"].get("credit_impairment_charge", ()))
    movement = [re.compile(rf"(?<![a-z]){re.escape(w)}(?![a-z])") for w in words["movement"]]
    causal = [re.compile(rf"(?<![a-z]){re.escape(w)}(?![a-z])") for w in words["causal"]]
    scored: dict[str, list[Passage]] = {topic: [] for topic in wanted.values()}
    for document, meta in documents:
        # The results document is where each division is written up page by page.
        if meta.period.upper() != spec.period.upper() or meta.doc_type != "RESULTS_DOCUMENT":
            continue
        by_page = division_pages(document, divisions)
        in_scope = {page for page, key in by_page.items() if key in wanted}
        if not in_scope:
            continue
        for item in placed_sentences(
            document, spec, fy_end_month=fy_end_month, config_dir=config_dir, pages=in_scope
        ):
            topic = wanted[by_page[item.page.page_no]]
            sentence, lowered = item.sentence, _plain(item.sentence)
            count = len(sentence.split())
            if item.placed == "other" or not _MIN_WORDS <= count <= MAX_WORDS:
                continue
            if not reads_as_prose(sentence) or not _printed_on(sentence, item.page.text or ""):
                continue
            hits = [phrase for phrase in phrases if _has(f" {lowered} ", phrase)]
            if not hits or not any(pattern.search(lowered) for pattern in movement):
                continue
            score = 3.0
            position = _first_position(lowered, hits)
            if position is not None and position > 12:
                score -= 1.5
            elif position is not None and position <= 6:
                score += 1.5
            if any(pattern.search(lowered) for pattern in causal):
                score += 2.5
            score += 2.0 if item.placed == "pcp" else 0.0
            if any(form in sentence for form in _figure_forms(figures.get(topic, ()))):
                score += 2.0
            # The loss rate ("as a percentage of average gross loans") restates the charge
            # against the book, and says nothing about why it moved.
            if _LOSS_RATE.search(lowered):
                score -= 1.5
            scored[topic].append(
                Passage(
                    text=sentence,
                    topic=topic,
                    doc_id=meta.doc_id,
                    doc_type=meta.doc_type,
                    page_no=item.page.page_no,
                    publish_date=meta.publish_date,
                    score=score,
                )
            )
    chosen: list[Passage] = []
    for topic in wanted.values():
        ranked = sorted(scored[topic], key=lambda p: (-p.score, p.doc_id, p.page_no))
        taken = 0
        for passage in ranked:
            if taken >= per_topic or passage.score < min_score:
                break
            if any(_same_sentence(passage.text, kept.text) for kept in chosen):
                continue
            chosen.append(passage)
            taken += 1
    return chosen


def _condensed(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _plain(text))


def _printed_on(sentence: str, page_text: str) -> bool:
    """Whether a sentence is on its page word for word, ignoring only spacing, line breaks,
    hyphenation and typographic quotes."""
    return _condensed(sentence) in _condensed_page(page_text)


@lru_cache(maxsize=256)
def _condensed_page(page_text: str) -> str:
    """A page condensed once, not once per sentence checked against it."""
    return _condensed(page_text)


def _other_topics(words: dict[str, Any], topics: Sequence[str]) -> tuple[str, ...]:
    """Every topic phrase that belongs only to topics outside this request."""
    wanted = {phrase for topic in topics for phrase in words["topics"].get(topic, ())}
    return tuple(
        phrase
        for topic, phrases in words["topics"].items()
        if topic not in topics
        for phrase in phrases
        if phrase not in wanted and not any(phrase in mine for mine in wanted)
    )


def _first_position(lowered: str, phrases: Sequence[str]) -> int | None:
    """The word position at which the first of these phrases appears."""
    starts = [
        match.start()
        for phrase in phrases
        if (match := re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", lowered))
    ]
    return len(lowered[: min(starts)].split()) if starts else None


def _same_sentence(first: str, second: str) -> bool:
    a = set(re.findall(r"[a-z0-9$.,%]+", first.lower()))
    b = set(re.findall(r"[a-z0-9$.,%]+", second.lower()))
    return bool(a and b) and len(a & b) / len(a | b) >= 0.75


def reads_as_prose(sentence: str) -> bool:
    """Whether a "sentence" is prose, rather than a table row or a chart's labels run together.

    A flattened table has more figures than words; a run of labels is mostly capitalised:
    "Other interest-earning assets NIM (Excluding Notable Items, %) NIM Treasury & Markets
    impact on NIM". Acronyms don't count against it, because prose is full of them.
    """
    tokens = sentence.split()
    numeric = sum(1 for token in tokens if re.fullmatch(r"[\(\-$]?[\d,.]+%?\)?", token))
    if numeric / max(len(tokens), 1) > 0.35:
        return False
    words = [token.strip("()\"',.;:") for token in tokens[1:]]
    words = [word for word in words if len(word) > 2 and word.isalpha() and not word.isupper()]
    if not words:
        return False
    return sum(1 for word in words if word[0].isupper()) / len(words) <= 0.45


def _topics_named(lowered: str, topics: Sequence[str], words: dict[str, Any]) -> tuple[str, ...]:
    padded = f" {lowered} "
    return tuple(
        topic
        for topic in topics
        if (phrases := tuple(words["topics"].get(topic, ())))
        and _any_of(phrases).search(padded) is not None
    )
