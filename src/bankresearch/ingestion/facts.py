"""Reads figures out of registered tables."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, cast

import yaml

from ..domain.models import Basis, DocMeta, Fact
from ..domain.registry import get_registry
from ..domain.units import AUD_UNITS, Unit, normalise_aud, value_tolerance
from ..settings import get_settings
from .basis import detect_basis, headline_earnings_basis
from .parse_pdf import ParsedDocument, ParsedTable
from .sections import (
    ASX_STAMP_WORDS,
    Section,
    SectionRules,
    is_heading_label,
    is_marker_row,
    is_wrap_candidate,
)
from .tables import detect_units, map_columns

__all__ = [
    "LineItemIndex",
    "cell_under",
    "clean_label",
    "extract_facts",
    "label_candidates",
    "parse_cell",
    "without_stamp",
]

_PUNCTUATION = re.compile(r"[^a-z0-9]+")
#: A heading's own section number: "2.1. Consumer", "4 Business Banking".
_SECTION_NUMBER = re.compile(r"^\s*\d{1,2}(?:\.\d{1,2})*\.?\s+")
_SUPERSCRIPT = re.compile(r"[\u00b9\u00b2\u00b3\u2070-\u2079\u2080-\u2089*\u2020\u2021]")
_NUMERIC = re.compile(r"^\(?-?\$?\s?[\d,]+(?:\.\d+)?\)?%?$")

# A trailing unit parenthetical belongs to the heading, not to the name:
# "Average interest earning assets ($M)", "Common Equity Tier 1 (APRA) (%)". A footnote
# marker can be fused to it, as in ANZ's "Collectively assessed credit impairment
# charge/(release) ($M)6", which matched nothing until the marker went with the unit.
_TRAILING_UNITS = re.compile(
    r"\s*\((?:a?\$\s?[mb]n?|nz\$\s?[mb]n?|%|bps?|cents?|apra|\$ ?million|\$ ?billion)\)"
    r"(?:\d{1,2}(?:,\d{1,2})*)?\s*$",
    re.IGNORECASE,
)
# The same unit written bare at the end of the label, as Bendigo does: "Average interest
# earning assets $m", "Common Equity Tier 1 %".
_TRAILING_BARE_UNIT = re.compile(r"\s+(?:a?\$\s?[mb]n?|%|bps?|\u00a2|cents?)\s*$", re.IGNORECASE)
# A label that is nothing but a unit: the line a wrapped name leaves between its halves.
_UNIT_ONLY = re.compile(r"^\(?(?:a?\$\s?[mb]n?|%|bps?|\u00a2|cents?|x|times)\)?$", re.IGNORECASE)
# A trailing footnote marker: "(1)", a lone letter, or a short digit run.
_TRAILING_FOOTNOTE = re.compile(r"(?:\(\d{1,2}\)|[a-z]|\d{1,2}(?:,\d{1,2})*)$", re.IGNORECASE)
# Figures that leaked into the label when a column anchor sat too far right.
# Placeholders count as figures here. When the label column runs into a neighbouring block,
# CBA's loan impairment row ends "(319)" followed by a dash standing in for an unquoted
# movement, and leaving it on stopped the earnings bridge's impairment line matching at all.
_TRAILING_FIGURES = re.compile(
    r"(?:\s+(?:\(?-?[\d,]+(?:\.\d+)?\)?%?|[-\u2013\u2014]|n/a|\(?large\)?|n\.?m\.?)){1,8}$",
    re.IGNORECASE,
)
# A unit named in the label's own parenthetical, with anything short after it.
_LABEL_UNIT = re.compile(
    r"\((a?\$\s?[mb]n?|nz\$\s?[mb]n?|%|bps?|cents?|\$ ?million|\$ ?billion)\)"
    r"(?:[^()]{0,6})$",
    re.IGNORECASE,
)
# A footnote marker set before the label, as Westpac does: "2 common equity Tier 1".
_LEADING_FOOTNOTE = re.compile(r"^\s*\(?\d{1,2}\)?[\s.]+(?=[A-Za-z])")
# A trailing abbreviation in brackets: "Return on average ordinary equity (ROE)".
_TRAILING_ABBREVIATION = re.compile(r"\s*\([A-Z][A-Za-z0-9]{1,6}\)\s*$")
# Digits fused to the end of a word, as a footnote marker usually is.
_FUSED_DIGITS = re.compile(r"(?<=[a-z])\d{1,2}(?:,\d{1,2})*$", re.IGNORECASE)
# A bracketed footnote fused to the last word: "Income tax expense(1)". The bracketed form
# is only handled where a space comes before it, because `_TRAILING_FOOTNOTE` matches a
# whole word and "expense(1)" isn't one. NAB writes every footnote this way, so its income
# tax line matched nothing, the earnings identity was one component short, and the run
# reported no drivers for NAB's headline earnings while ANZ and Westpac (same identity, same
# pack layout) each resolved six.
_FUSED_FOOTNOTE = re.compile(r"(?<=[a-z])\((?:\d{1,2}|[a-z])\)$", re.IGNORECASE)
# "Less:", "Add:" and "Plus:" are accounting presentation, not part of the line item's name.
# NAB writes "Less: Non-controlling interests", so its minority interests matched nothing.
# That only showed up once the tax line was fixed: the earnings identity then had five of
# six components and was out by exactly the missing one (19m in 1H26, 20m in 1H25), so the
# whole group was quarantined rather than published incomplete.
_LEADING_QUALIFIER = re.compile(r"^(?:less|add|plus|deduct|net of)\s*:\s*", re.IGNORECASE)
# "Net profit after tax from continuing operations" is the same line item as "Net
# profit after tax"; the qualifier states a basis, which `sections.basis_dimensions`
# reads off the same label. The discontinued row is refused by `exclusions`, so
# stripping the qualifier here cannot turn it into a headline reading.
_OPERATIONS_QUALIFIER = re.compile(
    r"\s+(?:from|for)\s+(?:continuing|discontinued)\s+operations\b", re.IGNORECASE
)
# A trailing basis qualifier, set off by a dash: "Net profit after tax - cash basis".
# The basis itself is read off the same label by `sections.basis_dimensions`, so
# taking it off the name loses nothing.
_TRAILING_BASIS = re.compile(
    # CBA sets the qualifier in quotes on its half-year page and bare on its full-year
    # page, so the quotes are optional and both curly and straight forms are allowed.
    r"\s*[\u2013\u2014-]\s*[\"\u201c\u2018\']?"
    r"(?:cash|statutory|underlying|cash earnings)(?:\s+(?:earnings|profit))?\s+basis"
    r"[\"\u201d\u2019\']?\s*$",
    re.IGNORECASE,
)


def _normalise(label: str) -> str:
    return _PUNCTUATION.sub(" ", _SUPERSCRIPT.sub(" ", label or "").lower()).strip()


def unit_in_label(label: str) -> Unit | None:
    """A unit the row names for itself, e.g. "Average interest earning assets ($M) 3".

    Searched for as a parenthetical rather than a suffix, because a footnote marker usually
    follows it. Getting it wrong is costly: a balance of 448,489 read as a percentage
    (because the table was captioned "%") makes the margin identity fail by 4,000 basis
    points.
    """
    match = _LABEL_UNIT.search(label or "")
    if match is None:
        return None
    return detect_units(match.group(1))


#: A two-sided label set the other way round. Bendigo's FY25 results print "Credit
#: reversals/(expenses)" and its FY26 report "Credit Expenses/(Reversals)": the same row. The
#: sign is still read from whichever side the label as printed brackets (ADR-039).
_TWO_SIDED_RIGHT = re.compile(
    r"^(?P<lead>.*?)(?P<a>[A-Za-z-]+)\s*/\s*\((?P<b>[A-Za-z-]+)\)(?P<tail>.*)$"
)
_TWO_SIDED_LEFT = re.compile(
    r"^(?P<lead>.*?)\((?P<a>[A-Za-z-]+)\)\s*/\s*(?P<b>[A-Za-z-]+)(?P<tail>.*)$"
)


def mirrored(label: str) -> str | None:
    """The same two-sided label with its sides swapped, or ``None`` for any other label."""
    match = _TWO_SIDED_RIGHT.match(label)
    if match is not None:
        return f"{match['lead']}{match['b']}/({match['a']}){match['tail']}"
    match = _TWO_SIDED_LEFT.match(label)
    if match is not None:
        return f"{match['lead']}({match['b']})/{match['a']}{match['tail']}"
    return None


def label_candidates(label: str) -> list[str]:
    """Readings of a row label to try, most faithful first.

    Footnote markers are often set as plain text with no space ("Average interest earning
    assets4", "Expense to income ratioa"), and nothing in the text separates a stray
    trailing letter from the real last letter of a word. So instead of guessing, several
    readings are generated and only one that actually matches a known component is accepted.
    A wrong strip costs nothing, and a name that really does end in a digit, like "Common
    Equity Tier 1", still matches on the first reading.
    """
    readings: list[str] = []

    def add(candidate: str) -> None:
        text = candidate.strip(" .,:;")
        if text and text not in readings:
            readings.append(text)

    add(label)
    cleaned = clean_label(label)
    add(cleaned)
    # CBA prints the group's profit twice on one page: once "from continuing operations" and
    # once as the unqualified total. Only the first is the basis its headline uses, and
    # without this reading the qualified row matched no component, so the figure stored was
    # the total including discontinued operations: $10,253m instead of the $10,252m the
    # bank's own commentary moves from.
    qualified = _TRAILING_BASIS.sub("", _OPERATIONS_QUALIFIER.sub("", cleaned)).strip()
    add(qualified)
    # A footnote marker fused to the final word, bare or in brackets.
    add(_FUSED_DIGITS.sub("", cleaned))
    add(_FUSED_FOOTNOTE.sub("", cleaned))
    # And the accounting prefix, which may sit in front of either reading.
    add(_LEADING_QUALIFIER.sub("", cleaned))
    add(_LEADING_QUALIFIER.sub("", _FUSED_FOOTNOTE.sub("", cleaned)))
    words = cleaned.split()
    if words and len(words[-1]) > 3 and words[-1][-1].isalpha():
        add(" ".join([*words[:-1], words[-1][:-1]]))
    # The ASX's "For personal use only" stamp, when a word of it has landed in front of the
    # label. Tried after every other reading, so a name that really starts with one of the
    # words still matches as printed first.
    for reading in list(readings):
        unstamped = without_stamp(reading)
        if unstamped != reading:
            add(unstamped)
    # And a two-sided label the other way round, tried last of all.
    for reading in list(readings):
        swapped = mirrored(reading)
        if swapped is not None:
            add(swapped)
    return readings


def without_stamp(label: str) -> str:
    """A label with any words of the ASX's "For personal use only" stamp taken off its
    front: "only Corporate tax expense" is "Corporate tax expense"."""
    words = label.split()
    stamp = 0
    while stamp < len(words) - 1 and words[stamp].lower() in ASX_STAMP_WORDS:
        stamp += 1
    return " ".join(words[stamp:]) if stamp else label


def clean_label(label: str) -> str:
    """A row label with the page furniture taken off.

    Real packs decorate labels in three ways that stop them matching: a trailing unit in
    brackets, a footnote marker, and (when layout extraction misjudges a column edge) the
    row's own figures. Each is stripped in that order, and the result is only tried as an
    alternative. The raw label is matched first, so a name that really ends in a digit, like
    "Common Equity Tier 1", is never damaged.
    """
    text = _SUPERSCRIPT.sub("", label or "").strip()
    text = _LEADING_FOOTNOTE.sub("", text).strip()
    text = _TRAILING_FIGURES.sub("", text).strip()
    text = _TRAILING_ABBREVIATION.sub("", text).strip()
    while True:
        stripped = _TRAILING_BARE_UNIT.sub("", _TRAILING_UNITS.sub("", text)).strip()
        if stripped == text:
            break
        text = stripped
    words = text.split()
    if len(words) > 1 and _TRAILING_FOOTNOTE.fullmatch(words[-1]):
        text = " ".join(words[:-1])
    return text.strip(" .,:;")


def parse_cell(text: str) -> Decimal | None:
    """A cell's value, or ``None`` if the cell isn't a figure.

    Brackets mean negative: a results table prints a deduction or write-back as ``(42)``,
    and reading it as positive flips the sign of a driver.
    """
    cleaned = (text or "").strip()
    if not cleaned or not _NUMERIC.match(cleaned):
        return None
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    cleaned = cleaned.strip("()").replace(",", "").replace("$", "").replace("%", "").strip()
    if not cleaned or cleaned in {"-", "."}:
        return None
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    return -value if negative else value


def cell_under(row: Sequence[str], position: int, headed: Iterable[int]) -> str:
    """The cell under the column heading at ``position``, or a figure set just off it.

    A table read from a PDF can put a figure a column away from its heading when the
    heading is wider than the figures under it. BNK Banking Corporation's key metrics (FY26
    report, p.9) set every comparative one column left of "30 June 2025", so the cell under
    the heading was blank and the whole prior year went unread. When it's blank, a figure
    in the next column either side is taken, provided no heading sits over that column and
    this heading is nearer to it than any other.
    """
    cell = row[position] if 0 <= position < len(row) else ""
    if cell.strip():
        return cell
    headings = set(headed)
    for neighbour in (position - 1, position + 1):
        if neighbour in headings or not 0 < neighbour < len(row):
            continue
        candidate = row[neighbour]
        if parse_cell(candidate) is None:
            continue
        distance = abs(neighbour - position)
        if all(abs(neighbour - other) > distance for other in headings - {position}):
            return candidate
    return cell


@dataclass(frozen=True, slots=True)
class SignPolicy:
    """Which components are deductions, and how to tell a table's convention."""

    deductions: frozenset[str]
    anchors: tuple[str, ...]
    anchor_labels: frozenset[str] = frozenset()
    """Normalised labels of costs no component stores, read only to settle a convention."""

    def flip_needed(self, values: dict[str, Decimal]) -> bool:
        """Whether this table signs its deductions negatively.

        Decided from a cost that's never legitimately a credit: operating expenses first,
        then the tax charge. If one comes out negative, the table prints its deductions in
        brackets.

        The second anchor is needed because an appendix income statement may label its costs
        in a way no synonym matches. Westpac's presentation heads its cost line simply
        "Expenses", and with no anchor at all the table's impairment charge kept the sign
        printed on the page: a $193m increase in the charge went into the bridge as a $193m
        benefit, and the bridge missed by twice that.
        """
        return self.decided_by_anchors(values) is True

    def decided_by_anchors(self, values: dict[str, Decimal]) -> bool | None:
        """The convention the anchors show, or ``None`` if no anchor is in the table."""
        for anchor in self.anchors:
            value = values.get(anchor)
            if value is not None and value != 0:
                return value < 0
        return None


#: Phrases that state a figure's notable-items treatment, longest first. A label carrying one
#: is read as the measure it names, on that basis (the section rules read the basis off the
#: same words).
_NOTABLE_ITEMS_PHRASES: tuple[str, ...] = (
    "excluding large notable items",
    "excluding notable items",
    "ex large notable items",
    "ex notable items",
    "before notable items",
)


@dataclass(frozen=True, slots=True)
class LineItemIndex:
    """Normalised row label to component, for one issuer."""

    by_label: dict[str, tuple[str, ...]]
    """Normalised label to every component it feeds.

    One row often feeds two: "Credit impairment charge" is both the headline metric and a
    line in the earnings bridge, and binding it to just one quietly switches off the
    bridge's identity check.
    """
    units: dict[str, Unit]
    unit_suffixes: tuple[tuple[str, Unit], ...]
    signs: SignPolicy
    exclusions: tuple[str, ...]
    plausible: dict[str, tuple[Decimal, Decimal, Unit]]
    division_markers: tuple[str, ...]
    group_markers: tuple[str, ...]
    group_is_max: frozenset[str]
    sections: SectionRules
    preference: dict[str, tuple[str, ...]]
    coherent_groups: tuple[tuple[str, ...], ...]
    basis_neutral: frozenset[str]
    balances: frozenset[str] = frozenset()
    exempt_labels: frozenset[str] = frozenset()
    """Labels that name their component with a phrase the exclusions refuse elsewhere:
    Judo's "Profit before tax" is its headline, where "before tax" usually marks a line
    that isn't the figure."""
    """Components that are a balance at the period end, so a half and a year ending on the
    same date report the same figure. An average is not one of them."""

    @property
    def summed(self) -> frozenset[str]:
        """The lines of a statement that adds up: the earnings bridge (notable items included)
        and the impairment split. A dash in one of their period columns means nil."""
        return frozenset(component for group in self.coherent_groups for component in group)

    def is_divisional_page(self, page_text: str, heading_path: tuple[str, ...] = ()) -> bool:
        """Whether a page's section title marks it as divisional rather than group.

        The PDF's own outline is checked first and decides it when it has an entry. It's the
        issuer's own structure rather than a guess from the text, and on a pack that repeats
        its masthead on every page it's the only reliable signal. Westpac's Consumer
        division sits on a page whose first four lines are "PERFORMANCE REVIEW / DIRECTORS'
        REPORT / 2026 INTERIM FINANCIAL REPORT / ADDITIONAL INFORMATION", and its margin of
        1.74% is nothing like the group's 1.89%.
        """
        for entry in reversed(heading_path):
            if self._names_division(entry):
                return True
        head = " ".join(line.strip() for line in (page_text or "").splitlines() if line.strip())[
            :300
        ]
        haystack = f" {_normalise(head)} "
        if any(f" {marker} " in haystack for marker in self.group_markers):
            return False
        if any(f" {marker} " in haystack for marker in self.division_markers):
            return True
        # A divisional page is usually titled with just the division's name (NAB heads page
        # 44 "Business and Private Banking"), so the same names the section machine uses
        # inside a table also mark a page.
        return self._names_division(
            next((line for line in (page_text or "").splitlines() if line.strip()), "")
        )

    def _names_division(self, title: str) -> bool:
        # A numbered heading ("2.1. Consumer") is still the division it names. Westpac's
        # 2024 interim report numbers its segment pages, and the Consumer division's margin
        # of 1.69% was read as the group's.
        normalised = _normalise(_SECTION_NUMBER.sub("", title or ""))
        return any(
            normalised == name or normalised.startswith(f"{name} ")
            for name in self.sections.divisions
        )

    def is_plausible(self, component: str, value: Decimal, unit: Unit) -> bool:
        """Whether this figure could be the group-level component it matched.

        A label match isn't enough. Packs repeat the same row label on divisional pages in
        different units and at different sizes, and a plausibility range rejects the absurd
        (net interest income of 8.5, a margin read from a percentage-growth column) without
        second-guessing the issuer.
        """
        bounds = self.plausible.get(component)
        if bounds is None:
            return True
        low, high, expected = bounds
        if unit in AUD_UNITS and expected in AUD_UNITS:
            value, _ = normalise_aud(value, unit)
        elif unit != expected:
            return False
        return low <= value <= high

    def is_excluded(self, row_label: str) -> bool:
        """Whether this label names a non-headline variant.

        Margin excluding Markets and Treasury, core net interest income, a New Zealand
        subsidiary's balances, a divisional cut, a pre-tax line: each bridges
        differently from the group headline figure, so none of them may be extracted as
        that figure.
        """
        haystack = f" {_normalise(row_label)} "
        return any(f" {phrase} " in haystack for phrase in self.exclusions)

    def rank_for(self, component: str, row_label: str) -> int:
        """How good a reading of ``component`` this row label is, lower being better.

        One table can print the same component on two rows that differ in a way the
        component name doesn't capture. NAB's return-on-equity note gives "Average equity"
        of $62,937m and, two rows down, "Total average equity (attributable to owners of the
        Company)" of $62,227m. Both are average equity, but only the second is the
        denominator of the ratio the bank published. Taking the first leaves the
        non-controlling interests in, and the identity fails by a basis point.
        """
        preferred = self.preference.get(component)
        if not preferred:
            return 0
        normalised = _normalise(row_label)
        for position, label in enumerate(preferred):
            if normalised == label:
                return position
        return len(preferred)

    def components_for(self, row_label: str) -> tuple[str, ...]:
        """Components this row feeds, trying each reading of the label in turn.

        A label that names the notable-items treatment it's on ("Net profit excluding
        Notable Items") isn't a different measure, it's the headline measure on another
        basis, and the basis is read off the same words. It was refused as an excluded
        variant, so Westpac's net profit excluding notable items was never held, and the
        investor pack's identical $3,483m looked like an unexplained disagreement with the
        $3,414m net profit.

        A label configured for the bank with a phrase the exclusions refuse elsewhere is read
        before them, since for that bank the phrase names the figure.
        """
        exempt = self._exempt_reading(row_label)
        if exempt is not None:
            return self.by_label[exempt]
        if self.is_excluded(row_label):
            restated = self._without_notable_items(row_label)
            if restated is None or self.is_excluded(restated):
                return ()
            row_label = restated
        for reading in label_candidates(row_label):
            found = self.by_label.get(_normalise(reading))
            if found is not None:
                return found
        return ()

    def _exempt_reading(self, row_label: str) -> str | None:
        """The reading of this label configured as exempt from the exclusions, if any."""
        for reading in label_candidates(row_label):
            if _normalise(reading) in self.exempt_labels:
                return _normalise(reading)
        return None

    def is_exempt(self, row_label: str) -> bool:
        return self._exempt_reading(row_label) is not None

    def _without_notable_items(self, row_label: str) -> str | None:
        """The label with a notable-items phrase taken out, or ``None`` if it has none.

        Only the treatment of notable items: every other excluded qualifier (excluding
        Markets, per share, a division) names a different figure, not a different basis.
        """
        lowered = f" {_normalise(row_label)} "
        for phrase in _NOTABLE_ITEMS_PHRASES:
            if f" {phrase} " in lowered:
                return " ".join(lowered.replace(f" {phrase} ", " ").split())
        return None

    def unit_for(
        self,
        component: str,
        row_label: str,
        table_unit: Unit | None,
        cell: str = "",
    ) -> Unit | None:
        """The unit of this cell, using the most specific evidence first.

        The cell itself wins: a figure printed ``68.5%`` is a percentage whatever the
        table's caption says. Next is the row's own suffix, because a key-metrics table
        captioned "$M" still prints its ratios in per cent, and without this a margin of
        1.86 would be stored as 1.86 million dollars. Then the table's caption, which is the
        only thing that knows whether a balance sheet is in $M or $B. The component's
        declared unit is the last resort, when nothing on the page says.
        """
        if cell.strip().endswith("%"):
            return "pct"
        from_label = unit_in_label(row_label)
        if from_label is not None:
            return from_label
        lowered = (row_label or "").lower().strip()
        for suffix, unit in self.unit_suffixes:
            if lowered.endswith(suffix):
                return unit
        declared = self.units.get(component)
        if (
            table_unit is not None
            and declared is not None
            and ((table_unit in AUD_UNITS) != (declared in AUD_UNITS))
        ):
            # A caption can't turn an amount into a ratio, or a ratio into an amount. It only
            # says which of a kind: millions or billions, per cent or basis points. Macquarie's
            # FY26 results table is captioned with its movement column's "%", and its profit of
            # A$4,847m was read as a percentage and refused.
            return declared
        return table_unit or declared


@lru_cache(maxsize=16)
def _raw_config(config_dir: str) -> dict[str, Any]:
    loaded = yaml.safe_load((Path(config_dir) / "line_items.yaml").read_text(encoding="utf-8"))
    return cast("dict[str, Any]", loaded)


@lru_cache(maxsize=32)
def line_item_index(bank_id: str, config_dir: str | None = None) -> LineItemIndex:
    """Build the label index for one issuer, overrides last so they win."""
    raw = _raw_config(config_dir or str(get_settings().config_dir))
    by_label: dict[str, tuple[str, ...]] = {}
    units: dict[str, Unit] = {}
    exempt: set[str] = set()

    def add(label: str, component: str) -> None:
        key = _normalise(label)
        existing = by_label.get(key, ())
        if component not in existing:
            by_label[key] = (*existing, component)

    for section in ("headline", "components"):
        for component, spec in raw.get(section, {}).items():
            units[component] = spec.get("unit")
            labels = [
                *(spec.get("synonyms", ()) or ()),
                *((spec.get("overrides", {}) or {}).get(bank_id, ()) or ()),
            ]
            for label in labels:
                add(label, component)
            if spec.get("exempt_from_exclusions"):
                exempt.update(_normalise(label) for label in labels)

    # A bank no one configured brings the labels a model matched for it, and is held to
    # ranges that allow for a much smaller bank (DESIGN.md ADR-063).
    profile = get_registry().banks.get(bank_id)
    discovered = profile is not None and profile.discovered
    for label, component in profile.row_labels if profile is not None else ():
        add(label, component)
    ranges = {
        **(raw.get("plausible", {}) or {}),
        **((raw.get("plausible_discovered", {}) or {}) if discovered else {}),
    }

    suffixes = tuple(
        sorted(
            ((suffix, unit) for suffix, unit in (raw.get("unit_suffixes", {}) or {}).items()),
            key=lambda item: len(item[0]),
            reverse=True,
        )
    )
    return LineItemIndex(
        by_label=by_label,
        units=units,
        unit_suffixes=suffixes,
        division_markers=tuple(_normalise(m) for m in (raw.get("division_markers", ()) or ())),
        group_markers=tuple(_normalise(m) for m in (raw.get("group_markers", ()) or ())),
        group_is_max=frozenset(raw.get("group_is_max", ()) or ()),
        sections=SectionRules.from_config(raw),
        basis_neutral=frozenset(raw.get("basis_neutral", ()) or ()),
        balances=frozenset(raw.get("balances", ()) or ()),
        exempt_labels=frozenset(exempt),
        coherent_groups=tuple(tuple(group) for group in (raw.get("coherent_groups", ()) or ())),
        preference={
            component: tuple(_normalise(label) for label in spec["prefer"])
            for section in ("headline", "components")
            for component, spec in (raw.get(section, {}) or {}).items()
            if spec.get("prefer")
        },
        plausible={
            component: (
                Decimal(str(spec["min"])),
                Decimal(str(spec["max"])),
                spec.get("unit", "aud_m"),
            )
            for component, spec in ranges.items()
        },
        exclusions=tuple(_normalise(phrase) for phrase in (raw.get("exclusions", ()) or ())),
        signs=SignPolicy(
            deductions=frozenset(raw.get("deductions", ()) or ()),
            anchors=tuple(raw.get("sign_anchors", ()) or ("operating_expenses",)),
            anchor_labels=frozenset(
                _normalise(label) for label in (raw.get("sign_anchor_labels", ()) or ())
            ),
        ),
    )


def extract_facts(
    document: ParsedDocument,
    meta: DocMeta,
    periods: tuple[str, ...],
    point_in_time: bool = False,
) -> list[Fact]:
    """Every figure in this document that maps to a known component and period.

    ``periods`` limits which columns may be read; in a run, that's the two periods being
    compared. Facts come back unvalidated, and :mod:`bankresearch.ingestion.validate`
    decides which reach the fact store.
    """
    registry = get_registry()
    profile = registry.bank(meta.bank_id)
    index = line_item_index(meta.bank_id)
    wanted = {p.upper() for p in periods}

    earnings_rule = registry.metrics["headline_earnings"].label_for(profile, meta.period)
    headline_basis = headline_earnings_basis(
        earnings_rule.label if earnings_rule is not None else profile.headline_earnings_label
    )

    # One reading per component, period and notable-items treatment. The issuer's headline
    # and its excluding-notable-items variant are both its own figures, and a ratio
    # published only on the variant (Westpac's expense-to-income ratio) can only be checked,
    # or explained, against inputs on the same basis.
    found: dict[tuple[str, str, bool], Fact] = {}
    candidates: list[Fact] = []
    rejected: list[Fact] = []
    divisions = _DivisionSplit(profile.impairment_divisions)
    for page in document.pages:
        # A page whose section title names a division carries that division's figures under
        # the same row labels the group uses. Nothing in the rows tells them apart, so the
        # page isn't read for the group. For a bank whose impairment charge is only split
        # by division, its impairment line is read for the division instead.
        if index.is_divisional_page(page.text, page.heading_path):
            divisions.read(page, index, meta, profile.fy_end_month, wanted, point_in_time)
            continue
        for table in page.tables:
            headers = [h for h in table.header if h and h.strip()]
            if not headers:
                continue
            basis = detect_basis(page.text, table.caption)
            try:
                columns = map_columns(
                    headers,
                    fy_end_month=profile.fy_end_month,
                    doc_period=meta.period,
                    point_in_time=point_in_time,
                )
                balance_columns = map_columns(
                    headers,
                    fy_end_month=profile.fy_end_month,
                    doc_period=meta.period,
                    point_in_time=True,
                )
            except ValueError:
                # A misread header row. Skip the table rather than attribute a figure
                # to a period it may not belong to.
                continue

            # In a table of halves the year-end column is the second half for a flow, and
            # still the year end for a balance, so the two are read separately where the
            # readings differ. Signs are settled over the whole table either way.
            readings: list[tuple[dict[str, str], bool | None]] = [(columns, None)]
            if balance_columns != columns:
                readings = [(columns, False), (balance_columns, True)]
            table_facts: list[Fact] = []
            for mapping, balances in readings:
                usable = {
                    header: label for header, label in mapping.items() if label.upper() in wanted
                }
                if not usable:
                    continue
                table_facts.extend(
                    fact
                    for fact in _normalise_signs(
                        _facts_from_table(table, usable, index, meta, basis), index, table
                    )
                    if balances is None or (fact.component in index.balances) is balances
                )
            for fact in table_facts:
                # Page order: every pack puts its group summary before its divisional
                # detail, so the earliest table with a component is the group one.
                # Preferring the table with the most components was tried and is worse,
                # because in a 150-page pack that's a divisional page.
                if not index.is_plausible(fact.component, fact.value, fact.unit):
                    rejected.append(fact)
                    continue
                candidates.append(fact)
                key = (fact.component, fact.period, fact.basis.ex_notable_items)
                existing = found.get(key)
                if existing is None or _prefer(fact, existing, headline_basis, index):
                    found[key] = fact

    # Each treatment is settled on its own, tables and all, so a bridge is never assembled
    # from the headline's lines and the variant's.
    kept: list[Fact] = divisions.complete(wanted)
    for treatment in (False, True):
        chosen = [fact for fact in found.values() if fact.basis.ex_notable_items is treatment]
        if not chosen:
            continue
        pool = [fact for fact in candidates if fact.basis.ex_notable_items is treatment]
        kept.extend(
            _one_table_per_component(
                chosen, wanted, index.coherent_groups, pool, headline_basis, index
            )
        )
    return kept


#: How many of a coherent group's lines a table needs before it counts as the bridge.
_BRIDGE_LINES = 3


def _one_table_per_component(
    facts: list[Fact],
    wanted: set[str],
    coherent: tuple[tuple[str, ...], ...] = (),
    candidates: list[Fact] | None = None,
    headline_basis: str = "n/a",
    index: LineItemIndex | None = None,
) -> list[Fact]:
    """Keep each component's periods from a single table.

    Within one document, both periods of a comparison are printed side by side in the same
    table. If they come from different tables, one of them came from somewhere else (a
    divisional repeat, an appendix segment note) and comparing them mixes sources. CBA's
    profit announcement prints a group margin of 2.05% and a Retail Banking margin of 2.50%
    under almost identical section titles, and taking one period from each gives a movement
    that never happened.

    So the table that supplies the most requested periods wins, and a component whose
    periods can't come from one table is reported as incomplete rather than mixed.

    The same reasoning ties some components to each other. The lines of the earnings bridge
    are one statement, and a pack prints that statement several times: statutory and cash,
    before and after notable items, group and division. Each line is correct in each
    version, so no single figure says which version it came from, and a bridge assembled one
    line at a time won't reconcile to any of them. CBA's full-year summary sets three such
    blocks side by side, and taking cash net interest income with the statutory tax charge
    left the bridge $74m short. Where a group of components is declared coherent, they're
    all read from one table or not at all, and an identity missing an input is skipped
    rather than failed.
    """
    by_component: dict[str, list[Fact]] = {}
    for fact in facts:
        by_component.setdefault(fact.component, []).append(fact)

    # The coherent groups are settled from every reading, not just the winners of the
    # per-component choice. Choosing per component first and grouping afterwards loses the
    # figure the group needed: CBA's cash operating income of $30,224m beats the statutory
    # $30,153m on basis, so the statutory reading is dropped, and then the table that
    # supplies the rest of the bridge is the statutory one, which no longer has an operating
    # income to offer.
    pool: dict[str, list[Fact]] = {}
    for fact in candidates or facts:
        pool.setdefault(fact.component, []).append(fact)

    kept: list[Fact] = []
    for group in coherent:
        members = [component for component in group if component in by_component]
        if len(members) < 2:
            continue
        grouped: dict[str, list[Fact]] = {}
        for component in members:
            by_component.pop(component, None)
            for fact in pool.get(component, ()):
                grouped.setdefault(fact.table_id or "", []).append(fact)
        if not grouped:
            continue
        kept.extend(
            _best_reading_per_period(
                max(
                    grouped.values(),
                    # A table with fewer than three of the group's lines isn't a bridge,
                    # whatever its basis. Bendigo's other-income analysis is tagged "cash"
                    # and holds one line, and with basis deciding first it beat the cash
                    # bridge on page 10 that holds four, so the group kept one line and
                    # the earnings identity had nothing to run on. Between two real
                    # bridges, basis still decides first, as it must for CBA's statutory
                    # and cash pages.
                    key=lambda rows: (
                        len({fact.component for fact in rows}) >= _BRIDGE_LINES,
                        _states(rows) == headline_basis,
                        len(
                            {
                                (fact.component, fact.period)
                                for fact in rows
                                if fact.period in wanted
                            }
                        ),
                        -min(fact.page_no for fact in rows),
                    ),
                ),
                headline_basis,
                index,
            )
        )

    for candidates in by_component.values():
        by_table: dict[str, list[Fact]] = {}
        for fact in candidates:
            by_table.setdefault(fact.table_id or "", []).append(fact)
        best = max(
            by_table.values(),
            key=lambda group: (
                len({fact.period for fact in group} & wanted),
                -group[0].page_no,
            ),
        )
        kept.extend(best)
    return kept


def _states(rows: list[Fact]) -> str:
    """The earnings basis most of a table's rows are on.

    Basis decides the coherent group before completeness does, because a bridge built on the
    wrong basis is useless however complete it is. ANZ states its income statement twice,
    and its statutory page has one more line than its cash page, so ranking on completeness
    alone built the bridge from the statutory lines and compared it with the cash headline
    the bank leads with. That gave a $204m residual from figures that were each exactly
    right.
    """
    tally = Counter(fact.basis.earnings_basis for fact in rows)
    return tally.most_common(1)[0][0] if tally else "n/a"


def _best_reading_per_period(
    rows: list[Fact], headline_basis: str, index: LineItemIndex | None
) -> list[Fact]:
    """One reading per component and period, from the rows of a single table."""
    chosen: dict[tuple[str, str], Fact] = {}
    for fact in rows:
        key = (fact.component, fact.period)
        incumbent = chosen.get(key)
        if incumbent is None or (
            index is not None and _prefer(fact, incumbent, headline_basis, index)
        ):
            chosen[key] = fact
    return list(chosen.values())


def _prefer(candidate: Fact, incumbent: Fact, headline_basis: str, index: LineItemIndex) -> bool:
    """Whether a later reading should replace an earlier one.

    Basis comes first: the same figure appears on both the statutory and the cash page, and
    the issuer's headline basis wins so the bridge isn't built from one page's lines and
    another page's total.

    Then, for a component the group reports as the sum of its divisions, the larger value
    wins. That's what makes a divisional repeat harmless on a page whose section title gives
    no hint which it is.

    Then the headline reading beats the excluding-notable-items one. Both are the issuer's
    own and both are correct, but the headline figure is the one the question asks for; the
    variant is what the reconciler needs to call the difference a basis difference rather
    than an error. (Extraction keeps one reading of each treatment, so this only decides
    between two readings that claim the same one.)

    Then a reading that says it's from continuing operations, or that states its regulatory
    level, beats one that's silent on it.

    Otherwise page order stands, since every pack puts its group summary first.
    """
    if candidate.basis.earnings_basis != incumbent.basis.earnings_basis:
        return (
            candidate.basis.earnings_basis == headline_basis
            and incumbent.basis.earnings_basis != headline_basis
        )
    if candidate.basis.ex_notable_items != incumbent.basis.ex_notable_items:
        return not candidate.basis.ex_notable_items
    # An issuer that splits continuing from discontinued operations is telling you which one
    # its headline is on. Where one reading says so and the other doesn't, the one that says
    # so wins: the figures differ only by the discontinued result, which is exactly what the
    # headline leaves out.
    if candidate.basis.continuing_ops is not incumbent.basis.continuing_ops:
        return candidate.basis.continuing_ops is True
    # Likewise a reading that states its regulatory level beats one that doesn't. Westpac's
    # key-metrics table prints its CET1 ratio with no Level 2 heading above it, four pages
    # before the capital page that does, and taking the first by page order dropped the
    # level the figure is reported at.
    if candidate.basis.scope != incumbent.basis.scope:
        return candidate.basis.scope == "level2" and incumbent.basis.scope == "group"
    if candidate.component in index.group_is_max:
        candidate_value, _ = normalise_aud(candidate.value, candidate.unit)
        incumbent_value, _ = normalise_aud(incumbent.value, incumbent.unit)
        # Materially larger, not just larger. The rule is there to beat a divisional repeat,
        # which differs by a multiple, and it mustn't choose between two readings of the
        # same figure at different precision. ANZ gives average interest earning assets as
        # "$1,142.1bn" on its average balance sheet and as 1,142,128 in an appendix. Letting
        # the appendix win for one period split the component across two tables, and the
        # coherence rule then dropped the period the margin identity needed.
        slack = value_tolerance(candidate.unit, reported=candidate.value) + value_tolerance(
            incumbent.unit, reported=incumbent.value
        )
        # And larger by the margin a division falls short of the group by, not by a few
        # tenths of a per cent. NAB's FY25 pack prints its full-year average interest
        # earning assets as $1,001.2bn and, in a half-year table the layout reader mapped
        # to the same column, the second half's $1,005.3bn. Taking the larger one put a
        # half-year balance under a full-year margin, the identity failed, and both
        # figures were quarantined.
        return candidate_value > incumbent_value + slack and (
            incumbent_value <= candidate_value * _DIVISION_SHARE
        )
    return False


#: A division's share of the group, at most. Any reading within this of a larger one is
#: another reading of the same group figure, not a division's.
_DIVISION_SHARE: Final = Decimal("0.9")


def _normalise_signs(
    facts: list[Fact], index: LineItemIndex, table: ParsedTable | None = None
) -> list[Fact]:
    """Put a table's deduction lines on the magnitude convention the identities use.

    The anchors decide wherever the table has one. Where it has none, the table is asked:
    a cost line no component stores, or a deduction's own label stating which way it's
    printed, settles it (see :func:`_stated_convention`).
    """
    if not facts:
        return facts
    by_period: dict[str, dict[str, Decimal]] = {}
    for fact in facts:
        by_period.setdefault(fact.period, {})[fact.component] = fact.value

    stated = _stated_convention(table, index) if table is not None else None
    flip_periods = set()
    for period, values in by_period.items():
        decided = index.signs.decided_by_anchors(values)
        if decided is None:
            decided = stated
        if decided:
            flip_periods.add(period)
    if not flip_periods:
        return facts
    return [
        fact.model_copy(update={"value": -fact.value})
        if fact.period in flip_periods and fact.component in index.signs.deductions
        else fact
        for fact in facts
    ]  # published_value is deliberately left as the page printed it


#: A bracketed word in a deduction's label says what the brackets mean. "Impairment
#: (charges)/benefits" prints charges in brackets; "charges/(benefits)" prints benefits there.
_BRACKETED_COST = re.compile(r"\((?:charges?|expenses?|costs?)\)", re.IGNORECASE)
_BRACKETED_CREDIT = re.compile(
    r"\((?:benefits?|releases?|reversals?|write-?backs?|credits?)\)", re.IGNORECASE
)


def _stated_convention(table: ParsedTable, index: LineItemIndex) -> bool | None:
    """Whether a table prints its costs negative, from what its own rows say, or ``None``.

    Asked only where no anchor settles it. First a deduction row whose label says which way
    it's printed: Westpac's "Total impairment (charges)/benefits (443)" on page 21, in a
    table with no expense or tax line. Then a cost line no component stores, from
    ``sign_anchor_labels``: its investor pack prints "Expenses (5,698)". A label that only
    describes a ratio ("Impairment charges/(benefits) to average loans") isn't a deduction,
    so it's never asked.
    """
    deductions = index.signs.deductions
    for row in table.rows:
        label = row[0] if row else ""
        if not label or not set(index.components_for(label)) & deductions:
            continue
        if _BRACKETED_COST.search(label):
            return True
        if _BRACKETED_CREDIT.search(label):
            return False
    for row in table.rows:
        if not row or _normalise(row[0] or "") not in index.signs.anchor_labels:
            continue
        for cell in row[1:]:
            value = parse_cell(cell)
            if value is not None and value != 0:
                return value < 0
    return None


def _facts_from_table(
    table: ParsedTable,
    columns: dict[str, str],
    index: LineItemIndex,
    meta: DocMeta,
    basis: Basis,
    divisional: bool = False,
) -> list[Fact]:
    header_positions = {header: position for position, header in enumerate(table.header) if header}
    headed = [position for position, header in enumerate(table.header) if header.strip()]
    out: list[Fact] = []
    # Within one table, a better-ranked row label replaces a worse-ranked one for the
    # same component and period; across tables that is settled later, by basis and page
    # order, where the ranking has nothing to say.
    best: dict[tuple[str, str, str], tuple[int, Fact]] = {}
    section = Section()
    # The words over the label column describe every row beneath them, as a heading above
    # the first row would. Westpac sets "Excluding Notable Items, $m" there, on the same
    # line as the periods. A label header that only names a variant this tool doesn't store
    # (excluding Markets, per share) rules out the whole table, as it would rule out a row.
    label_header = (table.header[0] if table.header else "").strip()
    if label_header and is_heading_label(label_header):
        if index.sections.states(label_header):
            section = index.sections.advance(section, label_header, None)
        elif index.is_excluded(label_header):
            return []
    wrapped = ""
    recoveries_included = False
    row_of: dict[tuple[str, str, str], int] = {}
    for row_index, row in enumerate(table.rows):
        if not row:
            continue
        label = row[0]
        components = index.components_for(label)

        if is_marker_row(row[1:]) and _UNIT_ONLY.match(label.strip()):
            # Bendigo wraps "Net interest margin after revenue share / % / arrangements"
            # over three lines, the middle one just its unit. Treating that line as a new
            # wrap prefix replaced the real first line, so the 1.92% below was read under
            # the name "% arrangements" and matched nothing.
            continue
        if is_marker_row(row[1:]):
            heading = index.sections.states(label) or bool(components)
            if heading and is_heading_label(label):
                section = index.sections.advance(
                    section, label, components[0] if components else None
                )
            # A label-only row that says nothing about the rows below it isn't a heading at
            # all. It's the first line of a name that wrapped, so the next row is read under
            # both readings of its name.
            wrapped = "" if heading else (label if is_wrap_candidate(label) else "")
            continue
        if not (section.readable or (divisional and section.scope == "division")):
            wrapped = ""
            continue

        # Westpac sets "Common Equity Tier 1 (CET1) capital after / deductions ($m)" over
        # two lines, so the row carrying $56,936m is labelled just "deductions ($m)". When
        # there's a pending line above, the joined reading is the row's real name and is
        # tried first, and if that name is an excluded variant the row is rejected outright
        # rather than falling back. Westpac also writes "Net profit attributable to owners
        # of WBC (adjusted for RSP dividends) excluding / Notable Items", and reading the
        # second line alone booked its $3,480m net profit as the notable items themselves.
        # A line that is itself configured to carry an excluded phrase keeps its reading
        # whatever sits above it: the "For personal use only" stamped down the margin of
        # Judo's ASX copy leaves a stray "use" over "Profit before tax" (2026 annual report
        # p.33), and the joined label was refused for the phrase that names the figure.
        if wrapped:
            joined = f"{wrapped} {label}".strip()
            if index.is_excluded(joined) and not index.is_exempt(label):
                wrapped = ""
                continue
            from_joined = index.components_for(joined)
            if from_joined:
                components, label = from_joined, joined
        wrapped = ""

        # A bare qualifier ("Cash basis", "Statutory basis") takes its name from the heading
        # above it. Only a label that states a basis and nothing else can inherit, so an
        # "Operating expenses" heading can never lend its name to the "Personnel" and
        # "Occupancy" rows below it.
        row_dimensions = index.sections.basis_dimensions(label)
        if (
            not components
            and row_dimensions
            and section.component is not None
            and index.sections.is_bare_qualifier(label)
        ):
            components = (section.component,)

        row_basis = section.applied_to(basis)
        if row_dimensions:
            row_dimensions.pop("scope", None)
            row_basis = row_basis.model_copy(update=row_dimensions)

        for component in components:
            for header, period in columns.items():
                position = header_positions.get(header)
                if position is None or position >= len(row):
                    continue
                cell = cell_under(row, position, headed)
                value = parse_cell(cell)
                if value is None and component in index.summed and _DASH.match(cell.strip()):
                    # In a statement that adds up, a dash in a period column is nil.
                    # Westpac prints its 2024 non-controlling interests as "-", and
                    # without reading it the full year's earnings bridge was $17m short.
                    value = Decimal(0)
                if value is None:
                    continue
                # The cell is consulted for the unit: a real pack writes
                # "Return on average ordinary shareholders' equity" with no "(%)"
                # suffix and prints "10.3%" in the cell.
                unit = index.unit_for(component, label, section.unit or table.units, cell)
                if unit is None:
                    continue
                # Thousands are held as millions, like every other amount, and the page's
                # own figure is kept for the citation.
                stored = value
                if unit == "aud_k":
                    stored, unit = normalise_aud(value, unit)
                # Keyed by basis too, so the statutory, cash and excluding-notable-items
                # readings of one row label sit side by side, and the choice between them is
                # made once against the issuer's headline basis rather than by whichever
                # section came first. A stock has no earnings basis to inherit; carrying one
                # made NAB's average equity statutory and so incomparable with the cash
                # earnings it's the denominator of. Nor does it have notable items to
                # exclude, so a balance read in an "excluding" section is the same balance.
                fact_basis = (
                    row_basis.model_copy(
                        update={"earnings_basis": "n/a", "ex_notable_items": False}
                    )
                    if component in index.basis_neutral
                    else row_basis
                )
                key = (component, period, fact_basis.model_dump_json())
                rank = index.rank_for(component, label)
                incumbent = best.get(key)
                if incumbent is not None and incumbent[0] <= rank:
                    continue
                best[key] = (
                    rank,
                    Fact(
                        fact_id="",
                        bank_id=meta.bank_id,
                        period=period,
                        component=component,
                        value=stored,
                        published_value=value,
                        unit=unit,
                        basis=fact_basis,
                        doc_id=meta.doc_id,
                        doc_type=meta.doc_type,
                        page_no=table.page_no,
                        table_id=table.table_id,
                        column_header=header,
                        publish_date=meta.publish_date,
                        validated=False,
                        is_restated=table.restated and period != meta.period,
                    ),
                )
                row_of[key] = row_index
                if component == "impairment_individual" and "recover" in _normalise(label):
                    recoveries_included = True
    # Westpac's 2025 full-year layout lists its recoveries and then totals them into
    # "Total IAPs, write-backs and recoveries". An individual line that already includes
    # them leaves nothing for the recoveries line to add, and counting it again took $247m
    # off a $424m charge.
    # Notable items printed below the headline total are a memo of what the lines above
    # already include: Westpac's "Notable Items (140)" sits under net profit (1H25 p.13).
    # Judo's non-recurring costs sit above statutory profit, as a line of the bridge
    # (2025 annual report p.44). Only the second is stored as a line that adds up.
    headline_rows = [row_of[key] for key in row_of if key[0] == "headline_earnings"]
    headline_at = min(headline_rows) if headline_rows else None
    out.extend(
        fact
        for key, (_, fact) in best.items()
        if not (recoveries_included and fact.component == "impairment_recoveries")
        and not (
            fact.component == "notable_items"
            and headline_at is not None
            and row_of.get(key, -1) > headline_at
        )
    )
    return out


#: The group lines a division's impairment row can map to, best first.
_DIVISION_IMPAIRMENT_LINES = ("credit_impairment_charge", "credit_impairment")
_CONTINUED = re.compile(r"\(continued\)\s*$", re.IGNORECASE)


class _DivisionSplit:
    """A bank's impairment charge by division, read from each division's first page.

    Only for a bank configured with ``impairment_divisions`` (DESIGN.md ADR-038): CBA
    reports its loan impairment expense by division and not as individually and
    collectively assessed. Each division's own page prints its charge in the same row the
    group uses, and the page names the division just under its section title. The first
    page for each division is the one read: the pages marked "(continued)" after it hold the
    detail, and for New Zealand they restate the figures in NZ$.

    The split is kept only if every division is found for every period asked about. A
    partial split would put the missing divisions in the residual and still look like an
    answer, and validation then checks the complete split adds up to the group charge.
    """

    def __init__(self, names: dict[str, str]) -> None:
        self._components = {
            _normalise(name): f"impairment_div_{segment}" for name, segment in names.items()
        }
        self._found: dict[str, list[Fact]] = {}

    def read(
        self,
        page: Any,
        index: LineItemIndex,
        meta: DocMeta,
        fy_end_month: int,
        wanted: set[str],
        point_in_time: bool,
    ) -> None:
        if not self._components:
            return
        component = self._division_of(page.text)
        if component is None or component in self._found:
            return
        basis = detect_basis(page.text).model_copy(update={"scope": "division"})
        for table in page.tables:
            headers = [h for h in table.header if h and h.strip()]
            try:
                columns = map_columns(
                    headers,
                    fy_end_month=fy_end_month,
                    doc_period=meta.period,
                    point_in_time=point_in_time,
                )
            except ValueError:
                continue
            usable = {header: label for header, label in columns.items() if label.upper() in wanted}
            if {label.upper() for label in usable.values()} != wanted:
                continue  # one table has to give every period, or the periods could mix
            # Signs are settled over the whole table, so the division's expenses decide which
            # way its impairment row is printed, as they would for the group.
            facts = _normalise_signs(
                _facts_from_table(table, usable, index, meta, basis, divisional=True),
                index,
                table,
            )
            for line in _DIVISION_IMPAIRMENT_LINES:
                rows = [f for f in facts if f.component == line]
                if rows:
                    rows += _nil_where_dashed(table, line, usable, rows, index)
                if {f.period for f in rows} == wanted:
                    self._found[component] = [
                        f.model_copy(update={"component": component}) for f in rows
                    ]
                    return

    def _division_of(self, text: str) -> str | None:
        """The division a page is for, from the lines just under its section title.

        A division named with "(continued)" after it is a later page, never its first. The
        section title above it says "(continued)" on every page ("Divisional Performance
        (continued)"), so only the line that names the division counts.
        """
        lines = [line.strip() for line in (text or "").splitlines() if line.strip()][:6]
        for line in lines:
            component = self._components.get(_normalise(_CONTINUED.sub("", line)))
            if component is not None:
                return None if _CONTINUED.search(line) else component
        return None

    def complete(self, wanted: set[str]) -> list[Fact]:
        """Every division's figures, or nothing."""
        if not self._components or set(self._found) != set(self._components.values()):
            return []
        return [fact for facts in self._found.values() for fact in facts]


_DASH = re.compile("^[-\u2013\u2014]$")


def _nil_where_dashed(
    table: ParsedTable,
    line: str,
    columns: dict[str, str],
    read: list[Fact],
    index: LineItemIndex,
) -> list[Fact]:
    """Zero for a period whose cell in the division's impairment row is a bare dash.

    In a figure column a dash means nil. CBA's Corporate Centre printed a dash for its 1H26
    charge, and without reading it as zero the half's split had a division missing and
    couldn't be used. Only this row of the divisional pass reads a dash this way, and the
    split it completes still has to add up to the group's charge before it's stored.
    """
    have = {fact.period for fact in read}
    template = read[0]
    added: list[Fact] = []
    positions = {header: position for position, header in enumerate(table.header) if header}
    for row in table.rows:
        if not row or line not in index.components_for(row[0]):
            continue
        for header, period in columns.items():
            position = positions.get(header)
            if period in have or position is None or position >= len(row):
                continue
            if _DASH.match((row[position] or "").strip()):
                added.append(
                    template.model_copy(
                        update={
                            "period": period,
                            "value": Decimal(0),
                            "published_value": Decimal(0),
                            "column_header": header,
                        }
                    )
                )
                have.add(period)
        break
    return added
