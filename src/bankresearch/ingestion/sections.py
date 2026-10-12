"""What a label-only row says about the rows below it."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Final, Literal

from ..domain.models import Basis
from ..domain.units import Unit

__all__ = [
    "Section",
    "SectionRules",
    "is_heading_label",
    "is_marker_row",
    "is_wrap_candidate",
]

_PUNCTUATION = re.compile(r"[^a-z0-9]+")
_SUPERSCRIPT = re.compile(r"[¹²³⁰-⁹₀-₉*†‡]")
# A unit the section names for the rows under it: "Volumes ($bn)", "Balances ($m)".
# Words a bare qualifier may carry besides its basis phrase: "On a cash basis".
_QUALIFIER_FILLER = frozenset({"on", "a", "the", "and", "basis"})
_SECTION_UNIT = re.compile(r"\((a?\$\s?[mb]n?|nz\$\s?[mb]n?|%)\)", re.IGNORECASE)

Scope = Literal["group", "level2", "level1", "division"]


def _normalise(label: str) -> str:
    return _PUNCTUATION.sub(" ", _SUPERSCRIPT.sub(" ", label or "").lower()).strip()


@dataclass(frozen=True, slots=True)
class Section:
    """The reading in force at a given row, from the marker rows above it."""

    scope: Scope = "group"
    earnings_basis: str | None = None
    ex_notable_items: bool = False
    continuing_ops: bool | None = None
    unit: Unit | None = None
    component: str | None = None
    """The component a heading row named, for the bare qualifier rows beneath it."""

    @property
    def readable(self) -> bool:
        """Whether figures in this section are the group figures we want.

        A divisional cut and a Level 1 capital structure are both correct, and neither is
        the group headline, so they aren't stored at all rather than stored and ranked.
        Ranking them was tried: a division that happens to lead the table wins on page
        order, and the only thing that would catch it is the identity check it breaks.
        """
        return self.scope in {"group", "level2"}

    def applied_to(self, basis: Basis) -> Basis:
        """The page's basis refined by this section's own."""
        update: dict[str, Any] = {"ex_notable_items": self.ex_notable_items}
        if self.earnings_basis is not None:
            update["earnings_basis"] = self.earnings_basis
        if self.continuing_ops is not None:
            update["continuing_ops"] = self.continuing_ops
        if self.scope == "level2":
            update["scope"] = "level2"
        return basis.model_copy(update=update)


@dataclass(frozen=True, slots=True)
class SectionRules:
    """The marker phrases, from ``config/line_items.yaml``."""

    divisions: tuple[str, ...]
    group: tuple[str, ...]
    basis: tuple[tuple[str, dict[str, Any]], ...]

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> SectionRules:
        spec = raw.get("sections", {}) or {}
        return cls(
            divisions=tuple(_normalise(name) for name in (spec.get("divisions", ()) or ())),
            group=tuple(_normalise(name) for name in (spec.get("group", ()) or ())),
            # Longest phrase first, so "excluding large notable items" is tested before
            # "notable items" and the more specific reading wins.
            basis=tuple(
                sorted(
                    (
                        (_normalise(phrase), dims)
                        for phrase, dims in (spec.get("basis", {}) or {}).items()
                    ),
                    key=lambda item: len(item[0]),
                    reverse=True,
                )
            ),
        )

    def _names_division(self, label: str) -> bool:
        normalised = _normalise(label)
        return any(
            normalised == name or normalised.startswith(f"{name} ") for name in self.divisions
        )

    def _names_group(self, label: str) -> bool:
        haystack = f" {_normalise(label)} "
        return any(f" {name} " in haystack for name in self.group)

    def basis_dimensions(self, label: str) -> dict[str, Any]:
        """Every basis dimension this label states, most specific phrase first."""
        haystack = f" {_normalise(label)} "
        found: dict[str, Any] = {}
        for phrase, dimensions in self.basis:
            if f" {phrase} " in haystack:
                for key, value in dimensions.items():
                    found.setdefault(key, value)
        return found

    def is_bare_qualifier(self, label: str) -> bool:
        """Whether this label states a basis and says nothing else.

        Only such a row can take its name from the heading above it. Westpac follows its
        "Notable Items" heading with prose that the layout rebuild carves into a row,
        "Notable Items reduced net profit after tax in | 2025", and because the line names
        a basis it inherited the heading's component: the year was stored as the notable
        items themselves.
        """
        remainder = f" {_normalise(label)} "
        for phrase, _dimensions in self.basis:
            remainder = remainder.replace(f" {phrase} ", " ")
        return not [word for word in remainder.split() if word not in _QUALIFIER_FILLER]

    def states(self, label: str) -> bool:
        """Whether this label says something about the rows below it.

        A label that does is a heading, and a heading is never a wrapped name. Westpac sets
        "Business performance - excluding Notable Items" above the margin it leads with, and
        reading that as the first line of the margin's own name made the joined label an
        excluded variant, which threw the margin away entirely.
        """
        return bool(
            self._names_division(label)
            or self._names_group(label)
            or self.basis_dimensions(label)
            or _SECTION_UNIT.search(label or "")
        )

    def advance(self, section: Section, label: str, component: str | None) -> Section:
        """The section in force after this marker row.

        A division or group name opens a top-level section and clears what the previous one
        said. Anything else refines the current section, because "Volumes ($bn)" sits inside
        a division and mustn't end it.
        """
        if self._names_division(label):
            return Section(scope="division")
        if self._names_group(label):
            section = Section(scope="group")
        dimensions = self.basis_dimensions(label)
        if dimensions:
            scope = dimensions.pop("scope", None)
            section = replace(section, **dimensions)
            if scope is not None:
                section = replace(section, scope=scope)
        unit = _SECTION_UNIT.search(label or "")
        if unit is not None:
            from .tables import detect_units

            section = replace(section, unit=detect_units(unit.group(1)))
        # A heading that names a component lends it to the bare qualifier rows under it;
        # one that names none ends any such block rather than letting it run on.
        return replace(section, component=component)


# A heading is short and isn't a sentence. A mixed page's prose reaches the section logic as
# a label with no figures, just like a heading, and a paragraph that happens to mention New
# Zealand mustn't switch the reading to a division.
_MAX_HEADING_CHARS = 90
_SENTENCE = re.compile(r"[.!?]\s|[.!?]$")


def is_heading_label(label: str) -> bool:
    """Whether a label with no figures beside it reads as a section heading."""
    text = (label or "").strip()
    if not text or len(text) > _MAX_HEADING_CHARS:
        return False
    return _SENTENCE.search(text.rstrip("¹²³*†‡")) is None


#: The words of "For personal use only", which the ASX stamps down the margin of every copy
#: on its announcements site. Read with the page, they land among the row labels: CBA's FY24
#: tax line reads "only Corporate tax expense", and Westpac's 1H24 non-controlling interests
#: arrive as "only Profit attributable to non-controlling" over "interests (NCI)".
ASX_STAMP_WORDS: Final = frozenset({"for", "personal", "use", "only"})


def is_wrap_candidate(label: str) -> bool:
    """Whether a label with no figures could be the first line of a label that wrapped.

    Looser than :func:`is_heading_label` on purpose, because the two questions are
    different. A heading changes how every row below it is read, so it has to be short and
    look like a heading. A wrap prefix only offers the next row a second reading of its own
    name, which costs nothing if it's wrong, and the one that matters here is 91 characters
    long. Westpac writes "divided by: Net profit attributable to owners of WBC (adjusted for
    RSP dividends) excluding / Notable Items", and with that line ignored, the $3,480m of
    net profit below it was booked as the notable items themselves.

    A line that ends a sentence isn't a wrap. It's a footnote, and a footnote mentioning
    notable items mustn't suppress the row after it.
    """
    text = (label or "").strip()
    if text.lower() in ASX_STAMP_WORDS:
        # A word of the stamp down the margin of every ASX copy, never half of a name.
        return False
    return bool(text) and _SENTENCE.search(text.rstrip("¹²³*†‡")) is None


def is_marker_row(cells: tuple[str, ...] | list[str]) -> bool:
    """Whether a row states a section rather than a figure.

    A marker row has a label and no values at all. Requiring no values, rather than just a
    few, is what makes this safe: a data row can never be mistaken for a heading, so no
    figure is ever relabelled by the rows around it.
    """
    return not any((cell or "").strip() for cell in cells)
