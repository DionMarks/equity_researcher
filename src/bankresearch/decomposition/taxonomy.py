"""Maps an issuer's own wording onto the canonical driver taxonomy."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Final

from ..domain.registry import get_registry, other_driver_id, residual_driver_id

__all__ = [
    "STAND_INS",
    "canonical_ids",
    "display_label",
    "driver_label",
    "map_label",
    "map_labels",
    "roll_up",
    "validate_driver_id",
    "walk_driver_ids",
]

#: The drivers that stand in for a breakdown the run didn't read (DESIGN.md ADR-005): CET1
#: capital as one figure rather than its walk, risk-weighted assets rather than their split
#: by risk type, a margin's income against its balances, the collective charge unsplit, the
#: equity behind a return. When one of these is a computed driver with nothing beneath it,
#: it's arithmetic about a total, not a cause, and its confidence is capped at Medium.
#: Cost-to-income's expense and income effects aren't here: they are the economic drivers
#: of that ratio, not stand-ins for one.
STAND_INS: Final[frozenset[str]] = frozenset(
    {
        "cet1.capital",
        "cet1.rwa",
        "nim.income",
        "nim.balance_growth",
        "cic.collective",
        "roe.equity",
    }
)

_PUNCTUATION = re.compile(r"[^a-z0-9]+")


def _normalise(text: str) -> str:
    """Lowercase, "&" as "and", strip punctuation, collapse whitespace to single spaces.

    Charts abbreviate with ampersands far more than tables do: Westpac's margin walk has
    "Capital & other" and "T&M", ANZ's "Capital & replicating portfolio". Dropping the
    ampersand as punctuation made "Treasury & Markets" a different phrase from "markets and
    treasury" for no reason.
    """
    return _PUNCTUATION.sub(" ", text.lower().replace("&", " and ")).strip()


def _singular(normalised: str) -> str:
    """Each word without a plural "s", so "Assets pricing" finds "asset pricing".

    Applied to both sides of the match, so it only has to be consistent rather than
    correct English: "basis" becomes "basi" in the label and in the index alike.
    """
    return " ".join(
        word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word
        for word in normalised.split()
    )


def _variants(label: str) -> list[str]:
    """The forms of a label worth matching, most literal first.

    A bracketed expansion is matched on its own as well as with the rest: a chart labels a
    bar "WSF" and spells it out in a footnote, and a reader that has seen both returns
    "WSF (wholesale funding costs)".
    """
    forms = [label]
    bracketed = re.findall(r"\(([^)]*)\)", label)
    if bracketed:
        forms.append(re.sub(r"\([^)]*\)", " ", label))
        forms.extend(bracketed)
    out: list[str] = []
    for form in forms:
        normalised = _normalise(form)
        for candidate in (normalised, _singular(normalised)):
            if candidate and candidate not in out:
                out.append(candidate)
    return out


@lru_cache(maxsize=32)
def _synonym_index(metric_id: str) -> dict[str, str]:
    """Normalised synonym (and canonical id) -> canonical driver id."""
    registry = get_registry()
    group = registry.drivers.get(metric_id, {})
    index: dict[str, str] = {}
    for driver_id, spec in group.items():
        index[_normalise(driver_id)] = driver_id
        # An aggregate node is declared with an empty synonym list because a walk never
        # labels a bar with it, so its label isn't matched either. Matching it let a bar
        # called "Net interest income" (an earnings walk) pass as a margin driver, which is
        # what the empty list in config exists to prevent.
        if spec.get("synonyms", None) != []:
            index[_normalise(spec.get("label", ""))] = driver_id
        for synonym in spec.get("synonyms", ()) or ():
            key = _normalise(synonym)
            # First declaration wins, so the taxonomy order in config decides
            # ties rather than dict iteration order. The singular form goes in too, so
            # a label matches whichever way the issuer pluralised it.
            index.setdefault(key, driver_id)
            index.setdefault(_singular(key), driver_id)
    for template in registry.any_metric_drivers:
        resolved = template.format(metric=_prefix(metric_id))
        index[_normalise(resolved)] = resolved
    index.pop("", None)
    return index


def _prefix(metric_id: str) -> str:
    return residual_driver_id(metric_id).rsplit(".", 1)[0]


def canonical_ids(metric_id: str) -> tuple[str, ...]:
    """Every driver id valid for this metric, including `.other` and `.residual`."""
    registry = get_registry()
    group = tuple(registry.drivers.get(metric_id, {}))
    extras = (other_driver_id(metric_id), residual_driver_id(metric_id))
    return group + extras


def map_label(metric_id: str, label: str) -> str | None:
    """The canonical driver id for an issuer's wording, or ``None``.

    Exact match first; then an unambiguous phrase match. Never a guess.
    """
    if not label:
        return None
    index = _synonym_index(metric_id)
    variants = _variants(label)
    for variant in variants:
        exact = index.get(variant)
        if exact is not None:
            return exact
    for variant in variants:
        found = _phrase_match(metric_id, variant)
        if found is not None:
            return found
    return None


# Short phrases match too loosely to be trusted inside a longer label.
_MIN_PHRASE_LENGTH = 6


#: Parts of a compound label that add nothing to which driver it is.
_NEUTRAL_PARTS = frozenset({"other", "others", "other item", "sundry", "miscellaneou", "misc"})


def _phrase_match(metric_id: str, normalised: str) -> str | None:
    """A driver whose synonym appears in the label, only if exactly one does.

    A label joining parts with "and" is refused when its parts name different drivers,
    even if only one of them contains a synonym. ANZ's "Assets & funding mix" contains
    "funding mix", a deposit-mix synonym, but the bar covers asset mix too, and booking all
    of it to deposit mix would name a cause the issuer didn't state.

    Each part is read with the head noun it shares with the next one, because "and" in a
    label usually joins modifiers: "Wholesale and term funding costs" is wholesale funding
    costs and term funding costs, one driver, while "Assets & funding mix" is asset mix
    and funding mix, two. An "and other" tail is ignored.
    """
    parts = [
        part
        for part in normalised.split(" and ")
        if part and part not in _NEUTRAL_PARTS and _singular(part) not in _NEUTRAL_PARTS
    ]
    if len(parts) > 1:
        named = {
            found
            for i, part in enumerate(parts)
            if (found := _part_driver(metric_id, part, parts[i + 1] if i + 1 < len(parts) else ""))
        }
        if len(named) > 1:
            return None
    return _single_phrase(metric_id, " and ".join(parts) if parts else normalised)


def _part_driver(metric_id: str, part: str, following: str) -> str | None:
    """The driver one part of a compound label names, borrowing a head noun if it needs one."""
    index = _synonym_index(metric_id)
    for candidate in (part, _singular(part)):
        if candidate in index:
            return index[candidate]
    found = _single_phrase(metric_id, part)
    if found is not None:
        return found
    words = following.split()
    for k in range(1, len(words)):
        borrowed = f"{part} {' '.join(words[-k:])}"
        for candidate in (borrowed, _singular(borrowed)):
            if candidate in index:
                return index[candidate]
        found = _single_phrase(metric_id, borrowed)
        if found is not None:
            return found
    return None


def _single_phrase(metric_id: str, normalised: str) -> str | None:
    """The one driver with a synonym inside this text, or ``None``."""
    hits: dict[str, int] = {}
    for phrase, driver_id in _synonym_index(metric_id).items():
        if len(phrase) < _MIN_PHRASE_LENGTH:
            continue
        if f" {phrase} " in f" {normalised} ":
            hits[driver_id] = max(hits.get(driver_id, 0), len(phrase))
    if len(hits) != 1:
        return None
    return next(iter(hits))


def map_labels(metric_id: str, labels: list[str]) -> tuple[dict[str, str], list[str]]:
    """Map many labels at once.

    Returns the mappings that worked and the labels that didn't, so the caller can decide
    what to do with the rest: classify them with a restricted model, or leave them unmapped.
    """
    mapped: dict[str, str] = {}
    unmapped: list[str] = []
    for label in labels:
        hit = map_label(metric_id, label)
        if hit is None:
            unmapped.append(label)
        else:
            mapped[label] = hit
    return mapped, unmapped


def walk_driver_ids(metric_id: str) -> tuple[str, ...]:
    """The ids a walk bar can be mapped to: the metric's drivers less its aggregate nodes.

    The aggregates (declared with no synonyms) are the identity's two halves, never a bar,
    and ``.residual`` and ``.other`` are what's left over rather than a cause.
    """
    group = get_registry().drivers.get(metric_id, {})
    return tuple(driver_id for driver_id, spec in group.items() if spec.get("synonyms", None) != [])


def roll_up(metric_id: str, driver_id: str) -> str | None:
    """The driver's parent, where the taxonomy declares one."""
    return get_registry().driver_parent(metric_id, driver_id)


def driver_label(metric_id: str, driver_id: str) -> str:
    return get_registry().driver_label(metric_id, driver_id)


#: The note on a driver's sign the taxonomy carries for the model ("higher costs =
#: negative"). A reader sees the sign in the contribution itself.
_SIGN_NOTE: Final = re.compile(r"\s*\((?:higher|lower)\b[^)]*=\s*negative\)\s*$")


def display_label(metric_id: str, driver_id: str) -> str:
    """A driver's name as a reader sees it: the taxonomy's label without its sign note."""
    return _SIGN_NOTE.sub("", driver_label(metric_id, driver_id))


def validate_driver_id(metric_id: str, driver_id: str) -> bool:
    """Whether a driver id is in this metric's taxonomy.

    Used to reject a driver id proposed by a model before it reaches the output, so an
    invented id can't appear in a report.
    """
    return driver_id in canonical_ids(metric_id)
