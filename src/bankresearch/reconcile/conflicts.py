"""When two sources disagree: classify first, resolve second."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Final

from ..domain.models import Basis, CandidateValue, ConflictClass, DataConflict
from ..domain.registry import get_registry
from ..domain.units import AUD_UNITS, RATIO_UNITS, Unit, to_aud_m, to_bps, value_tolerance

__all__ = ["CLASSIFICATION_ORDER", "classify", "normalised_value", "resolve"]

CLASSIFICATION_ORDER: Final[tuple[ConflictClass, ...]] = (
    "UNIT",
    "ROUNDING",
    "BASIS",
    "SCOPE",
    "PRO_FORMA",
    "RESTATEMENT",
    "TIMING",
    "UNEXPLAINED",
)

# Fallback mirror of config/precedence.yaml, used only if config is unavailable.
# A test asserts the two agree.
_DEFAULT_DOC_PRECEDENCE: Final[dict[str, int]] = {
    "RESULTS_DOCUMENT": 1,
    "RESTATEMENT_NOTICE": 1,
    "RESULTS_RELEASE": 2,
    "INVESTOR_PRESENTATION": 2,
    "PILLAR3": 3,
    "ANNUAL_REPORT": 3,
}


def _doc_precedence(doc_type: str) -> int:
    try:
        return int(get_registry().precedence["doc_type_precedence"][doc_type])
    except Exception:  # pragma: no cover - config always present in this repo
        return _DEFAULT_DOC_PRECEDENCE.get(doc_type, 9)


def normalised_value(candidate: CandidateValue) -> Decimal:
    """The candidate's value in a unit shared across the comparison.

    Ratios normalise to basis points and amounts to AUD millions, so a figure
    quoted as ``1.95%`` and one quoted as ``195bp`` compare as the same number.
    """
    if candidate.unit in RATIO_UNITS:
        return to_bps(candidate.value, candidate.unit)
    if candidate.unit in AUD_UNITS:
        return to_aud_m(candidate.value, candidate.unit)
    return candidate.value


def _normalised_unit(unit: Unit) -> str:
    if unit in RATIO_UNITS:
        return "bps"
    if unit in AUD_UNITS:
        return "aud_m"
    return unit


def _own_tolerance(candidate: CandidateValue) -> Decimal:
    """Half the last digit this candidate was actually reported to."""
    return value_tolerance(candidate.unit, reported=candidate.value)


def _spread(candidates: list[CandidateValue]) -> Decimal:
    values = [normalised_value(c) for c in candidates]
    return max(values) - min(values)


def _signs_differ(candidates: list[CandidateValue]) -> bool:
    signs = {
        1 if normalised_value(c) > 0 else -1 if normalised_value(c) < 0 else 0 for c in candidates
    }
    return 1 in signs and -1 in signs


def _most_precise(candidates: list[CandidateValue]) -> CandidateValue:
    return min(candidates, key=lambda c: (_own_tolerance(c), c.source.doc_id))


def _pick_by_precedence(candidates: list[CandidateValue]) -> CandidateValue:
    return min(
        candidates,
        key=lambda c: (
            _doc_precedence(c.source.doc_type),
            _own_tolerance(c),
            c.source.doc_id,
        ),
    )


def _matching_basis(
    candidates: list[CandidateValue], headline: Basis | None, attribute: str
) -> CandidateValue | None:
    if headline is None:
        return None
    wanted = getattr(headline, attribute)
    hits = [c for c in candidates if getattr(c.basis, attribute) == wanted]
    return _pick_by_precedence(hits) if hits else None


def _closest_to_headline(
    candidates: list[CandidateValue], headline: Basis | None, attributes: tuple[str, ...]
) -> CandidateValue | None:
    """The candidate on the headline's basis, narrowing one dimension at a time.

    Matching on the first dimension that differs and then falling back to document
    precedence isn't enough once a component is held both with and without notable items.
    Where the candidates differ by earnings basis, the ones on the headline's earnings basis
    can still include the variant, and precedence alone could pick it. Each dimension
    narrows the set only where something matches, so a dimension nothing matches on is
    skipped rather than emptying it.
    """
    if headline is None:
        return None
    pool = candidates
    for attribute in attributes:
        wanted = getattr(headline, attribute)
        hits = [c for c in pool if getattr(c.basis, attribute) == wanted]
        if hits:
            pool = hits
    return _pick_by_precedence(pool) if pool is not candidates else None


def classify(
    component: str,
    candidates: list[CandidateValue],
    *,
    headline_basis: Basis | None = None,
    measurement_dates: dict[str, date] | None = None,
    period_end: date | None = None,
) -> DataConflict | None:
    """Classify a disagreement, or return ``None`` if there isn't one.

    There's a conflict when two candidates differ by more than the strictest tolerance among
    them, which is the precision of the most precisely reported figure. Classification then
    uses the loosest, because "equal after rounding to the coarser precision" is a statement
    about the coarser figure.
    """
    if len(candidates) < 2:
        return None

    tolerances = [_own_tolerance(c) for c in candidates]
    spread = _spread(candidates)
    units = {_normalised_unit(c.unit) for c in candidates}
    raw_units = {c.unit for c in candidates}

    # A unit difference is recorded as a classified non-conflict rather than ignored, so the
    # report can say the figures were reconciled by normalising them. The class has no
    # effect on confidence.
    units_differ_only = len(raw_units) > 1 and len(units) == 1 and spread <= max(tolerances)
    if not units_differ_only and spread <= min(tolerances):
        return None  # agreement, not a conflict

    def build(
        classification: ConflictClass, chosen: CandidateValue, rule: str, impact: str = ""
    ) -> DataConflict:
        return DataConflict(
            component=component,
            candidates=list(candidates),
            classification=classification,
            chosen=chosen,
            rule_applied=rule,
            impact=impact,
        )

    # 1. UNIT: the same quantity written differently. Checked first so a unit difference is
    #    never mistaken for a disagreement about the figure.
    if units_differ_only:
        return build(
            "UNIT",
            _most_precise(candidates),
            "normalised to a common unit; the values agree",
        )

    # 2. ROUNDING: equal once rounded to the coarser precision.
    if spread <= max(tolerances):
        return build(
            "ROUNDING",
            _most_precise(candidates),
            "equal after rounding to the coarser precision; took the more precise value",
        )

    # 3. BASIS: cash vs statutory, with or without notable items, continuing operations.
    for attribute, label in (
        ("earnings_basis", "earnings basis"),
        ("ex_notable_items", "notable-items treatment"),
        ("continuing_ops", "continuing-operations treatment"),
    ):
        if len({getattr(c.basis, attribute) for c in candidates}) > 1:
            narrowing = (
                ("earnings_basis", "ex_notable_items")
                if attribute != "continuing_ops"
                else ("continuing_ops",)
            )
            chosen = _closest_to_headline(
                candidates, headline_basis, narrowing
            ) or _pick_by_precedence(candidates)
            return build(
                "BASIS",
                chosen,
                f"candidates differ by {label}; took the headline basis, "
                "others recorded as alternative bases",
            )

    # 4. SCOPE: Level 1 vs Level 2, group vs division.
    if len({c.basis.scope for c in candidates}) > 1:
        chosen = _matching_basis(candidates, headline_basis, "scope") or _pick_by_precedence(
            candidates
        )
        return build(
            "SCOPE",
            chosen,
            "candidates differ by consolidation scope; took the bank's headline scope",
        )

    # 5. PRO_FORMA: the reported figure wins; pro forma becomes context.
    if len({c.basis.pro_forma for c in candidates}) > 1:
        reported = [c for c in candidates if not c.basis.pro_forma]
        return build(
            "PRO_FORMA",
            _pick_by_precedence(reported),
            "took the reported figure; pro forma recorded as an alternative basis",
        )

    # 6. RESTATEMENT: a later vintage of the same period differs.
    vintages = {c.vintage for c in candidates}
    if len(vintages) > 1:
        latest = max(vintages)
        newest = [c for c in candidates if c.vintage == latest]
        chosen = _pick_by_precedence(newest)
        unfootnoted = any(c.is_restated and not c.footnoted for c in newest)
        return build(
            "RESTATEMENT",
            chosen,
            "took the vintage in the current period's documents; original recorded as a caveat",
            impact="restated comparative used without an explaining footnote: -0.05"
            if unfootnoted
            else "restatement footnoted by the issuer; no confidence penalty",
        )

    # 7. TIMING: measured at different dates.
    if measurement_dates and period_end is not None:
        dated = {c.source.doc_id: measurement_dates.get(c.source.doc_id) for c in candidates}
        if len({d for d in dated.values() if d is not None}) > 1:
            matching = [
                c for c in candidates if measurement_dates.get(c.source.doc_id) == period_end
            ]
            if matching:
                return build(
                    "TIMING",
                    _pick_by_precedence(matching),
                    "candidates measured at different dates; took the value at the period end",
                )

    # 8. UNEXPLAINED: none of the known causes. Both values stay visible.
    chosen = _pick_by_precedence(candidates)
    cap = "0.45" if _signs_differ(candidates) else "0.75"
    detail = (
        "the conflict flips the driver's sign"
        if _signs_differ(candidates)
        else "dependent drivers capped at Medium"
    )
    return build(
        "UNEXPLAINED",
        chosen,
        "no known cause; resolved by source precedence, both values reported",
        impact=f"dependent drivers capped at {cap} ({detail})",
    )


def resolve(
    component: str,
    candidates: list[CandidateValue],
    *,
    as_of: date | None = None,
    headline_basis: Basis | None = None,
    measurement_dates: dict[str, date] | None = None,
    period_end: date | None = None,
) -> tuple[CandidateValue, list[DataConflict]]:
    """Settle on one value, recording any conflict that had to be resolved.

    Raises if a candidate was published after the as-of date. That's look-ahead leakage, and
    refusing it here is cheaper than detecting it later.
    """
    if not candidates:
        raise ValueError(f"no candidates for {component}")
    if as_of is not None:
        late = [c for c in candidates if c.source.publish_date > as_of]
        if late:
            raise ValueError(
                f"{component}: {len(late)} candidate(s) published after as-of {as_of.isoformat()} "
                f"({', '.join(sorted(c.source.doc_id for c in late))}); point-in-time "
                "integrity forbids using them"
            )

    # Stage one: the headline's own readings, settled among themselves.
    own = _on_headline_basis(candidates, headline_basis)
    inner = classify(
        component,
        own,
        headline_basis=headline_basis,
        measurement_dates=measurement_dates,
        period_end=period_end,
    )
    # A reading whose table doesn't say its basis, but whose figure is exactly the bank's
    # own reading on another basis, is that reading. Westpac's investor pack prints net
    # profit of $3,483m in a table headed only "Net profit", and its results document
    # prints $3,483m as "Net profit excluding Notable Items": not a disagreement with the
    # $3,414m headline, and calling it unexplained capped every driver at Medium and told
    # the reader the documents disagreed for no reason. The same figure in a later document
    # looked like a restatement of the prior half too: both results documents print 1H25's
    # $3,317m, and the memo said a restatement had been "footnoted by the issuer".
    matched: list[CandidateValue] = []
    if inner is not None and inner.classification in ("UNEXPLAINED", "RESTATEMENT"):
        elsewhere = [c for c in candidates if not any(c is o for o in own)]
        matched = [
            c
            for c in own
            if any(_agrees(c, o) and o.basis.differs_from(c.basis) for o in elsewhere)
        ]
        rest = [c for c in own if not any(c is m for m in matched)]
        if matched and rest:
            own = rest
            inner = classify(
                component,
                own,
                headline_basis=headline_basis,
                measurement_dates=measurement_dates,
                period_end=period_end,
            )
        else:
            matched = []
    if inner is not None:
        chosen = inner.chosen
    else:
        # They agree, but which one is carried forward still matters, because its basis
        # travels with it. NAB prints its 1H25 return on equity as 11.7% with and without
        # large notable items; carrying the variant forward made the ratio incomparable
        # with the headline earnings it's built from, and the drivers disappeared.
        chosen = _closest_to_headline(
            own, headline_basis, ("earnings_basis", "ex_notable_items")
        ) or _pick_by_precedence(own)
    conflicts = [inner] if inner is not None else []

    # Stage two: readings on another basis, recorded where they differ from what was chosen.
    differing = [c for c in candidates if not any(c is o for o in own) and not _agrees(c, chosen)]
    if differing:
        how = _how_bases_differ(
            [c for c in differing if not any(c is m for m in matched)] or differing, chosen
        )
        rule = (
            f"candidates differ by {how}; took the headline basis, others recorded as "
            "alternative bases"
        )
        if matched:
            rule += (
                "; "
                + ", ".join(f"{m.source.doc_id} p.{m.source.page}" for m in matched)
                + " states no basis but prints exactly the issuer's own figure on the other "
                "basis, so it is read as that figure"
            )
        conflicts.insert(
            0,
            DataConflict(
                component=component,
                candidates=list(candidates),
                classification="BASIS",
                chosen=chosen,
                rule_applied=rule,
            ),
        )
    return chosen, conflicts


def _on_headline_basis(
    candidates: list[CandidateValue], headline: Basis | None
) -> list[CandidateValue]:
    """The candidates on the headline's earnings basis and notable-items treatment.

    Narrowed one dimension at a time and only where something matches, so a set with
    nothing on the headline's basis is kept whole rather than emptied, and is then settled
    by the classes in order, as it always was.
    """
    if headline is None:
        return candidates
    pool = candidates
    for attribute in ("earnings_basis", "ex_notable_items"):
        wanted = getattr(headline, attribute)
        hits = [c for c in pool if getattr(c.basis, attribute) == wanted]
        if hits:
            pool = hits
    return pool


def _agrees(candidate: CandidateValue, chosen: CandidateValue) -> bool:
    """The same figure to the precision either was printed to."""
    gap = abs(normalised_value(candidate) - normalised_value(chosen))
    return gap <= max(_own_tolerance(candidate), _own_tolerance(chosen))


def _how_bases_differ(differing: list[CandidateValue], chosen: CandidateValue) -> str:
    for attribute, label in (
        ("earnings_basis", "earnings basis"),
        ("ex_notable_items", "notable-items treatment"),
        ("continuing_ops", "continuing-operations treatment"),
        ("scope", "consolidation scope"),
        ("pro_forma", "pro forma treatment"),
    ):
        if any(getattr(c.basis, attribute) != getattr(chosen.basis, attribute) for c in differing):
            return label
    return "basis"
