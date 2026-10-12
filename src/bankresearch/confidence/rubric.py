"""Confidence from evidence: every figure and driver earns its score from what stands behind
it, and named problems take from it or cap it (DESIGN.md ADR-069 and ADR-072)."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Final

from .features import ConfidenceFeatures, DriverEvidence, FigureEvidence

__all__ = [
    "CLAMP_HIGH",
    "CLAMP_LOW",
    "LEVEL_TARGETS",
    "ScoredConfidence",
    "level_for",
    "overall_confidence",
    "score_driver",
    "score_figures",
]

CLAMP_LOW: Final = Decimal("0.05")
#: A result read from a finite set of documents always carries some risk of a misread cell
#: or an unread footnote, so nothing reaches certainty.
CLAMP_HIGH: Final = Decimal("0.98")

_HIGH_FLOOR: Final = Decimal("0.85")
_MEDIUM_FLOOR: Final = Decimal("0.60")

# Target share of claims correct at each level, checked by the eval harness.
LEVEL_TARGETS: Final[dict[str, Decimal | None]] = {
    "High": Decimal("0.90"),
    "Medium": Decimal("0.70"),
    "Low": None,
}

#: What a printed figure is worth with nothing behind it but the page it's on. Everything
#: above this is earned from evidence, in proportion to the share of the figures it's true
#: of, and the three together reach the ceiling: none of them alone can.
_PRINTED: Final = Decimal("0.86")
_PRIMARY: Final = Decimal("0.04")
_CORROBORATED: Final = Decimal("0.04")
_CONFIRMED: Final = Decimal("0.04")
#: A row a model named. Its figure still had to pass every check, but its label was never
#: read off the page by the person who configured the bank.
_MODEL_MATCHED: Final = Decimal("0.06")
_RESTATED: Final = Decimal("0.02")
_RESTATED_UNFOOTNOTED: Final = Decimal("0.05")
#: Printing precision against the movement: ANZ prints CET1 to one decimal place, so its
#: 60bps rise could be anywhere from 50 to 70. The figures are still exactly as printed, so
#: this takes off at most 0.04.
_ROUNDING_WEIGHT: Final = Decimal("0.12")
_ROUNDING_MAX: Final = Decimal("0.04")

#: How a driver's number came from its figures.
_CHART_CHECKED: Final = Decimal("0.03")
_CHART_UNCHECKED: Final = Decimal("0.15")
_CHAINED: Final = Decimal("0.04")
_PATHS_DISAGREE: Final = Decimal("0.15")
#: A total standing in for a split, or a sub-split that rebuilds its parent only roughly,
#: says what moved but not why. From the ceiling this leaves 0.83, so it can't read as High.
_NOT_A_CAUSE: Final = Decimal("0.15")

_RESIDUAL_PENALTY_PER_MULTIPLE: Final = Decimal("0.10")
_RESIDUAL_MAX_MULTIPLES: Final = Decimal(3)
_LARGE_RESIDUAL_SHARE: Final = Decimal("0.20")
_LARGE_RESIDUAL_PENALTY: Final = Decimal("0.10")

#: Above this share of the movement left unexplained, the answer is really "it moved and we
#: can't say why", and a well-read headline doesn't make that a confident answer.
_UNEXPLAINED_SHARE_FOR_LOW: Final = Decimal("0.80")
_UNEXPLAINED_CAP: Final = Decimal("0.45")
#: A nil movement with no attribution: the figure is known, the explanation isn't.
_FLAT_UNATTRIBUTED_CAP: Final = Decimal("0.75")


def level_for(score: Decimal) -> str:
    if score >= _HIGH_FLOOR:
        return "High"
    if score >= _MEDIUM_FLOOR:
        return "Medium"
    return "Low"


@dataclass(frozen=True, slots=True)
class ScoredConfidence:
    """A score, its level, and every rule that produced it."""

    score: Decimal
    level: str
    reasons: list[str] = field(default_factory=list)
    removed: bool = False
    """Set when the verifier found the claim unsupported. The caller drops it."""


def _finish(score: Decimal, reasons: list[str], removed: bool = False) -> ScoredConfidence:
    clamped = min(max(score, CLAMP_LOW), CLAMP_HIGH)
    if clamped != score:
        reasons.append(f"clamped to the range {CLAMP_LOW}-{CLAMP_HIGH}")
    return ScoredConfidence(
        score=clamped, level=level_for(clamped), reasons=reasons, removed=removed
    )


def _earned(evidence: FigureEvidence) -> tuple[Decimal, list[str]]:
    """What a set of printed figures is worth, from what stands behind each of them."""
    counted = max(evidence.figures, 1)
    score = _PRINTED
    if evidence.figures == 0:
        reasons = [f"{_PRINTED}: none of the figures behind it were found in the store"]
    else:
        figures = "1 figure" if counted == 1 else f"{counted} figures"
        reasons = [f"{_PRINTED} for {figures} as printed, then what stands behind them"]
    for count, weight, said in (
        (evidence.primary, _PRIMARY, "printed in a primary filing"),
        (evidence.corroborated, _CORROBORATED, "printed the same in another filing"),
        (evidence.confirmed, _CONFIRMED, f"confirmed, {evidence.confirmed_by}"),
    ):
        if count > 0:
            share = Decimal(min(count, counted)) / counted
            earned = (weight * share).quantize(Decimal("0.001"))
            score += earned
            reasons.append(f"{min(count, counted)} of {counted} {said}: +{earned}")
    if evidence.model_matched:
        score -= _MODEL_MATCHED
        reasons.append(f"a row was named by a model, not configured: -{_MODEL_MATCHED}")
    if evidence.restated:
        penalty = _RESTATED if evidence.restatement_footnoted else _RESTATED_UNFOOTNOTED
        score -= penalty
        reasons.append(
            "the prior figure is a restated comparative"
            + ("" if evidence.restatement_footnoted else " with no footnote saying why")
            + f": -{penalty}"
        )
    rounding = min(_ROUNDING_MAX, _ROUNDING_WEIGHT * evidence.rounding_share)
    rounding = rounding.quantize(Decimal("0.001"))
    if rounding > 0:
        score -= rounding
        share = min(evidence.rounding_share, Decimal(9)) * 100
        reasons.append(
            f"the printed figures' rounding is {share.quantize(Decimal('1'))}% of the "
            f"movement: -{rounding}"
        )
    return score, reasons


def _capped(score: Decimal, caps: list[tuple[Decimal, str]], reasons: list[str]) -> Decimal:
    """The strictest cap that binds. A cap is a ceiling, never a target, so one the score is
    already under does nothing and isn't listed."""
    for cap, why in sorted(caps):
        if score > cap:
            score = cap
            reasons.append(f"capped at {cap}: {why}")
    return score


def _problem_caps(features: ConfidenceFeatures, subject: str) -> list[tuple[Decimal, str]]:
    """The caps for a named problem, which apply to figures and drivers alike."""
    caps: list[tuple[Decimal, str]] = []
    if features.unexplained_conflict:
        caps.append(
            (Decimal("0.45"), f"UNEXPLAINED conflict flips the {subject}'s sign")
            if features.conflict_flips_sign
            else (Decimal("0.75"), f"UNEXPLAINED conflict on the {subject}")
        )
    if features.alternative_basis_used:
        caps.append((Decimal("0.75"), "headline basis unavailable, an alternative was used"))
    if features.definition_break:
        caps.append((Decimal("0.80"), "definition break between the two periods"))
    if features.input_failed_validation:
        caps.append((Decimal("0.50"), "a figure failed validation"))
    return caps


def score_figures(evidence: FigureEvidence, features: ConfidenceFeatures) -> ScoredConfidence:
    """How sure the two headline figures are, from what stands behind them, capped for any
    named problem."""
    score, reasons = _earned(evidence)
    score = _capped(score, _problem_caps(features, "figure"), reasons)
    return _finish(score, reasons)


def score_driver(evidence: DriverEvidence, features: ConfidenceFeatures) -> ScoredConfidence:
    """How sure one driver is: what stands behind the figures it rests on, less what was
    risked getting its number from them, capped for any named problem."""
    # An unsupported claim isn't downgraded; it's removed from the report. A claim the
    # verifier couldn't ground isn't weak, it's unfounded.
    if features.verifier == "unsupported":
        return ScoredConfidence(
            score=Decimal(0),
            level="Low",
            reasons=["verifier rated the claim unsupported; claim removed"],
            removed=True,
        )

    score, reasons = _earned(evidence.figures)

    if evidence.read_from == "chart":
        if evidence.checksum_ok:
            score -= _CHART_CHECKED
            reasons.append(f"read off a chart, which added up to its endpoints: -{_CHART_CHECKED}")
        else:
            score -= _CHART_UNCHECKED
            reasons.append(f"read off a chart with no checksum that held: -{_CHART_UNCHECKED}")
    if evidence.chained:
        score -= _CHAINED
        reasons.append(f"joined from two half-year walks at the balance between them: -{_CHAINED}")

    multiples = features.residual_multiples
    if multiples > 0:
        applied = min(_RESIDUAL_MAX_MULTIPLES, multiples)
        penalty = _RESIDUAL_PENALTY_PER_MULTIPLE * applied
        score -= penalty
        reasons.append(f"Path B residual {applied.normalize()}x tolerance: -{penalty.normalize()}")
    if features.paths_disagree:
        score -= _PATHS_DISAGREE
        reasons.append(
            f"Paths A and B disagree on this driver by more than 2x tolerance: -{_PATHS_DISAGREE}"
        )
    if features.inputs_aggregated:
        score -= _NOT_A_CAUSE
        reasons.append(
            "a total standing in for a split the bank doesn't give, so it says what moved but "
            f"not why: -{_NOT_A_CAUSE}"
        )
    if features.indicative_split:
        score -= _NOT_A_CAUSE
        reasons.append(
            "an indicative split: its identity rebuilds the parent only approximately: "
            f"-{_NOT_A_CAUSE}"
        )

    caps = _problem_caps(features, "driver")
    if features.uncited:
        caps.append((Decimal("0.60"), "no evidence card supports this figure"))
    if features.verifier == "partial":
        caps.append((Decimal("0.60"), "verifier rated the claim partial"))
    score = _capped(score, caps, reasons)
    return _finish(score, reasons)


def overall_confidence(
    headline: ScoredConfidence,
    driver_scores: list[tuple[Decimal, Decimal]],
    residual: Decimal,
    headline_change: Decimal,
) -> ScoredConfidence:
    """The lower of the headline score and the share-weighted driver average.

    Taking the lower means a well-read headline can't prop up poorly attributed drivers, and
    a confident attribution can't prop up a shaky headline. Another 0.10 comes off when the
    residual is more than 20% of the movement, because then most of the change is
    unexplained however good the explained part is.
    """
    reasons = [f"headline {headline.score}"]

    if driver_scores:
        weights = sum((abs(w) for _, w in driver_scores), Decimal(0))
        if weights > 0:
            weighted = sum((s * abs(w) for s, w in driver_scores), Decimal(0)) / weights
        else:
            weighted = sum((s for s, _ in driver_scores), Decimal(0)) / len(driver_scores)
        weighted = weighted.quantize(Decimal("0.001"))
        reasons.append(f"share-weighted driver mean {weighted}")
        score = min(headline.score, weighted)
    else:
        reasons.append("no drivers; overall follows the headline")
        score = headline.score

    # A flat figure with nothing attributed. The figure may be exactly right and the
    # explanation still unknown: ANZ's group margin was unchanged at 1.56% in 1H25 while its
    # "Banking" margin fell 12bps, so offsetting drivers were there to be found.
    if headline_change == 0 and not driver_scores and score > _FLAT_UNATTRIBUTED_CAP:
        score = _FLAT_UNATTRIBUTED_CAP
        reasons.append(
            f"the movement is nil but nothing was attributed, and a flat figure can hide "
            f"offsetting drivers: capped at {_FLAT_UNATTRIBUTED_CAP}"
        )

    if headline_change != 0:
        # With no drivers, nothing was explained, whatever the residual field says: a
        # declined decomposition carries a residual of zero simply because nothing set one.
        share = Decimal(1) if not driver_scores else abs(residual) / abs(headline_change)
        if share >= _UNEXPLAINED_SHARE_FOR_LOW:
            if score > _UNEXPLAINED_CAP:
                score = _UNEXPLAINED_CAP
            reasons.append(
                f"{share:.0%} of the movement is unexplained, so the answer is capped at "
                f"{_UNEXPLAINED_CAP} however well the headline reads"
            )
        elif share > _LARGE_RESIDUAL_SHARE:
            score -= _LARGE_RESIDUAL_PENALTY
            reasons.append(
                f"residual is {(share * 100).quantize(Decimal('1'))}% of the movement, "
                f"over 20%: -{_LARGE_RESIDUAL_PENALTY}"
            )

    return _finish(score, reasons)
