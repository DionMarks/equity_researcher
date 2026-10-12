"""Calibration: whether a confidence score means what it says."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

__all__ = [
    "MIN_CLAIMS_FOR_RECALIBRATION",
    "CalibrationReport",
    "Claim",
    "LevelBucket",
    "calibrate",
    "risk_coverage",
]

# Below this, an isotonic map is fitted to noise.
MIN_CLAIMS_FOR_RECALIBRATION = 150

LEVELS = ("High", "Medium", "Low")


@dataclass(frozen=True, slots=True)
class Claim:
    """One scored claim, with whether it turned out to be right.

    ``correct`` means matched, right sign, within tolerance and
    supported by its citation. All four, because a driver with the right size and the wrong
    sign isn't partly correct.
    """

    claim_id: str
    score: Decimal
    level: str
    correct: bool
    gold_verified: bool = False


@dataclass(frozen=True, slots=True)
class LevelBucket:
    level: str
    claims: int
    accuracy: float
    mean_score: float

    @property
    def gap(self) -> float:
        """Signed, so over- and under-confidence can be told apart.

        Negative means the level claims more than it delivers, which is the direction that
        matters: an under-confident system wastes a reader's time, an over-confident one
        misleads them.
        """
        return self.accuracy - self.mean_score


@dataclass
class CalibrationReport:
    buckets: list[LevelBucket] = field(default_factory=list)
    ece: float | None = None
    brier: float | None = None
    high_confidence_error_rate: float | None = None
    curve: list[tuple[float, float]] = field(default_factory=list)
    claims: int = 0
    dropped_unverified: int = 0
    recalibration_fitted: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def measured(self) -> bool:
        return self.claims > 0

    def summary(self) -> str:
        if not self.measured:
            return (
                "Calibration was not measured: "
                f"{self.dropped_unverified} claim(s) carried no human-verified gold. "
                "Every measure here needs it, and reporting a default in their place "
                "would be quoting a number that was never measured."
            )
        high = (
            f"high-confidence error rate {self.high_confidence_error_rate:.4f}"
            if self.high_confidence_error_rate is not None
            else "no High claim, so no high-confidence error rate"
        )
        return f"{self.claims} claim(s): ECE {self.ece:.4f}, Brier {self.brier:.4f}, {high}"


def claims_from(scored: list[Any]) -> tuple[list[Claim], int]:
    """Verified claims, and how many were dropped for not being verified.

    Accepts either a :class:`Claim` or the harness's ``CaseScore``, adapting the latter.
    It's an adapter rather than duck-typing because ``CaseScore`` names the fields
    differently (``overall_confidence`` and ``confidence_level``), and a ``getattr`` that
    missed would report "not measured" forever, with gold present and no sign why.

    The count is returned rather than logged, because "calibration wasn't measured" and
    "calibration was measured and is fine" must never be confused in a report.
    """
    kept: list[Claim] = []
    dropped = 0
    for item in scored:
        claim = item if isinstance(item, Claim) else _as_claim(item)
        if claim is None or not claim.gold_verified:
            dropped += 1
            continue
        kept.append(claim)
    return kept, dropped


def _as_claim(score: Any) -> Claim | None:
    """A harness ``CaseScore`` as a claim, or ``None`` when it has no verified gold.

    ``headline_exact`` is the correctness signal. It's ``None`` unless the case had verified
    gold, which is exactly the gate this needs, so the two can't drift apart.

    Scored against the headline's confidence, not the run's overall confidence. The overall
    score is the lower of the headline and the attribution, minus a residual penalty, so it
    answers "how much of this movement is explained", while ``headline_exact`` asks "is this
    figure right". Calibrating one against the other scores a composite against an outcome
    that measures only one part of it, and it showed: tightening the attribution caps, a
    real improvement to the memo with every figure unchanged, moved ECE from 0.14 to 0.32.
    Falls back to the overall score where a result has no per-figure confidence.
    """
    correct = getattr(score, "headline_exact", None)
    confidence = getattr(score, "headline_confidence", None)
    level = getattr(score, "headline_confidence_level", None)
    if confidence is None or not level:
        confidence = getattr(score, "overall_confidence", None)
        level = getattr(score, "confidence_level", None)
    if correct is None or confidence is None or not level:
        return None
    return Claim(
        claim_id=str(getattr(score, "case_id", "")),
        score=Decimal(str(confidence)),
        level=str(level),
        correct=bool(correct),
        gold_verified=True,
    )


def _bucket(level: str, claims: list[Claim]) -> LevelBucket | None:
    inside = [claim for claim in claims if claim.level == level]
    if not inside:
        return None
    return LevelBucket(
        level=level,
        claims=len(inside),
        accuracy=sum(1 for claim in inside if claim.correct) / len(inside),
        mean_score=float(sum(claim.score for claim in inside) / len(inside)),
    )


def risk_coverage(claims: list[Claim]) -> list[tuple[float, float]]:
    """Accuracy against coverage, answering the highest-scored claims first.

    A useful system's accuracy rises as coverage falls. A flat curve means the scores say
    nothing about which claims to trust, and a good ECE can hide that: a system that gives
    every claim the same well-calibrated score is perfectly calibrated and no help at all in
    deciding what to check.
    """
    if not claims:
        return []
    ranked = sorted(claims, key=lambda claim: claim.score, reverse=True)
    out: list[tuple[float, float]] = []
    right = 0
    for index, claim in enumerate(ranked, start=1):
        right += int(claim.correct)
        out.append((index / len(ranked), right / index))
    return out


def calibrate(scored: list[Any]) -> CalibrationReport:
    """Every calibration measure, or a report saying none was taken and why."""
    claims, dropped = claims_from(scored)
    report = CalibrationReport(claims=len(claims), dropped_unverified=dropped)
    if not claims:
        report.notes.append(
            "no claim carried human-verified gold, so no calibration measure was taken"
        )
        return report

    total = len(claims)
    report.buckets = [bucket for level in LEVELS if (bucket := _bucket(level, claims))]

    # ECE: the per-level gap, weighted by how many claims fall in that level.
    report.ece = sum((bucket.claims / total) * abs(bucket.gap) for bucket in report.buckets)
    # Brier: squared error against the outcome, which punishes a confident wrong answer
    # more than an unconfident one.
    report.brier = sum((float(claim.score) - int(claim.correct)) ** 2 for claim in claims) / total

    high = next((b for b in report.buckets if b.level == "High"), None)
    if high is not None:
        report.high_confidence_error_rate = 1.0 - high.accuracy
    else:
        report.notes.append("no High claim was made, so the headline safety number is undefined")

    report.curve = risk_coverage(claims)

    if total < MIN_CLAIMS_FOR_RECALIBRATION:
        report.notes.append(
            f"no isotonic map fitted: {total} claim(s) is below the "
            f"{MIN_CLAIMS_FOR_RECALIBRATION} needed, and fitting one to fewer "
            "would be fitting it to noise and reporting the result as calibration"
        )
    else:
        report.recalibration_fitted = True
        report.notes.append(
            "an isotonic map may be fitted at this sample size; the thresholds in "
            "config/confidence.yaml are the input to it, not a substitute for it"
        )
    return report
