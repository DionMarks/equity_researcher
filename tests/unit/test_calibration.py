"""Calibration: whether a confidence score means what it says."""

from __future__ import annotations

from decimal import Decimal

import pytest

from bankresearch.evals.calibration import (
    MIN_CLAIMS_FOR_RECALIBRATION,
    Claim,
    calibrate,
    claims_from,
    risk_coverage,
)


def _claims(level: str, score: str, n: int, wrong_every: int, verified: bool = True):
    return [
        Claim(f"{level}{i}", Decimal(score), level, i % wrong_every != 0, verified)
        for i in range(n)
    ]


# --- nothing is measured on unverified gold --------------------------------


def test_an_unverified_claim_is_dropped_and_counted():
    """ "Calibration was not measured" and "calibration was measured and is fine" must
    never be confusable in a report."""
    kept, dropped = claims_from(_claims("High", "0.92", 5, 10, verified=False))
    assert kept == [] and dropped == 5


def test_with_no_verified_gold_no_measure_is_reported():
    report = calibrate(_claims("High", "0.92", 5, 10, verified=False))
    assert not report.measured
    assert report.ece is None and report.brier is None
    assert report.high_confidence_error_rate is None
    assert "no human-verified gold" in report.summary()


def test_the_dropped_count_survives_into_the_report():
    report = calibrate(
        [*_claims("High", "0.92", 3, 10), *_claims("Low", "0.4", 2, 10, verified=False)]
    )
    assert report.claims == 3
    assert report.dropped_unverified == 2


# --- the measures ----------------------------------------------------------


def test_a_perfectly_calibrated_set_has_no_expected_error():
    """Nine in ten High claims right, at a score of 0.9."""
    report = calibrate(_claims("High", "0.9", 20, 10))
    assert report.ece == pytest.approx(0.0, abs=1e-9)
    assert report.high_confidence_error_rate == pytest.approx(0.1)


def test_over_confidence_shows_as_a_negative_gap():
    """The direction matters: an under-confident system wastes a reader's time, an
    over-confident one misleads them."""
    report = calibrate(_claims("High", "0.95", 10, 2))  # claims 0.95, delivers 0.5
    bucket = next(b for b in report.buckets if b.level == "High")
    assert bucket.gap < 0
    assert report.ece == pytest.approx(0.45)


def test_brier_punishes_a_confident_wrong_answer_more_than_an_unconfident_one():
    """The asymmetry ECE does not have, and the one that matters here."""
    confident_wrong = calibrate([Claim("a", Decimal("0.95"), "High", False, True)])
    unconfident_wrong = calibrate([Claim("b", Decimal("0.40"), "Low", False, True)])
    assert confident_wrong.brier > unconfident_wrong.brier


def test_the_high_confidence_error_rate_is_undefined_rather_than_zero_with_no_high_claims():
    """Reporting 0.0 would say the headline safety number is perfect when it was never
    measured."""
    report = calibrate(_claims("Low", "0.4", 5, 10))
    assert report.high_confidence_error_rate is None
    assert any("undefined" in note for note in report.notes)
    # The summary says so rather than formatting a missing number. It raised a TypeError,
    # which only showed when five live runs' drivers were all Medium.
    assert "no High claim" in report.summary()


def test_each_level_is_weighted_by_how_many_claims_fall_in_it():
    report = calibrate([*_claims("High", "0.9", 90, 10), *_claims("Low", "0.5", 10, 1)])
    high = next(b for b in report.buckets if b.level == "High")
    low = next(b for b in report.buckets if b.level == "Low")
    assert high.claims == 90 and low.claims == 10
    # The Low bucket is badly wrong but small, so it contributes a tenth of its gap.
    assert report.ece == pytest.approx(0.9 * 0.0 + 0.1 * 0.5, abs=1e-9)


# --- risk and coverage -----------------------------------------------------


def test_accuracy_rises_as_coverage_falls_when_the_scores_rank_anything():
    """A useful system's accuracy rises as it answers less."""
    good = [
        Claim("a", Decimal("0.95"), "High", True, True),
        Claim("b", Decimal("0.90"), "High", True, True),
        Claim("c", Decimal("0.50"), "Low", False, True),
        Claim("d", Decimal("0.40"), "Low", False, True),
    ]
    curve = risk_coverage(good)
    assert curve[0][1] == 1.0, "the highest-scored claim is right"
    assert curve[-1][1] == 0.5, "answering everything gives the base rate"
    assert curve[0][1] > curve[-1][1]


def test_a_flat_curve_is_what_a_good_ece_can_hide():
    """A system that gives every claim the same well-calibrated score is perfectly
    calibrated and no help at all in deciding what to check.
    """
    uninformative = [Claim(f"c{i}", Decimal("0.5"), "Low", i % 2 == 0, True) for i in range(20)]
    report = calibrate(uninformative)
    assert report.ece == pytest.approx(0.0, abs=1e-9), "well calibrated"
    coverages = [accuracy for _, accuracy in report.curve[5:]]
    assert max(coverages) - min(coverages) < 0.2, "and carries no ranking information"


def test_an_empty_set_has_no_curve():
    assert risk_coverage([]) == []


# --- recalibration ---------------------------------------------------------


def test_no_isotonic_map_is_fitted_below_the_plans_sample_size():
    report = calibrate(_claims("High", "0.9", 20, 10))
    assert not report.recalibration_fitted
    assert any(str(MIN_CLAIMS_FOR_RECALIBRATION) in note for note in report.notes)
    assert any("noise" in note for note in report.notes)


def test_at_the_plans_sample_size_a_map_may_be_fitted():
    report = calibrate(_claims("High", "0.9", MIN_CLAIMS_FOR_RECALIBRATION, 10))
    assert report.recalibration_fitted


def test_the_levels_match_the_rubric():
    from bankresearch.evals.calibration import LEVELS

    report = calibrate(
        [
            *_claims("High", "0.92", 2, 10),
            *_claims("Medium", "0.75", 2, 10),
            *_claims("Low", "0.4", 2, 10),
        ]
    )
    assert [bucket.level for bucket in report.buckets] == list(LEVELS)


# --- the adapter from the harness's own score type -------------------------


def test_a_harness_case_score_is_adapted_rather_than_duck_typed():
    """``CaseScore`` names its fields ``overall_confidence`` and ``confidence_level``, so a
    ``getattr`` that missed would have reported "not measured" forever, with gold present
    and no sign why.
    """
    from bankresearch.evals.scorers import CaseScore

    scored = [
        CaseScore(
            case_id=f"c{i}",
            status="OK",
            expected_status=("OK",),
            status_correct=True,
            overall_confidence=0.9,
            confidence_level="High",
            headline_exact=i % 10 != 0,
        )
        for i in range(20)
    ]
    report = calibrate(scored)
    assert report.measured
    assert report.claims == 20
    assert report.high_confidence_error_rate == pytest.approx(0.1)


def test_a_case_score_without_verified_gold_is_dropped():
    """``headline_exact`` is None unless the gold is verified, which is the gate, so the
    correctness signal and the verification gate can't drift apart.
    """
    from bankresearch.evals.scorers import CaseScore

    scored = [
        CaseScore(
            case_id="c",
            status="OK",
            expected_status=("OK",),
            status_correct=True,
            overall_confidence=0.9,
            confidence_level="High",
            headline_exact=None,
        )
    ]
    report = calibrate(scored)
    assert not report.measured and report.dropped_unverified == 1


# --- calibration scores the headline, not the composite -------------------


def test_calibration_scores_the_headline_confidence_not_the_overall():
    """``headline_exact`` asks "is this figure right"; the overall score answers "how much
    of this movement is explained". They're different questions.

    Calibrating one against the other scores a composite against an outcome that measures
    only one part of it, and it showed: tightening the attribution caps, a real improvement
    to the memo with every figure unchanged, moved ECE from 0.14 to 0.32. Scoring the
    headline's own confidence instead brought it to 0.09.
    """
    from bankresearch.evals.calibration import claims_from
    from bankresearch.evals.scorers import CaseScore

    score = CaseScore(
        case_id="x",
        status="OK",
        expected_status=("OK",),
        status_correct=True,
        headline_exact=True,
        # The figure is well evidenced; the attribution is not.
        headline_confidence=0.92,
        headline_confidence_level="High",
        overall_confidence=0.45,
        confidence_level="Low",
    )
    claims, dropped = claims_from([score])
    assert dropped == 0
    assert len(claims) == 1
    # Stored as a Decimal; `Decimal("0.92") == 0.92` is false, 0.92 having no exact
    # binary form. Comparing through str is the point of keeping it a Decimal.
    assert claims[0].score == Decimal("0.92"), "the figure's own confidence is scored"
    assert claims[0].level == "High"


def test_it_falls_back_to_the_overall_where_there_is_no_headline_score():
    """An older run, or one that abstained before scoring a figure."""
    from bankresearch.evals.calibration import claims_from
    from bankresearch.evals.scorers import CaseScore

    claims, _ = claims_from(
        [
            CaseScore(
                case_id="x",
                status="OK",
                expected_status=("OK",),
                status_correct=True,
                headline_exact=True,
                overall_confidence=0.82,
                confidence_level="Medium",
            )
        ]
    )
    assert claims[0].score == Decimal("0.82")


def test_the_headline_confidence_reaches_the_result():
    """It was computed on every run and discarded; only the composite was returned."""
    from bankresearch.output.schema import ResearchResult

    assert "headline_confidence" in ResearchResult.model_fields
