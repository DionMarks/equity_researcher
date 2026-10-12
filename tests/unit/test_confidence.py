"""The rubric: every figure and driver earns its score from what stands behind it, and named
problems take from it or cap it (ADR-069 and ADR-072)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from bankresearch.confidence.features import ConfidenceFeatures, DriverEvidence, FigureEvidence
from bankresearch.confidence.rubric import (
    LEVEL_TARGETS,
    ScoredConfidence,
    level_for,
    overall_confidence,
    score_driver,
    score_figures,
)

NONE = ConfidenceFeatures()


def problems(**named: object) -> ConfidenceFeatures:
    return ConfidenceFeatures(**named)  # type: ignore[arg-type]


def behind(**counted: object) -> DriverEvidence:
    """A driver resting on two figures, with whatever else the test says about them."""
    return DriverEvidence(figures=FigureEvidence(**counted))  # type: ignore[arg-type]


# --- what stands behind the figures ---------------------------------------


def test_a_figure_with_nothing_behind_it_but_its_page_scores_the_floor():
    assert score_figures(FigureEvidence(primary=0), NONE).score == Decimal("0.86")


def test_everything_behind_the_figures_reaches_the_ceiling_and_nothing_alone_does():
    full = FigureEvidence(primary=2, corroborated=2, confirmed=2)
    assert score_figures(full, NONE).score == Decimal("0.98")
    for alone in (
        FigureEvidence(primary=2),
        FigureEvidence(primary=0, corroborated=2),
        FigureEvidence(primary=0, confirmed=2),
    ):
        assert score_figures(alone, NONE).score == Decimal("0.90")


def test_evidence_is_earned_in_proportion_to_the_figures_it_is_true_of():
    """Four figures behind a driver, three in a primary filing and one rebuilt by an
    identity, earn three quarters and one quarter of each step."""
    evidence = FigureEvidence(figures=4, primary=3, confirmed=1)
    assert score_figures(evidence, NONE).score == Decimal("0.86") + Decimal("0.03") + Decimal(
        "0.01"
    )


def test_more_behind_the_figures_scores_higher_and_less_scores_lower():
    """A fixed table of bases gave every clean headline 0.92 and every computed driver the
    same. Each thing counted has to move the score the way it says."""
    base = score_figures(FigureEvidence(), NONE).score
    assert base == Decimal("0.90")
    up = [FigureEvidence(corroborated=1), FigureEvidence(confirmed=1)]
    down = [
        FigureEvidence(model_matched=True),
        FigureEvidence(restated=True),
        FigureEvidence(rounding_share=Decimal("0.1")),
        FigureEvidence(primary=1),
    ]
    assert all(score_figures(e, NONE).score > base for e in up)
    assert all(score_figures(e, NONE).score < base for e in down)
    footnoted = score_figures(FigureEvidence(restated=True), NONE).score
    unfootnoted = score_figures(
        FigureEvidence(restated=True, restatement_footnoted=False), NONE
    ).score
    assert unfootnoted < footnoted < base


def test_figures_printed_in_a_primary_filing_and_rebuilt_by_an_identity_are_high():
    result = score_figures(FigureEvidence(confirmed=2), NONE)
    assert result.level == "High" and result.score == Decimal("0.94")


def test_a_figure_only_in_a_presentation_on_a_model_named_row_is_not_high():
    result = score_figures(
        FigureEvidence(primary=0, model_matched=True, rounding_share=Decimal("0.5")), NONE
    )
    assert result.score == Decimal("0.76") and result.level == "Medium"


def test_printing_can_take_off_at_most_a_little_because_the_figures_are_still_exact():
    flat = score_figures(FigureEvidence(rounding_share=Decimal(9)), NONE)
    assert flat.score == Decimal("0.86") and flat.level == "High"


def test_the_caps_for_a_named_problem_still_apply_to_the_figures():
    strong = FigureEvidence(corroborated=2, confirmed=2)
    assert score_figures(strong, problems(unexplained_conflict=True)).score == Decimal("0.75")
    assert score_figures(strong, problems(definition_break=True)).score == Decimal("0.80")
    assert score_figures(strong, problems(input_failed_validation=True)).score == Decimal("0.50")


def test_every_rule_that_moved_the_figures_is_said():
    result = score_figures(
        FigureEvidence(corroborated=1, confirmed=2, restated=True, rounding_share=Decimal("0.2")),
        NONE,
    )
    said = " ".join(result.reasons)
    for words in ("primary filing", "printed the same", "identity", "restated", "rounding"):
        assert words in said


# --- a driver: what it rests on, and how its number came from it ----------


def test_two_drivers_with_different_evidence_score_differently():
    """Under the fixed table every computed driver scored 0.92 and every capped one 0.75,
    whatever its inputs. Now a driver is as sure as the figures behind it."""
    weak = score_driver(behind(primary=1), NONE)
    plain = score_driver(behind(), NONE)
    strong = score_driver(behind(corroborated=2, confirmed=2), NONE)
    assert weak.score < plain.score < strong.score
    assert (weak.score, plain.score, strong.score) == (
        Decimal("0.88"),
        Decimal("0.90"),
        Decimal("0.98"),
    )


def test_a_walk_step_is_confirmed_by_its_walk_adding_up():
    step = DriverEvidence(
        figures=FigureEvidence(
            figures=1, primary=1, confirmed=1, confirmed_by="the walk adds up to the movement"
        ),
        read_from="table",
    )
    result = score_driver(step, NONE)
    assert result.score == Decimal("0.94")
    assert any("the walk adds up" in reason for reason in result.reasons)


def test_a_step_small_against_its_printing_is_less_sure_of_its_size():
    """A 1bp step printed to whole basis points could be anywhere from 0.5 to 1.5."""
    big = score_driver(behind(figures=1, primary=1, rounding_share=Decimal("0.02")), NONE)
    small = score_driver(behind(figures=1, primary=1, rounding_share=Decimal("0.5")), NONE)
    assert small.score == Decimal("0.86") < big.score


def test_a_chart_reading_that_passed_its_checksum_costs_three_points():
    result = score_driver(DriverEvidence(read_from="chart", checksum_ok=True), NONE)
    assert result.score == Decimal("0.87")
    assert any("chart" in r for r in result.reasons)


def test_a_chart_reading_without_a_checksum_costs_fifteen():
    for checked in (False, None):
        reading = DriverEvidence(read_from="chart", checksum_ok=checked)
        assert score_driver(reading, NONE).score == Decimal("0.75")


def test_a_walk_joined_from_two_halves_costs_four_points():
    assert score_driver(DriverEvidence(chained=True), NONE).score == Decimal("0.86")


def test_a_path_b_residual_beyond_tolerance_scales_with_how_far_beyond():
    over = problems(path_b_residual=Decimal("2"), path_b_tolerance=Decimal("1"))
    assert score_driver(DriverEvidence(), over).score == Decimal("0.70")  # 0.90 - 0.10*2


def test_the_residual_penalty_is_capped_at_three_multiples():
    far = problems(path_b_residual=Decimal("50"), path_b_tolerance=Decimal("1"))
    assert score_driver(DriverEvidence(), far).score == Decimal("0.60")  # 0.90 - 0.10*3


def test_a_residual_within_tolerance_costs_nothing():
    within = problems(path_b_residual=Decimal("0.4"), path_b_tolerance=Decimal("1"))
    assert score_driver(DriverEvidence(), within).score == Decimal("0.90")


def test_paths_that_disagree_on_a_driver_cost_fifteen_points():
    assert score_driver(DriverEvidence(), problems(paths_disagree=True)).score == Decimal("0.75")


def test_an_unfootnoted_restatement_costs_five_points():
    restated = behind(restated=True, restatement_footnoted=False)
    assert score_driver(restated, NONE).score == Decimal("0.85")


def test_a_driver_computed_from_an_aggregate_is_not_an_attribution():
    """Westpac discloses its CET1 capital movement as one number, so "CET1 capital
    +42.5bps" is a bucket rather than a cause. However much stands behind its figures, it
    can't read as High."""
    plain = score_driver(behind(), problems(inputs_aggregated=True))
    strongest = score_driver(behind(corroborated=2, confirmed=2), problems(inputs_aggregated=True))
    assert plain.score == Decimal("0.75")
    assert strongest.score == Decimal("0.83") and strongest.level == "Medium"


def test_a_sub_split_called_indicative_is_not_high():
    """CBA's net interest income split misses the reported figure by up to A$49m, and the
    decomposition says so."""
    split = score_driver(DriverEvidence(), problems(indicative_split=True))
    assert split.score == Decimal("0.75") and split.level == "Medium"
    assert any("indicative split" in reason for reason in split.reasons)


# --- caps -------------------------------------------------------------------


def test_an_unexplained_conflict_caps_at_medium():
    strong = behind(corroborated=2, confirmed=2)
    assert score_driver(strong, problems(unexplained_conflict=True)).score == Decimal("0.75")


def test_a_sign_flipping_conflict_caps_harder():
    flipped = problems(unexplained_conflict=True, conflict_flips_sign=True)
    unaffected = score_driver(DriverEvidence(), NONE)
    assert score_driver(DriverEvidence(), flipped).score == Decimal("0.45") < unaffected.score


def test_using_an_alternative_basis_caps_at_medium():
    assert score_driver(DriverEvidence(), problems(alternative_basis_used=True)).score == Decimal(
        "0.75"
    )


def test_a_definition_break_caps_at_point_eight():
    assert score_driver(DriverEvidence(), problems(definition_break=True)).score == Decimal("0.80")


def test_a_partially_supported_claim_caps_at_point_six():
    assert score_driver(DriverEvidence(), problems(verifier="partial")).score == Decimal("0.60")


def test_an_unsupported_claim_is_removed_not_downgraded():
    assert score_driver(DriverEvidence(), problems(verifier="unsupported")).removed is True


def test_a_failed_input_validation_caps_at_point_five():
    failed = problems(input_failed_validation=True)
    assert score_driver(DriverEvidence(), failed).score == Decimal("0.50")


def test_an_uncited_driver_is_not_high_confidence():
    """ANZ's walk has a row mapping to no canonical driver, so it becomes "other" and
    carries 3bp of a 3bp movement with nothing attached."""
    uncited = score_driver(DriverEvidence(), problems(uncited=True))
    assert uncited.score <= Decimal("0.60") < score_driver(DriverEvidence(), NONE).score
    assert any("no evidence card" in r for r in uncited.reasons)


def test_caps_do_not_raise_a_low_score():
    """A cap is a ceiling, not a target."""
    unchecked = DriverEvidence(read_from="chart", checksum_ok=False)
    assert score_driver(unchecked, problems(definition_break=True)).score == Decimal("0.75")


def test_the_strictest_cap_wins():
    both = problems(definition_break=True, input_failed_validation=True)
    assert score_driver(DriverEvidence(), both).score == Decimal("0.50")


# --- clamping and reasons ---------------------------------------------------


def test_the_score_never_reaches_certainty():
    assert score_driver(behind(corroborated=9, confirmed=9), NONE).score <= Decimal("0.98")


def test_the_score_never_reaches_zero():
    worst = DriverEvidence(
        figures=FigureEvidence(
            primary=0, model_matched=True, restated=True, restatement_footnoted=False
        ),
        read_from="chart",
        checksum_ok=False,
        chained=True,
    )
    everything = problems(
        paths_disagree=True,
        inputs_aggregated=True,
        indicative_split=True,
        path_b_residual=Decimal("99"),
        path_b_tolerance=Decimal("1"),
    )
    assert score_driver(worst, everything).score == Decimal("0.05")


def test_every_applied_rule_is_named_in_the_reasons():
    # 0.90 - 0.05 - 0.03 = 0.82, then capped to 0.80: the floor, the primary filing, the
    # restatement, the chart and the cap.
    result = score_driver(
        DriverEvidence(
            figures=FigureEvidence(restated=True, restatement_footnoted=False),
            read_from="chart",
            checksum_ok=True,
        ),
        problems(definition_break=True),
    )
    assert result.score == Decimal("0.80")
    assert len(result.reasons) == 5
    said = " ".join(result.reasons)
    for words in ("as printed", "primary filing", "restated", "chart", "capped at 0.80"):
        assert words in said


def test_a_cap_that_does_not_bind_is_not_listed():
    """A cap is a ceiling. If penalties already took the score below it, it did nothing,
    and saying otherwise would overstate what was checked."""
    result = score_driver(
        DriverEvidence(read_from="chart", checksum_ok=True),
        problems(paths_disagree=True, definition_break=True),
    )
    assert result.score == Decimal("0.72")
    assert not any("capped" in r for r in result.reasons)


# --- levels -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "level"),
    [
        ("0.98", "High"),
        ("0.85", "High"),
        ("0.8499", "Medium"),
        ("0.60", "Medium"),
        ("0.5999", "Low"),
        ("0.05", "Low"),
    ],
)
def test_level_thresholds(score, level):
    assert level_for(Decimal(score)) == level


def test_level_targets():
    assert LEVEL_TARGETS["High"] == Decimal("0.90")
    assert LEVEL_TARGETS["Medium"] == Decimal("0.70")
    assert LEVEL_TARGETS["Low"] is None


# --- the answer as a whole --------------------------------------------------


def _figures() -> ScoredConfidence:
    """A pair of figures scored 0.92: printed in a primary filing, one of them twice."""
    return score_figures(FigureEvidence(corroborated=1), NONE)


def test_overall_is_the_lower_of_headline_and_weighted_driver_mean():
    overall = overall_confidence(
        headline=_figures(),
        driver_scores=[(Decimal("0.90"), Decimal("0.5")), (Decimal("0.60"), Decimal("0.5"))],
        residual=Decimal("0"),
        headline_change=Decimal("40"),
    )
    assert overall.score == Decimal("0.75")  # the weighted mean, below the headline


def test_overall_is_penalised_when_the_residual_is_large():
    overall = overall_confidence(
        headline=_figures(),
        driver_scores=[(Decimal("0.90"), Decimal("1"))],
        residual=Decimal("10"),
        headline_change=Decimal("40"),  # residual is 25% of the change
    )
    assert overall.score == Decimal("0.80")
    assert any("residual" in r for r in overall.reasons)


def test_a_small_residual_is_not_penalised():
    overall = overall_confidence(
        headline=_figures(),
        driver_scores=[(Decimal("0.90"), Decimal("1"))],
        residual=Decimal("4"),
        headline_change=Decimal("40"),  # 10%
    )
    assert overall.score == Decimal("0.90")


def test_no_drivers_means_nothing_explained_whatever_the_residual_says():
    """A residual of zero with no drivers doesn't mean the movement reconciled; it means
    nothing set a residual. Westpac's expense ratio, whose identity is refused across bases,
    came back with no drivers and a residual of zero and scored High, while ANZ's margin, with
    no drivers and the whole movement as residual, scored Low."""
    headline = _figures()
    declined = overall_confidence(
        headline=headline,
        driver_scores=[],
        residual=Decimal("0"),  # nothing set one
        headline_change=Decimal("40"),
    )
    attributed_none = overall_confidence(
        headline=headline,
        driver_scores=[],
        residual=Decimal("40"),  # the whole movement
        headline_change=Decimal("40"),
    )
    assert declined.score == attributed_none.score, (
        "two ways of explaining nothing must score the same"
    )
    assert declined.score <= Decimal("0.45")
    assert declined.score < headline.score
    assert any("unexplained" in reason for reason in declined.reasons)


def test_a_reconciled_walk_still_earns_the_headline_score():
    """The cap must not punish a decomposition that genuinely ties."""
    headline = _figures()
    overall = overall_confidence(
        headline=headline,
        driver_scores=[(Decimal("0.92"), Decimal("40"))],
        residual=Decimal("0"),
        headline_change=Decimal("40"),
    )
    assert overall.score == min(headline.score, Decimal("0.92"))


def test_a_movement_that_is_mostly_unexplained_cannot_be_confident():
    """The headline was read well, but there was no explanation, and the score a reader acts
    on is the one next to the explanation."""
    headline = ScoredConfidence(level="High", score=Decimal("0.92"), reasons=[])
    nothing_explained = overall_confidence(
        headline=headline,
        driver_scores=[],
        residual=Decimal("-3"),
        headline_change=Decimal("-3"),
    )
    assert nothing_explained.score <= Decimal("0.45")
    assert nothing_explained.level == "Low"
    assert any("unexplained" in r for r in nothing_explained.reasons)

    # A third unexplained is a real caveat, not a disqualification.
    mostly_explained = overall_confidence(
        headline=headline,
        driver_scores=[(Decimal("0.90"), Decimal("2"))],
        residual=Decimal("-1"),
        headline_change=Decimal("-3"),
    )
    assert Decimal("0.45") < mostly_explained.score < headline.score


def test_a_flat_figure_with_nothing_attributed_is_not_high():
    """ANZ's group margin was flat at 1.56% while its Banking margin fell 12bps: the figure
    was right and the explanation unknown, and the answer came out High."""
    headline = _figures()
    flat = overall_confidence(headline, [], Decimal(0), Decimal(0))
    assert headline.level == "High"
    assert flat.score == Decimal("0.75") and flat.level == "Medium"
    explained = overall_confidence(
        headline, [(Decimal("0.92"), Decimal("1"))], Decimal(0), Decimal(0)
    )
    assert explained.level == "High", "a flat figure whose drivers were found keeps its score"


# --- a conflict caps only the drivers that depend on it ---------------------


def test_a_conflict_on_an_unrelated_component_does_not_cap_a_driver():
    """A Westpac capital run came out Low because its investor pack and results document
    disagree on the sign of the impairment charge, a figure no part of a CET1 ratio uses."""
    from datetime import date

    from bankresearch.agent.orchestrator import Agent
    from bankresearch.memory.db import connect
    from bankresearch.memory.fact_store import FactStore
    from bankresearch.settings import get_settings

    settings = get_settings()
    if not settings.db_path.exists():
        pytest.skip("no ingested corpus: run `make ingest` first")

    with connect(settings.db_path) as connection:
        result = (
            Agent(store=FactStore(connection), settings=settings, persist=False)
            .run("WBC", "1H26", "cet1_ratio", as_of=date(2026, 5, 31))
            .result
        )

    conflicted = {conflict.component for conflict in result.data_conflicts}
    assert "credit_impairment_charge" not in conflicted and "headline_earnings" not in conflicted
    assert "cet1_capital" not in conflicted and "rwa" not in conflicted, (
        "the premise: nothing a CET1 ratio is computed from is in conflict"
    )
    for driver in result.top_level_drivers:
        assert driver.confidence.score > Decimal("0.45"), (
            f"{driver.driver_id} was capped by a conflict it does not depend on: "
            f"{driver.confidence.reasons}"
        )
