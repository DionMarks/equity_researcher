"""Source conflicts are classified before they're resolved.

The tests cover one fixture per class, plus the ordering: a rounding difference must
never fall through to UNEXPLAINED.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from bankresearch.domain.models import Basis, CandidateValue, SourceRef
from bankresearch.reconcile.conflicts import (
    CLASSIFICATION_ORDER,
    classify,
    resolve,
)


def ref(
    doc_id: str,
    doc_type: str = "RESULTS_DOCUMENT",
    page: int = 10,
    published: date = date(2026, 5, 1),
) -> SourceRef:
    return SourceRef(doc_id=doc_id, doc_type=doc_type, page=page, publish_date=published)


def cand(
    value: str,
    unit: str = "pct",
    *,
    doc_id: str = "d1",
    doc_type: str = "RESULTS_DOCUMENT",
    basis: Basis | None = None,
    published: date = date(2026, 5, 1),
    restated: bool = False,
    footnoted: bool = False,
) -> CandidateValue:
    return CandidateValue(
        value=Decimal(value),
        unit=unit,
        basis=basis or Basis(),
        source=ref(doc_id, doc_type, published=published),
        vintage=published,
        is_restated=restated,
        footnoted=footnoted,
    )


# --- one fixture per class ------------------------------------------------


def test_the_same_quantity_in_different_units_is_not_a_conflict():
    conflict = classify("nim.headline", [cand("1.95", "pct"), cand("195", "bps", doc_id="d2")])
    assert conflict.classification == "UNIT"
    assert conflict.impact == ""


def test_equal_after_rounding_is_rounding():
    conflict = classify("cet1.headline", [cand("12.39"), cand("12.4", doc_id="d2")])
    assert conflict.classification == "ROUNDING"


def test_rounding_takes_the_more_precise_value():
    conflict = classify("cet1.headline", [cand("12.4", doc_id="d2"), cand("12.39")])
    assert conflict.chosen.value == Decimal("12.39")


def test_differing_basis_tags_are_a_basis_conflict():
    conflict = classify(
        "earn.headline",
        [
            cand("5016", "aud_m", basis=Basis(earnings_basis="cash")),
            cand("4800", "aud_m", doc_id="d2", basis=Basis(earnings_basis="statutory")),
        ],
    )
    assert conflict.classification == "BASIS"


def test_basis_conflict_takes_the_headline_basis():
    conflict = classify(
        "earn.headline",
        [
            cand("4800", "aud_m", doc_id="d2", basis=Basis(earnings_basis="statutory")),
            cand("5016", "aud_m", basis=Basis(earnings_basis="cash")),
        ],
        headline_basis=Basis(earnings_basis="cash"),
    )
    assert conflict.chosen.value == Decimal("5016")


def test_level_one_versus_level_two_is_a_scope_conflict():
    conflict = classify(
        "cet1.headline",
        [
            cand("12.39", basis=Basis(scope="level2")),
            cand("11.20", doc_id="d2", basis=Basis(scope="level1")),
        ],
    )
    assert conflict.classification == "SCOPE"


def test_scope_conflict_takes_the_headline_scope():
    conflict = classify(
        "cet1.headline",
        [
            cand("11.20", doc_id="d2", basis=Basis(scope="level1")),
            cand("12.39", basis=Basis(scope="level2")),
        ],
        headline_basis=Basis(scope="level2"),
    )
    assert conflict.chosen.value == Decimal("12.39")


def test_a_pro_forma_flag_is_its_own_class():
    conflict = classify(
        "cet1.headline",
        [
            cand("12.39", basis=Basis(scope="level2")),
            cand("13.10", doc_id="d2", basis=Basis(scope="level2", pro_forma=True)),
        ],
    )
    assert conflict.classification == "PRO_FORMA"
    assert conflict.chosen.value == Decimal("12.39")  # reported, not pro forma


def test_a_later_vintage_of_the_same_period_is_a_restatement():
    conflict = classify(
        "earn.headline",
        [
            cand("5016", "aud_m", doc_id="old", published=date(2025, 5, 1)),
            cand("4990", "aud_m", doc_id="new", published=date(2026, 5, 1), restated=True),
        ],
    )
    assert conflict.classification == "RESTATEMENT"
    assert conflict.chosen.source.doc_id == "new"


def test_an_unfootnoted_restatement_costs_confidence():
    conflict = classify(
        "earn.headline",
        [
            cand("5016", "aud_m", doc_id="old", published=date(2025, 5, 1)),
            cand(
                "4990",
                "aud_m",
                doc_id="new",
                published=date(2026, 5, 1),
                restated=True,
                footnoted=False,
            ),
        ],
    )
    assert "-0.05" in conflict.impact


def test_a_footnoted_restatement_does_not():
    conflict = classify(
        "earn.headline",
        [
            cand("5016", "aud_m", doc_id="old", published=date(2025, 5, 1)),
            cand(
                "4990",
                "aud_m",
                doc_id="new",
                published=date(2026, 5, 1),
                restated=True,
                footnoted=True,
            ),
        ],
    )
    assert "-0.05" not in conflict.impact


def test_different_measurement_dates_are_a_timing_conflict():
    conflict = classify(
        "cet1.headline",
        [
            cand("12.39", doc_type="RESULTS_DOCUMENT"),
            cand("12.10", doc_id="p3", doc_type="PILLAR3"),
        ],
        measurement_dates={"d1": date(2026, 3, 31), "p3": date(2025, 12, 31)},
        period_end=date(2026, 3, 31),
    )
    assert conflict.classification == "TIMING"
    assert conflict.chosen.source.doc_id == "d1"


def test_an_otherwise_unexplained_difference_falls_through():
    conflict = classify("nim.headline", [cand("1.95"), cand("1.88", doc_id="d2")])
    assert conflict.classification == "UNEXPLAINED"


# --- ordering -------------------------------------------------------------


def test_the_classification_order_matches_the_plan():
    assert CLASSIFICATION_ORDER == (
        "UNIT",
        "ROUNDING",
        "BASIS",
        "SCOPE",
        "PRO_FORMA",
        "RESTATEMENT",
        "TIMING",
        "UNEXPLAINED",
    )


def test_a_rounding_difference_never_falls_through_to_unexplained():
    """The ordering test: a difference of precision is ROUNDING, whatever comes after."""
    for a, b in [("12.39", "12.4"), ("1.95", "1.9"), ("5016", "5016.4")]:
        conflict = classify("x", [cand(a), cand(b, doc_id="d2")])
        assert conflict.classification != "UNEXPLAINED", (a, b)


def test_a_unit_difference_outranks_everything_else():
    """1.95% vs 195bp is not a basis disagreement even if the tags differ."""
    conflict = classify(
        "nim.headline",
        [
            cand("1.95", "pct", basis=Basis(earnings_basis="cash")),
            cand("195", "bps", doc_id="d2", basis=Basis(earnings_basis="statutory")),
        ],
    )
    assert conflict.classification == "UNIT"


def test_a_single_candidate_is_not_a_conflict():
    assert classify("x", [cand("1.95")]) is None


def test_values_within_tolerance_are_not_a_conflict():
    assert classify("x", [cand("1.95"), cand("1.95", doc_id="d2")]) is None


# --- precedence, only for UNEXPLAINED -------------------------------------


def test_unexplained_resolves_by_precedence_and_caps_confidence():
    conflict = classify(
        "nim.headline",
        [
            cand("1.88", doc_id="pack", doc_type="INVESTOR_PRESENTATION"),
            cand("1.95", doc_id="fr", doc_type="RESULTS_DOCUMENT"),
        ],
    )
    assert conflict.classification == "UNEXPLAINED"
    assert conflict.chosen.source.doc_type == "RESULTS_DOCUMENT"
    assert "0.75" in conflict.impact


def test_a_sign_flipping_unexplained_conflict_caps_harder():
    conflict = classify(
        "cic.headline",
        [
            cand("-50", "aud_m", doc_id="pack", doc_type="INVESTOR_PRESENTATION"),
            cand("120", "aud_m", doc_id="fr", doc_type="RESULTS_DOCUMENT"),
        ],
    )
    assert conflict.classification == "UNEXPLAINED"
    assert "0.45" in conflict.impact


def test_pillar3_loses_to_the_results_document_when_unexplained():
    conflict = classify(
        "cet1.headline",
        [
            cand("12.10", doc_id="p3", doc_type="PILLAR3"),
            cand("12.39", doc_id="fr", doc_type="RESULTS_DOCUMENT"),
        ],
    )
    assert conflict.chosen.source.doc_type == "RESULTS_DOCUMENT"


def test_values_are_never_averaged():
    """Never average, never silently pick one."""
    conflict = classify("nim.headline", [cand("1.95"), cand("1.85", doc_id="d2")])
    assert conflict.chosen.value in {Decimal("1.95"), Decimal("1.85")}
    assert conflict.rule_applied


def test_resolve_returns_the_chosen_value_and_records_the_conflict():
    chosen, conflicts = resolve(
        "nim.headline",
        [cand("1.95", doc_id="fr"), cand("1.88", doc_id="pack", doc_type="INVESTOR_PRESENTATION")],
    )
    assert chosen.value == Decimal("1.95")
    assert len(conflicts) == 1


def test_resolve_with_no_disagreement_records_nothing():
    chosen, conflicts = resolve("nim.headline", [cand("1.95")])
    assert chosen.value == Decimal("1.95")
    assert conflicts == []


def test_a_figure_published_after_as_of_is_refused_outright():
    """Never allowed: invariant 5 makes it a hard rule."""
    with pytest.raises(ValueError):
        resolve(
            "nim.headline",
            [
                cand("1.95", published=date(2026, 5, 1)),
                cand("1.90", doc_id="d2", published=date(2026, 11, 1)),
            ],
            as_of=date(2026, 5, 1),
        )


def test_an_unlabelled_figure_on_the_other_basis_is_not_a_restatement():
    """Both of Westpac's results documents print 1H25 net profit as A$3,317m. Its investor
    pack prints A$3,457m in a table headed only "Net profit", which is exactly the bank's own
    figure excluding notable items. The memo had called that a restatement "footnoted by the
    issuer"; it's the other basis."""
    original = cand("3317", "aud_m", doc_id="wbc-1h25", published=date(2025, 5, 5))
    comparative = cand("3317", "aud_m", doc_id="wbc-1h26", published=date(2026, 5, 5))
    unlabelled = cand(
        "3457",
        "aud_m",
        doc_id="wbc-1h26-pack",
        doc_type="INVESTOR_PRESENTATION",
        published=date(2026, 5, 5),
    )
    excluding = cand(
        "3457",
        "aud_m",
        doc_id="wbc-1h26-p10",
        basis=Basis(ex_notable_items=True),
        published=date(2026, 5, 5),
    )
    chosen, conflicts = resolve(
        "headline_earnings",
        [original, unlabelled, excluding, comparative],
        headline_basis=Basis(),
    )
    assert chosen.value == Decimal("3317")
    assert [c.classification for c in conflicts] == ["BASIS"]
    assert "states no basis but prints exactly the issuer's own figure" in conflicts[0].rule_applied


def test_a_real_restatement_is_still_one():
    original = cand("3317", "aud_m", doc_id="wbc-1h25", published=date(2025, 5, 5))
    restated = cand(
        "3290",
        "aud_m",
        doc_id="wbc-1h26",
        published=date(2026, 5, 5),
        restated=True,
        footnoted=True,
    )
    excluding = cand(
        "3457",
        "aud_m",
        doc_id="wbc-1h26-p10",
        basis=Basis(ex_notable_items=True),
        published=date(2026, 5, 5),
    )
    chosen, conflicts = resolve(
        "headline_earnings", [original, restated, excluding], headline_basis=Basis()
    )
    assert chosen.value == Decimal("3290")
    assert "RESTATEMENT" in [c.classification for c in conflicts]
