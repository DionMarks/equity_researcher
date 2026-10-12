"""Fact extraction and identity validation."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from bankresearch.domain.models import DocMeta, Fact
from bankresearch.ingestion.facts import extract_facts, line_item_index, parse_cell
from bankresearch.ingestion.parse_pdf import PyMuPdfParser
from bankresearch.ingestion.validate import validate_facts

FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthetic_results_document.pdf"
PUB = date(2026, 5, 1)
PERIODS = ("1H26", "1H25")


@pytest.fixture(scope="module")
def meta():
    return DocMeta(
        doc_id="fx-1h26-rd",
        bank_id="ANZ",
        period="1H26",
        doc_type="RESULTS_DOCUMENT",
        publish_date=PUB,
        url="https://example.invalid/fx",
        page_count=5,
    )


@pytest.fixture(scope="module")
def facts(meta):
    document = PyMuPdfParser().parse(FIXTURE, meta.doc_id)
    return extract_facts(document, meta, PERIODS)


@pytest.fixture(scope="module")
def by_key(facts):
    return {(f.component, f.period): f for f in facts}


# --- cell parsing ---------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("5016", "5016"),
        ("12.39", "12.39"),
        ("1,035,000", "1035000"),
        ("$5,016", "5016"),
        ("43.33%", "43.33"),
        ("(42)", "-42"),
        ("-7.0", "-7.0"),
    ],
)
def test_a_figure_cell_parses_to_decimal(text, expected):
    assert parse_cell(text) == Decimal(expected)


def test_brackets_mean_negative():
    """A deduction or a write-back is printed as (42); reading it positive flips a sign."""
    assert parse_cell("(1,200)") == Decimal("-1200")


def test_published_precision_survives_parsing():
    assert str(parse_cell("1.80")) == "1.80"


@pytest.mark.parametrize("text", ["", "n/a", "-", "Cash profit", "large", None])
def test_a_non_figure_cell_is_not_a_value(text):
    assert parse_cell(text) is None


# --- the label index -----------------------------------------------------


def test_one_row_can_feed_several_components():
    """ "Credit impairment charge" is the headline metric and a bridge line."""
    index = line_item_index("ANZ")
    assert set(index.components_for("Credit impairment charge")) == {
        "credit_impairment_charge",
        "credit_impairment",
    }


def test_an_unknown_row_label_maps_to_nothing():
    assert line_item_index("ANZ").components_for("Dividend per ordinary share") == ()


def test_a_near_miss_label_is_not_matched():
    """Fuzzy matching would bind this to operating_expenses, a real figure from
    the wrong line."""
    assert line_item_index("ANZ").components_for("Operating expenses before notable items") == ()


def test_a_row_suffix_overrides_the_tables_unit():
    """A key-metrics table captioned $M still prints its ratios in per cent."""
    index = line_item_index("ANZ")
    assert index.unit_for("nim", "Net interest margin (%)", "aud_m") == "pct"
    assert index.unit_for("nii", "Net interest income", "aud_m") == "aud_m"


def test_an_issuer_override_is_applied():
    assert "headline_earnings" in line_item_index("CBA").components_for("Cash NPAT")


# --- extraction ----------------------------------------------------------


def test_only_the_requested_periods_are_extracted(facts):
    assert {f.period for f in facts} == {"1H26", "1H25"}


def test_the_headline_metrics_are_all_found(by_key):
    for metric in (
        "nim",
        "cet1_ratio",
        "roe",
        "cost_to_income",
        "headline_earnings",
        "credit_impairment_charge",
    ):
        assert (metric, "1H26") in by_key, metric
        assert (metric, "1H25") in by_key, metric


def test_the_decomposition_line_items_are_found(by_key):
    for component in (
        "nii",
        "other_income",
        "operating_income",
        "operating_expenses",
        "tax",
        "aiea",
        "average_equity",
        "cet1_capital",
        "rwa",
        "rwa_credit",
        "rwa_irrbb",
        "rwa_market",
        "rwa_operational",
    ):
        assert (component, "1H26") in by_key, component


def test_ratios_are_stored_in_per_cent_not_millions(by_key):
    assert by_key[("nim", "1H26")].unit == "pct"
    assert by_key[("nim", "1H26")].value == Decimal("1.86")
    assert by_key[("cet1_ratio", "1H26")].unit == "pct"


def test_the_pcp_column_feeds_the_pcp_facts(by_key):
    assert by_key[("cet1_capital", "1H25")].value == Decimal("48840")
    assert by_key[("headline_earnings", "1H25")].value == Decimal("4300")


def test_the_adjacent_half_does_not_leak_into_the_pcp(by_key):
    """Sep 25 holds 49803; reading it as the pcp would be a silent error."""
    assert by_key[("cet1_capital", "1H25")].value != Decimal("49803")


def test_a_restatement_footnote_marks_the_comparative_not_the_current(by_key):
    assert by_key[("headline_earnings", "1H25")].is_restated is True
    assert by_key[("headline_earnings", "1H26")].is_restated is False


def test_every_fact_carries_its_page_and_vintage(facts):
    for fact in facts:
        assert fact.page_no >= 1
        assert fact.publish_date == PUB
        assert fact.doc_id == "fx-1h26-rd"


def test_facts_arrive_unvalidated(facts):
    assert all(not f.validated for f in facts)


def test_the_summary_table_wins_over_a_later_restatement_of_the_same_component(facts):
    """One fact per component and period; page order decides."""
    keys = [(f.component, f.period) for f in facts]
    assert len(keys) == len(set(keys))


# --- validation ----------------------------------------------------------


def test_every_identity_in_a_consistent_document_passes(facts):
    outcome = validate_facts(facts, "ANZ", PERIODS)
    assert outcome.failed_checks == []
    assert outcome.quarantined == []
    assert len(outcome.validated) == len(facts)


def test_the_identities_that_ran_are_the_ones_the_plan_names(facts):
    outcome = validate_facts(facts, "ANZ", PERIODS)
    ran = {c.name for c in outcome.checks if not c.skipped}
    assert {"cet1_ratio", "nim", "roe", "cost_to_income", "headline_earnings", "rwa_sum"} <= ran


def test_validated_facts_are_marked_validated(facts):
    outcome = validate_facts(facts, "ANZ", PERIODS)
    assert all(f.validated for f in outcome.validated)


def test_tolerance_is_derived_from_published_precision(facts):
    """A ratio published to two decimals tolerates about half a basis point, plus
    the rounding propagated from its components."""
    outcome = validate_facts(facts, "ANZ", PERIODS)
    cet1 = next(c for c in outcome.checks if c.name == "cet1_ratio" and c.period == "1H26")
    assert Decimal("0.5") < cet1.tolerance < Decimal("1")


def test_a_misread_component_quarantines_the_components_but_keeps_the_headline(facts):
    """A wrong CET1 capital must not reach Path B, but the published ratio stands:
    The reported headline is the comparison target regardless."""
    broken = [
        f.model_copy(update={"value": Decimal("60000")})
        if (f.component, f.period) == ("cet1_capital", "1H26")
        else f
        for f in facts
    ]
    outcome = validate_facts(broken, "ANZ", PERIODS)

    failed = {c.name for c in outcome.failed_checks}
    assert "cet1_ratio" in failed
    assert outcome.any_input_failed

    quarantined = {(f.component, f.period) for f, _ in outcome.quarantined}
    assert ("cet1_capital", "1H26") in quarantined
    assert ("rwa", "1H26") in quarantined
    assert ("cet1_ratio", "1H26") not in quarantined


def test_the_quarantine_reason_names_the_identity_and_the_gap(facts):
    broken = [
        f.model_copy(update={"value": Decimal("60000")})
        if (f.component, f.period) == ("cet1_capital", "1H26")
        else f
        for f in facts
    ]
    outcome = validate_facts(broken, "ANZ", PERIODS)
    reason = next(r for f, r in outcome.quarantined if f.component == "cet1_capital")
    assert "cet1_ratio" in reason
    assert "tolerance" in reason


def test_a_broken_earnings_bridge_quarantines_its_lines(facts):
    broken = [
        f.model_copy(update={"value": Decimal("9000")})
        if (f.component, f.period) == ("nii", "1H26")
        else f
        for f in facts
    ]
    outcome = validate_facts(broken, "ANZ", PERIODS)
    assert "headline_earnings" in {c.name for c in outcome.failed_checks}
    quarantined = {f.component for f, _ in outcome.quarantined}
    assert "nii" in quarantined
    assert "headline_earnings" not in quarantined


def test_a_missing_component_skips_its_identity_rather_than_failing_it():
    """Nothing to check is not the same as a check that failed."""
    only_headline = [
        Fact(
            fact_id="",
            bank_id="ANZ",
            period="1H26",
            component="cet1_ratio",
            value=Decimal("12.39"),
            unit="pct",
            doc_id="d",
            doc_type="RESULTS_DOCUMENT",
            page_no=7,
            publish_date=PUB,
        )
    ]
    outcome = validate_facts(only_headline, "ANZ", PERIODS)
    assert outcome.failed_checks == []
    assert len(outcome.validated) == 1


def test_the_summary_reports_what_happened(facts):
    assert "identity checks passed" in validate_facts(facts, "ANZ", PERIODS).summary()


# --- continuing operations is a basis, not a different line item ----------


@pytest.mark.parametrize(
    ("label", "component", "dimensions"),
    [
        # CBA prints the group's profit three ways on one page. All three are the same
        # line item; the qualifier says which basis each is on, and the basis decides
        # which one answers the question.
        (
            "Net profit after tax from continuing operations",
            ("headline_earnings",),
            {"continuing_ops": True},
        ),
        (
            'Net profit after tax from continuing operations - "cash basis"',
            ("headline_earnings",),
            {"continuing_ops": True, "earnings_basis": "cash"},
        ),
        (
            "Net profit after tax from continuing operations - cash basis",
            ("headline_earnings",),
            {"continuing_ops": True, "earnings_basis": "cash"},
        ),
        (
            'Net profit after tax - "statutory basis"',
            ("headline_earnings",),
            {"earnings_basis": "statutory"},
        ),
        ("Net profit after tax", ("headline_earnings",), {}),
        # The discontinued result is not a reading of the headline at any scale: CBA's
        # FY26 statutory line is a $45m loss and its FY25 cash line is $1m.
        ("Net (loss)/profit after tax from discontinued operations", (), {}),
        ("Net profit after tax from discontinued operations", (), {}),
    ],
)
def test_an_operations_qualifier_is_read_as_a_basis(label, component, dimensions):
    from bankresearch.ingestion.facts import line_item_index

    index = line_item_index("CBA")
    assert index.components_for(label) == component
    assert index.sections.basis_dimensions(label) == dimensions


def test_the_continuing_operations_reading_beats_the_unqualified_total():
    """Page 16 gives FY25 as $10,252m from continuing operations and $10,253m in total.

    They differ by the $1m discontinued result. CBA presents its headline on a
    continuing-operations basis (page 18 heads the block "Group Performance from continuing
    operations"), and its own commentary moves the year by $730m, which is 10,982 - 10,252.
    Taking the total made the movement $729m against a published $730m.
    """
    from bankresearch.domain.models import Basis, Fact
    from bankresearch.ingestion.facts import _prefer, line_item_index

    index = line_item_index("CBA")

    def fact(value: str, continuing: bool | None) -> Fact:
        return Fact(
            fact_id="",
            bank_id="CBA",
            period="FY25",
            component="headline_earnings",
            value=Decimal(value),
            published_value=Decimal(value),
            unit="aud_m",
            basis=Basis(earnings_basis="cash", scope="group", continuing_ops=continuing),
            doc_id="cba-fy26-results-document",
            doc_type="RESULTS_DOCUMENT",
            page_no=16,
            publish_date=date(2026, 8, 12),
        )

    total = fact("10253", None)
    continuing = fact("10252", True)
    assert _prefer(continuing, total, "cash", index)
    assert not _prefer(total, continuing, "cash", index)
