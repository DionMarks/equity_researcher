"""The column-to-period map.

"The column-to-period map is what identifies the pcp column, so it MUST be tested." A
results pack prints three comparison columns side by side, and a figure read from the wrong
one is correctly extracted but answers a different question, so this is the most
consequential part of extraction.
"""

from __future__ import annotations

from datetime import date

import pytest

from bankresearch.domain.calendar import PeriodKind
from bankresearch.ingestion.tables import (
    detect_restated,
    detect_units,
    map_columns,
    parse_column_period,
)

ANZ = 9  # September year end
CBA = 6  # June year end
BOQ = 8  # August year end


# --- explicit fiscal labels ----------------------------------------------


@pytest.mark.parametrize(
    ("header", "expected"),
    [("1H26", "1H26"), ("2H25", "2H25"), ("FY25", "FY25"), ("HY26", "1H26"), ("FY2025", "FY25")],
)
def test_an_explicit_fiscal_label_is_unambiguous(header, expected):
    column = parse_column_period(header, fy_end_month=ANZ)
    assert column.candidates == (expected,)
    assert column.resolve("1H26") == expected


# --- dates ---------------------------------------------------------------


@pytest.mark.parametrize(
    "header",
    ["Mar 26", "Mar-26", "Mar 2026", "March 2026", "31 Mar 26", "31 March 2026", "Mar'26"],
)
def test_a_half_end_date_resolves_for_a_september_year_end(header):
    column = parse_column_period(header, fy_end_month=ANZ)
    assert column.end_date == date(2026, 3, 31)
    assert column.candidates == ("1H26",)


def test_the_same_month_means_a_different_half_for_a_june_year_end():
    """'Dec 25' is CBA's first half of FY26; it is nothing for ANZ."""
    cba = parse_column_period("Dec 25", fy_end_month=CBA)
    assert cba.candidates == ("1H26",)
    assert parse_column_period("Dec 25", fy_end_month=ANZ).candidates == ()


def test_a_full_year_end_date_is_ambiguous_between_the_second_half_and_the_year():
    """Both end on the same day, which is why the balance-sheet value at 2H end
    equals the FY-end value."""
    column = parse_column_period("Sep 25", fy_end_month=ANZ)
    assert column.end_date == date(2025, 9, 30)
    assert set(column.candidates) == {"2H25", "FY25"}


def test_an_august_year_end_resolves_its_own_halves():
    assert parse_column_period("Feb 26", fy_end_month=BOQ).candidates == ("1H26",)
    assert set(parse_column_period("Aug 25", fy_end_month=BOQ).candidates) == {"2H25", "FY25"}


def test_a_quarter_end_matches_no_reporting_period():
    column = parse_column_period("Dec 25", fy_end_month=ANZ)
    assert column.candidates == ()
    assert column.unmatched_reason


# --- explicit kind narrows the ambiguity ---------------------------------


def test_half_year_ended_narrows_to_the_half():
    column = parse_column_period("Half year ended 30 September 2025", fy_end_month=ANZ)
    assert column.explicit_kind is PeriodKind.H2
    assert column.candidates == ("2H25",)


def test_full_year_ended_narrows_to_the_year():
    column = parse_column_period("Full year ended 30 September 2025", fy_end_month=ANZ)
    assert column.explicit_kind is PeriodKind.FY
    assert column.candidates == ("FY25",)


@pytest.mark.parametrize("phrase", ["6 months ended 31 Mar 26", "Six months to 31 March 2026"])
def test_six_months_implies_a_half(phrase):
    assert parse_column_period(phrase, fy_end_month=ANZ).candidates == ("1H26",)


@pytest.mark.parametrize("phrase", ["12 months ended 30 Sep 25", "Twelve months to September 2025"])
def test_twelve_months_implies_a_full_year(phrase):
    assert parse_column_period(phrase, fy_end_month=ANZ).candidates == ("FY25",)


# --- resolution against the document's own period ------------------------


def test_an_ambiguous_column_resolves_to_the_half_in_a_half_year_document():
    column = parse_column_period("Sep 25", fy_end_month=ANZ)
    assert column.resolve("1H26") == "2H25"


def test_an_ambiguous_column_resolves_to_the_year_in_a_full_year_document():
    column = parse_column_period("Sep 25", fy_end_month=ANZ)
    assert column.resolve("FY26") == "FY25"


def test_a_stock_metric_prefers_the_balance_date_reading():
    """CET1 at 30 September is the same number whether you call it 2H25 or FY25."""
    column = parse_column_period("Sep 25", fy_end_month=ANZ)
    assert column.resolve("1H26", point_in_time=True) in {"2H25", "FY25"}


def test_an_unmatched_column_resolves_to_nothing():
    assert parse_column_period("Movement", fy_end_month=ANZ).resolve("1H26") is None


# --- the pcp column, which is the point of all this ----------------------


def test_the_pcp_column_is_identified_among_three_comparatives():
    """An ANZ interim table: current half, prior half, prior corresponding half."""
    mapping = map_columns(["Mar 26", "Sep 25", "Mar 25"], fy_end_month=ANZ, doc_period="1H26")
    assert mapping == {"Mar 26": "1H26", "Sep 25": "2H25", "Mar 25": "1H25"}


def test_the_adjacent_half_is_not_mistaken_for_the_pcp():
    mapping = map_columns(["Mar 26", "Sep 25", "Mar 25"], fy_end_month=ANZ, doc_period="1H26")
    pcp_columns = [header for header, label in mapping.items() if label == "1H25"]
    assert pcp_columns == ["Mar 25"]
    assert mapping["Sep 25"] != "1H25"


def test_a_full_year_table_maps_its_own_comparatives():
    mapping = map_columns(["Sep 26", "Sep 25", "Sep 24"], fy_end_month=ANZ, doc_period="FY26")
    assert mapping == {"Sep 26": "FY26", "Sep 25": "FY25", "Sep 24": "FY24"}


def test_a_full_year_packs_half_year_table_is_read_as_halves():
    """ANZ's 2025 full-year pack heads its half-year block ``Sep 25 | Mar 25`` with "Half
    Year" on the line above, where no column reads it. Read by the document's own period,
    "Sep 25" was the full year, and the second half's average assets ($1,177bn) were stored
    as the year's ($1,160bn). Set beside March, September is the second half.
    """
    mapping = map_columns(["Sep 25", "Mar 25"], fy_end_month=ANZ, doc_period="FY25")
    assert mapping == {"Sep 25": "2H25", "Mar 25": "1H25"}


def test_a_balance_in_a_half_year_table_is_still_the_year_end_balance():
    """A balance on 30 September is the same figure for the half and the year."""
    mapping = map_columns(
        ["Sep 25", "Mar 25"], fy_end_month=ANZ, doc_period="FY25", point_in_time=True
    )
    assert mapping == {"Sep 25": "FY25", "Mar 25": "1H25"}


def test_a_full_year_block_beside_a_half_year_block_keeps_its_years():
    mapping = map_columns(["Sep 25", "Sep 24"], fy_end_month=ANZ, doc_period="FY25")
    assert mapping == {"Sep 25": "FY25", "Sep 24": "FY24"}


def test_a_column_named_as_the_full_year_stays_the_full_year_beside_halves():
    mapping = map_columns(["FY25", "2H25", "1H25"], fy_end_month=ANZ, doc_period="FY25")
    assert mapping == {"FY25": "FY25", "2H25": "2H25", "1H25": "1H25"}


def test_a_cba_full_year_packs_half_year_table_is_read_as_halves():
    mapping = map_columns(["Jun 26", "Dec 25", "Jun 25"], fy_end_month=CBA, doc_period="FY26")
    assert mapping == {"Jun 26": "2H26", "Dec 25": "1H26", "Jun 25": "2H25"}


def test_a_cba_interim_table_maps_to_june_year_end_periods():
    mapping = map_columns(["Dec 25", "Jun 25", "Dec 24"], fy_end_month=CBA, doc_period="1H26")
    assert mapping == {"Dec 25": "1H26", "Jun 25": "2H25", "Dec 24": "1H25"}


def test_non_period_columns_are_left_out_rather_than_guessed():
    mapping = map_columns(
        ["Mar 26", "Mar 25", "Movement", "Change %", "$M"], fy_end_month=ANZ, doc_period="1H26"
    )
    assert mapping == {"Mar 26": "1H26", "Mar 25": "1H25"}


def test_a_repeated_period_keeps_its_leftmost_column():
    """Real packs label their movement columns with periods too. Westpac's key-metrics table
    reads ``March 2026 | Sept 2025 | March 2025 | Mar 26 | Sept 25 | Mar 25``, with the last
    three heading a "% Mov't" block, and in every pack seen the period columns come first.
    Raising on the duplicate was tried and threw away that whole table, and with it
    Westpac's group margin, return on equity and capital ratio.
    """
    mapping = map_columns(
        ["March 2026", "Sept 2025", "March 2025", "Mar 26", "Sept 25", "Mar 25"],
        fy_end_month=ANZ,
        doc_period="1H26",
    )
    assert mapping == {"March 2026": "1H26", "Sept 2025": "2H25", "March 2025": "1H25"}
    assert "Mar 26" not in mapping


def test_two_spellings_of_one_period_do_not_both_map():
    mapping = map_columns(["Mar 26", "31 March 2026"], fy_end_month=ANZ, doc_period="1H26")
    assert mapping == {"Mar 26": "1H26"}


# --- units and restatement flags -----------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$M", "aud_m"),
        ("A$m", "aud_m"),
        ("$ million", "aud_m"),
        ("$bn", "aud_bn"),
        ("%", "pct"),
        ("bps", "bps"),
        ("basis points", "bps"),
    ],
)
def test_units_are_detected_from_the_caption_or_header(text, expected):
    assert detect_units(text) == expected


def test_an_undetectable_unit_is_none_rather_than_a_guess():
    assert detect_units("Group performance") is None


@pytest.mark.parametrize(
    "text",
    [
        "Comparatives have been restated",
        "restated for the sale of the insurance business",
        "Prior periods have been re-presented",
        "1H25 (restated)",
    ],
)
def test_a_restatement_footnote_is_detected(text):
    assert detect_restated(text)


def test_ordinary_text_is_not_a_restatement():
    assert not detect_restated("Net interest margin increased 4 basis points")
