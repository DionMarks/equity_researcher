"""Fiscal periods, and how each bank's calendar turns a label into dates."""

from __future__ import annotations

from datetime import date

import pytest

from bankresearch.domain.calendar import (
    AmbiguousPeriodError,
    PeriodKind,
    fiscal_period,
    parse_period,
)

# --- FY boundaries ---------------------------------------------------------


def test_cba_fy_runs_july_to_june():
    p = fiscal_period("FY26", fy_end_month=6)
    assert (p.start, p.end) == (date(2025, 7, 1), date(2026, 6, 30))


def test_september_year_end_fy_runs_october_to_september():
    p = fiscal_period("FY26", fy_end_month=9)
    assert (p.start, p.end) == (date(2025, 10, 1), date(2026, 9, 30))


def test_boq_august_year_end():
    p = fiscal_period("FY26", fy_end_month=8)
    assert (p.start, p.end) == (date(2025, 9, 1), date(2026, 8, 31))


# --- halves ----------------------------------------------------------------


def test_cba_first_half_is_july_to_december():
    p = fiscal_period("1H26", fy_end_month=6)
    assert (p.start, p.end) == (date(2025, 7, 1), date(2025, 12, 31))


def test_september_year_end_first_half_is_october_to_march():
    p = fiscal_period("1H26", fy_end_month=9)
    assert (p.start, p.end) == (date(2025, 10, 1), date(2026, 3, 31))


def test_second_half_ends_on_the_fy_end():
    half = fiscal_period("2H26", fy_end_month=6)
    full = fiscal_period("FY26", fy_end_month=6)
    assert half.end == full.end
    assert half.start == date(2026, 1, 1)


def test_the_same_label_is_a_different_window_for_each_issuer():
    """The whole reason periods are resolved per bank."""
    assert (
        fiscal_period("1H26", fy_end_month=6).start != fiscal_period("1H26", fy_end_month=9).start
    )


# --- the comparison period -------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [("1H26", "1H25"), ("2H25", "2H24"), ("FY25", "FY24"), ("1H23", "1H22")],
)
def test_pcp_is_the_same_period_one_financial_year_back(label, expected):
    assert fiscal_period(label, fy_end_month=9).pcp().label == expected


def test_pcp_is_never_the_adjacent_half():
    """Never the adjacent half. Invariant 3."""
    assert fiscal_period("1H26", fy_end_month=9).pcp().label != "2H25"


def test_pcp_window_is_one_year_earlier():
    p = fiscal_period("1H26", fy_end_month=9)
    prior = p.pcp()
    assert (prior.start, prior.end) == (date(2024, 10, 1), date(2025, 3, 31))


# --- day counts ------------------------------------------------------------


def test_day_count_is_inclusive_of_both_ends():
    assert fiscal_period("FY25", fy_end_month=6).days == 365


def test_leap_year_makes_pcp_halves_differ_by_a_day():
    """'Leap years make pcp halves differ by a day.'"""
    # A September year end: 2H24 covers Apr-Sep 2024, which includes no 29 Feb,
    # while 1H24 covers Oct 2023-Mar 2024, which does.
    first = fiscal_period("1H24", fy_end_month=9)
    assert first.days == 183
    assert fiscal_period("1H23", fy_end_month=9).days == 182
    assert first.days - fiscal_period("1H23", fy_end_month=9).days == 1


# --- parsing ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1H26", "1H26"),
        ("2H25", "2H25"),
        ("FY25", "FY25"),
        ("H1 FY26", "1H26"),
        ("1HFY26", "1H26"),
        ("HY26", "1H26"),
        ("FY2025", "FY25"),
        ("fy25", "FY25"),
        ("  1h26 ", "1H26"),
        ("H2 FY25", "2H25"),
    ],
)
def test_accepted_input_forms(raw, expected):
    assert parse_period(raw) == expected


@pytest.mark.parametrize("raw", ["H1 2025", "H1 2026", "1H 2025", "first half 2025"])
def test_calendar_style_input_is_rejected_as_ambiguous(raw):
    """Rejected with AMBIGUOUS_INPUT, offering the fiscal reading."""
    with pytest.raises(AmbiguousPeriodError) as exc:
        parse_period(raw)
    # The error must offer the fiscal reading rather than just refusing.
    assert "FY" in str(exc.value) or "1H" in str(exc.value)


@pytest.mark.parametrize("raw", ["Q1 26", "3H26", "", "banana", "FY", "1H"])
def test_unparseable_input_is_rejected(raw):
    with pytest.raises(AmbiguousPeriodError):
        parse_period(raw)


def test_kind_is_exposed():
    assert fiscal_period("1H26", fy_end_month=9).kind is PeriodKind.H1
    assert fiscal_period("2H26", fy_end_month=9).kind is PeriodKind.H2
    assert fiscal_period("FY26", fy_end_month=9).kind is PeriodKind.FY


def test_balance_date_of_second_half_equals_the_full_year():
    """Balance-sheet metrics at 2H end equal the FY-end values."""
    assert (
        fiscal_period("2H25", fy_end_month=9).balance_date
        == fiscal_period("FY25", fy_end_month=9).balance_date
    )
