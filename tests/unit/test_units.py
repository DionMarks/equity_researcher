""""""

from __future__ import annotations

from decimal import Decimal

import pytest

from bankresearch.domain.units import (
    normalise_numbers,
    to_aud_m,
    to_bps,
    to_pct,
    value_tolerance,
    walk_tolerance,
)


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        ("1.95", "pct", "195"),
        ("195", "bps", "195"),
        ("0.0195", "ratio", "195.00"),
    ],
)
def test_ratio_units_convert_to_bps(value, unit, expected):
    assert to_bps(Decimal(value), unit) == Decimal(expected)


def test_percent_and_ratio_round_trip():
    assert to_pct(Decimal("195"), "bps") == Decimal("1.95")
    assert to_pct(Decimal("0.0195"), "ratio") == Decimal("1.9500")


def test_aud_billions_convert_to_millions():
    assert to_aud_m(Decimal("12.5"), "aud_bn") == Decimal("12500.0")
    assert to_aud_m(Decimal("12500"), "aud_m") == Decimal("12500")


def test_a_ratio_unit_is_not_an_aud_unit():
    with pytest.raises(ValueError):
        to_aud_m(Decimal("1"), "pct")
    with pytest.raises(ValueError):
        to_bps(Decimal("1"), "aud_m")


# --- tolerances ------------------------------------------------------------


def test_two_decimal_ratio_tolerates_half_a_basis_point():
    assert value_tolerance("pct", reported=Decimal("12.39")) == Decimal("0.5")


def test_one_decimal_ratio_tolerates_five_basis_points():
    """Precision drives the tolerance, rather than a fixed constant."""
    assert value_tolerance("pct", reported=Decimal("12.4")) == Decimal("5.0")


def test_whole_millions_tolerate_half_a_million():
    assert value_tolerance("aud_m", reported=Decimal("5016")) == Decimal("0.5")


def test_walk_tolerance_grows_with_the_number_of_rounded_steps():
    """Five independently rounded steps can legitimately miss by more than one."""
    assert walk_tolerance("bps", 1) == Decimal("1")
    assert walk_tolerance("bps", 2) == Decimal("1")
    assert walk_tolerance("bps", 8) == Decimal("4.0")
    assert walk_tolerance("aud_m", 2) == Decimal("2")
    assert walk_tolerance("aud_m", 10) == Decimal("5.0")


def test_walk_tolerance_rejects_negative_steps():
    with pytest.raises(ValueError):
        walk_tolerance("bps", -1)


# --- citation normalisation ------------------------------------------------


def test_a_percentage_also_reads_as_basis_points():
    assert Decimal("195") in normalise_numbers("NIM of 1.95% for the half")


def test_basis_points_also_read_as_a_percentage():
    assert Decimal("1.95") in normalise_numbers("up 195 bps")


def test_thousands_separators_are_stripped():
    assert Decimal("11371") in normalise_numbers("cash earnings of $11,371m")


def test_billions_also_read_as_millions():
    assert Decimal("12500") in normalise_numbers("RWA of $12.5bn")


def test_a_bracketed_number_reads_as_negative():
    assert Decimal("-42") in normalise_numbers("impairment of (42)")


def test_several_numbers_are_all_found():
    found = normalise_numbers("1H26 NIM 1.95%, up from 1.88% in 1H25")
    assert {Decimal("1.95"), Decimal("1.88"), Decimal("195"), Decimal("188")} <= found


# --- comparing two readings of one figure ---------------------------------


@pytest.mark.parametrize(
    ("left", "left_unit", "right", "right_unit", "same"),
    [
        # The bug this guards. Westpac's Level 2 CET1 is 12.42% and its Level 1 is
        # 12.75%: 33 basis points apart. Held against a tolerance of half a basis point
        # the two agree only if the difference is measured in percentage points, which
        # is the unit `value_tolerance` does not answer in.
        (Decimal("12.42"), "pct", Decimal("12.75"), "pct", False),
        (Decimal("12.42"), "pct", Decimal("12.42"), "pct", True),
        (Decimal("1.53"), "pct", Decimal("1.89"), "pct", False),
        # One decimal place tolerates five basis points, so 12.4 and 12.42 agree.
        (Decimal("12.4"), "pct", Decimal("12.42"), "pct", True),
        # Amounts, where the tolerance is already in millions.
        (Decimal("3414"), "aud_m", Decimal("3483"), "aud_m", False),
        (Decimal("10252"), "aud_m", Decimal("10253"), "aud_m", False),
        (Decimal("3414"), "aud_m", Decimal("3414"), "aud_m", True),
        # Across units: the same figure written two ways.
        (Decimal("12.4"), "pct", Decimal("1240"), "bps", True),
        (Decimal("426.4"), "aud_bn", Decimal("426400"), "aud_m", True),
        # A ratio and an amount are never the same figure.
        (Decimal("12.42"), "pct", Decimal("12.42"), "aud_m", False),
    ],
)
def test_two_readings_are_compared_in_the_unit_the_tolerance_is_given_in(
    left, left_unit, right, right_unit, same
):
    from bankresearch.domain.units import same_published_figure

    assert same_published_figure(left, left_unit, right, right_unit) is same
    assert same_published_figure(right, right_unit, left, left_unit) is same
