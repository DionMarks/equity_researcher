"""Exact Shapley attribution."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bankresearch.decomposition.shapley import (
    TooManyFactorsError,
    ratio_split,
    shapley,
    shapley_exact,
    weighted_average_split,
)

# Bank-sized magnitudes, kept away from zero denominators.
amounts = st.decimals(min_value="100", max_value="100000", places=1, allow_nan=False)
balances = st.decimals(min_value="1000", max_value="2000000", places=1, allow_nan=False)


def _ratio(x: dict[str, Fraction]) -> Fraction:
    return x["a"] / x["b"]


def _sum(x: dict[str, Fraction]) -> Fraction:
    return sum(x.values(), Fraction(0))


# --- efficiency: the axiom the whole design leans on ----------------------


@settings(max_examples=200, deadline=None)
@given(a0=amounts, a1=amounts, b0=balances, b1=balances)
def test_contributions_sum_to_the_total_change_for_a_ratio(a0, a1, b0, b1):
    x0 = {"a": a0, "b": b0}
    x1 = {"a": a1, "b": b1}
    parts = shapley_exact(_ratio, x0, x1)
    total = Fraction(a1) / Fraction(b1) - Fraction(a0) / Fraction(b0)
    assert sum(parts.values(), Fraction(0)) == total


@settings(max_examples=100, deadline=None)
@given(
    v0=st.lists(amounts, min_size=2, max_size=5),
    v1=st.lists(amounts, min_size=2, max_size=5),
)
def test_contributions_sum_to_the_total_change_for_a_sum(v0, v1):
    n = min(len(v0), len(v1))
    keys = [f"k{i}" for i in range(n)]
    x0 = dict(zip(keys, v0[:n], strict=True))
    x1 = dict(zip(keys, v1[:n], strict=True))
    parts = shapley_exact(_sum, x0, x1)
    expected = sum((Fraction(v) for v in v1[:n]), Fraction(0)) - sum(
        (Fraction(v) for v in v0[:n]), Fraction(0)
    )
    assert sum(parts.values(), Fraction(0)) == expected


def test_the_decimal_facing_api_also_sums_to_the_total():
    parts = shapley(
        _ratio,
        {"a": Decimal("1000"), "b": Decimal("50000")},
        {"a": Decimal("1100"), "b": Decimal("52000")},
    )
    total = Decimal("1100") / Decimal("52000") - Decimal("1000") / Decimal("50000")
    assert abs(sum(parts.values()) - total) < Decimal("1e-20")


# --- symmetry and the null player -----------------------------------------


def test_symmetric_factors_get_equal_shares():
    """Two factors entering a sum identically must split the change equally."""
    parts = shapley_exact(
        _sum, {"x": Decimal("10"), "y": Decimal("10")}, {"x": Decimal("20"), "y": Decimal("20")}
    )
    assert parts["x"] == parts["y"]


def test_symmetric_factors_of_a_product_get_equal_shares():
    def product(x: dict[str, Fraction]) -> Fraction:
        return x["x"] * x["y"]

    parts = shapley_exact(
        product, {"x": Decimal("2"), "y": Decimal("2")}, {"x": Decimal("3"), "y": Decimal("3")}
    )
    assert parts["x"] == parts["y"]


@settings(max_examples=100, deadline=None)
@given(a0=amounts, a1=amounts, b=balances)
def test_a_factor_that_does_not_change_gets_exactly_zero(a0, a1, b):
    parts = shapley_exact(_ratio, {"a": a0, "b": b}, {"a": a1, "b": b})
    assert parts["b"] == Fraction(0)


def test_no_change_at_all_gives_every_factor_zero():
    parts = shapley_exact(
        _ratio, {"a": Decimal("5"), "b": Decimal("7")}, {"a": Decimal("5"), "b": Decimal("7")}
    )
    assert all(v == Fraction(0) for v in parts.values())


# --- the closed form for a ratio -----------------------------------------


@settings(max_examples=200, deadline=None)
@given(a0=amounts, a1=amounts, b0=balances, b1=balances)
def test_ratio_split_matches_the_generic_enumeration(a0, a1, b0, b1):
    """The closed midpoint form must BE the two-factor Shapley."""
    generic = shapley_exact(_ratio, {"a": a0, "b": b0}, {"a": a1, "b": b1})
    numerator, denominator = ratio_split(a0, b0, a1, b1)
    assert numerator == generic["a"]
    assert denominator == generic["b"]


def test_ratio_split_is_the_documented_midpoint_formula():
    a0, b0, a1, b1 = Decimal("1000"), Decimal("50000"), Decimal("1100"), Decimal("52000")
    numerator, denominator = ratio_split(a0, b0, a1, b1)
    expected_num = (
        Fraction(a1 - a0) * (Fraction(1, 1) / Fraction(b0) + Fraction(1, 1) / Fraction(b1)) / 2
    )
    expected_den = (
        -Fraction(b1 - b0) * (Fraction(a0) + Fraction(a1)) / (2 * Fraction(b0) * Fraction(b1))
    )
    assert numerator == expected_num
    assert denominator == expected_den


@settings(max_examples=100, deadline=None)
@given(
    n0=st.lists(amounts, min_size=2, max_size=4),
    n1=st.lists(amounts, min_size=2, max_size=4),
    b0=balances,
    b1=balances,
)
def test_the_numerator_effect_splits_linearly_across_components(n0, n1, b0, b1):
    """The ratio 'splits linearly across the components of a'."""
    k = min(len(n0), len(n1))
    whole_num, _ = ratio_split(sum(n0[:k]), b0, sum(n1[:k]), b1)
    pieces = sum((ratio_split(n0[i], b0, n1[i], b1)[0] for i in range(k)), Fraction(0))
    assert pieces == whole_num


# --- the weighted-average mix/rate split ----------------------------------


@settings(max_examples=100, deadline=None)
@given(
    w0=st.lists(st.decimals(min_value="0.05", max_value="1", places=3), min_size=2, max_size=4),
    w1=st.lists(st.decimals(min_value="0.05", max_value="1", places=3), min_size=2, max_size=4),
    m0=st.lists(st.decimals(min_value="0.5", max_value="5", places=3), min_size=2, max_size=4),
    m1=st.lists(st.decimals(min_value="0.5", max_value="5", places=3), min_size=2, max_size=4),
)
def test_mix_and_rate_effects_sum_to_the_weighted_average_change(w0, w1, m0, m1):
    """Exact for the weighted average."""
    k = min(len(w0), len(w1), len(m0), len(m1))
    keys = [f"d{i}" for i in range(k)]
    mix, rate = weighted_average_split(
        dict(zip(keys, w0[:k], strict=True)),
        dict(zip(keys, m0[:k], strict=True)),
        dict(zip(keys, w1[:k], strict=True)),
        dict(zip(keys, m1[:k], strict=True)),
    )
    before = sum((Fraction(w0[i]) * Fraction(m0[i]) for i in range(k)), Fraction(0))
    after = sum((Fraction(w1[i]) * Fraction(m1[i]) for i in range(k)), Fraction(0))
    assert sum(mix.values(), Fraction(0)) + sum(rate.values(), Fraction(0)) == after - before


# --- guard rails -----------------------------------------------------------


def test_more_than_five_factors_is_refused():
    """The enumeration is bounded at 5 factors, 120 orderings."""
    x = {f"k{i}": Decimal(i + 1) for i in range(6)}
    with pytest.raises(TooManyFactorsError):
        shapley_exact(_sum, x, x)


def test_mismatched_factor_sets_are_refused():
    with pytest.raises(ValueError):
        shapley_exact(_sum, {"a": Decimal(1)}, {"b": Decimal(1)})
