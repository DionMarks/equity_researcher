"""Per-metric identities, annualisation and residual classification."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest

from bankresearch.decomposition.identities import (
    IDENTITIES,
    ResidualClass,
    cet1_pct,
    classify_residual,
    cost_to_income_pct,
    credit_impairment_charge,
    headline_earnings,
    nim_pct,
    roe_pct,
)

# A consistent half-year, September year end: 1 Oct to 31 Mar = 182 days.
HALF_DAYS = 182
FULL_DAYS = 365


# --- NIM -------------------------------------------------------------------


def test_nim_reproduces_a_reported_margin():
    # NII 9,000m over average interest-earning assets of 1,000,000m, half year.
    # Annualised: 9000 * 365/182 / 1_000_000 = 1.8049...%
    value = nim_pct(nii=Decimal("9000"), aiea=Decimal("1000000"), days=HALF_DAYS)
    assert abs(value - Fraction(9000 * 365, 182 * 1000000) * 100) == 0


def test_nim_annualises_the_half_year():
    """Omitting annualisation halves every figure."""
    half = nim_pct(nii=Decimal("9000"), aiea=Decimal("1000000"), days=HALF_DAYS)
    unannualised = Fraction(9000, 1000000) * 100
    assert half > unannualised * Fraction(19, 10)


def test_a_full_year_uses_its_own_day_count():
    leap = nim_pct(nii=Decimal("18000"), aiea=Decimal("1000000"), days=366)
    plain = nim_pct(nii=Decimal("18000"), aiea=Decimal("1000000"), days=365)
    assert leap < plain  # the same income spread over one more day


def test_nim_rejects_a_zero_balance():
    with pytest.raises(ZeroDivisionError):
        nim_pct(nii=Decimal("9000"), aiea=Decimal("0"), days=HALF_DAYS)


# --- CET1 ------------------------------------------------------------------


def test_cet1_is_capital_over_rwa():
    assert cet1_pct(capital=Decimal("50000"), rwa=Decimal("400000")) == Fraction(125, 10)


def test_cet1_does_not_annualise():
    """A stock at the balance date, not a flow. No day count enters."""
    import inspect

    assert "days" not in inspect.signature(cet1_pct).parameters


# --- ROE -------------------------------------------------------------------


def test_roe_annualises_earnings_over_average_equity():
    value = roe_pct(earnings=Decimal("5000"), average_equity=Decimal("70000"), days=HALF_DAYS)
    assert value == Fraction(5000 * 365, 182 * 70000) * 100


# --- cost-to-income --------------------------------------------------------


def test_cost_to_income_is_expenses_over_income():
    assert cost_to_income_pct(
        expenses=Decimal("6000"), operating_income=Decimal("15000")
    ) == Fraction(40)


# --- additive identities ---------------------------------------------------


def test_the_earnings_bridge_sums():
    value = headline_earnings(
        nii=Decimal("9000"),
        other_income=Decimal("2500"),
        operating_expenses=Decimal("5000"),
        credit_impairment=Decimal("400"),
        tax=Decimal("1800"),
        minority_interests=Decimal("20"),
    )
    assert value == Fraction(9000 + 2500 - 5000 - 400 - 1800 - 20)


def test_expenses_impairment_tax_and_nci_all_reduce_earnings():
    """Sign conventions."""
    base = dict(
        nii=Decimal("9000"),
        other_income=Decimal("2500"),
        operating_expenses=Decimal("5000"),
        credit_impairment=Decimal("400"),
        tax=Decimal("1800"),
        minority_interests=Decimal("20"),
    )
    for line in ("operating_expenses", "credit_impairment", "tax", "minority_interests"):
        raised = dict(base)
        raised[line] = base[line] + Decimal("100")
        assert headline_earnings(**raised) < headline_earnings(**base), line


def test_credit_impairment_is_additive():
    assert credit_impairment_charge(
        individual=Decimal("250"), collective=Decimal("150")
    ) == Fraction(400)


def test_a_net_write_back_is_a_negative_charge():
    assert credit_impairment_charge(
        individual=Decimal("-80"), collective=Decimal("30")
    ) == Fraction(-50)


# --- the registry of identities -------------------------------------------


def test_every_metric_has_an_identity():
    """One per metric, and the earnings bridge again struck before tax, for a bank that
    leads with profit before tax."""
    assert set(IDENTITIES) == {
        "nim",
        "headline_earnings",
        "profit_before_tax",
        "roe",
        "cet1_ratio",
        "credit_impairment_charge",
        "cost_to_income",
    }


def test_the_pre_tax_bridge_is_the_after_tax_one_less_what_follows_tax():
    from bankresearch.decomposition.identities import identity_for

    after, before = IDENTITIES["headline_earnings"], IDENTITIES["profit_before_tax"]
    assert set(before.components) == set(after.components) - {"tax"}
    assert not {"minority_interests", "discontinued_ops"} & set(before.optional_components)
    assert identity_for("headline_earnings", pre_tax=True) is before
    assert identity_for("headline_earnings") is after
    assert identity_for("roe", pre_tax=True) is IDENTITIES["roe"]


def test_each_identity_declares_its_required_components():
    for metric_id, spec in IDENTITIES.items():
        assert spec.components, metric_id
        assert spec.change_unit in {"bps", "aud_m"}, metric_id


# --- residual classification ----------------------------------------------


def test_a_residual_within_tolerance_is_rounding_not_a_finding():
    assert (
        classify_residual(Decimal("0.4"), tolerance=Decimal("1"))
        is ResidualClass.DEFINITION_OR_ROUNDING
    )


def test_a_residual_beyond_tolerance_is_unexplained():
    assert classify_residual(Decimal("3.2"), tolerance=Decimal("1")) is ResidualClass.UNEXPLAINED


def test_classification_uses_the_absolute_residual():
    assert classify_residual(Decimal("-3.2"), tolerance=Decimal("1")) is ResidualClass.UNEXPLAINED


def test_exactly_at_tolerance_is_rounding():
    assert (
        classify_residual(Decimal("1"), tolerance=Decimal("1"))
        is ResidualClass.DEFINITION_OR_ROUNDING
    )
