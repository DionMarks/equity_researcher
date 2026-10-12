"""Path B per metric, taxonomy mapping and sign conventions."""

from __future__ import annotations

from decimal import Decimal

import pytest

from bankresearch.decomposition.metrics import decompose_path_b
from bankresearch.decomposition.taxonomy import map_label, roll_up, validate_driver_id

HALF = 182
PCP_HALF = 182


def sums_to_total(result) -> bool:
    """Top-level drivers plus the residual must equal the movement exactly.

    Nested drivers (those with a ``parent_id``) are detail under a top-level line and are
    left out on purpose, since they break down a different identity and would otherwise be
    double counted.
    """
    quantified = sum(
        (
            d.contribution
            for d in result.drivers
            if d.contribution is not None and d.parent_id is None
        ),
        Decimal(0),
    )
    return abs(quantified + result.residual - result.headline_change) < Decimal("1e-9")


# --- taxonomy -------------------------------------------------------------


def test_an_exact_synonym_maps_to_its_canonical_driver():
    assert map_label("nim", "deposit pricing") == "nim.deposit_pricing"
    assert map_label("nim", "Replicating portfolio") == "nim.hedging"
    assert map_label("cet1_ratio", "organic capital generation") == "cet1.organic_capital"


def test_matching_ignores_case_and_punctuation():
    assert map_label("cet1_ratio", "Credit RWA") == "cet1.rwa_credit"
    assert map_label("credit_impairment_charge", "Write-Backs") == "cic.individual"
    assert map_label("cost_to_income", "Cost   growth") == "cti.expense_growth"


def test_a_canonical_id_maps_to_itself():
    assert map_label("nim", "nim.deposit_pricing") == "nim.deposit_pricing"


def test_an_unrecognised_label_is_left_unmapped_rather_than_forced():
    """A wrong mapping attributes a real number to the wrong cause."""
    assert map_label("nim", "something the taxonomy has never seen") is None


def test_synonyms_do_not_leak_between_metrics():
    assert map_label("nim", "organic capital generation") is None


def test_children_roll_up_to_their_parent():
    assert roll_up("cost_to_income", "cti.exp_wages_inflation") == "cti.expense_growth"
    assert roll_up("cost_to_income", "cti.inc_nii") == "cti.income_growth"
    assert roll_up("nim", "nim.deposit_pricing") is None


def test_driver_ids_are_validated_against_the_taxonomy():
    assert validate_driver_id("nim", "nim.deposit_pricing")
    assert validate_driver_id("nim", "nim.residual")
    assert validate_driver_id("nim", "nim.other")
    assert not validate_driver_id("nim", "nim.invented")
    assert not validate_driver_id("nim", "cet1.buyback")


# --- CET1 -----------------------------------------------------------------


def test_cet1_capital_and_rwa_effects_sum_to_the_movement():
    result = decompose_path_b(
        "cet1_ratio",
        prior={"cet1_capital": Decimal("50000"), "rwa": Decimal("400000")},
        current={"cet1_capital": Decimal("52000"), "rwa": Decimal("410000")},
        reported_prior=Decimal("12.50"),
        reported_current=Decimal("12.68"),
    )
    assert result.change_unit == "bps"
    assert sums_to_total(result)


def test_rwa_growth_reduces_the_capital_ratio():
    result = decompose_path_b(
        "cet1_ratio",
        prior={"cet1_capital": Decimal("50000"), "rwa": Decimal("400000")},
        current={"cet1_capital": Decimal("50000"), "rwa": Decimal("420000")},
        reported_prior=Decimal("12.50"),
        reported_current=Decimal("11.90"),
    )
    rwa_drivers = [d for d in result.drivers if "rwa" in d.driver_id]
    assert rwa_drivers
    assert all(d.contribution < 0 for d in rwa_drivers if d.contribution is not None)


def test_rwa_splits_by_risk_type_when_disclosed():
    result = decompose_path_b(
        "cet1_ratio",
        prior={
            "cet1_capital": Decimal("50000"),
            "rwa": Decimal("400000"),
            "rwa_credit": Decimal("330000"),
            "rwa_irrbb": Decimal("20000"),
            "rwa_market": Decimal("10000"),
            "rwa_operational": Decimal("40000"),
        },
        current={
            "cet1_capital": Decimal("52000"),
            "rwa": Decimal("410000"),
            "rwa_credit": Decimal("338000"),
            "rwa_irrbb": Decimal("21000"),
            "rwa_market": Decimal("11000"),
            "rwa_operational": Decimal("40000"),
        },
        reported_prior=Decimal("12.50"),
        reported_current=Decimal("12.68"),
    )
    ids = {d.driver_id for d in result.drivers}
    assert {"cet1.rwa_credit", "cet1.rwa_irrbb", "cet1.rwa_market"} <= ids
    # Operational RWA did not move, so it must contribute exactly zero.
    operational = next(d for d in result.drivers if d.driver_id == "cet1.rwa_operational")
    assert operational.contribution == Decimal("0")
    assert sums_to_total(result)


def test_capital_splits_into_the_disclosed_movements():
    result = decompose_path_b(
        "cet1_ratio",
        prior={"cet1_capital": Decimal("50000"), "rwa": Decimal("400000")},
        current={
            "cet1_capital": Decimal("52000"),
            "rwa": Decimal("400000"),
            "organic_capital": Decimal("3500"),
            "dividends_net_drp": Decimal("-1200"),
            "buybacks": Decimal("-500"),
            "deductions": Decimal("200"),
        },
        reported_prior=Decimal("12.50"),
        reported_current=Decimal("13.00"),
    )
    ids = {d.driver_id for d in result.drivers}
    assert {"cet1.organic_capital", "cet1.dividends", "cet1.buyback"} <= ids
    assert sums_to_total(result)


# --- cost-to-income -------------------------------------------------------


def test_higher_expenses_push_cost_to_income_up():
    """Sign convention: higher expenses are a POSITIVE cost-to-income contribution."""
    result = decompose_path_b(
        "cost_to_income",
        prior={"operating_expenses": Decimal("6000"), "operating_income": Decimal("15000")},
        current={"operating_expenses": Decimal("6300"), "operating_income": Decimal("15000")},
        reported_prior=Decimal("40.00"),
        reported_current=Decimal("42.00"),
    )
    expense = next(d for d in result.drivers if d.driver_id == "cti.expense_growth")
    assert expense.contribution > 0
    assert sums_to_total(result)


def test_higher_income_pushes_cost_to_income_down():
    result = decompose_path_b(
        "cost_to_income",
        prior={"operating_expenses": Decimal("6000"), "operating_income": Decimal("15000")},
        current={"operating_expenses": Decimal("6000"), "operating_income": Decimal("16000")},
        reported_prior=Decimal("40.00"),
        reported_current=Decimal("37.50"),
    )
    income = next(d for d in result.drivers if d.driver_id == "cti.income_growth")
    assert income.contribution < 0


def test_income_splits_into_nii_and_other_income():
    result = decompose_path_b(
        "cost_to_income",
        prior={
            "operating_expenses": Decimal("6000"),
            "operating_income": Decimal("15000"),
            "nii": Decimal("12000"),
            "other_income": Decimal("3000"),
        },
        current={
            "operating_expenses": Decimal("6300"),
            "operating_income": Decimal("16000"),
            "nii": Decimal("12600"),
            "other_income": Decimal("3400"),
        },
        reported_prior=Decimal("40.00"),
        reported_current=Decimal("39.375"),
    )
    ids = {d.driver_id for d in result.drivers}
    assert {"cti.inc_nii", "cti.inc_other_income"} <= ids
    nii = next(d for d in result.drivers if d.driver_id == "cti.inc_nii")
    assert nii.parent_id == "cti.income_growth"
    assert sums_to_total(result)


# --- earnings -------------------------------------------------------------


def test_the_earnings_bridge_attributes_each_line():
    result = decompose_path_b(
        "headline_earnings",
        prior={
            "nii": Decimal("9000"),
            "other_income": Decimal("2500"),
            "operating_expenses": Decimal("5000"),
            "credit_impairment": Decimal("400"),
            "tax": Decimal("1800"),
        },
        current={
            "nii": Decimal("9600"),
            "other_income": Decimal("2400"),
            "operating_expenses": Decimal("5200"),
            "credit_impairment": Decimal("500"),
            "tax": Decimal("1850"),
        },
        reported_prior=Decimal("4300"),
        reported_current=Decimal("4450"),
        days=HALF,
        pcp_days=PCP_HALF,
    )
    by_id = {d.driver_id: d.contribution for d in result.drivers}
    assert by_id["earn.other_income"] == Decimal("-100")
    assert by_id["earn.operating_expenses"] == Decimal("-200")
    assert by_id["earn.credit_impairment"] == Decimal("-100")
    assert by_id["earn.tax"] == Decimal("-50")
    assert sums_to_total(result)


def test_net_interest_income_splits_into_volume_margin_and_day_count():
    result = decompose_path_b(
        "headline_earnings",
        prior={
            "nii": Decimal("9000"),
            "other_income": Decimal("2500"),
            "operating_expenses": Decimal("5000"),
            "credit_impairment": Decimal("400"),
            "tax": Decimal("1800"),
            "nim": Decimal("1.80"),
            "aiea": Decimal("1000000"),
        },
        current={
            "nii": Decimal("9600"),
            "other_income": Decimal("2500"),
            "operating_expenses": Decimal("5000"),
            "credit_impairment": Decimal("400"),
            "tax": Decimal("1800"),
            "nim": Decimal("1.86"),
            "aiea": Decimal("1030000"),
        },
        reported_prior=Decimal("4300"),
        reported_current=Decimal("4900"),
        days=183,
        pcp_days=182,
    )
    ids = {d.driver_id for d in result.drivers}
    assert {"earn.nii_volume", "earn.nii_margin", "earn.nii_day_count"} <= ids
    day_count = next(d for d in result.drivers if d.driver_id == "earn.nii_day_count")
    assert day_count.contribution > 0  # one extra day of income
    assert all(
        d.parent_id is None or d.parent_id == "earn.nii_margin" or d.parent_id
        for d in result.drivers
    )


# --- ROE, impairment, NIM -------------------------------------------------


def test_roe_splits_into_earnings_and_equity_effects():
    result = decompose_path_b(
        "roe",
        prior={"headline_earnings": Decimal("5000"), "average_equity": Decimal("70000")},
        current={"headline_earnings": Decimal("5400"), "average_equity": Decimal("72000")},
        reported_prior=Decimal("14.33"),
        reported_current=Decimal("15.03"),
        days=HALF,
        pcp_days=PCP_HALF,
    )
    ids = {d.driver_id for d in result.drivers}
    assert ids == {"roe.earnings", "roe.equity"}
    earnings = next(d for d in result.drivers if d.driver_id == "roe.earnings")
    equity = next(d for d in result.drivers if d.driver_id == "roe.equity")
    assert earnings.contribution > 0
    assert equity.contribution < 0  # a bigger denominator reduces the return
    assert sums_to_total(result)


def test_the_earnings_lines_under_return_on_equity_are_in_basis_points_of_return():
    """ANZ's 1H26 memo said lower costs "added 208.0 bps" to its return on equity: the
    lines were A$m movements shown under a parent in basis points. Each is its share of the
    earnings effect, at the same rate, and together they make up the parent."""
    from bankresearch.domain.registry import get_registry

    bridge = {
        "nii": "9000",
        "other_income": "2000",
        "operating_expenses": "5000",
        "credit_impairment": "300",
        "tax": "1700",
    }
    moved = {
        "nii": "9100",
        "other_income": "2200",
        "operating_expenses": "4800",
        "credit_impairment": "400",
        "tax": "1800",
    }
    prior = {"headline_earnings": Decimal("4000"), "average_equity": Decimal("70000")}
    current = {"headline_earnings": Decimal("4300"), "average_equity": Decimal("71000")}
    result = decompose_path_b(
        "roe",
        prior={**prior, **{k: Decimal(v) for k, v in bridge.items()}},
        current={**current, **{k: Decimal(v) for k, v in moved.items()}},
        reported_prior=Decimal("11.43"),
        reported_current=Decimal("12.11"),
        days=HALF,
        pcp_days=HALF,
    )
    earnings = next(d for d in result.drivers if d.driver_id == "roe.earnings")
    lines = {d.driver_id: d.contribution for d in result.drivers if d.parent_id == "roe.earnings"}
    assert set(lines) == {f"earn.{name}" for name in bridge}
    assert earnings.contribution is not None
    # A$200m less in costs is worth far less than 200bps of return on A$70bn of equity.
    assert Decimal("0") < lines["earn.operating_expenses"] < Decimal("100")
    assert abs(sum(lines.values()) - earnings.contribution) < Decimal("0.01")
    assert get_registry().driver_label("roe", "earn.nii") == "Net interest income"


def test_a_buyback_raises_return_on_equity():
    result = decompose_path_b(
        "roe",
        prior={"headline_earnings": Decimal("5000"), "average_equity": Decimal("72000")},
        current={"headline_earnings": Decimal("5000"), "average_equity": Decimal("70000")},
        reported_prior=Decimal("13.93"),
        reported_current=Decimal("14.33"),
        days=HALF,
        pcp_days=PCP_HALF,
    )
    equity = next(d for d in result.drivers if d.driver_id == "roe.equity")
    assert equity.contribution > 0


def test_credit_impairment_is_attributed_additively():
    result = decompose_path_b(
        "credit_impairment_charge",
        prior={"impairment_individual": Decimal("250"), "impairment_collective": Decimal("150")},
        current={"impairment_individual": Decimal("300"), "impairment_collective": Decimal("90")},
        reported_prior=Decimal("400"),
        reported_current=Decimal("390"),
    )
    by_id = {d.driver_id: d.contribution for d in result.drivers}
    assert by_id["cic.individual"] == Decimal("50")
    assert by_id["cic.collective"] == Decimal("-60")
    assert sums_to_total(result)


def test_the_collective_charge_splits_when_disclosed():
    result = decompose_path_b(
        "credit_impairment_charge",
        prior={
            "impairment_individual": Decimal("250"),
            "impairment_collective": Decimal("150"),
            "collective_volume": Decimal("60"),
            "collective_quality": Decimal("50"),
            "collective_outlook": Decimal("30"),
            "overlays": Decimal("10"),
        },
        current={
            "impairment_individual": Decimal("300"),
            "impairment_collective": Decimal("90"),
            "collective_volume": Decimal("40"),
            "collective_quality": Decimal("35"),
            "collective_outlook": Decimal("10"),
            "overlays": Decimal("5"),
        },
        reported_prior=Decimal("400"),
        reported_current=Decimal("390"),
    )
    ids = {d.driver_id for d in result.drivers}
    assert {"cic.collective_volume", "cic.collective_quality", "cic.overlays"} <= ids
    assert sums_to_total(result)


def test_nim_path_b_cross_checks_mix_against_rate():
    result = decompose_path_b(
        "nim",
        prior={
            "nii": Decimal("9000"),
            "aiea": Decimal("1000000"),
            "divisional_weights": {"retail": Decimal("0.6"), "business": Decimal("0.4")},
            "divisional_margins": {"retail": Decimal("1.70"), "business": Decimal("1.95")},
        },
        current={
            "nii": Decimal("9600"),
            "aiea": Decimal("1030000"),
            "divisional_weights": {"retail": Decimal("0.58"), "business": Decimal("0.42")},
            "divisional_margins": {"retail": Decimal("1.76"), "business": Decimal("2.00")},
        },
        reported_prior=Decimal("1.80"),
        reported_current=Decimal("1.86"),
        days=182,
        pcp_days=182,
    )
    ids = {d.driver_id for d in result.drivers}
    assert "nim.asset_mix" in ids
    assert sums_to_total(result)


# --- residual -------------------------------------------------------------


def test_a_gap_between_reported_and_recomputed_becomes_the_residual():
    result = decompose_path_b(
        "cet1_ratio",
        prior={"cet1_capital": Decimal("50000"), "rwa": Decimal("400000")},
        current={"cet1_capital": Decimal("52000"), "rwa": Decimal("410000")},
        # Inconsistent with the components on purpose, by 10bp.
        reported_prior=Decimal("12.50"),
        reported_current=Decimal("12.78"),
    )
    assert result.residual != Decimal("0")
    assert sums_to_total(result)


def test_missing_required_components_are_reported_not_guessed():
    result = decompose_path_b(
        "cet1_ratio",
        prior={"cet1_capital": Decimal("50000")},
        current={"cet1_capital": Decimal("52000")},
        reported_prior=Decimal("12.50"),
        reported_current=Decimal("12.68"),
    )
    assert "rwa" in result.missing_inputs
    assert result.drivers == []


def test_an_unknown_metric_is_refused():
    with pytest.raises(KeyError):
        decompose_path_b(
            "not_a_metric",
            prior={},
            current={},
            reported_prior=Decimal("1"),
            reported_current=Decimal("2"),
        )


# --- the two deterministic mapping tiers (DESIGN.md ADR-011) ---------------


def test_tier_one_matches_a_canonical_label_exactly():
    assert map_label("credit_impairment_charge", "Collective provision, portfolio growth") == (
        "cic.collective_volume"
    )


def test_tier_two_matches_a_synonym_inside_a_longer_label():
    """Issuers write more than the taxonomy does."""
    assert map_label("nim", "Replicating portfolio and capital hedge") == "nim.hedging"
    assert map_label("nim", "Wholesale and term funding costs") == "nim.funding_costs"
    assert map_label("cet1_ratio", "Organic capital generation and other") == (
        "cet1.organic_capital"
    )
    assert map_label("cet1_ratio", "On-market share buyback") == "cet1.buyback"


def test_tier_two_refuses_when_two_drivers_match():
    """ "Deposit pricing and funding costs" names two causes; picking one would
    attribute a real number to a cause the issuer did not state."""
    assert map_label("nim", "Deposit pricing and funding costs") is None


def test_a_short_token_does_not_match_inside_a_longer_label():
    """A two- or three-letter abbreviation matches far too loosely as a phrase."""
    assert map_label("headline_earnings", "NII") == "earn.nii"
    assert map_label("cet1_ratio", "Movement in RWA") is None
    # The longer phrase still counts: "model changes" is the "model change" synonym, and
    # the bar is a methodology change whatever it moved.
    assert map_label("cet1_ratio", "Movement in RWA from model changes") == "cet1.regulatory"


def test_chart_labels_from_the_four_majors_map_to_their_drivers():
    """The wording on the real margin and capital walks, as printed.

    Westpac's walk (wbc-1h26-results-document p.13) labels its bars "Loans", "Deposits",
    "WSF", "Capital & other" and "T&M", with the abbreviations spelled out in footnotes; ANZ
    writes "Assets pricing" (anz-1h26-investor-presentation p.25). None of these mapped,
    so all but one of Westpac's seven bars would have been reported as "other".
    """
    expected = {
        "Loans": "nim.lending_pricing",
        "Assets pricing": "nim.lending_pricing",
        "Deposits": "nim.deposit_pricing",
        "Deposits pricing": "nim.deposit_pricing",
        "WSF": "nim.funding_costs",
        "WSF (wholesale funding costs)": "nim.funding_costs",
        "Liquid assets": "nim.liquids",
        "Group Centre liquids": "nim.liquids",
        "Capital & other": "nim.hedging",
        "Capital, Replicating and Other": "nim.hedging",
        "Capital & replicating portfolio": "nim.hedging",
        "T&M": "nim.markets_treasury",
        "T&M (Treasury & Markets contribution)": "nim.markets_treasury",
        "Treasury & Markets": "nim.markets_treasury",
        "Markets activities": "nim.markets_treasury",
        "Portfolio mix": "nim.asset_mix",
    }
    for label, driver_id in expected.items():
        assert map_label("nim", label) == driver_id, label
    assert map_label("cet1_ratio", "Net profit") == "cet1.organic_capital"


def test_a_label_whose_parts_name_different_drivers_stays_unmapped():
    """ANZ's "Assets & funding mix" holds a deposit-mix synonym but covers asset mix too."""
    assert map_label("nim", "Assets & funding mix") is None
    assert map_label("nim", "Deposit pricing and funding costs") is None
    # Bars the taxonomy has no driver for stay unmapped rather than taking the nearest one.
    assert map_label("nim", "Timing difference") is None
    assert map_label("nim", "Basis risk") is None


def test_an_aggregate_node_is_never_matched_by_its_label():
    """A walk bar labelled "Net interest income" is an earnings walk, not a margin driver."""
    assert map_label("nim", "Net interest income") is None
    assert map_label("nim", "Average interest-earning assets") is None


def test_aggregate_synonyms_are_specific_enough_not_to_block_their_children():
    """A bare "capital" or "risk weighted assets" would make the phrase tier
    refuse almost every capital label as ambiguous."""
    assert map_label("cet1_ratio", "Credit risk weighted assets") == "cet1.rwa_credit"
    assert map_label("cet1_ratio", "Interest rate risk in the banking book") == "cet1.rwa_irrbb"
    assert map_label("cet1_ratio", "Total risk weighted assets") == "cet1.rwa"


# --- invariant 2: a computed driver records what it was computed from ------

_ALL_METRICS = {
    "cet1_ratio": (
        dict(
            cet1_capital=Decimal(48840),
            rwa=Decimal(400000),
            rwa_credit=Decimal(330000),
            rwa_irrbb=Decimal(20000),
            rwa_market=Decimal(10000),
            rwa_operational=Decimal(40000),
        ),
        dict(
            cet1_capital=Decimal(50799),
            rwa=Decimal(410000),
            rwa_credit=Decimal(338000),
            rwa_irrbb=Decimal(21000),
            rwa_market=Decimal(11000),
            rwa_operational=Decimal(40000),
        ),
        Decimal("12.21"),
        Decimal("12.39"),
        None,
        None,
    ),
    "cost_to_income": (
        dict(
            operating_expenses=Decimal(5000),
            operating_income=Decimal(11500),
            nii=Decimal(9000),
            other_income=Decimal(2500),
        ),
        dict(
            operating_expenses=Decimal(5200),
            operating_income=Decimal(12000),
            nii=Decimal(9600),
            other_income=Decimal(2400),
        ),
        Decimal("43.48"),
        Decimal("43.33"),
        None,
        None,
    ),
    "headline_earnings": (
        dict(
            nii=Decimal(9000),
            other_income=Decimal(2500),
            operating_expenses=Decimal(5000),
            credit_impairment=Decimal(400),
            tax=Decimal(1800),
            nim=Decimal("1.80"),
            aiea=Decimal(1003000),
        ),
        dict(
            nii=Decimal(9600),
            other_income=Decimal(2400),
            operating_expenses=Decimal(5200),
            credit_impairment=Decimal(500),
            tax=Decimal(1850),
            nim=Decimal("1.86"),
            aiea=Decimal(1035000),
        ),
        Decimal(4300),
        Decimal(4450),
        182,
        182,
    ),
    "roe": (
        dict(headline_earnings=Decimal(4300), average_equity=Decimal(60180)),
        dict(headline_earnings=Decimal(4450), average_equity=Decimal(59380)),
        Decimal("14.33"),
        Decimal("15.03"),
        182,
        182,
    ),
    "credit_impairment_charge": (
        dict(impairment_individual=Decimal(250), impairment_collective=Decimal(150)),
        dict(impairment_individual=Decimal(300), impairment_collective=Decimal(90)),
        Decimal(400),
        Decimal(390),
        None,
        None,
    ),
    "nim": (
        dict(
            nii=Decimal(9000),
            aiea=Decimal(1003000),
            divisional_weights={"retail": Decimal("0.6"), "business": Decimal("0.4")},
            divisional_margins={"retail": Decimal("1.70"), "business": Decimal("1.95")},
        ),
        dict(
            nii=Decimal(9600),
            aiea=Decimal(1035000),
            divisional_weights={"retail": Decimal("0.58"), "business": Decimal("0.42")},
            divisional_margins={"retail": Decimal("1.76"), "business": Decimal("2.00")},
        ),
        Decimal("1.80"),
        Decimal("1.86"),
        182,
        182,
    ),
}


@pytest.mark.parametrize("metric_id", sorted(_ALL_METRICS))
def test_every_computed_driver_records_the_components_behind_it(metric_id):
    """Invariant 2: a number produced by arithmetic is only as traceable as its
    inputs, so each driver names them and the orchestrator turns them into
    citations."""
    prior, current, reported_prior, reported_current, days, pcp_days = _ALL_METRICS[metric_id]
    result = decompose_path_b(
        metric_id,
        prior=prior,
        current=current,
        reported_prior=reported_prior,
        reported_current=reported_current,
        days=days,
        pcp_days=pcp_days,
    )
    assert result.drivers
    uncited = [d.driver_id for d in result.drivers if not d.from_components]
    assert uncited == []


@pytest.mark.parametrize("metric_id", sorted(_ALL_METRICS))
def test_every_metric_reconciles_to_its_reported_movement(metric_id):
    prior, current, reported_prior, reported_current, days, pcp_days = _ALL_METRICS[metric_id]
    result = decompose_path_b(
        metric_id,
        prior=prior,
        current=current,
        reported_prior=reported_prior,
        reported_current=reported_current,
        days=days,
        pcp_days=pcp_days,
    )
    assert sums_to_total(result)


# --- the reconciliation tolerance -----------------------------------------


def test_the_tolerance_includes_the_reported_figures_own_rounding():
    """A movement is the difference of two published figures, so it inherits their
    precision. ANZ publishes CET1 to one decimal place, which makes a 60bp movement carry
    ten basis points of rounding before any driver is considered."""
    from bankresearch.decomposition.metrics import reconciliation_tolerance

    one_decimal = reconciliation_tolerance("bps", 2, Decimal("11.8"), Decimal("12.4"), "pct")
    two_decimals = reconciliation_tolerance("bps", 2, Decimal("11.80"), Decimal("12.40"), "pct")
    assert one_decimal == Decimal("11.0")  # 1.0 from the steps + 5.0 + 5.0
    assert two_decimals == Decimal("2.0")  # 1.0 from the steps + 0.5 + 0.5
    assert one_decimal > two_decimals


def test_a_coarsely_published_ratio_does_not_report_rounding_as_unexplained():
    """The defect this fixes: a 1bp residual against figures published to one decimal
    place was reported as beyond tolerance."""
    result = decompose_path_b(
        "cet1_ratio",
        prior={"cet1_capital": Decimal("55229"), "rwa": Decimal("469000")},
        current={"cet1_capital": Decimal("57472"), "rwa": Decimal("464000")},
        reported_prior=Decimal("11.8"),
        reported_current=Decimal("12.4"),
    )
    assert abs(result.residual) < Decimal("12")
    assert result.within_tolerance


def test_the_tolerance_still_grows_with_the_number_of_drivers():
    from bankresearch.decomposition.metrics import reconciliation_tolerance

    few = reconciliation_tolerance("bps", 1, Decimal("12.39"), Decimal("12.40"), "pct")
    many = reconciliation_tolerance("bps", 10, Decimal("12.39"), Decimal("12.40"), "pct")
    assert many > few


def test_an_amount_movement_uses_aud_tolerances():
    from bankresearch.decomposition.metrics import reconciliation_tolerance

    tolerance = reconciliation_tolerance("aud_m", 5, Decimal("4300"), Decimal("4450"), "aud_m")
    assert tolerance == Decimal("3.5")  # max(2, 2.5) from steps + 0.5 + 0.5
