"""Resolving a request, and the abstention codes it can produce."""

from __future__ import annotations

from datetime import date

import pytest

from bankresearch.agent.state import ABSTAIN_REASONS, STATES, Step
from bankresearch.agent.steps.resolve import AbstainSignal, resolve_request


def test_a_well_formed_request_resolves_to_the_pcp():
    spec = resolve_request("NAB", "1H26", "nim", as_of=date(2026, 5, 10))
    assert spec.bank_id == "NAB"
    assert spec.period == "1H26"
    assert spec.comparison_period == "1H25"
    assert (spec.period_start, spec.period_end) == (date(2025, 10, 1), date(2026, 3, 31))
    assert (spec.comparison_start, spec.comparison_end) == (date(2024, 10, 1), date(2025, 3, 31))


def test_each_issuer_gets_its_own_window_for_the_same_label():
    nab = resolve_request("NAB", "1H26", "nim", as_of=date(2026, 5, 10))
    cba = resolve_request("CBA", "1H26", "nim", as_of=date(2026, 5, 10))
    assert nab.period_start != cba.period_start
    assert cba.period_start == date(2025, 7, 1)


def test_the_comparison_is_never_the_adjacent_half():
    for label, expected in [("1H26", "1H25"), ("2H25", "2H24"), ("FY25", "FY24")]:
        spec = resolve_request("ANZ", label, "cet1_ratio", as_of=date(2026, 5, 10))
        assert spec.comparison_period == expected


def test_a_bank_can_be_named_rather_than_tickered():
    assert resolve_request("Westpac", "FY25", "nim", as_of=date(2026, 1, 1)).bank_id == "WBC"


def test_a_metric_synonym_resolves():
    assert resolve_request("NAB", "1H26", "margin", as_of=date(2026, 5, 10)).metric_id == "nim"
    assert (
        resolve_request("NAB", "1H26", "cash earnings", as_of=date(2026, 5, 10)).metric_id
        == "headline_earnings"
    )


@pytest.mark.parametrize(
    ("asked", "bank_id"),
    [
        ("CommBank", "CBA"),
        ("ANZ Bank", "ANZ"),
        ("Australia and New Zealand Banking Group", "ANZ"),
        ("Judo Bank", "JDO"),
        ("Bendigo Bank", "BEN"),
    ],
)
def test_a_bank_can_be_asked_for_by_its_other_names(asked, bank_id):
    """Each was refused as an unknown bank, though each is how people name it."""
    assert resolve_request(asked, "FY25", "nim", as_of=date(2026, 1, 1)).bank_id == bank_id


@pytest.mark.parametrize(
    ("asked", "metric_id"),
    [
        ("cash_earnings", "headline_earnings"),
        ("cash-earnings", "headline_earnings"),
        ("net_interest_margin", "nim"),
        ("return_on_equity", "roe"),
        ("credit_impairment", "credit_impairment_charge"),
        ("CET1-ratio", "cet1_ratio"),
    ],
)
def test_a_metric_written_like_an_identifier_resolves(asked, metric_id):
    """Written the way the ids are, "cash_earnings" was refused as out of scope though
    "cash earnings" was understood."""
    assert resolve_request("NAB", "1H26", asked, as_of=date(2026, 5, 10)).metric_id == metric_id


def test_a_bank_or_metric_out_of_scope_still_abstains():
    with pytest.raises(AbstainSignal, match="BANK_UNKNOWN"):
        resolve_request("Macquarie", "FY25", "nim", as_of=date(2026, 1, 1))
    with pytest.raises(AbstainSignal, match="METRIC_OUT_OF_SCOPE"):
        resolve_request("NAB", "1H26", "dividend_payout", as_of=date(2026, 5, 10))


def test_period_input_variants_resolve_identically():
    base = resolve_request("NAB", "1H26", "nim", as_of=date(2026, 5, 10))
    for variant in ["1HFY26", "H1 FY26", "HY26", "  1h26 "]:
        assert resolve_request("NAB", variant, "nim", as_of=date(2026, 5, 10)).period == base.period


def test_the_headline_label_is_looked_up_for_the_period():
    early = resolve_request("WBC", "FY22", "headline_earnings", as_of=date(2023, 1, 1))
    later = resolve_request("WBC", "1H23", "headline_earnings", as_of=date(2023, 6, 1))
    assert early.headline_label == "Cash earnings"
    assert later.headline_label == "Net profit"


# --- abstentions ----------------------------------------------------------


def test_calendar_style_input_abstains_as_ambiguous():
    with pytest.raises(AbstainSignal) as exc:
        resolve_request("CBA", "H1 2025", "nim", as_of=date(2026, 1, 1))
    assert exc.value.reason == "AMBIGUOUS_INPUT"
    assert "1H25" in exc.value.detail or "1H26" in exc.value.detail


def test_an_unknown_bank_abstains():
    with pytest.raises(AbstainSignal) as exc:
        resolve_request("Barclays", "1H26", "nim", as_of=date(2026, 5, 10))
    assert exc.value.reason == "BANK_UNKNOWN"


def test_a_metric_out_of_scope_abstains():
    with pytest.raises(AbstainSignal) as exc:
        resolve_request("NAB", "1H26", "dividend yield", as_of=date(2026, 5, 10))
    assert exc.value.reason == "METRIC_OUT_OF_SCOPE"


def test_every_abstention_says_what_is_missing():
    with pytest.raises(AbstainSignal) as exc:
        resolve_request("NAB", "1H26", "dividend yield", as_of=date(2026, 5, 10))
    assert exc.value.detail
    assert "in scope" in exc.value.detail.lower()


def test_the_as_of_date_defaults_to_the_expected_release_month():
    """As-of defaults to the period's results release date."""
    spec = resolve_request("NAB", "1H26", "nim", as_of=None)
    assert spec.as_of >= spec.period_end


def test_a_run_id_is_assigned():
    spec = resolve_request("NAB", "1H26", "nim", as_of=date(2026, 5, 10))
    assert spec.run_id


# --- the state machine ----------------------------------------------------


def test_the_state_machine_has_the_twelve_steps_in_order():
    assert [s.value for s in STATES] == [
        "resolve",
        "availability",
        "discover",
        "locate",
        "extract",
        "validate",
        "reconcile",
        "decompose",
        "explain",
        "score",
        "verify",
        "render",
    ]


def test_every_abstain_reason_in_the_plan_exists():
    assert ABSTAIN_REASONS == (
        "AMBIGUOUS_INPUT",
        "METRIC_OUT_OF_SCOPE",
        "BANK_UNKNOWN",
        "PERIOD_NOT_YET_REPORTED",
        "DOCS_UNAVAILABLE",
        "METRIC_NOT_REPORTED",
        "INSUFFICIENT_EVIDENCE",
    )


def test_only_five_steps_may_call_a_model():
    """Discover, Locate, Extract, Explain and Verify, plus Resolve
    for free text only."""
    llm_steps = {s for s in STATES if s.uses_llm}
    assert llm_steps == {
        Step.DISCOVER,
        Step.LOCATE,
        Step.EXTRACT,
        Step.EXPLAIN,
        Step.VERIFY,
    }


def test_the_deterministic_steps_are_explicitly_model_free():
    for step in (
        Step.AVAILABILITY,
        Step.VALIDATE,
        Step.RECONCILE,
        Step.DECOMPOSE,
        Step.SCORE,
        Step.RENDER,
    ):
        assert not step.uses_llm


def test_every_path_ends_in_render():
    assert Step.RENDER.is_terminal
    assert not Step.RESOLVE.is_terminal


def test_a_metric_an_issuer_does_not_report_abstains_with_its_own_code():
    """METRIC_NOT_REPORTED and INSUFFICIENT_EVIDENCE are different findings. One will
    still be true after more documents are ingested; the other will not."""
    from dataclasses import replace

    from bankresearch.domain.registry import get_registry

    registry = get_registry()
    profile = registry.banks["ANZ"]
    patched = replace(profile, not_reported={"nim": "Its business mix makes it meaningless."})
    registry_with = replace(registry, banks={**registry.banks, "ANZ": patched})

    with pytest.raises(AbstainSignal) as exc:
        resolve_request("ANZ", "1H26", "nim", as_of=date(2026, 5, 31), registry=registry_with)
    assert exc.value.reason == "METRIC_NOT_REPORTED"
    assert "meaningless" in exc.value.detail


def test_all_seven_abstention_codes_are_reachable():
    """Every abstention code must be producible by some step, or it is documentation
    rather than behaviour."""
    import pathlib

    from bankresearch.agent.state import ABSTAIN_REASONS

    expected = {
        "AMBIGUOUS_INPUT",
        "METRIC_OUT_OF_SCOPE",
        "BANK_UNKNOWN",
        "PERIOD_NOT_YET_REPORTED",
        "DOCS_UNAVAILABLE",
        "METRIC_NOT_REPORTED",
        "INSUFFICIENT_EVIDENCE",
    }
    assert set(ABSTAIN_REASONS) == expected

    agent = pathlib.Path(__file__).parents[2] / "src" / "bankresearch" / "agent"
    emitted = "".join(path.read_text(encoding="utf-8") for path in agent.rglob("*.py"))
    for code in expected:
        assert f'"{code}"' in emitted, f"{code} is never emitted by any step"


# --- the request surface a reviewer actually types -----------------------


@pytest.mark.parametrize(
    ("typed", "metric"),
    [
        ("efficiency ratio", "cost_to_income"),
        ("cost income ratio", "cost_to_income"),
        ("operating efficiency", "cost_to_income"),
        ("bad debts", "credit_impairment_charge"),
        ("loan losses", "credit_impairment_charge"),
        ("provisions", "credit_impairment_charge"),
        ("capital ratio", "cet1_ratio"),
        ("interest margin", "nim"),
        ("return on average equity", "roe"),
        ("NIM", "nim"),
        ("Cash Profit", "headline_earnings"),
    ],
)
def test_ordinary_industry_terms_resolve(typed, metric):
    """ "Efficiency ratio" is what bankers usually call cost-to-income, and it wasn't a
    synonym.

    Someone typing the ordinary term got METRIC_OUT_OF_SCOPE from a system that computes it
    for four banks. Requests and documents use different name lists
    (``config/line_items.yaml`` matches row labels inside filings), and the two had drifted
    apart.
    """
    from bankresearch.domain.registry import get_registry

    assert get_registry().metric(typed).metric_id == metric


@pytest.mark.parametrize("typed", ["statutory profit", "net profit after tax", "underlying profit"])
def test_no_alias_names_a_basis(typed):
    """``headline_earnings`` is whatever each issuer leads with: cash for three of the
    majors, and net profit for Westpac since 1H23.

    So "statutory profit" isn't a synonym for it. Mapping it would give the cash figure to
    someone who explicitly asked for the statutory one, which is exactly the basis error the
    system is built to prevent. Requests can't name a basis yet, and until they can,
    refusing is safer than guessing.
    """
    from bankresearch.domain.registry import UnknownMetricError, get_registry

    with pytest.raises(UnknownMetricError):
        get_registry().metric(typed)


@pytest.mark.parametrize("typed", ["dividend yield", "branch count", "tier 2 ratio", "nps"])
def test_a_metric_out_of_scope_is_still_refused(typed):
    """Widening the aliases must not widen the scope."""
    from bankresearch.domain.registry import UnknownMetricError, get_registry

    with pytest.raises(UnknownMetricError):
        get_registry().metric(typed)
