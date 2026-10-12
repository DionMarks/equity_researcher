"""Every request carries its step's output limit, and an answer cut off at it is not a verdict."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from bankresearch.domain.registry import get_registry
from bankresearch.llm.anthropic_client import build_payload
from bankresearch.llm.client import LLMRequest, LLMResponse, ScriptedClient, Usage, cost_usd
from bankresearch.llm.tracing import Tracer


@pytest.fixture
def spec() -> object:
    from bankresearch.domain.models import RequestSpec

    return RequestSpec(
        run_id="r1",
        bank_id="ANZ",
        metric_id="nim",
        period="1H26",
        comparison_period="1H25",
        period_start=date(2025, 10, 1),
        period_end=date(2026, 3, 31),
        comparison_start=date(2024, 10, 1),
        comparison_end=date(2025, 3, 31),
        as_of=date(2026, 5, 31),
    )


def _request(step: str = "explain", **overrides: object) -> LLMRequest:
    base: dict[str, object] = {
        "step": step,
        "model": "claude-fable-5-1",
        "effort": "high",
        "system": "s",
        "messages": ({"role": "user", "content": "u"},),
    }
    base.update(overrides)
    return LLMRequest(**base)  # type: ignore[arg-type]


def test_each_request_carries_its_steps_own_limit():
    registry = get_registry()
    assert build_payload(_request("explain"))["max_tokens"] == registry.output_limit("explain")
    assert build_payload(_request("verify"))["max_tokens"] == registry.output_limit("verify")
    assert registry.output_limit("explain") == 16000


def test_a_limit_named_on_the_request_wins():
    assert build_payload(_request("explain", max_tokens=1000))["max_tokens"] == 1000


def test_a_step_with_no_cap_is_refused_rather_than_sent_with_a_guess():
    with pytest.raises(KeyError, match="no output cap"):
        build_payload(_request("a_step_nobody_configured"))


def test_every_routed_step_has_room_to_think_and_answer():
    registry = get_registry()
    for step in registry.models["routing"]:
        assert registry.output_limit(step) >= 4000, step


def test_every_step_runs_on_fable_at_its_published_price():
    models = get_registry().models
    assert {entry["model"] for entry in models["routing"].values()} == {"claude-fable-5-1"}
    assert models["pricing"]["claude-fable-5-1"] == {
        "input": "10.00",
        "output": "50.00",
        "cache_read": "0.25",
        "verification": "verified",
    }
    assert models["fallback_model"] != "claude-fable-5-1"


def test_discovery_searches_at_medium_effort():
    """At low effort Fable answers from memory more often than it searches, and a cold-path
    search that doesn't search finds nothing."""
    assert get_registry().models["routing"]["discover_classify"]["effort"] == "medium"


def test_a_narrative_cut_off_at_its_limit_is_not_used(spec):
    from bankresearch.agent.steps.explain import write_narrative
    from bankresearch.domain.models import Decomposition, ResolvedInputs

    client = ScriptedClient(
        responses=[
            LLMResponse(
                text="The margin fell 3.0 bps to 1.53% because",
                stop_reason="max_tokens",
                usage=Usage(10, 16000),
            )
        ]
    )
    narrative, note = write_narrative(
        spec,
        Decomposition(metric_id="nim", change_unit="bps", headline_change=Decimal("-3.0")),
        ResolvedInputs(current={"nim": Decimal("1.53")}, prior={"nim": Decimal("1.56")}),
        [],
        client,
    )
    assert narrative == ""
    assert "cut off" in note


def _card():
    from bankresearch.domain.models import EvidenceCard, SourceRef

    return EvidenceCard(
        evidence_id="ev_001",
        component="nim",
        period="1H26",
        value=Decimal("1.89"),
        unit="pct",
        statement="A figure from the results document.",
        source=SourceRef(
            doc_id="wbc-1h26-results-document",
            doc_type="RESULTS_DOCUMENT",
            page=13,
            publish_date=date(2026, 5, 4),
        ),
        method="table",
    )


@pytest.mark.parametrize(
    ("stop_reason", "structured", "why"),
    [
        ("max_tokens", None, "cut off"),
        ("end_turn", None, "no verdicts"),
    ],
)
def test_a_check_with_no_verdicts_keeps_the_memo_and_says_it_was_not_checked(
    stop_reason: str, structured: object, why: str
):
    from bankresearch.agent.steps.explain import check_narrative

    narrative = "The margin fell to 1.89% [ev_001]."
    client = ScriptedClient(
        responses=[LLMResponse(stop_reason=stop_reason, structured=structured)]  # type: ignore[arg-type]
    )
    check = check_narrative(narrative, [_card()], {}, client)
    assert check.narrative == narrative
    assert check.unsupported == [] and check.partial == []
    assert why in check.skipped


def test_the_trace_says_what_each_call_cost():
    tracer = Tracer()
    usage = Usage(input_tokens=2000, output_tokens=1000, cache_read_tokens=4000)
    event = tracer.model_call(_request("verify"), LLMResponse(usage=usage))
    prices = get_registry().models["pricing"]["claude-fable-5-1"]
    assert event["cost_usd"] == str(cost_usd(usage, prices))
    assert Decimal(event["cost_usd"]) == Decimal("0.071")


def test_a_replayed_call_is_traced_as_free():
    tracer = Tracer()
    event = tracer.model_call(_request(), LLMResponse(usage=Usage(10, 10), cached=True))
    assert event["cost_usd"] == "0"


def test_a_driver_cited_by_its_id_is_cited_by_its_evidence():
    """A narrative citing "[cti.income_growth]" escaped the check and printed a raw id."""
    from bankresearch.agent.steps.explain import _cite_drivers
    from bankresearch.domain.models import Decomposition, DriverContribution

    decomposition = Decomposition(
        metric_id="cost_to_income",
        change_unit="bps",
        headline_change=Decimal("-16"),
        drivers=[
            DriverContribution(
                driver_id="cti.income_growth",
                contribution=Decimal("-135.2"),
                evidence_ids=["ev_005", "ev_042"],
            )
        ],
    )
    text = "Income growth took 135.2 bps off the ratio [cti.income_growth]. Costs [ev_009]."
    assert _cite_drivers(text, decomposition) == (
        "Income growth took 135.2 bps off the ratio [ev_005, ev_042]. Costs [ev_009]."
    )
    assert _cite_drivers("See [nim.unknown].", decomposition) == "See [nim.unknown]."


def test_a_period_is_annualised_over_its_own_financial_year():
    """ANZ's 1H24 runs October 2023 to March 2024 and takes in 29 February; annualising it
    over 365 days put its margin at 1.551% against the 1.56% published."""
    from bankresearch.decomposition.identities import nim_pct
    from bankresearch.domain.calendar import fiscal_period

    half = fiscal_period("1H24", 9)
    assert (half.days, half.year_days) == (183, 366)
    assert fiscal_period("1H26", 9).year_days == 365
    margin = nim_pct(Decimal(7899), Decimal(1015621), half.days, half.year_days)
    assert round(float(margin), 2) == 1.56
    assert round(float(nim_pct(Decimal(7899), Decimal(1015621), half.days)), 2) == 1.55
