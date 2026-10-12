"""What stands behind the headline figures and each driver, counted off the stored figures and
the evidence cards (ADR-069 and ADR-072)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from bankresearch.agent.steps.resolve import resolve_request
from bankresearch.agent.steps.score import figure_evidence
from bankresearch.domain.models import Fact, ResolvedInputs


def _fact(component: str, period: str, value: str, doc: str, **extra: object) -> Fact:
    unit = "pct" if component.endswith("ratio") or component == "nim" else "aud_m"
    fields: dict[str, object] = {
        "fact_id": f"{doc}:{period}:{component}",
        "bank_id": "NAB",
        "period": period,
        "component": component,
        "value": Decimal(value),
        "published_value": Decimal(value),
        "unit": unit,
        "doc_id": doc,
        "doc_type": "RESULTS_DOCUMENT",
        "page_no": 22,
        "publish_date": date(2026, 5, 7),
        "validated": True,
    }
    fields.update(extra)
    return Fact.model_validate(fields)


def _capital(period: str, ratio: str, capital: str, doc: str) -> list[Fact]:
    return [
        _fact("cet1_ratio", period, ratio, doc),
        _fact("cet1_capital", period, capital, doc),
        _fact("rwa", period, "1000", doc),
    ]


def test_a_second_filing_and_an_identity_are_counted_period_by_period():
    spec = resolve_request("NAB", "1H26", "cet1_ratio")
    facts = [
        *_capital("1H25", "12.00", "120", "nab-1h26-results-document"),
        _fact("cet1_ratio", "1H25", "12.00", "nab-1h25-results-document"),
        *_capital("1H26", "11.50", "115", "nab-1h26-results-document"),
    ]
    inputs = ResolvedInputs(
        current={"cet1_ratio": Decimal("11.50")}, prior={"cet1_ratio": Decimal("12.00")}
    )
    evidence = figure_evidence(spec, facts, inputs, movement=Decimal("-50"))
    assert evidence.primary == 2
    assert evidence.corroborated == 1  # only 1H25 is printed twice
    assert evidence.confirmed == 2  # capital over risk-weighted assets, both periods
    assert not evidence.model_matched and not evidence.restated
    # Printed to two decimal places: half a basis point each, 1bp of a 50bp movement.
    assert evidence.rounding_share == Decimal("0.02")


def test_a_figure_that_disagrees_isnt_counted_as_agreeing():
    spec = resolve_request("NAB", "1H26", "cet1_ratio")
    facts = [
        _fact("cet1_ratio", "1H25", "12.00", "nab-1h26-results-document"),
        _fact("cet1_ratio", "1H25", "12.10", "nab-1h26-investor-presentation"),
        _fact("cet1_ratio", "1H26", "11.50", "nab-1h26-results-document"),
    ]
    inputs = ResolvedInputs(
        current={"cet1_ratio": Decimal("11.50")}, prior={"cet1_ratio": Decimal("12.00")}
    )
    evidence = figure_evidence(spec, facts, inputs, movement=Decimal("-50"))
    assert evidence.corroborated == 0
    assert evidence.confirmed == 0  # no capital or risk-weighted assets to rebuild it from


def test_a_figure_printed_only_in_a_deck_on_a_row_a_model_named_says_both():
    from bankresearch.agent.orchestrator import _found_profile, _register
    from bankresearch.domain.registry import get_registry

    named = (("Net interest margin (average)", "nim"),)
    _register(_found_profile("BBC", "BNK Banking Corporation Limited", 6, (), (), named))
    try:
        spec = resolve_request("BBC", "FY26", "nim")
        deck = {"doc_type": "INVESTOR_PRESENTATION", "bank_id": "BBC"}
        facts = [
            _fact("nim", "FY25", "1.52", "bbc-fy26-investor-presentation", **deck),
            _fact("nim", "FY26", "1.96", "bbc-fy26-investor-presentation", **deck),
        ]
        inputs = ResolvedInputs(current={"nim": Decimal("1.96")}, prior={"nim": Decimal("1.52")})
        evidence = figure_evidence(spec, facts, inputs, movement=Decimal("44"))
        assert evidence.primary == 0
        assert evidence.model_matched
    finally:
        get_registry().banks.pop("BBC", None)


def test_a_flat_figure_counts_all_of_its_rounding():
    spec = resolve_request("NAB", "1H26", "cet1_ratio")
    facts = [
        _fact("cet1_ratio", "1H25", "12.00", "nab-1h26-results-document"),
        _fact("cet1_ratio", "1H26", "12.00", "nab-1h26-results-document"),
    ]
    inputs = ResolvedInputs(
        current={"cet1_ratio": Decimal("12.00")}, prior={"cet1_ratio": Decimal("12.00")}
    )
    evidence = figure_evidence(spec, facts, inputs, movement=Decimal(0))
    assert evidence.rounding_share == Decimal(9)


def test_only_a_stand_in_with_nothing_beneath_it_is_capped_whatever_the_notes_say():
    """CET1 capital read as one figure is a total, not a cause, so it's held below High; the
    risk-weighted assets split by type beside it are causes and aren't. This used to be read
    off the notes by the words "in aggregate", which let a caveat's wording move the scores."""
    from bankresearch.agent.steps.score import score_run
    from bankresearch.domain.models import Decomposition, DriverContribution

    spec = resolve_request("NAB", "1H26", "cet1_ratio")
    inputs = ResolvedInputs(
        current={"cet1_ratio": Decimal("11.50")}, prior={"cet1_ratio": Decimal("12.00")}
    )

    def run(notes: list[str]) -> dict[str, Decimal]:
        decomposition = Decomposition(
            metric_id="cet1_ratio",
            change_unit="bps",
            headline_change=Decimal("-50"),
            drivers=[
                DriverContribution(
                    driver_id="cet1.capital",
                    contribution=Decimal("20"),
                    direction="+",
                    from_components=["cet1_capital", "rwa"],
                ),
                DriverContribution(
                    driver_id="cet1.rwa_credit",
                    contribution=Decimal("-70"),
                    direction="-",
                    from_components=["rwa_credit", "rwa", "cet1_capital"],
                ),
            ],
            notes=notes,
        )
        scored = score_run(spec, decomposition, inputs)
        return {driver_id: score.score for driver_id, score in scored.drivers.items()}

    plain = run([])
    worded = run(["The capital movement is attributed in aggregate: no breakdown was read."])
    assert plain == worded
    assert plain["cet1.capital"] == plain["cet1.rwa_credit"] - Decimal("0.15")
    assert plain["cet1.capital"] < Decimal("0.85")  # never High
