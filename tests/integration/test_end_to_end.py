"""The whole deterministic pipeline on one document, with no model."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from bankresearch.agent.orchestrator import Agent
from bankresearch.domain.models import DocMeta
from bankresearch.ingestion.pipeline import ingest_document
from bankresearch.memory.db import connect
from bankresearch.memory.fact_store import FactStore
from bankresearch.output.render_md import render_markdown
from bankresearch.settings import Settings

FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthetic_results_document.pdf"
PUBLISHED = date(2026, 5, 1)
AS_OF = date(2026, 5, 31)


@pytest.fixture(scope="module")
def ingested(tmp_path_factory):
    """A store and settings with the fixture ingested as XMPL 1H26."""
    root = tmp_path_factory.mktemp("bankresearch")
    settings = Settings(
        config_dir=Path(__file__).parents[2] / "config",
        data_dir=root / "data",
        runs_dir=root / "runs",
        evals_dir=root / "evals",
    )
    store = FactStore(connect(settings.db_path))
    meta = DocMeta(
        doc_id="xmpl-1h26-results-document",
        bank_id="XMPL",
        period="1H26",
        doc_type="RESULTS_DOCUMENT",
        publish_date=PUBLISHED,
        url=FIXTURE.resolve().as_uri(),
    )
    report = ingest_document(FIXTURE, meta, store, processed_root=settings.processed_dir)
    return settings, store, report


# --- ingestion acceptance ------------------------------------


def test_the_page_count_matches_the_pdf(ingested):
    _, _, report = ingested
    assert report.page_count == 5


def test_every_identity_check_passes(ingested):
    _, _, report = ingested
    assert report.checks_run >= 10
    assert report.checks_passed == report.checks_run


def test_nothing_is_quarantined_from_a_consistent_document(ingested):
    _, _, report = ingested
    assert report.facts_quarantined == 0
    assert report.facts_stored >= 35


def test_a_headline_label_is_recorded_for_every_metric_found(ingested):
    _, store, report = ingested
    assert report.headline_labels == 6
    label = store.headline_label("XMPL", "1H26", "cet1_ratio")
    assert label is not None
    assert label.page_no >= 1


def test_processed_artefacts_are_written_so_a_query_never_parses_a_pdf(ingested):
    settings, _, _ = ingested
    target = settings.processed_dir / "xmpl-1h26-results-document"
    assert {p.name for p in target.iterdir()} == {"pages.jsonl", "tables.jsonl", "outline.json"}


# --- a walk-driven run (Path A) -------------------------------------------


@pytest.fixture(scope="module")
def nim_run(ingested):
    settings, store, _ = ingested
    return Agent(store=store, settings=settings).run("XMPL", "1H26", "nim", as_of=AS_OF)


def test_the_nim_run_succeeds(nim_run):
    assert nim_run.result.status == "OK"
    assert nim_run.exit_code == 0


def test_the_movement_is_struck_against_the_prior_corresponding_period(nim_run):
    result = nim_run.result
    assert result.request.comparison_period == "1H25"
    assert result.prior.value == Decimal("1.80")
    assert result.current.value == Decimal("1.86")
    assert result.change.absolute == Decimal("6.00")


def test_the_issuers_own_walk_is_used_where_it_published_one(nim_run):
    methods = {driver.method for driver in nim_run.result.drivers}
    assert methods == {"management_walk"}


def test_the_walk_drivers_map_onto_the_canonical_taxonomy(nim_run):
    ids = {driver.driver_id for driver in nim_run.result.drivers}
    assert ids == {
        "nim.lending_pricing",
        "nim.deposit_pricing",
        "nim.hedging",
        "nim.funding_costs",
    }


def test_the_walk_reconciles_to_the_reported_movement(nim_run):
    reconciliation = nim_run.result.reconciliation
    assert reconciliation.within_tolerance
    assert reconciliation.sum_of_drivers == Decimal("6.0")
    assert abs(reconciliation.residual) <= reconciliation.tolerance


def test_every_driver_cites_evidence_that_exists(nim_run):
    known = {card.evidence_id for card in nim_run.result.evidence}
    for driver in nim_run.result.drivers:
        assert driver.evidence_ids, driver.driver_id
        assert set(driver.evidence_ids) <= known, driver.driver_id


def test_every_cited_card_quotes_the_cell_it_came_from(nim_run):
    cited = nim_run.result.cited_evidence_ids()
    for card in nim_run.result.evidence:
        if card.evidence_id in cited:
            assert card.quote, card.evidence_id
            assert "|" in card.quote


def test_the_restated_comparative_is_disclosed(nim_run):
    assert nim_run.result.prior.restated is True
    assert any("restated" in caveat for caveat in nim_run.result.caveats)


def test_no_model_was_called_and_the_memo_says_so(nim_run):
    assert nim_run.result.diagnostics.llm_calls == 0
    assert any("No model configured" in w for w in nim_run.result.diagnostics.warnings)
    assert nim_run.result.narrative == ""


def test_the_run_writes_its_three_artefacts(nim_run):
    assert nim_run.artefacts is not None
    assert {p.name for p in nim_run.artefacts.iterdir()} == {
        "result.json",
        "result.md",
        "trace.jsonl",
    }


def test_the_trace_records_every_step(nim_run):
    steps = [event["step"] for event in nim_run.trace]
    for step in ("resolve", "availability", "extract", "reconcile", "decompose", "render"):
        assert step in steps, step


def test_the_memo_renders_with_the_comparison_basis_stated(nim_run):
    markdown = render_markdown(nim_run.result)
    assert "same period last financial year" in markdown
    assert "| Net interest margin | 1.80% | 1.86% | +6.0 bps |" in markdown


# --- an identity-driven run (Path B) --------------------------------------


@pytest.fixture(scope="module")
def cet1_run(ingested):
    settings, store, _ = ingested
    return Agent(store=store, settings=settings).run("XMPL", "1H26", "cet1_ratio", as_of=AS_OF)


def test_the_cet1_run_succeeds(cet1_run):
    assert cet1_run.result.status == "OK"


def test_the_margin_walk_is_not_used_for_a_capital_question(cet1_run):
    """A margin waterfall and a capital walk are the same shape; applying one to the
    other reconciles against nothing and explains nothing."""
    assert {driver.method for driver in cet1_run.result.drivers} == {"computed"}
    assert not any(driver.driver_id.startswith("nim.") for driver in cet1_run.result.drivers)


def test_capital_and_risk_weighted_assets_are_attributed_separately(cet1_run):
    ids = {driver.driver_id for driver in cet1_run.result.drivers}
    assert "cet1.capital" in ids
    assert {"cet1.rwa_credit", "cet1.rwa_irrbb", "cet1.rwa_market"} <= ids


def test_risk_weighted_asset_growth_reduces_the_ratio(cet1_run):
    rwa = [d for d in cet1_run.result.drivers if d.driver_id.startswith("cet1.rwa")]
    assert rwa
    moved = [d for d in rwa if d.contribution != 0]
    assert all(d.contribution < 0 for d in moved)


def test_a_component_that_did_not_move_contributes_exactly_zero(cet1_run):
    operational = next(d for d in cet1_run.result.drivers if d.driver_id == "cet1.rwa_operational")
    assert operational.contribution == Decimal("0")


def test_the_identity_reconciles_to_the_reported_movement(cet1_run):
    reconciliation = cet1_run.result.reconciliation
    assert reconciliation.headline_change == Decimal("18.00")
    assert reconciliation.within_tolerance


def test_every_computed_driver_cites_the_figures_behind_it(cet1_run):
    known = {card.evidence_id for card in cet1_run.result.evidence}
    for driver in cet1_run.result.drivers:
        assert driver.evidence_ids, driver.driver_id
        assert set(driver.evidence_ids) <= known


# --- abstentions still produce a result -----------------------------------


def test_a_period_before_its_release_abstains_rather_than_guessing(ingested):
    settings, store, _ = ingested
    outcome = Agent(store=store, settings=settings).run(
        "XMPL", "FY26", "nim", as_of=date(2026, 10, 6)
    )
    assert outcome.result.status == "ABSTAIN"
    assert outcome.result.abstain_reason == "PERIOD_NOT_YET_REPORTED"
    assert outcome.exit_code == 2
    assert "FY26" in render_markdown(outcome.result)


def test_an_uningested_period_abstains_with_what_is_missing(ingested):
    settings, store, _ = ingested
    outcome = Agent(store=store, settings=settings).run(
        "XMPL", "1H25", "nim", as_of=date(2025, 5, 31)
    )
    assert outcome.result.abstain_reason == "DOCS_UNAVAILABLE"
    assert "1H25" in " ".join(outcome.result.caveats)


def test_calendar_style_input_abstains_as_ambiguous(ingested):
    settings, store, _ = ingested
    outcome = Agent(store=store, settings=settings).run("XMPL", "H1 2025", "nim")
    assert outcome.result.abstain_reason == "AMBIGUOUS_INPUT"
    assert outcome.exit_code == 2


def test_an_unknown_bank_abstains(ingested):
    settings, store, _ = ingested
    outcome = Agent(store=store, settings=settings).run("Barclays", "1H26", "nim")
    assert outcome.result.abstain_reason == "BANK_UNKNOWN"


def test_a_metric_out_of_scope_abstains(ingested):
    settings, store, _ = ingested
    outcome = Agent(store=store, settings=settings).run("XMPL", "1H26", "dividend yield")
    assert outcome.result.abstain_reason == "METRIC_OUT_OF_SCOPE"


def test_an_abstention_still_renders_a_memo(ingested):
    settings, store, _ = ingested
    outcome = Agent(store=store, settings=settings).run("XMPL", "H1 2025", "nim")
    markdown = render_markdown(outcome.result)
    assert "Status: ABSTAIN" in markdown
    assert "AMBIGUOUS_INPUT" in markdown


# --- point-in-time integrity ----------------------------------------------


def test_an_as_of_before_publication_cannot_see_the_document(ingested):
    settings, store, _ = ingested
    outcome = Agent(store=store, settings=settings).run(
        "XMPL", "1H26", "nim", as_of=date(2026, 4, 30)
    )
    assert outcome.result.status == "ABSTAIN"


def test_all_six_metrics_run_against_the_fixture(ingested):
    settings, store, _ = ingested
    agent = Agent(store=store, settings=settings)
    statuses = {
        metric: agent.run("XMPL", "1H26", metric, as_of=AS_OF).result.status
        for metric in (
            "nim",
            "headline_earnings",
            "roe",
            "cet1_ratio",
            "credit_impairment_charge",
            "cost_to_income",
        )
    }
    assert set(statuses) == {
        "nim",
        "headline_earnings",
        "roe",
        "cet1_ratio",
        "credit_impairment_charge",
        "cost_to_income",
    }
    assert all(status in {"OK", "PARTIAL"} for status in statuses.values()), statuses


# --- the inspection commands ------------------------------------------------------
#
# `facts` and `trace` were listed in the README and printed "not available yet" when run.


def _env(settings) -> dict[str, str]:  # type: ignore[no-untyped-def]
    return {
        "BANKRESEARCH_DATA_DIR": str(settings.data_dir),
        "BANKRESEARCH_RUNS_DIR": str(settings.runs_dir),
    }


def test_facts_lists_what_a_run_could_read_with_its_source(ingested, monkeypatch):
    from typer.testing import CliRunner

    from bankresearch import cli

    settings, _, _ = ingested
    monkeypatch.setattr(cli.console, "width", 160)  # wide enough that no cell wraps
    outcome = CliRunner().invoke(
        cli.app,
        ["facts", "--bank", "XMPL", "--period", "1H26", "--as-of", AS_OF.isoformat()],
        env=_env(settings),
    )
    assert outcome.exit_code == 0, outcome.output
    assert "cet1_ratio" in outcome.stdout
    assert "1h26-results-document p." in outcome.stdout
    assert "validated figure(s) from 1 document(s)" in " ".join(outcome.stdout.split())


def test_facts_reads_only_what_was_published_by_the_as_of_date(ingested):
    """The same scoped read a run makes: a date before the filing sees nothing."""
    from typer.testing import CliRunner

    from bankresearch.cli import app

    settings, _, _ = ingested
    outcome = CliRunner().invoke(
        app,
        ["facts", "--bank", "XMPL", "--period", "1H26", "--as-of", "2026-04-30"],
        env=_env(settings),
    )
    assert outcome.exit_code == 0, outcome.output
    assert "No validated figures" in outcome.stdout


def test_trace_prints_a_runs_steps_in_order(ingested):
    from typer.testing import CliRunner

    from bankresearch.cli import app

    settings, store, _ = ingested
    outcome = Agent(store=store, settings=settings).run("XMPL", "1H26", "nim", as_of=AS_OF)
    shown = CliRunner().invoke(app, ["trace", outcome.result.run_id], env=_env(settings))
    assert shown.exit_code == 0, shown.output
    steps = ["resolve", "availability", "extract", "reconcile", "decompose", "render"]
    positions = [shown.stdout.index(step) for step in steps]
    assert positions == sorted(positions)
    assert outcome.result.status in shown.stdout


def test_trace_of_an_unknown_run_is_a_plain_error(ingested):
    from typer.testing import CliRunner

    from bankresearch.cli import app

    settings, _, _ = ingested
    shown = CliRunner().invoke(app, ["trace", "not-a-run"], env=_env(settings))
    assert shown.exit_code == 1
    assert "No trace for run" in shown.stdout
