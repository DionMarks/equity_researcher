"""An answer says how it was reached: every model call it made, and every filing it found."""

from __future__ import annotations

import io
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from rich.console import Console

from bankresearch.agent.orchestrator import Agent, _trail
from bankresearch.domain.models import DocMeta
from bankresearch.ingestion.pipeline import ingest_document
from bankresearch.llm.client import LLMRequest, LLMResponse, ScriptedClient, Usage
from bankresearch.llm.tracing import Tracer
from bankresearch.memory.db import connect
from bankresearch.memory.fact_store import FactStore
from bankresearch.output.progress import PURPOSES
from bankresearch.output.render_md import render_markdown
from bankresearch.output.render_terminal import render_terminal
from bankresearch.settings import Settings

FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthetic_results_document.pdf"
AS_OF = date(2026, 5, 31)


@pytest.fixture(scope="module")
def answered(tmp_path_factory):
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
        publish_date=date(2026, 5, 1),
        url=FIXTURE.resolve().as_uri(),
    )
    ingest_document(FIXTURE, meta, store, processed_root=settings.processed_dir)
    client = ScriptedClient(
        on_exhausted=lambda request: LLMResponse(text="", usage=Usage(1000, 100))
    )
    outcome = Agent(store=store, settings=settings, client=client).run(
        "XMPL", "1H26", "nim", as_of=AS_OF
    )
    return outcome.result, client


def test_every_model_call_the_run_made_is_in_the_trail_in_order(answered):
    result, client = answered
    calls = result.trail.model_calls
    assert client.requests, "the scripted run made no model call to trace"
    assert [call.step for call in calls] == [request.step for request in client.requests]
    assert [call.model for call in calls] == [request.model for request in client.requests]
    for call in calls:
        assert (call.input_tokens, call.output_tokens) == (1000, 100)
        assert not call.cached
        assert call.cost_usd is not None and call.cost_usd > 0  # priced from config


def test_the_trail_is_saved_with_the_answer(answered):
    result, _ = answered
    saved = result.model_dump(mode="json")
    assert len(saved["trail"]["model_calls"]) == len(result.trail.model_calls)
    assert saved["trail"]["fetched"] == []  # the fixture was held, so nothing was found


def test_the_memo_lists_each_call_and_what_it_was_for(answered):
    result, _ = answered
    memo = render_markdown(result)
    calls = result.trail.model_calls
    assert "## How it was answered" in memo
    count = f"{len(calls)} model call{'' if len(calls) == 1 else 's'}"
    assert count in memo
    for call in calls:
        assert PURPOSES[call.step] in memo


def test_the_terminal_shows_the_calls_and_what_was_read(answered):
    result, _ = answered
    console = Console(file=io.StringIO(), width=160, color_system=None)
    render_terminal(result, console, artefacts="runs/example")
    assert isinstance(console.file, io.StringIO)
    shown = console.file.getvalue()
    assert "What was read" in shown
    calls = result.trail.model_calls
    assert f"Model calls   {len(calls)} call{'' if len(calls) == 1 else 's'}" in shown
    for call in calls:
        assert PURPOSES[call.step] in shown


def _request(step: str) -> LLMRequest:
    return LLMRequest(step=step, model="claude-fable-5-1", effort="high", system="", messages=())


def test_a_replayed_call_is_shown_as_costing_nothing():
    tracer = Tracer()
    tracer.model_call(_request("explain"), LLMResponse(text="", usage=Usage(1000, 100)))
    tracer.model_call(_request("verify"), LLMResponse(text="", usage=Usage(1000, 100), cached=True))
    tracer.skipped("locate", "nothing to search for")
    paid, replayed = _trail(tracer, []).model_calls  # the skipped step is not a call
    assert paid.cost_usd is not None and paid.cost_usd > 0
    assert replayed.cached and replayed.cost_usd == Decimal(0)


def test_a_filing_found_during_the_run_is_named_and_nothing_else_is():
    trace = [
        {"step": "discover", "notes": ["found and ingested bbc-fy26-annual-report from the ASX"]},
        {"step": "discover", "notes": ["searched the ASX list for 2026"]},
        {"step": "explain", "notes": ["found and ingested nothing, wrong step"]},
    ]
    assert _trail(None, trace).fetched == ["found and ingested bbc-fy26-annual-report from the ASX"]
