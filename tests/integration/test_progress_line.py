"""A model-backed run reports what it's doing as it goes, and is the same run either way."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from bankresearch.agent.orchestrator import Agent
from bankresearch.domain.models import DocMeta
from bankresearch.ingestion.pipeline import ingest_document
from bankresearch.llm.client import LLMRequest, LLMResponse, ScriptedClient, Usage
from bankresearch.memory.db import connect
from bankresearch.memory.fact_store import FactStore
from bankresearch.settings import Settings

FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthetic_results_document.pdf"
AS_OF = date(2026, 5, 31)


@pytest.fixture(scope="module")
def ingested(tmp_path_factory):
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
    return settings, store


class _Recorder:
    def __init__(self) -> None:
        self.heard: list[tuple[object, ...]] = []

    def doing(self, activity: str) -> None:
        self.heard.append(("doing", activity))

    def calling(self, step: str) -> None:
        self.heard.append(("calling", step))

    def called(self, step: str, cost_usd: Decimal) -> None:
        self.heard.append(("called", step, cost_usd))


def _model() -> ScriptedClient:
    """A model that answers everything with an empty reply: enough to drive every model
    step, and the same every time."""

    def reply(request: LLMRequest) -> LLMResponse:
        return LLMResponse(text="", usage=Usage(1000, 100))

    return ScriptedClient(on_exhausted=reply)


def _run(ingested, progress=None):
    settings, store = ingested
    client = _model()
    outcome = Agent(store=store, settings=settings, client=client, progress=progress).run(
        "XMPL", "1H26", "nim", as_of=AS_OF
    )
    return outcome, client


def test_the_line_hears_each_phase_in_order_and_every_model_call(ingested):
    progress = _Recorder()
    outcome, client = _run(ingested, progress)
    assert outcome.result.status in {"OK", "PARTIAL"}

    phases = [entry[1] for entry in progress.heard if entry[0] == "doing"]
    assert phases[0] == "Working out the question"
    assert phases[-1] == "Putting the answer together"
    order = ["Reading the saved figures", "Splitting the movement"]
    assert [phase for phase in phases if phase in order] == order

    calls = [entry for entry in progress.heard if entry[0] != "doing"]
    assert len(calls) == 2 * len(client.requests) > 0
    for before, after in zip(calls[::2], calls[1::2], strict=True):
        assert before[0] == "calling" and after[0] == "called" and before[1] == after[1]


def test_the_line_only_watches(ingested):
    """With or without it, the run makes the same calls and gives the same answer."""
    watched, watched_client = _run(ingested, _Recorder())
    plain, plain_client = _run(ingested)
    assert [r.step for r in watched_client.requests] == [r.step for r in plain_client.requests]
    assert watched.result.status == plain.result.status
    assert [(d.driver_id, d.contribution) for d in watched.result.drivers] == [
        (d.driver_id, d.contribution) for d in plain.result.drivers
    ]
