"""A run that can't use the model stops and says why, instead of answering without it."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from bankresearch.llm.anthropic_client import AnthropicClient
from bankresearch.llm.client import LLMRequest, ModelUnreachable
from bankresearch.llm.consent import AlwaysApprove, ConsentDenied

FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthetic_results_document.pdf"


def _request() -> LLMRequest:
    return LLMRequest(
        step="explain", model="claude-fable-5-1", effort="high", system="", messages=()
    )


class _Refusing:
    """Stands in for the provider and raises the same error on every call."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0
        self.messages = self

    def create(self, **payload: Any) -> Any:
        self.calls += 1
        raise self.error


class _ApiError(Exception):
    def __init__(self, message: str, status_code: int | None) -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.fixture
def no_waiting(monkeypatch):  # type: ignore[no-untyped-def]
    """Retries back off for real; a test doesn't need to wait them out."""
    monkeypatch.setattr(AnthropicClient._send.retry, "sleep", lambda seconds: None)


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (_ApiError("authentication_error: invalid x-api-key", 401), "key_rejected"),
        (_ApiError("permission_error: not allowed", 403), "key_rejected"),
        (
            _ApiError(
                "invalid_request_error: Your credit balance is too low to access the API", 400
            ),
            "out_of_credit",
        ),
    ],
)
def test_a_key_or_an_account_the_api_refuses_stops_the_call_at_once(error, kind):
    sdk = _Refusing(error)
    client = AnthropicClient(fallback_model=None, consent=AlwaysApprove(), _sdk=sdk)
    with pytest.raises(ModelUnreachable) as stopped:
        client.complete(_request())
    assert stopped.value.kind == kind
    assert sdk.calls == 1  # no retry: waiting won't fix a key


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (_ApiError("rate_limit_error: rate limit reached", 429), "rate_limited"),
        (_ApiError("overloaded_error: overloaded", 529), "unreachable"),
        (_ApiError("Connection error.", None), "unreachable"),
    ],
)
def test_a_limit_or_an_outage_that_outlasts_the_retries_stops_the_call(no_waiting, error, kind):
    sdk = _Refusing(error)
    client = AnthropicClient(fallback_model=None, consent=AlwaysApprove(), _sdk=sdk)
    with pytest.raises(ModelUnreachable) as stopped:
        client.complete(_request())
    assert stopped.value.kind == kind
    assert sdk.calls == 3


def test_no_key_is_its_own_reason(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from bankresearch.settings import Settings

    with pytest.raises(ModelUnreachable) as stopped:
        AnthropicClient(settings=Settings()).complete(_request())
    assert stopped.value.kind == "no_key"


def test_a_step_that_absorbs_other_model_failures_lets_this_one_through():
    from bankresearch.agent.steps.taxonomy_llm import map_labels_with_model
    from bankresearch.llm.client import LLMUnavailable, ScriptedClient

    def unavailable(request: LLMRequest) -> Any:
        raise LLMUnavailable("an answer that couldn't be used")

    def unreachable(request: LLMRequest) -> Any:
        raise ModelUnreachable("unreachable", "provider returned 503")

    # Any other failure costs the step: the labels go unmapped and the run carries on.
    carried_on = map_labels_with_model(
        "nim", ["Quarterly widget adjustment"], ScriptedClient(on_exhausted=unavailable)
    )
    assert any("unavailable" in note for note in carried_on.notes)
    with pytest.raises(ModelUnreachable):
        map_labels_with_model(
            "nim", ["Quarterly widget adjustment"], ScriptedClient(on_exhausted=unreachable)
        )


def _ingested(tmp_path: Path) -> dict[str, str]:
    """The synthetic filing, read into a data folder of its own; the environment that
    points the command at it."""
    from bankresearch.cli import app

    env = {
        "BANKRESEARCH_DATA_DIR": str(tmp_path / "data"),
        "BANKRESEARCH_RUNS_DIR": str(tmp_path / "runs"),
    }
    read = CliRunner().invoke(
        app,
        [
            "ingest",
            "--path",
            str(FIXTURE),
            "--bank",
            "XMPL",
            "--period",
            "1H26",
            "--publish-date",
            "2026-05-01",
        ],
        env=env,
    )
    assert read.exit_code == 0, read.output
    return env


ASK = ["run", "--bank", "XMPL", "--period", "1H26", "--metric", "nim", "--yes"]


@pytest.mark.parametrize(
    ("kind", "said"),
    [
        ("key_rejected", "didn't accept the key"),
        ("out_of_credit", "run out of credit"),
        ("rate_limited", "kept limiting requests"),
        ("unreachable", "couldn't be reached"),
    ],
)
def test_the_command_says_why_and_gives_no_answer(tmp_path, scripted_model, kind, said):
    from bankresearch.cli import app

    env = _ingested(tmp_path)

    def unreachable(request: LLMRequest) -> Any:
        raise ModelUnreachable(kind, "the API's own words")

    scripted_model.on_exhausted = unreachable
    result = CliRunner().invoke(app, ASK, env=env)
    assert result.exit_code == 1
    assert said in result.output
    assert "No answer was given" in result.output
    assert "Net interest margin" not in result.stdout  # no answer on stdout at all


def test_a_run_that_uses_up_what_was_approved_says_so(tmp_path, scripted_model):
    from bankresearch.cli import app

    env = _ingested(tmp_path)

    def over_budget(request: LLMRequest) -> Any:
        raise ConsentDenied("the approved ceiling would be passed")

    scripted_model.on_exhausted = over_budget
    result = CliRunner().invoke(app, ASK, env=env)
    assert result.exit_code == 1
    assert "used up the budget approved for it" in result.output
    assert "--grant" in result.output


def test_the_run_itself_raises_it_rather_than_answering_without_the_model(tmp_path):
    from bankresearch.agent.orchestrator import Agent
    from bankresearch.domain.models import DocMeta
    from bankresearch.ingestion.pipeline import ingest_document
    from bankresearch.llm.client import ScriptedClient
    from bankresearch.memory.db import connect
    from bankresearch.memory.fact_store import FactStore
    from bankresearch.settings import Settings

    settings = Settings(
        config_dir=Path(__file__).parents[2] / "config",
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        evals_dir=tmp_path / "evals",
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

    def unreachable(request: LLMRequest) -> Any:
        raise ModelUnreachable("key_rejected")

    agent = Agent(store=store, settings=settings, client=ScriptedClient(on_exhausted=unreachable))
    with pytest.raises(ModelUnreachable):
        agent.run("XMPL", "1H26", "nim", as_of=date(2026, 5, 31))
