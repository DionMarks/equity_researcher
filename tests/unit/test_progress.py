"""The line a terminal shows while a paid run works, and the wrapper that feeds it."""

from __future__ import annotations

import io
from decimal import Decimal

import pytest
from rich.console import Console

from bankresearch.agent.estimate import PATHS, Estimate
from bankresearch.agent.reporting import ReportingClient
from bankresearch.llm.client import LLMRequest, LLMResponse, ScriptedClient, Usage
from bankresearch.output.progress import ACTIVITIES, TerminalProgress, duration, usual_range


@pytest.mark.parametrize(
    ("seconds", "said"),
    [
        (25, "25 seconds"),
        (59, "59 seconds"),
        (60, "1 minute"),
        (90, "1.5 minutes"),
        (120, "2 minutes"),
        (180, "3 minutes"),
    ],
)
def test_a_length_of_time_is_said_the_way_a_person_says_it(seconds, said):
    assert duration(seconds) == said


def test_every_path_says_how_long_it_usually_takes():
    assert usual_range(*PATHS["saved"].seconds) == "25 seconds to 2.5 minutes"
    assert usual_range(*PATHS["new_bank"].seconds) == "1 minute to 3 minutes"
    for estimate in PATHS.values():
        low, high = estimate.seconds
        assert 0 < low < high


def test_the_estimate_says_how_long_before_it_asks():
    estimate = Estimate(asked="NAB 1H26 nim", path="saved", walk_led=True)
    assert estimate.lines()[-1] == "It usually takes 25 seconds to 2.5 minutes."


class _Recorder:
    def __init__(self) -> None:
        self.heard: list[tuple[object, ...]] = []

    def doing(self, activity: str) -> None:
        self.heard.append(("doing", activity))

    def calling(self, step: str) -> None:
        self.heard.append(("calling", step))

    def called(self, step: str, cost_usd: Decimal) -> None:
        self.heard.append(("called", step, cost_usd))


def _request(step: str) -> LLMRequest:
    return LLMRequest(step=step, model="claude-fable-5-1", effort="high", system="", messages=())


def test_the_wrapper_passes_each_call_through_and_says_what_it_cost():
    paid = LLMResponse(text="a memo", usage=Usage(input_tokens=1_000_000))
    replayed = LLMResponse(text="a memo", usage=Usage(input_tokens=1_000_000), cached=True)
    inner = ScriptedClient(responses=[paid, replayed])
    progress = _Recorder()
    client = ReportingClient(inner, progress)

    assert client.available
    request = _request("explain")
    assert client.complete(request) is paid
    assert client.complete(request) is replayed
    assert inner.requests == [request, request]  # nothing added, nothing changed

    calling, called, _, replay = progress.heard
    assert calling == ("calling", "explain")
    assert called[:2] == ("called", "explain") and called[2] > 0  # priced from config
    assert replay == ("called", "explain", Decimal(0))  # a cache hit cost nothing


def test_a_call_that_fails_is_not_counted():
    inner = ScriptedClient()  # nothing to say, so the call raises
    progress = _Recorder()
    with pytest.raises(Exception, match="ran out of responses"):
        ReportingClient(inner, progress).complete(_request("verify"))
    assert progress.heard == [("calling", "verify")]


def test_a_script_or_a_redirected_run_sees_nothing():
    out = io.StringIO()
    with TerminalProgress(Console(file=out, force_terminal=False), (25, 90)) as progress:
        progress.doing("Reading the saved figures")
        progress.calling("explain")
        progress.called("explain", Decimal("0.10"))
    assert out.getvalue() == ""


def test_a_terminal_sees_what_it_is_doing_how_long_it_usually_takes_and_the_spend():
    out = io.StringIO()
    console = Console(file=out, force_terminal=True, width=160)
    with TerminalProgress(console, (25, 90)) as progress:
        progress.calling("text_walk")
        progress.called("text_walk", Decimal("0.12"))
        assert progress.context() == (
            "usually 25 seconds to 1.5 minutes  ·  1 model call, A$0.17 so far"
        )
        progress.calling("explain")
        progress.called("explain", Decimal("0.2"))
        assert progress.context().endswith("2 model calls, A$0.46 so far")
    assert "Working out the question" in out.getvalue()


def test_every_model_step_a_run_routes_has_words_on_the_line():
    from bankresearch.domain.registry import get_registry

    routed = set(get_registry().models["routing"])
    assert routed <= set(ACTIVITIES)
