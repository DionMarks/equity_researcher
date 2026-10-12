"""What a live run spends, how it's counted, and what it records about itself.

Each test here comes from the first Opus 5 run against the real filings (Westpac 1H26 net
interest margin, 2026-10-10), which spent A$1.22 and reported A$1.224: cache writes were
left out of every cost, the locate loop resent a growing conversation with nothing cached,
the model and prompt behind each step went unrecorded, and ``--format json`` printed a line
of text ahead of the JSON.
"""

from __future__ import annotations

from decimal import Decimal

from bankresearch.llm.client import LLMRequest, LLMResponse, Usage, cost_usd

OPUS_5 = {"input": "5.00", "output": "25.00", "cache_read": "0.50"}


def test_a_cache_write_is_charged_at_a_quarter_over_the_input_price():
    usage = Usage(input_tokens=212, output_tokens=119, cache_write_tokens=2236)
    expected = (
        Decimal(212) * 5 + Decimal(119) * 25 + Decimal(2236) * 5 * Decimal("1.25")
    ) / 1_000_000
    assert cost_usd(usage, OPUS_5) == expected


def test_a_cache_read_is_charged_at_the_cache_read_price():
    usage = Usage(input_tokens=5157, output_tokens=146, cache_read_tokens=2236)
    expected = (Decimal(5157) * 5 + Decimal(146) * 25 + Decimal(2236) * Decimal("0.5")) / 1_000_000
    assert cost_usd(usage, OPUS_5) == expected


def _request(messages: tuple[dict, ...]) -> LLMRequest:
    return LLMRequest(
        step="locate", model="claude-opus-5", effort="medium", system="s", messages=messages
    )


def test_a_tool_loop_marks_the_conversation_so_far_for_caching():
    from bankresearch.llm.anthropic_client import build_payload

    loop = (
        {"role": "user", "content": "find the pages"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "r"}]},
    )
    payload = build_payload(_request(loop))
    assert payload["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    # Only the newest turn carries the marker, and the caller's messages aren't changed.
    assert "cache_control" not in str(payload["messages"][:-1])
    assert "cache_control" not in str(loop)


def test_a_single_turn_call_is_not_marked():
    """Nothing reads it back, and a cache write costs more than plain input."""
    from bankresearch.llm.anthropic_client import build_payload

    payload = build_payload(_request(({"role": "user", "content": "write the memo"},)))
    assert "cache_control" not in str(payload["messages"])


def test_the_ceiling_prices_a_cached_conversation_at_the_write_rate():
    """The estimate has to err high, and a cache write is the most input can cost."""
    from bankresearch.llm.consent import estimate_call

    pricing = {"claude-opus-5": OPUS_5}
    plain = {
        "model": "claude-opus-5",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x" * 3000}],
    }
    marked = {
        "model": "claude-opus-5",
        "max_tokens": 10,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "x" * 3000, "cache_control": {"type": "ephemeral"}}
                ],
            }
        ],
    }
    assert (
        estimate_call(marked, step="locate", pricing=pricing).input_usd
        > estimate_call(plain, step="locate", pricing=pricing).input_usd
    )


def test_a_grant_is_settled_to_what_the_call_cost_including_the_cache():
    from bankresearch.llm.anthropic_client import AnthropicClient
    from bankresearch.llm.consent import BudgetedGrant

    class _SDK:
        class messages:  # mirrors the SDK's attribute name
            @staticmethod
            def create(**payload: object) -> dict:
                return {
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {
                        "input_tokens": 1000,
                        "output_tokens": 100,
                        "cache_creation_input_tokens": 2000,
                        "cache_read_input_tokens": 0,
                    },
                }

    grant = BudgetedGrant(calls=5, ceiling_usd=Decimal("1.00"))
    client = AnthropicClient(consent=grant, pricing={"claude-opus-5": OPUS_5}, _sdk=_SDK())
    client.complete(_request(({"role": "user", "content": "q"},)))
    expected = cost_usd(Usage(1000, 100, 0, 2000), OPUS_5)
    assert grant.spent_usd == expected


def test_the_grant_notice_goes_to_stderr_so_json_output_stays_parseable(capsys):
    from bankresearch.cli import _consent_gate

    _consent_gate("2:1.00")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Granting up to 2 call(s)" in captured.err


def test_the_model_and_prompt_behind_each_step_are_recorded():
    from bankresearch.agent.orchestrator import _record_spend
    from bankresearch.domain.models import Diagnostics
    from bankresearch.llm.budget import Budget
    from bankresearch.llm.tracing import Tracer, default_library

    tracer = Tracer()
    prompt = default_library().get("explain")
    tracer.model_call(
        LLMRequest(
            step="explain", model="claude-opus-5", effort="high", system=prompt.text, messages=()
        ),
        LLMResponse(text="memo", usage=Usage(10, 10)),
        prompt=prompt,
    )
    diagnostics = Diagnostics(config_hash="x", model_ids={}, warnings=[])
    _record_spend(diagnostics, Budget(), tracer)
    assert diagnostics.model_ids == {"explain": "claude-opus-5"}
    assert diagnostics.prompt_hashes == {"explain": prompt.sha256}


def test_a_folder_that_cant_be_looked_at_doesnt_stop_a_paid_run(monkeypatch):
    """A live run crashed after a paid chart reading had come back, because macOS briefly
    refused every path under the Desktop and the trace's price lookup checked for a
    ``.env``. Neither the price nor the check is worth the run."""
    import os

    from bankresearch.domain import registry
    from bankresearch.llm import tracing

    found = registry._default_config_dir()

    def refused() -> str:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "getcwd", refused)
    assert registry._default_config_dir() == found
    monkeypatch.setattr(registry, "get_registry", lambda *a: (_ for _ in ()).throw(OSError("no")))
    assert tracing._priced("claude-fable-5-1", Usage(10, 10), cached=False) is None
