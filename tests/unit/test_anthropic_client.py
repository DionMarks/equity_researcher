"""The live client's contract, tested without a key, a network or the SDK."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from bankresearch.domain.registry import get_registry
from bankresearch.llm.anthropic_client import (
    AnthropicClient,
    ProviderError,
    build_payload,
    parse_response,
)
from bankresearch.llm.client import LLMRequest, LLMUnavailable
from bankresearch.llm.consent import AlwaysApprove


def _request(**overrides: Any) -> LLMRequest:
    base: dict[str, Any] = {
        "step": "explain",
        "model": "claude-opus-5-5",
        "effort": "high",
        "system": "You explain movements from evidence.",
        "messages": ({"role": "user", "content": "why did the margin fall?"},),
    }
    base.update(overrides)
    return LLMRequest(**base)


class _FakeSdk:
    """Stands in for the provider. Records payloads, returns scripted responses."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.payloads: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **payload: Any) -> Any:
        self.payloads.append(payload)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _message(text: str = "ok", stop: str = "end_turn", **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop,
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }
    body.update(extra)
    return body


# --- the payload -----------------------------------------------------------


def test_no_sampling_parameter_is_ever_sent():
    """Depth is set by output_config.effort. Sending both makes a run unreproducible for
    reasons that are hard to find later.
    """
    payload = build_payload(_request())
    for forbidden in ("temperature", "top_p", "top_k"):
        assert forbidden not in payload
    assert payload["output_config"] == {"effort": "high"}


def test_the_payload_matches_the_constraints_the_config_declares():
    constraints = get_registry().models["api_constraints"]
    payload = build_payload(_request(tools=({"name": "get_facts"},)))

    assert constraints["send_sampling_parameters"] is False
    assert constraints["control_depth_with"] == "output_config.effort"
    assert "output_config" in payload

    # Never forced: Opus 5.5 rejects `any` and a named tool.
    assert constraints["force_tool_choice"] is False
    assert payload["tool_choice"] == {"type": "auto"}

    # No assistant prefill means the last message is never an assistant turn we wrote.
    assert constraints["assistant_prefill"] is False
    assert payload["messages"][-1]["role"] == "user"

    assert constraints["prompt_caching"] is True
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_a_structured_output_is_requested_inside_output_config():
    """The schema goes in ``output_config.format``, next to ``effort``.

    It was being sent as a top-level ``output_format``, which isn't a Messages API
    parameter. That fails quietly: the call succeeds, the model is under no schema at all,
    and the output usually looks fine anyway.
    """
    schema = {"type": "object", "properties": {"drivers": {"type": "array"}}}
    payload = build_payload(_request(output_schema=schema))
    assert payload["output_config"]["effort"] == "high"
    assert payload["output_config"]["format"]["type"] == "json_schema"
    # `additionalProperties: false` is added on the way out, which strict mode requires;
    # see `test_the_payload_sanitises_any_schema_it_is_given`.
    assert payload["output_config"]["format"]["schema"] == {
        **schema,
        "additionalProperties": False,
    }
    assert "output_format" not in payload
    assert "response_format" not in payload


@pytest.mark.parametrize("level", ["low", "medium", "high", "xhigh", "max"])
def test_every_documented_effort_level_is_accepted(level):
    assert build_payload(_request(effort=level))["output_config"]["effort"] == level


def test_an_undocumented_effort_level_is_refused_before_the_call():
    """An unrecognised effort is a config typo that would fail every call in a step, and
    nobody reads a step's payload.
    """
    from bankresearch.llm.anthropic_client import EFFORT_LEVELS

    assert EFFORT_LEVELS == {"low", "medium", "high", "xhigh", "max"}
    with pytest.raises(ValueError, match="effort 'maximum'"):
        build_payload(_request(effort="maximum"))


def test_every_routed_effort_in_the_config_is_a_documented_level():
    import yaml

    from bankresearch.llm.anthropic_client import EFFORT_LEVELS

    routing = yaml.safe_load(Path("config/models.yaml").read_text(encoding="utf-8"))["routing"]
    for step, spec in routing.items():
        assert spec["effort"] in EFFORT_LEVELS, f"{step} routes to effort {spec['effort']!r}"


def test_the_model_can_be_overridden_for_the_fallback():
    assert build_payload(_request(), model="claude-sonnet-5-5")["model"] == "claude-sonnet-5-5"


# --- reading the response --------------------------------------------------


def test_content_blocks_are_read_by_type_not_by_position():
    """A response begins with thinking blocks, so content[0].text reads the model's
    reasoning as its answer."""
    response = parse_response(
        {
            "content": [
                {"type": "thinking", "thinking": "the margin fell because..."},
                {"type": "text", "text": "Deposit competition."},
            ],
            "stop_reason": "end_turn",
        }
    )
    assert response.text == "Deposit competition."
    assert "because" not in response.text


def test_text_blocks_are_joined_in_order():
    response = parse_response(
        {"content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]}
    )
    assert response.text == "one\ntwo"


def test_a_tool_call_keeps_the_id_its_result_must_carry_back():
    response = parse_response(
        {
            "content": [
                {"type": "thinking", "thinking": "I need the facts"},
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "get_facts",
                    "input": {"bank": "ANZ"},
                },
            ],
            "stop_reason": "tool_use",
        }
    )
    assert response.wants_tools
    assert response.tool_calls[0].id == "toolu_01"
    assert response.tool_calls[0].arguments == {"bank": "ANZ"}


def test_the_whole_block_list_is_kept_so_a_tool_loop_can_append_it_back():
    """A tool loop appends turns unchanged. Rewriting one (dropping the thinking
    blocks, reordering, summarising) is how a loop loses track.
    """
    blocks = [
        {"type": "thinking", "thinking": "..."},
        {"type": "tool_use", "id": "t1", "name": "f", "input": {}},
    ]
    assert list(parse_response({"content": blocks}).raw_blocks) == blocks


def test_a_structured_output_is_read_from_its_own_block():
    response = parse_response({"content": [{"type": "json", "json": {"drivers": []}}]})
    assert response.structured == {"drivers": []}


def test_a_json_body_in_text_is_accepted_but_prose_is_not_guessed_at():
    assert parse_response({"content": [{"type": "text", "text": '{"a": 1}'}]}).structured == {
        "a": 1
    }
    assert parse_response({"content": [{"type": "text", "text": "not json"}]}).structured is None


def test_usage_is_read_including_the_cache_counters():
    response = parse_response(
        {
            "content": [],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_read_input_tokens": 80,
                "cache_creation_input_tokens": 5,
            },
        }
    )
    assert response.usage.cache_read_tokens == 80
    assert response.usage.cache_write_tokens == 5


# --- availability and failure ----------------------------------------------


def test_without_a_key_the_client_is_unavailable_and_says_what_to_do(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from bankresearch.settings import Settings

    client = AnthropicClient(settings=Settings())
    assert not client.available
    with pytest.raises(LLMUnavailable, match="ANTHROPIC_API_KEY"):
        client.complete(_request())


def test_a_refusal_retries_once_on_the_fallback_model():
    sdk = _FakeSdk([_message(stop="refusal"), _message("answered")])
    client = AnthropicClient(fallback_model="claude-sonnet-5-5", consent=AlwaysApprove(), _sdk=sdk)
    assert client.complete(_request()).text == "answered"
    assert [p["model"] for p in sdk.payloads] == ["claude-opus-5-5", "claude-sonnet-5-5"]


def test_an_exceeded_context_window_retries_once_on_the_fallback_model():
    sdk = _FakeSdk([RuntimeError("model_context_window_exceeded"), _message("answered")])
    client = AnthropicClient(fallback_model="claude-sonnet-5-5", consent=AlwaysApprove(), _sdk=sdk)
    assert client.complete(_request()).text == "answered"
    assert len(sdk.payloads) == 2


def test_any_other_error_is_raised_rather_than_retried():
    """A client that retries everything turns a bad request into a slow bad request."""
    sdk = _FakeSdk([RuntimeError("invalid_request_error: bad schema")])
    client = AnthropicClient(fallback_model="claude-sonnet-5-5", consent=AlwaysApprove(), _sdk=sdk)
    with pytest.raises(ProviderError, match="bad schema"):
        client.complete(_request())
    assert len(sdk.payloads) == 1


def test_a_rate_limit_is_retried_on_the_same_model():
    sdk = _FakeSdk([RuntimeError("rate limit exceeded"), _message("answered")])
    client = AnthropicClient(fallback_model="claude-sonnet-5-5", consent=AlwaysApprove(), _sdk=sdk)
    assert client.complete(_request()).text == "answered"
    assert [p["model"] for p in sdk.payloads] == ["claude-opus-5-5", "claude-opus-5-5"]


def test_the_key_never_appears_in_an_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-must-not-leak")
    from bankresearch.settings import Settings

    sdk = _FakeSdk([RuntimeError("invalid_request_error")])
    client = AnthropicClient(
        settings=Settings(), fallback_model=None, consent=AlwaysApprove(), _sdk=sdk
    )
    with pytest.raises(ProviderError) as caught:
        client.complete(_request())
    assert "must-not-leak" not in str(caught.value)


# --- the search-result shape, as the provider documents it ----------------


def test_a_search_result_url_is_read_from_the_nested_result_block():
    """The provider nests ``web_search_result`` items in the result block's ``content``.

    Reading the model's text for links instead would let the pages it's searching influence
    discovery, which is what the structured field is there to prevent.
    """
    from bankresearch.acquisition.discover import search_errors, urls_from_search_results

    blocks = (
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_01",
            "content": [
                {
                    "type": "web_search_result",
                    "url": "https://www.commbank.com.au/profit-announcement.pdf",
                    "title": "Profit Announcement",
                    "page_age": "August 12, 2026",
                    "encrypted_content": "Eqgf...",
                },
                {
                    "type": "web_search_result",
                    "url": "https://example.invalid/scraped-copy.pdf",
                    "title": "CBA results (mirror)",
                },
            ],
        },
    )
    found = urls_from_search_results(blocks, ("commbank.com.au",))
    accepted = [c for c in found if not c.rejected_because]
    rejected = [c for c in found if c.rejected_because]
    assert [c.url for c in accepted] == ["https://www.commbank.com.au/profit-announcement.pdf"]
    assert accepted[0].title == "Profit Announcement"
    assert [c.url for c in rejected] == ["https://example.invalid/scraped-copy.pdf"]
    assert search_errors(blocks) == []


def test_a_failed_search_is_not_a_search_that_found_nothing():
    """On a failure the provider sends ``content`` as one error object, not a list.

    Both give no candidates, but they need different responses: one means the issuer really
    has nothing on its site, the other means we didn't look.
    """
    from bankresearch.acquisition.discover import search_errors, urls_from_search_results

    failed = (
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_02",
            "content": {
                "type": "web_search_tool_result_error",
                "error_code": "max_uses_exceeded",
            },
        },
    )
    assert urls_from_search_results(failed, ("commbank.com.au",)) == []
    assert search_errors(failed) == ["max_uses_exceeded"]

    no_matches = ({"type": "web_search_tool_result", "tool_use_id": "s3", "content": []},)
    assert urls_from_search_results(no_matches, ("commbank.com.au",)) == []
    assert search_errors(no_matches) == []


def test_the_search_tool_version_calls_the_api_directly():
    """``web_search_20250305`` defaults ``allowed_callers`` to direct.

    Later versions default to running inside code execution, which a model that doesn't
    support programmatic tool calling rejects with a 400 unless ``allowed_callers`` is set
    explicitly. Pinning the basic version keeps the call shape the tests check.
    """
    from bankresearch.acquisition.discover import WEB_SEARCH_TOOL, search_tool

    assert WEB_SEARCH_TOOL == "web_search_20250305"
    tool = search_tool(("commbank.com.au",))
    assert tool["type"] == WEB_SEARCH_TOOL
    assert "announcements.asx.com.au" in tool["allowed_domains"]
    assert "blocked_domains" not in tool, "allowed_domains and blocked_domains are exclusive"


def test_a_transient_retry_is_a_second_charge_and_a_second_question():
    """A rate-limit retry hits the provider again, so it's asked about again.

    Tenacity wraps ``_send``, which goes back through ``_call``. Gating ``complete`` instead
    would let one approval cover however many attempts the retry policy allowed.
    """
    import io

    from bankresearch.llm.consent import ConsentDenied, TerminalConsent

    sdk = _FakeSdk([RuntimeError("rate limit exceeded"), _message("answered")])
    gate = TerminalConsent(stream=io.StringIO("y\nn\n"), out=io.StringIO(), require_tty=False)
    client = AnthropicClient(fallback_model=None, consent=gate, _sdk=sdk)
    with pytest.raises(ConsentDenied):
        client.complete(_request())
    assert len(sdk.payloads) == 1, "the retry was refused, so only the first call happened"
    assert (gate.approvals, gate.refusals) == (1, 1)


def test_the_payload_sanitises_any_schema_it_is_given():
    """The check that makes the other two unnecessary.

    Two files sent a schema outside the subset ``strict`` accepts, and both failed with a
    400 on their first live call. A rule applied at six call sites will be missed at one, so
    it's applied here instead, at the single point everything goes through.
    """
    payload = build_payload(
        _request(
            tools=(
                {
                    "name": "bounded",
                    "description": "has bounds pydantic would emit",
                    "input_schema": {
                        "type": "object",
                        "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 5}},
                    },
                },
            ),
            output_schema={
                "type": "object",
                "properties": {"label": {"type": "string", "minLength": 1}},
            },
        )
    )
    tool_property = payload["tools"][0]["input_schema"]["properties"]["n"]
    assert "minimum" not in tool_property and "maximum" not in tool_property
    assert "minimum: 1" in tool_property["description"]
    assert payload["tools"][0]["input_schema"]["additionalProperties"] is False

    schema = payload["output_config"]["format"]["schema"]
    assert "minLength" not in schema["properties"]["label"]
    assert schema["additionalProperties"] is False


def test_a_server_side_tool_is_passed_through_untouched():
    """Web search has no input_schema; its shape is the provider's, not ours."""
    search = {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
    payload = build_payload(_request(tools=(search,)))
    assert payload["tools"][0] == search
