"""The live model client."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Final

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..domain.money import aud
from ..domain.registry import get_registry
from ..settings import Settings, get_settings
from .client import (
    LLMRequest,
    LLMResponse,
    LLMUnavailable,
    ModelUnreachable,
    ToolCall,
    Usage,
    blocks_by_type,
    cost_usd,
)
from .consent import (
    CallEstimate,
    ConsentDenied,
    ConsentGate,
    default_gate,
    estimate_call,
)
from .json_schema import strict_subset

__all__ = ["AnthropicClient", "ProviderError", "build_payload"]

# Errors worth one attempt on the fallback model, listed in config/models.yaml.
_FALLBACK_ON = ("refusal", "model_context_window_exceeded")
_TRANSIENT_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


class ProviderError(LLMUnavailable):
    """The provider refused or failed in a way no retry will fix."""


class TransientProviderError(LLMUnavailable):
    """A rate limit or a server error. Worth retrying."""


#: The documented values for ``output_config.effort``. Checked rather than passed through,
#: because an unrecognised effort is an easy config typo to miss: the API would reject every
#: call in that step, and nobody reads that step's payload.
EFFORT_LEVELS: Final[frozenset[str]] = frozenset({"low", "medium", "high", "xhigh", "max"})


def _sanitised_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """One tool definition with its input schema inside the accepted subset.

    Server-side tools (web search and so on) have no ``input_schema`` and are passed through
    as they are, because their shape belongs to the provider, not us.
    """
    out = dict(tool)
    schema = out.get("input_schema")
    if isinstance(schema, dict):
        out["input_schema"] = strict_subset(schema)
    return out


def build_payload(request: LLMRequest, *, model: str | None = None) -> dict[str, Any]:
    """The request body, with every API constraint applied.

    Built as a plain dict and returned rather than sent, so a test can check what would go
    over the wire without a key, a network or the SDK.
    """
    if request.effort not in EFFORT_LEVELS:
        raise ValueError(
            f"effort {request.effort!r} is not one of {', '.join(sorted(EFFORT_LEVELS))}; "
            "check config/models.yaml"
        )
    payload: dict[str, Any] = {
        "model": model or request.model,
        # The step's own cap unless the request names one. Every call used to go out at
        # the dataclass default of 4,096, thinking included, whatever the config said.
        "max_tokens": request.max_tokens or get_registry().output_limit(request.step),
        # Depth is set here, not with temperature. `api_constraints` records both halves
        # of that: send_sampling_parameters false, control_depth_with output_config.effort.
        "output_config": {"effort": request.effort},
        "messages": _with_conversation_cached(request.messages),
    }

    if request.system:
        # The system prompt is the largest stable prefix of every call in a step, so it
        # is the one worth caching. Marked here rather than by each caller.
        payload["system"] = [
            {
                "type": "text",
                "text": request.system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    if request.tools:
        # Cleaned up here rather than by each caller. Two separate files sent a schema
        # outside the subset ``strict`` accepts, and both failed with a 400 on their first
        # live call. A rule applied at six call sites will be missed at one, and the result
        # is a request the API rejects outright. ``strict_subset`` is idempotent, so a
        # caller that already applied it loses nothing.
        payload["tools"] = [_sanitised_tool(tool) for tool in request.tools]
        # Never forced: Opus 5.5 rejects `any` and a named tool, so the prompt says when
        # to call and the loop ends on its own budget instead.
        payload["tool_choice"] = {"type": "auto"}

    if request.output_schema is not None:  # sanitised for the same reason as the tools
        # Next to ``effort``, inside ``output_config``, not as a top-level
        # ``output_format``. The Messages API has no such top-level parameter, so the schema
        # would either be rejected or, worse, ignored: the call would succeed with the model
        # under no schema at all, and the only sign would be output that usually looks
        # right.
        payload["output_config"]["format"] = {
            "type": "json_schema",
            "schema": strict_subset(request.output_schema),
        }

    return payload


def _with_conversation_cached(messages: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    """The messages, with the conversation so far marked for caching in a tool loop.

    A tool loop resends every earlier turn each time, so its input grows with every call:
    Westpac's locate loop sent 212, 5,157, 16,598 and finally 28,183 tokens, and only the
    system prompt was cached. Marking the last block of the newest turn caches everything
    before it, and the next turn reads that back at a tenth of the input price.

    A single-turn call isn't marked: nothing reads it again, and a cache write costs a
    quarter more than plain input. Copies are made, so the caller's list isn't changed.
    """
    out = [dict(message) for message in messages]
    if len(out) < 3:
        return out
    last = out[-1]
    content = last.get("content")
    marker = {"type": "ephemeral"}
    if isinstance(content, str) and content:
        last["content"] = [{"type": "text", "text": content, "cache_control": marker}]
    elif isinstance(content, list) and content:
        blocks = [dict(block) for block in content]
        blocks[-1]["cache_control"] = marker
        last["content"] = blocks
    return out


def parse_response(payload: dict[str, Any]) -> LLMResponse:
    """A provider response, read by block type.

    Thinking blocks come first, so position is never used. Text blocks are joined in order,
    tool-use blocks become :class:`ToolCall`, and the whole block list is kept so a tool
    loop can append it back unchanged.
    """
    blocks = tuple(payload.get("content") or ())
    text = "\n".join(
        block.get("text", "") for block in blocks_by_type(blocks, "text") if block.get("text")
    )
    tool_calls = tuple(
        ToolCall(
            id=str(block.get("id", "")),
            name=str(block.get("name", "")),
            arguments=dict(block.get("input") or {}),
        )
        for block in blocks_by_type(blocks, "tool_use")
    )

    structured: dict[str, Any] | None = None
    # A structured output arrives as its own block where the API supports it, and as the
    # text body otherwise. Both are accepted; neither is guessed at.
    for block in blocks_by_type(blocks, "json"):
        candidate = block.get("json")
        if isinstance(candidate, dict):
            structured = candidate
            break
    if structured is None and text:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                loaded = json.loads(stripped)
            except json.JSONDecodeError:
                loaded = None
            if isinstance(loaded, dict):
                structured = loaded

    raw_usage = payload.get("usage") or {}
    return LLMResponse(
        text=text,
        tool_calls=tool_calls,
        structured=structured,
        stop_reason=str(payload.get("stop_reason") or "end_turn"),
        usage=Usage(
            input_tokens=int(raw_usage.get("input_tokens") or 0),
            output_tokens=int(raw_usage.get("output_tokens") or 0),
            cache_read_tokens=int(raw_usage.get("cache_read_input_tokens") or 0),
            cache_write_tokens=int(raw_usage.get("cache_creation_input_tokens") or 0),
            web_searches=int(
                (raw_usage.get("server_tool_use") or {}).get("web_search_requests") or 0
            ),
        ),
        raw_blocks=blocks,
    )


@dataclass
class AnthropicClient:
    """Live calls to the provider, behind the ``LLMClient`` protocol."""

    settings: Settings = field(default_factory=get_settings)
    fallback_model: str | None = None
    consent: ConsentGate = field(default_factory=default_gate)
    """Asked before every paid call. Defaults to refusing: see :mod:`.consent`."""
    pricing: dict[str, dict[str, str]] = field(default_factory=dict)
    _sdk: Any = None
    """Injected in tests, so the payload and the fallback can be tested without the SDK, a
    key or a network.
    """

    def __post_init__(self) -> None:
        models = get_registry().models
        if self.fallback_model is None:
            self.fallback_model = models.get("fallback_model")
        if not self.pricing:
            self.pricing = models.get("pricing") or {}

    @property
    def available(self) -> bool:
        return bool(self.settings.has_api_key or self._sdk is not None)

    def complete(self, request: LLMRequest) -> LLMResponse:
        if not self.available:
            raise ModelUnreachable(
                "no_key",
                "no API key was found (ANTHROPIC_API_KEY, .env or the keychain), so no model call "
                "can be made",
            )
        try:
            return self._send(request, model=request.model)
        except TransientProviderError as error:
            # Three attempts with backoff have already gone by. A limit that outlasts that,
            # or a service that does, is not something the run can wait out.
            limited = "429" in str(error) or "rate limit" in str(error)
            raise ModelUnreachable(
                "rate_limited" if limited else "unreachable", str(error)
            ) from error
        except ProviderError as error:
            reason = str(error)
            if self.fallback_model and any(name in reason for name in _FALLBACK_ON):
                # One retry, on the fallback model, for the two failures it usually gets
                # past. Not a general retry: that just turns a bad request into a slow one.
                # It's a second charge, so consent is asked again and says why.
                return self._send(
                    request,
                    model=self.fallback_model,
                    attempt=f"fallback model, after: {reason}",
                )
            raise

    # -- internals --------------------------------------------------------

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(TransientProviderError),
    )
    def _send(self, request: LLMRequest, model: str, attempt: str = "first attempt") -> LLMResponse:
        payload = build_payload(request, model=model)
        raw = self._call(payload, step=request.step, attempt=attempt)
        response = parse_response(raw)
        if response.stop_reason == "refusal":
            raise ProviderError(f"refusal on step {request.step}")
        return response

    def _call(
        self, payload: dict[str, Any], *, step: str = "", attempt: str = "first attempt"
    ) -> dict[str, Any]:
        # The gate goes here rather than in ``complete``, because this is the only method
        # that reaches the network. A tenacity retry and a fallback-model retry both come
        # back through here, and both are charged, so both are asked about.
        estimate = estimate_call(payload, step=step, pricing=self.pricing, attempt=attempt)
        if not self.consent.approve(estimate):
            raise ConsentDenied(
                f"consent was not given for the {step or 'unnamed'} call to "
                f"{estimate.model} (up to {aud(estimate.ceiling_usd, up=True)}), so it was not made"
            )
        client = self._client()
        try:
            message = client.messages.create(**payload)
        except Exception as error:  # the SDK's exception tree is not importable here
            # Not settled on purpose: a call that raised may still have been billed, so its
            # ceiling stays committed against the grant.
            raise self._classify(error) from error
        dump = getattr(message, "model_dump", None)
        result = dump() if callable(dump) else message
        raw = dict(result)
        self._settle(raw, estimate)
        return raw

    def _settle(self, raw: dict[str, Any], estimate: CallEstimate) -> None:
        """Tell the gate what the call cost, if it keeps a running total.

        Optional on the gate: ``DenyAll`` and ``AlwaysApprove`` have no budget to settle,
        and a gate written elsewhere shouldn't need a method it has no use for.
        """
        settle = getattr(self.consent, "settle", None)
        if not callable(settle):
            return
        # Through the same function the run's budget uses, so cache reads and writes are
        # counted the same way in both places.
        actual = cost_usd(parse_response(raw).usage, self.pricing.get(estimate.model, {}))
        try:
            settle(actual, estimate)
        except TypeError:  # a gate written for one call at a time
            settle(actual)

    def _client(self) -> Any:
        if self._sdk is not None:
            return self._sdk
        try:
            import anthropic
        except ModuleNotFoundError as error:  # pragma: no cover - the extra is optional
            raise LLMUnavailable(
                "the Anthropic SDK isn't installed. Run `make setup`, or `pip install -c constraints.txt -e '.[llm]'`"
            ) from error
        key = self.settings.anthropic_api_key
        assert key is not None  # guarded by `available`
        self._sdk = anthropic.Anthropic(api_key=key.get_secret_value())
        return self._sdk

    @staticmethod
    def _classify(error: Exception) -> LLMUnavailable:
        """A provider exception, classed as either transient or terminal.

        Classified by status code and message rather than by the SDK's exception classes,
        which can't be imported while the SDK is optional. The key never appears in either:
        it isn't in the message, and the payload isn't attached.
        """
        status = getattr(error, "status_code", None)
        text = str(error).lower()
        if status == 401 or "authentication_error" in text or "invalid x-api-key" in text:
            return ModelUnreachable("key_rejected", f"the API returned {status or 'an error'}")
        if status == 403 or "permission_error" in text:
            return ModelUnreachable("key_rejected", "the key has no permission for this model")
        if "credit balance" in text or "billing" in text or "insufficient credit" in text:
            return ModelUnreachable("out_of_credit", "the account's credit balance is too low")
        if isinstance(status, int) and status in _TRANSIENT_STATUS:
            return TransientProviderError(f"provider returned {status}")
        if status is None and ("connection" in text or "connect" in text):
            return TransientProviderError(f"could not connect: {type(error).__name__}")
        if any(name in text for name in _FALLBACK_ON):
            return ProviderError(text)
        if "rate limit" in text or "overloaded" in text or "timeout" in text:
            return TransientProviderError(text)
        return ProviderError(text)
