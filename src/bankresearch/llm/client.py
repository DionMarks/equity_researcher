"""The model boundary: one protocol, three implementations, no hidden calls."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol

__all__ = [
    "CacheMiss",
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "LLMUnavailable",
    "ModelUnreachable",
    "ScriptedClient",
    "ToolCall",
    "Usage",
    "request_hash",
]


class LLMUnavailable(RuntimeError):
    """A model call was needed and no model is available."""


class ModelUnreachable(LLMUnavailable):
    """The model can't be used at all, so the question can't be answered.

    Every other model failure (a refusal, an answer that doesn't parse, a step over its
    limit) costs that step and the run carries on. This one ends the run with a message
    saying which it is, because an answer without the model isn't the answer that was
    asked for:

    - ``no_key``: no API key was found;
    - ``key_rejected``: the API refused the key (wrong, revoked or without permission);
    - ``out_of_credit``: the account has no credit left to spend;
    - ``rate_limited``: the API kept limiting requests on this key after retrying;
    - ``unreachable``: the network or the service kept failing after retrying.
    """

    KINDS = ("no_key", "key_rejected", "out_of_credit", "rate_limited", "unreachable")

    def __init__(self, kind: str, detail: str = "") -> None:
        if kind not in self.KINDS:
            raise ValueError(f"unknown kind {kind!r}")
        super().__init__(detail or kind)
        self.kind = kind
        self.detail = detail


class CacheMiss(LLMUnavailable):
    """Replay was asked for a request it holds no response to."""


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    web_searches: int = 0
    """Server-side searches, billed per search rather than per token."""

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
            self.web_searches + other.web_searches,
        )


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool the model asked for, and the id its result must carry back."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """One call. Hashed to key the cache, so the key covers everything that matters."""

    step: str
    model: str
    effort: str
    system: str
    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...] = ()
    output_schema: dict[str, Any] | None = None
    max_tokens: int | None = None
    """The output limit, thinking included. ``None`` means the step's cap from
    ``config/budgets.yaml``, which is what every step uses."""

    def hashed(self) -> str:
        return request_hash(self)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """What came back: text, any tool calls, why it stopped, and what it cost."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    structured: dict[str, Any] | None = None
    stop_reason: str = "end_turn"
    usage: Usage = field(default=Usage())
    raw_blocks: tuple[dict[str, Any], ...] = ()
    """Assistant content blocks, kept verbatim so a tool loop can pass them back
    unmodified, as the API requires."""
    cached: bool = False
    """True when this came from the response cache and so cost nothing.

    A cache hit carries the original call's token counts, which is what makes it reusable
    and also what made it get billed twice: recording it like any other response reported
    A$0.095 for a run that made no calls at all. Over-reporting isn't harmless either,
    because the per-run limit would refuse real work to stay under a figure nobody was
    charged.
    """

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


def request_hash(request: LLMRequest) -> str:
    """A stable key over everything that could change the answer.

    Model, effort, prompt, inputs and schema. Two requests that differ in any of them are
    different requests, which stops a prompt change from reusing a stale recording (the
    usual way a replayed eval becomes untrustworthy).
    """
    payload = json.dumps(
        {
            "step": request.step,
            "model": request.model,
            "effort": request.effort,
            "system": request.system,
            "messages": request.messages,
            "tools": [tool.get("name") for tool in request.tools],
            "schema": request.output_schema,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class LLMClient(Protocol):
    """The only way to reach a model."""

    @property
    def available(self) -> bool: ...

    def complete(self, request: LLMRequest) -> LLMResponse: ...


@dataclass
class ScriptedClient:
    """Returns prepared responses in order, for testing the loop without a model."""

    responses: list[LLMResponse] = field(default_factory=list)
    requests: list[LLMRequest] = field(default_factory=list)
    on_exhausted: Callable[[LLMRequest], LLMResponse] | None = None

    @property
    def available(self) -> bool:
        return True

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.responses:
            return self.responses.pop(0)
        if self.on_exhausted is not None:
            return self.on_exhausted(request)
        raise LLMUnavailable(
            f"the script ran out of responses at call {len(self.requests)} ({request.step})"
        )


#: What writing a prompt to the cache costs, as a multiple of the base input price.
CACHE_WRITE_MULTIPLIER = Decimal("1.25")
#: One server-side web search, in US dollars as it's billed: A$14.33 per 1,000.
WEB_SEARCH_PRICE = Decimal("0.01")


def cost_usd(usage: Usage, pricing: dict[str, str]) -> Decimal:
    """Cost from token counts and the pricing table, as a Decimal.

    Reported in US dollars because that's how the provider prices. The research output is in
    AUD, and a cost in the wrong currency is easy to misread, so both are labelled.

    Cache writes are billed at 1.25 times the input price and were left out, so every run
    that wrote a system prompt to the cache reported less than it was charged.
    """
    per_million = Decimal(1_000_000)
    input_price = Decimal(pricing.get("input", "0"))
    return (
        Decimal(usage.input_tokens) * input_price / per_million
        + Decimal(usage.output_tokens) * Decimal(pricing.get("output", "0")) / per_million
        + Decimal(usage.cache_read_tokens) * Decimal(pricing.get("cache_read", "0")) / per_million
        + Decimal(usage.cache_write_tokens) * input_price * CACHE_WRITE_MULTIPLIER / per_million
        + Decimal(usage.web_searches) * WEB_SEARCH_PRICE
    )


def blocks_by_type(blocks: Sequence[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    """Content blocks of one type.

    Responses begin with thinking blocks, so a caller that assumed the first block was
    text would read the model's reasoning as its answer.
    """
    return [block for block in blocks if block.get("type") == kind]
