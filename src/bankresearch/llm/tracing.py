"""Structured events for a run, and the prompt provenance that makes it reproducible."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .client import LLMRequest, LLMResponse, Usage

__all__ = [
    "PROMPT_DIR",
    "PromptLibrary",
    "Tracer",
    "prompt_sha256",
]

PROMPT_DIR = Path(__file__).resolve().parents[1] / "agent" / "prompts"


def prompt_sha256(text: str) -> str:
    """The hash recorded against every event that used this prompt."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Prompt:
    """One prompt, with the hash that identifies it."""

    name: str
    text: str
    sha256: str


@dataclass
class PromptLibrary:
    """Prompts loaded from disk by name.

    Kept on disk rather than in code so a reviewer can read them without reading Python, and
    so the hash covers the file itself. A missing prompt is an error, not a default: a step
    that fell back to an empty system prompt would still answer, just worse, with no visible
    reason why.
    """

    root: Path = field(default_factory=lambda: PROMPT_DIR)

    def get(self, name: str) -> Prompt:
        path = self.root / f"{name}.md"
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError as error:
            available = sorted(p.stem for p in self.root.glob("*.md"))
            raise FileNotFoundError(
                f"no prompt named {name!r} in {self.root}. Available: {available}"
            ) from error
        return Prompt(name=name, text=text, sha256=prompt_sha256(text))

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(path.stem for path in self.root.glob("*.md")))


@lru_cache(maxsize=1)
def default_library() -> PromptLibrary:
    return PromptLibrary()


@dataclass
class Tracer:
    """Append-only events for one run.

    Append-only because a trace that can be rewritten isn't evidence. Events are plain dicts
    so they serialise to the ``trace.jsonl`` the orchestrator already writes, and the clock
    is monotonic so timestamps can't go backwards.
    """

    events: list[dict[str, Any]] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)

    def event(self, step: str, **fields: Any) -> dict[str, Any]:
        record = {"step": step, "at": round(time.perf_counter() - self.started, 6), **fields}
        self.events.append(record)
        return record

    def model_call(
        self,
        request: LLMRequest,
        response: LLMResponse,
        *,
        prompt: Prompt | None = None,
        cached: bool | None = None,
        cost_usd: str | None = None,
    ) -> dict[str, Any]:
        """One model call, with everything needed to reproduce or audit it.

        The request hash rather than the request: the hash identifies it exactly, and the
        body may hold page text that does not belong in a log. Whether it came from the
        response cache is read off the response unless the caller says otherwise; no caller
        did, so every replayed call was logged as a fresh one.
        """
        if cached is None:
            cached = bool(getattr(response, "cached", False))
        if cost_usd is None:
            cost_usd = _priced(request.model, response.usage, cached)
        return self.event(
            request.step,
            model=request.model,
            effort=request.effort,
            request_hash=request.hashed(),
            prompt=prompt.name if prompt else None,
            prompt_sha256=prompt.sha256 if prompt else None,
            cached=cached,
            stop_reason=response.stop_reason,
            tool_calls=[call.name for call in response.tool_calls],
            usage=_usage(response.usage),
            cost_usd=cost_usd,
        )

    def skipped(self, step: str, why: str, **fields: Any) -> dict[str, Any]:
        return self.event(step, skipped=why, **fields)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in self.events:
                handle.write(json.dumps(record, default=str) + "\n")


def _priced(model: str, usage: Usage, cached: bool) -> str | None:
    """What one call cost, at the configured price, so a trace says where the money went.

    Every call was traced with ``cost_usd: null``: the run's total was right, but nobody
    could tell which step had spent it. A replayed call cost nothing, and an unpriced model
    stays ``None`` rather than reading as free.
    """
    if cached:
        return "0"
    from ..domain.registry import get_registry
    from .client import cost_usd

    # Bookkeeping for the trace only: the budget prices the call separately, so a price
    # that can't be looked up here leaves the trace without one rather than stopping a run
    # whose call has already been paid for.
    try:
        prices = (get_registry().models.get("pricing") or {}).get(model)
    except Exception:
        return None
    return None if not prices else str(cost_usd(usage, prices))


def _usage(usage: Usage) -> dict[str, int]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "web_searches": usage.web_searches,
    }
