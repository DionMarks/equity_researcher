"""The response cache, which is what makes a replayed run free and identical."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from .client import CacheMiss, LLMClient, LLMRequest, LLMResponse, ToolCall, Usage

__all__ = ["CachedClient", "ResponseCache"]


@dataclass
class ResponseCache:
    """Stored model responses, in the same SQLite file as the facts."""

    connection: sqlite3.Connection

    def get(self, request: LLMRequest) -> LLMResponse | None:
        row = self.connection.execute(
            "SELECT response FROM llm_cache WHERE request_hash = ?", (request.hashed(),)
        ).fetchone()
        if row is None:
            return None
        return _decode(json.loads(row[0]))

    def put(self, request: LLMRequest, response: LLMResponse) -> None:
        self.connection.execute(
            """INSERT OR REPLACE INTO llm_cache
               (request_hash, model, effort, prompt_hash, schema_hash, response, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                request.hashed(),
                request.model,
                request.effort,
                _hash_of(request.system),
                _hash_of(json.dumps(request.output_schema, sort_keys=True)),
                json.dumps(_encode(response)),
                datetime.now(UTC).isoformat(),
            ),
        )

    def __len__(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0])


@dataclass
class CachedClient:
    """Reads through the cache, and in replay mode never beyond it."""

    cache: ResponseCache
    upstream: LLMClient | None = None
    replay_only: bool = True

    @property
    def available(self) -> bool:
        return self.replay_only or (self.upstream is not None and self.upstream.available)

    def complete(self, request: LLMRequest) -> LLMResponse:
        hit = self.cache.get(request)
        if hit is not None:
            # Marked here, because this is the only place that can tell. Everything above
            # counts spend from the response it's given and has no other way to know.
            return replace(hit, cached=True)
        if self.replay_only or self.upstream is None:
            raise CacheMiss(
                f"no recorded response for {request.step} "
                f"({request.hashed()[:12]}); a live run with a key records one"
            )
        fresh = self.upstream.complete(request)
        self.cache.put(request, fresh)
        return fresh


def _hash_of(text: str) -> str:
    import hashlib

    return hashlib.sha256((text or "").encode()).hexdigest()[:16]


def _encode(response: LLMResponse) -> dict[str, Any]:
    return {
        "text": response.text,
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in response.tool_calls
        ],
        "structured": response.structured,
        "stop_reason": response.stop_reason,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_read_tokens": response.usage.cache_read_tokens,
            "cache_write_tokens": response.usage.cache_write_tokens,
        },
        "raw_blocks": list(response.raw_blocks),
    }


def _decode(payload: dict[str, Any]) -> LLMResponse:
    usage = payload.get("usage", {}) or {}
    return LLMResponse(
        text=payload.get("text", ""),
        tool_calls=tuple(
            ToolCall(id=call["id"], name=call["name"], arguments=call["arguments"])
            for call in payload.get("tool_calls", [])
        ),
        structured=payload.get("structured"),
        stop_reason=payload.get("stop_reason", "end_turn"),
        usage=Usage(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_read_tokens=usage.get("cache_read_tokens", 0),
            cache_write_tokens=usage.get("cache_write_tokens", 0),
        ),
        raw_blocks=tuple(payload.get("raw_blocks", [])),
    )
