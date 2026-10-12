"""The subset of JSON Schema the provider accepts under ``strict``."""

from __future__ import annotations

import copy
from typing import Any, Final

__all__ = ["UNSUPPORTED_KEYWORDS", "strict_subset"]

#: Validation keywords outside the subset. Numeric bounds, string lengths and patterns,
#: and array constraints beyond the trivial ones.
UNSUPPORTED_KEYWORDS: Final[tuple[str, ...]] = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "minItems",
    "maxItems",
    "uniqueItems",
)


def strict_subset(schema: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``schema`` the API will accept under ``strict``.

    Every object gets ``additionalProperties: false``, which strict mode requires and which
    is worth having anyway. An argument a tool doesn't define should be refused rather than
    ignored, because ignoring it lets a model think it narrowed a search that was actually
    unfiltered.

    Copied rather than edited in place. A schema is often a module-level constant, and a
    function that rewrote one in place would make the second call behave differently from
    the first.
    """
    result = copy.deepcopy(schema)
    _apply(result)
    return result


def _apply(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            node["additionalProperties"] = False
        _move_to_description(node)
        for value in node.values():
            _apply(value)
    elif isinstance(node, list):
        for item in node:
            _apply(item)


def _move_to_description(node: dict[str, Any]) -> None:
    """Take the unsupported keywords off, and say what they were."""
    moved = {key: node.pop(key) for key in UNSUPPORTED_KEYWORDS if key in node}
    if not moved:
        return
    stated = ", ".join(f"{key}: {value}" for key, value in moved.items())
    existing = str(node.get("description", "")).strip()
    node["description"] = f"{existing} ({stated})" if existing else f"({stated})"
