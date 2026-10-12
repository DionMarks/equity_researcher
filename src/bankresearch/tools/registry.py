"""What the model is told it can call, and what it can't."""

from __future__ import annotations

from typing import Any

from .schemas import (
    GetOutlineInput,
    GetTableInput,
    ReadChartInput,
    ReadingListInput,
    ReadPagesInput,
    SearchDocsInput,
    tool_schema,
)

__all__ = ["LLM_CALLABLE", "ORCHESTRATOR_ONLY", "SUBMIT_READING_LIST", "tool_definitions"]

SUBMIT_READING_LIST = "submit_reading_list"

LLM_CALLABLE: tuple[str, ...] = (
    "get_outline",
    "search_docs",
    "read_pages",
    "get_table",
    "read_chart",
    SUBMIT_READING_LIST,
)

# Named so the separation can be tested, not just described. Each of these is either
# deterministic (where a model adds risk and nothing else) or security-relevant.
ORCHESTRATOR_ONLY: tuple[str, ...] = (
    "resolve_request",
    "check_availability",
    "ingest",
    "validate",
    "get_facts",
    "reconcile",
    "decompose",
    "score_confidence",
    "verify",
    "render",
    "fetch_url",
)

_DESCRIPTIONS: dict[str, tuple[str, type]] = {
    "get_outline": (
        "The document's section tree, its tables with their captions and the periods each "
        "reports, and the pages that may carry a walk. Call this first to find where to "
        "look rather than searching blind.",
        GetOutlineInput,
    ),
    "search_docs": (
        "Search the ingested text for passages relevant to a query. Returns ranked "
        "snippets, each with the document and page needed to cite it. Prefer several "
        "narrow searches over one broad one, and use the issuer's own vocabulary. The "
        "search is already scoped to this bank and the two periods under comparison.",
        SearchDocsInput,
    ),
    "read_pages": (
        "Read up to three pages in full, with their tables as markdown. Use when a search "
        "snippet is promising but truncated.",
        ReadPagesInput,
    ),
    "get_table": (
        "One table in full, with its units, its restatement flag, and the map from each "
        "column header to the period it reports. The column map is how you tell the prior "
        "corresponding period's column from the immediately prior half's.",
        GetTableInput,
    ),
    "read_chart": (
        "A management walk on this page, as start, steps and end, with whether its "
        "checksum held. A walk whose steps do not sum to its endpoints is not usable.",
        ReadChartInput,
    ),
    SUBMIT_READING_LIST: (
        "Submit the pages and tables that hold each component you were asked to find. "
        "This ends the loop. Call it once you have located the components in the "
        "checklist, or as soon as you are confident the remaining ones are not in these "
        "documents.",
        ReadingListInput,
    ),
}


def tool_definitions() -> list[dict[str, Any]]:
    """The tool definitions sent to the model, byte-identical across every request.

    Stability matters: the definitions and the system prompt form the prompt-cache
    prefix, so anything that varies per request must sit after them.
    """
    return [
        {
            "name": name,
            "description": description,
            "strict": True,
            "input_schema": tool_schema(model),
        }
        for name, (description, model) in _DESCRIPTIONS.items()
    ]
