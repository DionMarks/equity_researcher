"""Typed inputs and outputs for the tools a model can call."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..llm.json_schema import strict_subset

__all__ = [
    "GetOutlineInput",
    "GetTableInput",
    "OutlineResult",
    "PageResult",
    "ReadChartInput",
    "ReadPagesInput",
    "ReadingListInput",
    "SearchDocsInput",
    "SearchHit",
    "SearchResult",
    "TableResult",
    "ToolError",
    "WalkReading",
    "tool_schema",
]

_STRICT = ConfigDict(extra="forbid")


class ToolError(BaseModel):
    """A refusal the model can act on.

    "page 240 out of range: document has 212 pages" tells it what to do next; "invalid
    input" does not.
    """

    model_config = _STRICT

    error: str
    detail: str
    suggestion: str = ""


class Provenance(BaseModel):
    model_config = _STRICT

    doc_id: str
    doc_type: str
    page: int
    table_id: str | None = None
    publish_date: date


# --- get_outline ----------------------------------------------------------


class GetOutlineInput(BaseModel):
    model_config = _STRICT

    doc_id: str = Field(description="A document listed for this request.")
    max_depth: int = Field(default=3, ge=1, le=5)


class OutlineEntry(BaseModel):
    model_config = _STRICT

    heading: str
    first_page: int


class TableSummary(BaseModel):
    model_config = _STRICT

    table_id: str
    page: int
    caption: str
    periods: list[str] = Field(default_factory=list)


class OutlineResult(BaseModel):
    model_config = _STRICT

    doc_id: str
    doc_type: str
    page_count: int
    sections: list[OutlineEntry] = Field(default_factory=list)
    tables: list[TableSummary] = Field(default_factory=list)
    chart_pages: list[int] = Field(default_factory=list)
    truncated: bool = False
    cursor: int | None = None


# --- search_docs ----------------------------------------------------------


class SearchDocsInput(BaseModel):
    model_config = _STRICT

    query: str = Field(description="Search terms. Use the issuer's own vocabulary.")
    doc_types: list[str] | None = Field(default=None, description="Restrict to these types.")
    k: int = Field(default=8, ge=1, le=15)


class SearchHit(BaseModel):
    model_config = _STRICT

    source: Provenance
    kind: Literal["table", "prose", "boilerplate"]
    caption: str = ""
    snippet: str = Field(description="300 characters or fewer.")
    score: float


class SearchResult(BaseModel):
    model_config = _STRICT

    query: str
    hits: list[SearchHit] = Field(default_factory=list)
    searched_periods: list[str] = Field(default_factory=list)
    note: str = ""


# --- read_pages -----------------------------------------------------------


class ReadPagesInput(BaseModel):
    model_config = _STRICT

    doc_id: str
    pages: list[int] = Field(min_length=1, max_length=3)
    mode: Literal["text", "tables", "both"] = "both"


class PageContent(BaseModel):
    model_config = _STRICT

    source: Provenance
    kind: str
    heading_path: list[str] = Field(default_factory=list)
    text: str = ""
    tables: list[str] = Field(default_factory=list, description="Markdown per table.")


class PageResult(BaseModel):
    model_config = _STRICT

    pages: list[PageContent] = Field(default_factory=list)
    truncated: bool = False
    note: str = ""


# --- get_table ------------------------------------------------------------


class GetTableInput(BaseModel):
    model_config = _STRICT

    doc_id: str
    table_id: str


class TableResult(BaseModel):
    model_config = _STRICT

    source: Provenance
    caption: str
    units: str | None = None
    restated: bool = False
    column_periods: dict[str, str] = Field(
        default_factory=dict,
        description="Column header to the period it reports. Movement columns are absent.",
    )
    markdown: str = ""
    truncated: bool = False


# --- read_chart -----------------------------------------------------------


class ReadChartInput(BaseModel):
    model_config = _STRICT

    doc_id: str
    page: int


class WalkStepReading(BaseModel):
    model_config = _STRICT

    label: str
    value: float


class WalkReading(BaseModel):
    model_config = _STRICT

    source: Provenance
    start: float | None = None
    steps: list[WalkStepReading] = Field(default_factory=list)
    end: float | None = None
    unit: str = ""
    checksum_passed: bool = False
    note: str = ""


# --- submit_reading_list -------------------------------------------------


class ReadingItemInput(BaseModel):
    model_config = _STRICT

    component: str
    doc_id: str
    pages: list[int] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)
    rationale: str = ""


class ReadingListInput(BaseModel):
    model_config = _STRICT

    items: list[ReadingItemInput] = Field(min_length=1)


def tool_schema(model: type[BaseModel]) -> dict[str, Any]:
    """A JSON schema for a tool's input, within the subset ``strict: true`` accepts.

    The subset is defined by the provider, so the rules live in ``llm/json_schema.py`` and
    are shared with the chart reader's structured output, which hit the same 400 error for
    the same reason.
    """
    return strict_subset(model.model_json_schema())
