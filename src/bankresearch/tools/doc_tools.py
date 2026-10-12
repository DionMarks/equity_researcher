"""The tools a model can call, and the scope it can't widen."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..domain.models import DocMeta, ReadingItem, ReadingList, RequestSpec
from ..domain.registry import get_registry
from ..ingestion.parse_pdf import ParsedDocument, ParsedTable
from ..ingestion.tables import map_columns
from ..ingestion.walk_tables import walk_from_table
from ..retrieval.chunks import Chunk, chunk_document
from ..retrieval.hybrid import Retriever
from .schemas import (
    GetOutlineInput,
    GetTableInput,
    OutlineEntry,
    OutlineResult,
    PageContent,
    PageResult,
    Provenance,
    ReadChartInput,
    ReadingListInput,
    ReadPagesInput,
    SearchDocsInput,
    SearchHit,
    SearchResult,
    TableResult,
    TableSummary,
    ToolError,
    WalkReading,
    WalkStepReading,
)

__all__ = ["DOCUMENT_WRAPPER", "Toolkit"]

DOCUMENT_WRAPPER = "<document_content>\n{content}\n</document_content>"

_CHARS_PER_TOKEN = 4


def _result_token_cap() -> int:
    """The cap on one tool result, from ``config/budgets.yaml``.

    Read from config rather than hardcoded. It used to be both: the same 6000 in config and
    in this module, with nothing linking them, so raising the cap where anyone would look
    for it changed nothing.
    """
    from ..domain.registry import get_registry

    return int(get_registry().budgets.get("tool_result_token_cap", 6000))


_RESULT_TOKEN_CAP = _result_token_cap()
_MAX_RESULT_CHARS = _RESULT_TOKEN_CAP * _CHARS_PER_TOKEN


@dataclass
class Toolkit:
    """One request's tool surface. Scoped, bounded, cached and idempotent."""

    spec: RequestSpec
    documents: dict[str, ParsedDocument]
    metas: dict[str, DocMeta]
    retriever: Retriever | None = None
    reading_list: ReadingList | None = None
    call_log: list[dict[str, Any]] = field(default_factory=list)
    _cache: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.retriever is None:
            retriever = Retriever()
            chunks: list[Chunk] = []
            for doc_id, document in self.documents.items():
                meta = self.metas.get(doc_id)
                if meta is not None:
                    chunks.extend(chunk_document(document, meta))
            retriever.add(chunks)
            self.retriever = retriever

    # -- dispatch ---------------------------------------------------------

    def dispatch(self, name: str, arguments: dict[str, Any]) -> Any:
        """Run one tool. Cached within a run, so a repeated call is free and identical."""
        key = f"{name}:{sorted(arguments.items())!r}"
        if key in self._cache:
            self.call_log.append({"tool": name, "args": arguments, "cached": True})
            return self._cache[key]

        handler = {
            "get_outline": self._get_outline,
            "search_docs": self._search_docs,
            "read_pages": self._read_pages,
            "get_table": self._get_table,
            "read_chart": self._read_chart,
            "submit_reading_list": self._submit_reading_list,
        }.get(name)
        if handler is None:
            return ToolError(
                error="unknown_tool",
                detail=f"{name!r} is not a tool on this surface.",
                suggestion="Available: get_outline, search_docs, read_pages, get_table, "
                "read_chart, submit_reading_list.",
            )

        self.call_log.append({"tool": name, "args": arguments, "cached": False})
        result = handler(arguments)
        self._cache[key] = result
        return result

    # -- scope ------------------------------------------------------------

    def _document(self, doc_id: str) -> ParsedDocument | ToolError:
        """A document this request may read, or an error naming the ones it may."""
        document = self.documents.get(doc_id)
        if document is None:
            return ToolError(
                error="document_not_in_scope",
                detail=(
                    f"{doc_id!r} is not among the documents for {self.spec.bank_id} "
                    f"{self.spec.period} or {self.spec.comparison_period} published on or "
                    f"before {self.spec.as_of.isoformat()}."
                ),
                suggestion="Available: " + ", ".join(sorted(self.documents)) + ".",
            )
        return document

    def _provenance(self, doc_id: str, page: int, table_id: str | None = None) -> Provenance:
        meta = self.metas[doc_id]
        return Provenance(
            doc_id=doc_id,
            doc_type=meta.doc_type,
            page=page,
            table_id=table_id,
            publish_date=meta.publish_date,
        )

    def _periods_for(self, doc_id: str) -> str:
        return self.metas[doc_id].period

    # -- tools ------------------------------------------------------------

    def _get_outline(self, arguments: dict[str, Any]) -> OutlineResult | ToolError:
        request = GetOutlineInput.model_validate(arguments)
        document = self._document(request.doc_id)
        if isinstance(document, ToolError):
            return document

        tables: list[TableSummary] = []
        for table in document.tables():
            tables.append(
                TableSummary(
                    table_id=table.table_id,
                    page=table.page_no,
                    caption=table.caption[:120],
                    periods=sorted(set(self._periods_of(table, request.doc_id).values())),
                )
            )
        truncated = len(tables) > 60
        return OutlineResult(
            doc_id=request.doc_id,
            doc_type=self.metas[request.doc_id].doc_type,
            page_count=document.page_count,
            sections=[
                OutlineEntry(heading=heading, first_page=page)
                for heading, page in document.outline[:80]
            ],
            tables=tables[:60],
            chart_pages=list(document.walk_pages()[:40]),
            truncated=truncated,
            cursor=60 if truncated else None,
        )

    def _search_docs(self, arguments: dict[str, Any]) -> SearchResult | ToolError:
        request = SearchDocsInput.model_validate(arguments)
        assert self.retriever is not None
        hits = self.retriever.search(
            request.query,
            bank_id=self.spec.bank_id,
            periods=self.spec.periods,
            as_of=self.spec.as_of,
            doc_types=tuple(request.doc_types) if request.doc_types else None,
            limit=request.k,
        )
        return SearchResult(
            query=request.query,
            hits=[
                SearchHit(
                    source=self._provenance(
                        hit.chunk.doc_id, hit.chunk.page_no, hit.chunk.table_id
                    ),
                    kind=hit.chunk.kind,  # type: ignore[arg-type]
                    caption=hit.chunk.caption[:120],
                    snippet=hit.chunk.snippet(),
                    score=round(hit.score, 4),
                )
                for hit in hits
            ],
            searched_periods=list(self.spec.periods),
            note=(
                "Scoped to this issuer and the two periods under comparison; a search "
                "cannot reach another bank's documents."
            ),
        )

    def _read_pages(self, arguments: dict[str, Any]) -> PageResult | ToolError:
        request = ReadPagesInput.model_validate(arguments)
        document = self._document(request.doc_id)
        if isinstance(document, ToolError):
            return document

        out: list[PageContent] = []
        budget = _MAX_RESULT_CHARS
        truncated = False
        notes: list[str] = []

        for page_no in request.pages:
            page = document.page(page_no)
            if page is None:
                nearest = [
                    candidate.page_no
                    for candidate in document.pages
                    if abs(candidate.page_no - page_no) <= 2
                ]
                notes.append(
                    f"page {page_no} out of range: {request.doc_id} has "
                    f"{document.page_count} pages"
                    + (f"; nearest available are {nearest}" if nearest else "")
                )
                continue

            text = ""
            if request.mode in {"text", "both"}:
                text = DOCUMENT_WRAPPER.format(content=page.text[:budget])
                budget -= len(text)
            tables: list[str] = []
            if request.mode in {"tables", "both"}:
                for table in page.tables:
                    markdown = table.to_markdown()
                    if len(markdown) > budget:
                        truncated = True
                        break
                    tables.append(markdown)
                    budget -= len(markdown)

            out.append(
                PageContent(
                    source=self._provenance(request.doc_id, page_no),
                    kind=page.kind,
                    heading_path=list(page.heading_path),
                    text=text,
                    tables=tables,
                )
            )
            if budget <= 0:
                truncated = True
                break

        if truncated:
            notes.append(
                f"result truncated at about {_RESULT_TOKEN_CAP} tokens; request fewer pages"
            )
        return PageResult(pages=out, truncated=truncated, note=" ".join(notes))

    def _get_table(self, arguments: dict[str, Any]) -> TableResult | ToolError:
        request = GetTableInput.model_validate(arguments)
        document = self._document(request.doc_id)
        if isinstance(document, ToolError):
            return document

        table = document.table(request.table_id)
        if table is None:
            on_page = [t.table_id for t in document.tables()][:12]
            return ToolError(
                error="table_not_found",
                detail=f"{request.table_id!r} is not a table in {request.doc_id}.",
                suggestion="Call get_outline for the table ids. Some are: " + ", ".join(on_page),
            )

        markdown = table.to_markdown()
        truncated = len(markdown) > _MAX_RESULT_CHARS
        return TableResult(
            source=self._provenance(request.doc_id, table.page_no, table.table_id),
            caption=table.caption,
            units=table.units,
            restated=table.restated,
            column_periods=self._periods_of(table, request.doc_id),
            markdown=DOCUMENT_WRAPPER.format(content=markdown[:_MAX_RESULT_CHARS]),
            truncated=truncated,
        )

    def _read_chart(self, arguments: dict[str, Any]) -> WalkReading | ToolError:
        request = ReadChartInput.model_validate(arguments)
        document = self._document(request.doc_id)
        if isinstance(document, ToolError):
            return document

        page = document.page(request.page)
        if page is None:
            return ToolError(
                error="page_out_of_range",
                detail=f"page {request.page} out of range: {request.doc_id} has "
                f"{document.page_count} pages.",
                suggestion=f"Walk-bearing pages in this document: {document.walk_pages()[:12]}",
            )

        meta = self.metas[request.doc_id]
        for table in page.tables:
            walk = walk_from_table(table, self.spec.metric_id, meta)
            if walk is None:
                continue
            tolerance = 1.0 * max(len(walk.steps), 1)
            return WalkReading(
                source=self._provenance(request.doc_id, request.page, table.table_id),
                start=float(walk.opening_value),
                steps=[
                    WalkStepReading(label=step.label, value=float(step.value))
                    for step in walk.steps
                ],
                end=float(walk.closing_value),
                unit=walk.unit,
                checksum_passed=abs(float(walk.checksum_residual())) <= tolerance,
                note=f"Tabulated walk from {walk.opening_period} to {walk.closing_period}.",
            )

        return WalkReading(
            source=self._provenance(request.doc_id, request.page),
            note=(
                "No tabulated walk for this metric on this page. A walk published only as "
                "a chart image needs vision extraction, which is not configured."
            ),
        )

    def _submit_reading_list(self, arguments: dict[str, Any]) -> dict[str, Any]:
        request = ReadingListInput.model_validate(arguments)
        self.reading_list = ReadingList(
            items=[
                ReadingItem(
                    component=item.component,
                    doc_id=item.doc_id,
                    pages=item.pages,
                    table_ids=item.table_ids,
                    rationale=item.rationale,
                )
                for item in request.items
            ]
        )
        return {
            "accepted": True,
            "components": sorted({item.component for item in request.items}),
            "note": "Reading list recorded; the locate loop ends here.",
        }

    # -- helpers ----------------------------------------------------------

    def _periods_of(self, table: ParsedTable, doc_id: str) -> dict[str, str]:
        """Column header to period, for this issuer's calendar. Movements excluded."""
        profile = get_registry().bank(self.spec.bank_id)
        headers = [header for header in table.header if header and header.strip()]
        if not headers:
            return {}
        try:
            return map_columns(
                headers,
                fy_end_month=profile.fy_end_month,
                doc_period=self._periods_for(doc_id),
            )
        except ValueError:
            return {}
