"""Chunks an ingested document for retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ..domain.models import DocMeta
from ..ingestion.parse_pdf import ParsedDocument

__all__ = ["Chunk", "chunk_document"]

_MAX_CHARS = 3200  # about 800 tokens
_MIN_CHARS = 40


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    bank_id: str
    period: str
    doc_type: str
    page_no: int
    publish_date: date
    kind: str
    """``table``, ``prose`` or ``boilerplate``."""
    text: str
    caption: str = ""
    table_id: str | None = None
    heading_path: tuple[str, ...] = field(default=())

    def snippet(self, limit: int = 300) -> str:
        flat = " ".join(self.text.split())
        return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def chunk_document(document: ParsedDocument, meta: DocMeta) -> list[Chunk]:
    """Every retrievable unit in one document."""
    chunks: list[Chunk] = []

    def add(kind: str, text: str, page_no: int, **extra: object) -> None:
        if len(text.strip()) < _MIN_CHARS:
            return
        chunks.append(
            Chunk(
                chunk_id=f"{meta.doc_id}:{page_no}:{kind}:{len(chunks)}",
                doc_id=meta.doc_id,
                bank_id=meta.bank_id,
                period=meta.period,
                doc_type=meta.doc_type,
                page_no=page_no,
                publish_date=meta.publish_date,
                kind=kind,
                text=text.strip(),
                **extra,  # type: ignore[arg-type]
            )
        )

    for page in document.pages:
        for table in page.tables:
            add(
                "table",
                f"{table.caption}\n{table.to_markdown()}",
                page.page_no,
                caption=table.caption,
                table_id=table.table_id,
                heading_path=page.heading_path,
            )
        kind = "boilerplate" if page.kind == "boilerplate" else "prose"
        for block in _blocks(page.text):
            add(kind, block, page.page_no, heading_path=page.heading_path)
    return chunks


def _blocks(text: str) -> list[str]:
    """Split page text into section-sized blocks under the character cap."""
    paragraphs = [p.strip() for p in (text or "").split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [line for line in (text or "").splitlines() if line.strip()]

    blocks: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in paragraphs:
        if size + len(paragraph) > _MAX_CHARS and current:
            blocks.append("\n".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph)
    if current:
        blocks.append("\n".join(current))
    return blocks
