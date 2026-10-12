"""Processed artefacts on disk."""

from __future__ import annotations

import json
from pathlib import Path

from .parse_pdf import ParsedDocument, ParsedPage, ParsedTable

__all__ = ["processed_dir_for", "read_processed", "write_processed"]


def processed_dir_for(root: Path, doc_id: str) -> Path:
    return Path(root) / doc_id


def write_processed(document: ParsedDocument, root: Path) -> Path:
    """Write pages, tables and the outline for one document."""
    target = processed_dir_for(root, document.doc_id)
    target.mkdir(parents=True, exist_ok=True)

    with (target / "pages.jsonl").open("w", encoding="utf-8") as handle:
        for page in document.pages:
            handle.write(
                json.dumps(
                    {
                        "page_no": page.page_no,
                        "kind": page.kind,
                        "heading_path": list(page.heading_path),
                        "text": page.text,
                    }
                )
                + "\n"
            )

    with (target / "tables.jsonl").open("w", encoding="utf-8") as handle:
        for table in document.tables():
            handle.write(
                json.dumps(
                    {
                        "table_id": table.table_id,
                        "page_no": table.page_no,
                        "caption": table.caption,
                        "header": list(table.header),
                        "rows": [list(row) for row in table.rows],
                        "units": table.units,
                        "restated": table.restated,
                    }
                )
                + "\n"
            )

    (target / "outline.json").write_text(
        json.dumps(
            {
                "doc_id": document.doc_id,
                "sha256": document.sha256,
                "page_count": document.page_count,
                "outline": [[heading, page] for heading, page in document.outline],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def read_processed(doc_id: str, root: Path) -> ParsedDocument | None:
    """Read a document back, or ``None`` if it has not been ingested."""
    target = processed_dir_for(root, doc_id)
    outline_path = target / "outline.json"
    if not outline_path.exists():
        return None

    meta = json.loads(outline_path.read_text(encoding="utf-8"))

    tables_by_page: dict[int, list[ParsedTable]] = {}
    tables_path = target / "tables.jsonl"
    if tables_path.exists():
        for line in tables_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            tables_by_page.setdefault(raw["page_no"], []).append(
                ParsedTable(
                    table_id=raw["table_id"],
                    page_no=raw["page_no"],
                    caption=raw["caption"],
                    header=tuple(raw["header"]),
                    rows=tuple(tuple(row) for row in raw["rows"]),
                    units=raw["units"],
                    restated=raw["restated"],
                )
            )

    pages: list[ParsedPage] = []
    pages_path = target / "pages.jsonl"
    if pages_path.exists():
        for line in pages_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            pages.append(
                ParsedPage(
                    page_no=raw["page_no"],
                    text=raw["text"],
                    kind=raw["kind"],
                    heading_path=tuple(raw["heading_path"]),
                    tables=tuple(tables_by_page.get(raw["page_no"], ())),
                )
            )

    return ParsedDocument(
        doc_id=meta["doc_id"],
        sha256=meta["sha256"],
        page_count=meta["page_count"],
        pages=tuple(sorted(pages, key=lambda page: page.page_no)),
        outline=tuple((heading, page) for heading, page in meta["outline"]),
    )
