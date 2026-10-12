"""The manifest: which filings exist, where, and who checked that."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from ..domain.calendar import parse_period
from ..domain.models import DocMeta, DocType
from ..domain.registry import get_registry
from ..settings import get_settings

__all__ = ["ManifestEntry", "load_manifest", "manifest_path", "write_manifest"]

_DOC_TYPES: tuple[str, ...] = (
    "RESULTS_DOCUMENT",
    "RESTATEMENT_NOTICE",
    "RESULTS_RELEASE",
    "INVESTOR_PRESENTATION",
    "PILLAR3",
    "ANNUAL_REPORT",
)


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    doc_id: str
    bank_id: str
    period: str
    doc_type: DocType
    url: str
    publish_date: date
    sha256: str | None = None
    source: Literal["ir_site", "asx", "discovered"] = "ir_site"
    verified_by: Literal["human", "auto"] = "auto"
    note: str = ""

    def to_meta(self) -> DocMeta:
        return DocMeta(
            doc_id=self.doc_id,
            bank_id=self.bank_id,
            period=self.period,
            doc_type=self.doc_type,
            publish_date=self.publish_date,
            url=self.url,
            sha256=self.sha256,
        )

    def to_yaml(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "doc_id": self.doc_id,
            "bank_id": self.bank_id,
            "period": self.period,
            "doc_type": self.doc_type,
            "url": self.url,
            "publish_date": self.publish_date,
            "sha256": self.sha256,
            "source": self.source,
            "verified_by": self.verified_by,
        }
        if self.note:
            entry["note"] = self.note
        return entry


def manifest_path() -> Path:
    return get_settings().data_dir / "manifest.yaml"


def load_manifest(path: Path | None = None) -> list[ManifestEntry]:
    """Load and validate the manifest. Raises on an entry it cannot make sense of."""
    target = path or manifest_path()
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    registry = get_registry()

    entries: list[ManifestEntry] = []
    seen: set[str] = set()
    for index, item in enumerate(raw.get("documents", []) or []):
        where = f"{target.name} entry {index}"
        for required in ("doc_id", "bank_id", "period", "doc_type", "url", "publish_date"):
            if item.get(required) in (None, ""):
                raise ValueError(f"{where} is missing {required}")
        if item["doc_id"] in seen:
            raise ValueError(f"{where}: duplicate doc_id {item['doc_id']!r}")
        seen.add(item["doc_id"])

        profile = registry.bank(item["bank_id"])
        period = parse_period(item["period"])
        if item["doc_type"] not in _DOC_TYPES:
            raise ValueError(f"{where}: unknown doc_type {item['doc_type']!r}")

        published = item["publish_date"]
        if not isinstance(published, date):
            published = date.fromisoformat(str(published))

        entries.append(
            ManifestEntry(
                doc_id=item["doc_id"],
                bank_id=profile.bank_id,
                period=period,
                doc_type=cast(DocType, item["doc_type"]),
                url=item["url"],
                publish_date=published,
                sha256=item.get("sha256"),
                source=item.get("source", "ir_site"),
                verified_by=item.get("verified_by", "auto"),
                note=item.get("note", ""),
            )
        )
    return entries


def write_manifest(entries: list[ManifestEntry], path: Path | None = None) -> Path:
    """Write the manifest back, preserving its header comment."""
    target = path or manifest_path()
    header = []
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or not line.strip():
                header.append(line)
            else:
                break
    body = yaml.safe_dump(
        {"schema_version": "1.0", "documents": [e.to_yaml() for e in entries]},
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    target.write_text("\n".join(header).rstrip() + "\n\n" + body, encoding="utf-8")
    return target
