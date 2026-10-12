"""Per-metric retrieval checklists."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import yaml

from ..settings import get_settings

__all__ = ["Playbook", "load_playbooks", "playbook_for"]


@dataclass(frozen=True, slots=True)
class Playbook:
    """How to find one metric, and what to refuse."""

    metric_id: str
    queries: tuple[str, ...] = ()
    """Retrieval queries, best first."""
    confirm_labels: tuple[str, ...] = ()
    """Row labels that confirm a page carries the figure rather than discussing it."""
    identity_components: tuple[str, ...] = ()
    """The components whose pages the reading list should also name."""
    walk_hints: tuple[str, ...] = ()
    """Phrases that mark a published walk for this metric."""
    avoid: tuple[str, ...] = ()
    """Phrases that mark a page as the wrong one even though it matches everything else."""
    required: tuple[tuple[str, tuple[str, ...]], ...] = ()
    """What a complete answer needs, each with the document types it's usually in."""
    optional: tuple[tuple[str, tuple[str, ...]], ...] = ()
    """Evidence that strengthens an answer but isn't needed for one."""
    search_terms: tuple[str, ...] = ()
    """The issuers' own words for the metric and its components."""
    table_captions: tuple[str, ...] = ()
    """Captions the right tables are usually printed under."""

    def ranked_queries(self, bank_id: str, period: str) -> tuple[str, ...]:
        """The queries, with the issuer and period substituted in.

        Substituted rather than concatenated: "net interest margin ANZ 1H26" and
        "ANZ 1H26 net interest margin" rank differently, and the template fixes which.
        """
        return tuple(query.format(bank=bank_id, period=period) for query in self.queries) or (
            f"{bank_id} {period} {self.metric_id.replace('_', ' ')}",
        )

    def checklist(self, bank_id: str, period: str) -> str:
        """The playbook as plain lines, for the start of the locate loop.

        Every line is a hint about where to look, never a figure: the loop's only output is
        a list of pages to read, and nothing here could be mistaken for a number to report.
        """

        def evidence(items: tuple[tuple[str, tuple[str, ...]], ...]) -> str:
            return "; ".join(
                f"{name} ({' or '.join(types)})" if types else name for name, types in items
            )

        lines = [
            f"- Must find: {evidence(self.required)}" if self.required else "",
            f"- Worth finding: {evidence(self.optional)}" if self.optional else "",
            "- Search with: " + "; ".join(f'"{q}"' for q in self.ranked_queries(bank_id, period)),
            f"- The issuer's own words: {', '.join(self.search_terms)}"
            if self.search_terms
            else "",
            f"- The right table usually has a row labelled: {', '.join(self.confirm_labels)}"
            if self.confirm_labels
            else "",
            f"- Its caption is usually one of: {', '.join(self.table_captions)}"
            if self.table_captions
            else "",
            f"- A published walk is usually marked by: {', '.join(self.walk_hints)}"
            if self.walk_hints
            else "",
            f"- Not these pages, however well they match: {', '.join(self.avoid)}"
            if self.avoid
            else "",
        ]
        return "\n".join(line for line in lines if line)


@lru_cache(maxsize=1)
def load_playbooks(config_dir: str | None = None) -> dict[str, Playbook]:
    """Playbooks from ``config/playbooks.yaml``, keyed by metric."""
    from pathlib import Path

    root = Path(config_dir or str(get_settings().config_dir))
    path = root / "playbooks.yaml"
    if not path.exists():
        return {}
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, Playbook] = {}
    for metric_id, spec in (raw.get("playbooks") or {}).items():
        evidence = _evidence_spec(root, metric_id)
        out[metric_id] = Playbook(
            metric_id=metric_id,
            queries=tuple(spec.get("queries", ()) or ()),
            confirm_labels=tuple(spec.get("confirm_labels", ()) or ()),
            identity_components=tuple(spec.get("identity_components", ()) or ()),
            walk_hints=tuple(spec.get("walk_hints", ()) or ()),
            avoid=tuple(spec.get("avoid", ()) or ()),
            required=_items(evidence.get("required")),
            optional=_items(evidence.get("optional")),
            search_terms=tuple(evidence.get("search_terms", ()) or ()),
            table_captions=tuple(evidence.get("table_captions", ()) or ()),
        )
    return out


def _evidence_spec(root: Any, metric_id: str) -> dict[str, Any]:
    """``config/playbooks/<metric>.yaml``: what a complete answer needs."""
    path = root / "playbooks" / f"{metric_id}.yaml"
    if not path.exists():
        return {}
    loaded: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return loaded


def _items(raw: Any) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        (str(item.get("component", "")), tuple(item.get("doc_types", ()) or ()))
        for item in (raw or ())
        if isinstance(item, dict) and item.get("component")
    )


def playbook_for(metric_id: str, config_dir: str | None = None) -> Playbook:
    """The playbook for a metric, or an empty one.

    Empty rather than an error, because a metric with no playbook falls back to plain
    retrieval, which is weaker but still correct. Raising here would make adding a metric a
    two-file change with a broken step in between.
    """
    return load_playbooks(config_dir).get(metric_id) or Playbook(metric_id=metric_id)
