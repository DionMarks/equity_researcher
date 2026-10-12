"""Every conflict the eval cases raise, and whether each unexplained one has been reviewed."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..agent.orchestrator import Agent
from ..memory.db import connect
from ..memory.fact_store import FactStore
from ..settings import Settings
from .cases import EvalCase
from .runner import without_a_key

__all__ = ["Census", "ConflictSeen", "conflict_census", "reviewed_conflicts"]


@dataclass(frozen=True, slots=True)
class ConflictSeen:
    """One conflict, by what it is rather than by the run that raised it."""

    bank: str
    component: str
    classification: str
    readings: tuple[tuple[str, int | None, str], ...]
    """(document, page, value) for each candidate, sorted."""
    rule: str


@dataclass
class Census:
    seen: dict[ConflictSeen, set[str]] = field(default_factory=dict)
    """Each distinct conflict, with the cases that raised it."""
    reviewed: list[dict[str, Any]] = field(default_factory=list)

    def by_class(self) -> dict[str, int]:
        return dict(sorted(Counter(item.classification for item in self.seen).items()))

    def unreviewed(self) -> list[ConflictSeen]:
        """UNEXPLAINED conflicts that no review entry covers."""
        return [
            item
            for item in self.seen
            if item.classification == "UNEXPLAINED"
            and not any(_covers(entry, item) for entry in self.reviewed)
        ]


def _covers(entry: dict[str, Any], item: ConflictSeen) -> bool:
    """A review covers a conflict on the same bank and component whose readings it names."""
    if str(entry.get("bank", "")).upper() != item.bank.upper():
        return False
    if entry.get("component") != item.component:
        return False
    named = {str(doc) for doc in entry.get("documents", ())}
    return named <= {doc for doc, _page, _value in item.readings}


def reviewed_conflicts(config_dir: Path) -> list[dict[str, Any]]:
    path = config_dir / "conflicts_reviewed.yaml"
    if not path.exists():
        return []
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(loaded.get("reviewed", ()) or ())


def conflict_census(cases: list[EvalCase], settings: Settings) -> Census:
    """Run every answerable case without a model and collect what conflicted."""
    resolved = without_a_key(settings)
    census = Census(reviewed=reviewed_conflicts(resolved.config_dir))
    with connect(resolved.db_path) as connection:
        store = FactStore(connection)
        for case in cases:
            if case.expects_abstention:
                continue
            run = Agent(store=store, settings=resolved, persist=False).run(
                case.bank, case.period, case.metric, case.as_of
            )
            for conflict in run.result.data_conflicts:
                item = ConflictSeen(
                    bank=case.bank,
                    component=conflict.component,
                    classification=conflict.classification,
                    readings=tuple(
                        sorted(
                            (c.source.doc_id, c.source.page, str(c.value))
                            for c in conflict.candidates
                        )
                    ),
                    rule=conflict.rule_applied,
                )
                census.seen.setdefault(item, set()).add(case.case_id)
    return census
