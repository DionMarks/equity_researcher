"""Whether the same question answered three times gives the same answer."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

__all__ = ["RUNS", "StabilityReport", "jaccard", "measure_stability"]

RUNS = 3


def jaccard(sets: list[frozenset[str]]) -> float:
    """Mean pairwise Jaccard similarity. ``1.0`` when every run agreed.

    Pairwise rather than intersection-over-union of all three runs, because that drops to
    zero as soon as one run differs and says nothing about by how much.
    """
    pairs = [(a, b) for index, a in enumerate(sets) for b in sets[index + 1 :]]
    if not pairs:
        return 1.0
    scores = []
    for a, b in pairs:
        union = a | b
        scores.append(1.0 if not union else len(a & b) / len(union))
    return sum(scores) / len(scores)


@dataclass
class StabilityReport:
    """One case, run several times."""

    case_id: str
    statuses: list[str] = field(default_factory=list)
    levels: list[str] = field(default_factory=list)
    driver_sets: list[frozenset[str]] = field(default_factory=list)
    changes: list[Decimal | None] = field(default_factory=list)

    @property
    def driver_jaccard(self) -> float:
        return jaccard(self.driver_sets)

    @property
    def level_agreement(self) -> float:
        return 1.0 if len(set(self.levels)) <= 1 else 0.0

    @property
    def status_agreement(self) -> float:
        return 1.0 if len(set(self.statuses)) <= 1 else 0.0

    @property
    def change_agreement(self) -> float:
        """Whether the reported movement was identical, to the digit.

        The strictest of the three checks and the one that matters most: a driver set can
        legitimately vary once a model is involved, but a headline figure can't.
        """
        return 1.0 if len({str(c) for c in self.changes}) <= 1 else 0.0

    @property
    def stable(self) -> bool:
        return (
            self.status_agreement == 1.0
            and self.change_agreement == 1.0
            and self.level_agreement == 1.0
            and self.driver_jaccard == 1.0
        )


def measure_stability(outcomes_by_case: dict[str, list[Any]]) -> list[StabilityReport]:
    """Stability per case, from several runs of each."""
    reports: list[StabilityReport] = []
    for case_id, outcomes in sorted(outcomes_by_case.items()):
        report = StabilityReport(case_id=case_id)
        for outcome in outcomes:
            result = outcome.result
            report.statuses.append(result.status)
            overall = result.overall_confidence
            report.levels.append(getattr(overall, "level", "n/a") if overall else "n/a")
            report.driver_sets.append(
                frozenset(driver.driver_id for driver in result.drivers if driver.parent_id is None)
            )
            change = result.change
            report.changes.append(getattr(change, "value", None) if change else None)
        reports.append(report)
    return reports
