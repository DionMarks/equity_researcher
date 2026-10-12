"""Finds restated comparatives by comparing a period against its own later statement."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from ..domain.models import Fact, Restatement
from ..domain.units import normalise_aud, value_tolerance

__all__ = [
    "RestatementFinding",
    "find_restatements",
    "restated_components",
    "unfootnoted",
]


@dataclass(frozen=True, slots=True)
class RestatementFinding:
    """One component whose comparative moved between two of the issuer's documents."""

    component: str
    period: str
    original: Fact
    restated: Fact
    difference: Decimal
    tolerance: Decimal
    footnoted: bool

    @property
    def note(self) -> str:
        direction = "up" if self.difference > 0 else "down"
        return (
            f"{self.component} for {self.period} was restated {direction} by "
            f"{abs(self.difference)} between {self.original.doc_id} and {self.restated.doc_id}"
            + ("" if self.footnoted else "; the issuer did not footnote the change")
        )

    def as_record(self) -> Restatement:
        return Restatement(
            original_fact_id=self.original.fact_id,
            restated_fact_id=self.restated.fact_id,
            note=self.note,
            footnoted=self.footnoted,
        )


@dataclass
class _Pair:
    earlier: Fact | None = None
    later: Fact | None = None


def find_restatements(facts: list[Fact]) -> list[RestatementFinding]:
    """Components whose value for one period differs across two documents.

    ``facts`` is every stored fact for one issuer. Figures are grouped by component, period
    and basis, and within each group the earliest and latest publish dates are compared. The
    later document's comparative is the one that stands.
    """
    groups: dict[tuple[str, str, str], _Pair] = {}
    for fact in facts:
        key = (fact.component, fact.period, fact.basis.model_dump_json())
        pair = groups.setdefault(key, _Pair())
        if pair.earlier is None or fact.publish_date < pair.earlier.publish_date:
            pair.earlier = fact
        if pair.later is None or fact.publish_date > pair.later.publish_date:
            pair.later = fact

    findings: list[RestatementFinding] = []
    for (component, period, _), pair in sorted(groups.items()):
        earlier, later = pair.earlier, pair.later
        if earlier is None or later is None:
            continue
        if earlier.doc_id == later.doc_id or earlier.publish_date == later.publish_date:
            continue  # one statement of the figure, not two

        before, _ = normalise_aud(earlier.value, earlier.unit)
        after, _ = normalise_aud(later.value, later.unit)
        difference = after - before
        # Both figures' own precision, because a figure restated to the digit it was
        # already printed to has not been restated.
        tolerance = value_tolerance(earlier.unit, reported=earlier.value) + value_tolerance(
            later.unit, reported=later.value
        )
        if abs(difference) <= tolerance:
            continue

        findings.append(
            RestatementFinding(
                component=component,
                period=period,
                original=earlier,
                restated=later,
                difference=difference,
                tolerance=tolerance,
                # `is_restated` is set at extraction from the table's own footnote, so a
                # table that said "comparatives have been restated" is an explanation.
                footnoted=later.is_restated,
            )
        )
    return findings


def restated_components(findings: list[RestatementFinding]) -> tuple[str, ...]:
    """The component names, for the caveat the memo carries."""
    return tuple(sorted({finding.component for finding in findings}))


def unfootnoted(findings: list[RestatementFinding]) -> list[RestatementFinding]:
    """The restatements the issuer did not explain. These cost confidence."""
    return [finding for finding in findings if not finding.footnoted]
