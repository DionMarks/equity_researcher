"""Step 12: assembling the result and writing the run's files."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ...decomposition.taxonomy import display_label
from ...domain.models import (
    AbstainReason,
    Basis,
    Decomposition,
    Diagnostics,
    EvidenceCard,
    RequestSpec,
    ResolvedInputs,
    RunStatus,
)
from ...domain.money import aud_amounts
from ...domain.registry import get_registry, residual_driver_id
from ...output.render_md import render_markdown
from ...output.schema import (
    Confidence,
    Driver,
    HeadlineValue,
    Reconciliation,
    ResearchResult,
    build_change,
    share_of_change,
)
from .explain import _relevant_components
from .score import ScoredRun

__all__ = ["abstention_result", "build_result", "write_run"]


def _confidence(scored) -> Confidence:  # type: ignore[no-untyped-def]
    return Confidence(level=scored.level, score=float(scored.score), reasons=list(scored.reasons))


def build_result(
    spec: RequestSpec,
    inputs: ResolvedInputs,
    decomposition: Decomposition,
    scored: ScoredRun,
    cards: list[EvidenceCard],
    caveats: list[str],
    diagnostics: Diagnostics,
    narrative: str = "",
    *,
    causes: dict[str, tuple[str, list[str]]] | None = None,
    commentary: list[str] | None = None,
) -> ResearchResult:
    """One answered run, as the output contract.

    ``causes`` are the drivers' stated causes that survived the check, by driver id, and
    ``commentary`` the evidence ids of the issuer's own explanation of the movement.
    """
    causes = causes or {}
    relevant = _relevant_components(spec, decomposition)
    registry = get_registry()
    metric = registry.metrics[spec.metric_id]
    pair = inputs.pair(spec.metric_id)
    unit = inputs.units.get(spec.metric_id, metric.unit)

    current = prior = None
    change = None
    if pair is not None:
        prior_value, current_value = pair
        # The basis of the figure reconciliation chose, not a default. It used to be left
        # at the default, so every result said "group, as reported" whatever the figure
        # was: Westpac's expense-to-income ratio excluding Notable Items, a Level 2 capital
        # ratio, a cash profit.
        current = HeadlineValue(
            value=current_value,
            unit=unit,  # type: ignore[arg-type]
            label_at_bank=spec.headline_label or metric.display_name,
            basis=inputs.bases.get(f"{spec.period}:{spec.metric_id}", Basis()),
            evidence_ids=inputs.citations_for(spec.metric_id, spec.period),
        )
        prior = HeadlineValue(
            value=prior_value,
            unit=unit,  # type: ignore[arg-type]
            label_at_bank=spec.headline_label or metric.display_name,
            basis=inputs.bases.get(f"{spec.comparison_period}:{spec.metric_id}", Basis()),
            evidence_ids=inputs.citations_for(spec.metric_id, spec.comparison_period),
            restated=spec.metric_id in inputs.restated_components,
        )
        change = build_change(
            prior_value,
            current_value,
            decomposition.change_unit,
            show_pct=metric.show_pct,
        )

    drivers: list[Driver] = []
    for contribution in decomposition.drivers:
        score = scored.drivers.get(contribution.driver_id)
        if score is None or score.removed:
            continue
        # A part of a driver that moved nothing is noise in the table: CBA's day-count
        # effect between two 184-day halves is A$0m by construction.
        if contribution.parent_id is not None and not contribution.contribution:
            continue
        drivers.append(
            Driver(
                driver_id=contribution.driver_id,
                parent_id=contribution.parent_id,
                label=contribution.label or display_label(spec.metric_id, contribution.driver_id),
                contribution=contribution.contribution,
                direction=contribution.direction,
                share_of_change=share_of_change(
                    contribution.contribution,
                    decomposition.headline_change,
                    decomposition.tolerance,
                )
                if contribution.parent_id is None
                else None,
                method=contribution.method,
                evidence_type=contribution.evidence_type,
                explanation=contribution.note,
                cause=aud_amounts(causes.get(contribution.driver_id, ("", []))[0]),
                cause_evidence_ids=list(causes.get(contribution.driver_id, ("", []))[1]),
                evidence_ids=contribution.evidence_ids,
                confidence=_confidence(score),
            )
        )

    residual = None
    if decomposition.drivers:
        residual = Driver(
            driver_id=residual_driver_id(spec.metric_id),
            label="Residual",
            contribution=decomposition.residual,
            direction="+"
            if decomposition.residual > 0
            else "-"
            if decomposition.residual < 0
            else "0",
            method="computed",
            evidence_type="computed",
            explanation=(
                "Within tolerance: rounding or a definition difference."
                if decomposition.within_tolerance
                else "Beyond tolerance, and therefore unexplained."
            ),
            confidence=_confidence(scored.overall),
        )

    status: RunStatus = "OK"
    if pair is None:
        status = "ABSTAIN"
    elif not drivers or not decomposition.within_tolerance:
        status = "PARTIAL"

    return ResearchResult(
        run_id=spec.run_id,
        created_at=datetime.now(UTC),
        as_of=spec.as_of,
        request=spec,
        status=status,
        current=current,
        prior=prior,
        change=change,
        drivers=drivers,
        residual=residual,
        reconciliation=Reconciliation(
            headline_change=decomposition.headline_change,
            sum_of_drivers=decomposition.sum_of_drivers,
            residual=decomposition.residual,
            tolerance=decomposition.tolerance,
            within_tolerance=decomposition.within_tolerance,
        )
        if decomposition.drivers
        else None,
        headline_confidence=_confidence(scored.headline),
        overall_confidence=_confidence(scored.overall),
        narrative=aud_amounts(narrative),
        issuer_commentary=list(commentary or []),
        # Once each: two chained half-year walks name the same bars, and each bar matched by
        # the model was saying so twice.
        caveats=list(dict.fromkeys([*caveats, *inputs.caveats, *decomposition.notes])),
        # Only the disagreements about figures this answer uses. A margin memo used to list
        # a dozen conflicts about earnings lines nothing in it relied on.
        data_conflicts=[
            conflict for conflict in inputs.conflicts if conflict.component in relevant
        ],
        alternative_bases=inputs.alternative_bases,
        evidence=cards,
        diagnostics=diagnostics,
    )


def abstention_result(
    spec: RequestSpec,
    reason: AbstainReason,
    detail: str,
    caveats: list[str] | None = None,
    diagnostics: Diagnostics | None = None,
) -> ResearchResult:
    """A run that declined, stating what was found and what was missing."""
    return ResearchResult(
        run_id=spec.run_id,
        created_at=datetime.now(UTC),
        as_of=spec.as_of,
        request=spec,
        status="ABSTAIN",
        abstain_reason=reason,
        caveats=[detail, *(caveats or [])],
        diagnostics=diagnostics or Diagnostics(),
    )


def write_run(
    result: ResearchResult, runs_dir: Path, trace: list[dict[str, object]] | None = None
) -> Path:
    """Write ``result.json``, ``result.md`` and ``trace.jsonl`` for one run."""
    target = Path(runs_dir) / result.run_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "result.json").write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (target / "result.md").write_text(render_markdown(result), encoding="utf-8")
    with (target / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for event in trace or []:
            handle.write(json.dumps(event, default=str) + "\n")
    return target
