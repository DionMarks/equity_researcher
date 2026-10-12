"""Writes the eval results."""

from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path
from typing import Any

from ..domain.money import aud, to_aud
from .calibration import Claim, calibrate
from .cases import EvalCase
from .runner import EvalRun

__all__ = ["write_report"]


def write_report(run: EvalRun, cases: list[EvalCase], root: Path) -> Path:
    """Write ``results.csv`` and ``summary.md``, returning the directory."""
    stamp = run.started.strftime("%Y%m%dT%H%M%SZ")
    target = Path(root) / stamp
    target.mkdir(parents=True, exist_ok=True)
    by_id = {case.case_id: case for case in cases}

    with (target / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_id",
                "bank",
                "period",
                "metric",
                "categories",
                "expected_status",
                "status",
                "status_correct",
                "expected_reason",
                "abstain_reason",
                "reason_correct",
                "drivers",
                "reconciles",
                "residual",
                "tolerance",
                "cited_numbers",
                "cited_found",
                "citation_faithfulness",
                "confidence_level",
                "confidence",
                "latency_ms",
                "llm_calls",
                "cost_aud",
                "headline_exact",
                "gold_verified",
                "passed",
                "notes",
            ]
        )
        for score in run.scores:
            case = by_id.get(score.case_id)
            writer.writerow(
                [
                    score.case_id,
                    case.bank if case else "",
                    case.period if case else "",
                    case.metric if case else "",
                    "|".join(case.categories) if case else "",
                    "|".join(score.expected_status),
                    score.status,
                    score.status_correct,
                    (case.expectation(run.model_mode)[1] or "") if case else "",
                    score.abstain_reason or "",
                    "" if score.abstain_reason_correct is None else score.abstain_reason_correct,
                    score.drivers,
                    "" if score.reconciles is None else score.reconciles,
                    "" if score.residual is None else score.residual,
                    "" if score.tolerance is None else score.tolerance,
                    score.cited_numbers,
                    score.cited_numbers_found,
                    ""
                    if score.citation_faithfulness is None
                    else f"{score.citation_faithfulness:.3f}",
                    score.confidence_level or "",
                    "" if score.overall_confidence is None else f"{score.overall_confidence:.3f}",
                    score.latency_ms,
                    score.llm_calls,
                    f"{to_aud(score.cost_usd):.4f}",
                    "" if score.headline_exact is None else score.headline_exact,
                    case.gold_verified if case else "",
                    score.passed,
                    "; ".join(score.notes),
                ]
            )

    (target / "summary.md").write_text(_summary(run, cases), encoding="utf-8")
    return target


def _pct(value: float | None) -> str:
    return "not measured" if value is None else f"{value * 100:.1f}%"


def _labelling_provenance(cases: list[EvalCase]) -> list[str]:
    """Who checked the gold these accuracy figures are scored against.

    An accuracy number is only as good as the labels under it, so where the labels came from
    belongs next to the number, not in a separate document. Where the gold was checked by
    whoever built the system, the measure is still worth having (a figure taken off the
    wrong row or basis gets caught either way), but it isn't independent, and saying so is
    what makes it a measurement rather than a claim.
    """
    verified = [case for case in cases if case.gold_verified and case.has_gold]
    if not verified:
        return []
    by_labeller: dict[tuple[str, str], int] = {}
    for case in verified:
        key = (case.labeller, case.labeller_kind)
        by_labeller[key] = by_labeller.get(key, 0) + 1

    lines = ["#### Gold provenance", "", "| Labeller | Kind | Cases |", "| --- | --- | ---: |"]
    lines += [
        f"| {labeller} | {kind} | {count} |"
        for (labeller, kind), count in sorted(by_labeller.items())
    ]
    lines.append("")
    machine_checked = sum(count for (_, kind), count in by_labeller.items() if kind == "automated")
    if machine_checked:
        lines += [
            f"**{machine_checked} of {len(verified)} verified case(s) were checked by an "
            "automated pass, not a person.** Every figure was read off the page it cites, so a "
            "value taken from the wrong row, the wrong column or the wrong basis is "
            "caught. What this cannot catch is an error whose cause the labeller and the "
            "system share, and it is not an independent holdout. "
            "Treat these as a regression baseline; countersign them before quoting the "
            "accuracy anywhere it matters.",
            "",
        ]
    return lines


def _summary(run: EvalRun, cases: list[EvalCase]) -> str:
    by_id = {case.case_id: case for case in cases}
    unverified = sum(1 for case in cases if case.has_gold and not case.gold_verified)
    no_gold = sum(1 for case in cases if not case.has_gold)

    lines = [
        f"# Eval results: {run.split} split",
        "",
        f"_{run.started.isoformat()} · git `{run.git_sha}` · config `{run.config_hash}` · "
        f"model mode `{run.model_mode}` · {len(run.scores)} case(s) in "
        f"{run.elapsed_seconds:.1f}s_",
        "",
        "## Measured",
        "",
        "These need no labelled figure, so they are evidence.",
        "",
        "| Measure | Result |",
        "| --- | ---: |",
        f"| Cases behaved as asserted | **{run.passed}/{len(run.scores)}** |",
        f"| Status accuracy | {_pct(run.status_accuracy)} |",
        f"| Abstention reason correct | {_pct(run.abstention_accuracy)} |",
        f"| Drivers reconcile to the movement | {_pct(run.reconciliation_rate)} |",
        f"| Numeric citation faithfulness | {_pct(run.citation_faithfulness)} "
        f"({run.citations_checked} checked) |",
        f"| Citations with no document page | {run.dangling_citations} |",
        f"| Latency p50 / p95 | {run.latency_p50_ms}ms / {run.latency_p95_ms}ms |",
        f"| Model calls · cost | {sum(s.llm_calls for s in run.scores)} · "
        f"{aud(run.total_cost_usd)} |",
        "",
        ("## Not measured, and why" if run.gold_scored == 0 else "## Accuracy, against gold"),
        "",
    ]

    if run.gold_scored == 0:
        lines += [
            "**No accuracy measure is reported.** Headline exactness, change error, driver "
            "precision and recall, contribution error and every calibration measure need "
            "figures labelled by hand from the issuer's own documents and verified by a "
            "person. ",
            "",
            f"- {no_gold} case(s) carry no gold labels.",
            f"- {unverified} case(s) carry gold that no person has verified.",
            "",
            "The scorers for all of them are implemented and will run the moment verified "
            "gold exists. Reporting a default in their place would be quoting a number "
            "that was never measured.",
            "",
            "Calibration (expected calibration error, the Brier score, the "
            "high-confidence error rate, the headline safety number, and the "
            "risk-coverage curve) is implemented in `evals/calibration.py` and gated the "
            "same way. It refuses unverified labels rather than scoring against them, "
            "because scoring the system against its own output and calling the result "
            "calibration is worse than reporting nothing.",
            "",
        ]
    else:
        lines += [
            "Each figure is scored against gold within the gold's own tolerance, which is "
            "half the last digit the issuer printed.",
            "",
            "| Measure | Result | Cases scored |",
            "| --- | ---: | ---: |",
            f"| Headline exact (both periods) | {_pct(run.headline_exact_rate)} "
            f"| {run.gold_scored} |",
            f"| Movement within tolerance | {_pct(run.change_within_tolerance_rate)} "
            f"| {sum(1 for s in run.scores if s.change_error is not None)} |",
            f"| Driver precision (mean) | {_pct(run.mean_driver_precision)} "
            f"| {sum(1 for s in run.scores if s.driver_precision is not None)} |",
            f"| Driver recall (mean) | {_pct(run.mean_driver_recall)} "
            f"| {sum(1 for s in run.scores if s.driver_recall is not None)} |",
            _driver_values_row(run),
            "",
            *_labelling_provenance(cases),
        ]
        calibration = calibrate(run.scores)
        lines += ["### Calibration", "", calibration.summary(), ""]
        if calibration.buckets:
            lines += [
                "| Level | Claims | Accuracy | Mean score | Gap |",
                "| --- | ---: | ---: | ---: | ---: |",
                *(
                    f"| {bucket.level} | {bucket.claims} | {bucket.accuracy:.1%} | "
                    f"{bucket.mean_score:.3f} | {bucket.gap:+.3f} |"
                    for bucket in calibration.buckets
                ),
                "",
            ]
        lines += [f"- {note}" for note in calibration.notes] + [""]
        lines += _driver_calibration(run)

    counts = run.status_counts()
    lines += ["## Outcomes", "", "| Status | Cases |", "| --- | ---: |"]
    lines += [f"| {status} | {count} |" for status, count in counts.items()]
    lines += [
        "",
        "## Every case",
        "",
        "| Case | Request | Expected | Got | Drivers | Citations | Confidence | Pass |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for score in run.scores:
        case = by_id.get(score.case_id)
        request = f"{case.bank} {case.period} {case.metric}" if case else ""
        expected = "/".join(score.expected_status)
        reason = case.expectation(run.model_mode)[1] if case else None
        if reason and "ABSTAIN" in score.expected_status:
            expected += f" {reason}"
        got = score.status + (f" {score.abstain_reason}" if score.abstain_reason else "")
        faithful = score.citation_faithfulness
        lines.append(
            f"| `{score.case_id}` | {request} | {expected} | {got} | {score.drivers} | "
            f"{score.cited_numbers_found}/{score.cited_numbers}"
            + (f" ({faithful * 100:.0f}%)" if faithful is not None else "")
            + f" | {score.confidence_level or 'n/a'} | "
            + ("yes" if score.passed else "**no**")
            + " |"
        )

    if run.errors:
        lines += ["", "## Errors", ""]
        lines += [f"- `{case_id}`: {reason}" for case_id, reason in run.errors]

    failures = [score for score in run.scores if not score.passed]
    if failures:
        lines += ["", "## Cases that did not behave as asserted", ""]
        for score in failures:
            lines.append(
                f"- `{score.case_id}`: expected {'/'.join(score.expected_status)}, got "
                f"{score.status}"
                + (f" ({score.abstain_reason})" if score.abstain_reason else "")
                + (f". {'; '.join(score.notes[:2])}" if score.notes else "")
            )

    return "\n".join(lines) + "\n"


def _driver_values_row(run: Any) -> str:
    right, checked = run.driver_values
    share = _pct(right / checked) if checked else "n/a"
    cases = sum(1 for s in run.scores if s.driver_values_checked)
    return f"| Driver sized right (sign and tolerance) | {share} ({right}/{checked}) | {cases} |"


def _driver_calibration(run: Any) -> list[str]:
    """Calibration over drivers rather than headlines.

    Each material driver a run names in a case with driver gold is one claim, right when it
    is a gold driver with the same sign and within tolerance. It answers whether a driver's
    own confidence can be trusted, which the headline's calibration says nothing about.
    """
    claims = [
        Claim(
            claim_id=f"{score.case_id}:{driver_id}",
            score=Decimal(str(confidence)),
            level=level,
            correct=correct,
            gold_verified=True,
        )
        for score in run.scores
        for driver_id, confidence, level, correct in score.driver_claims
    ]
    if not claims:
        return []
    report = calibrate(claims)
    lines = ["### Calibration of drivers", "", report.summary(), ""]
    if report.buckets:
        lines += [
            "| Level | Claims | Accuracy | Mean score | Gap |",
            "| --- | ---: | ---: | ---: | ---: |",
            *(
                f"| {bucket.level} | {bucket.claims} | {bucket.accuracy:.1%} | "
                f"{bucket.mean_score:.3f} | {bucket.gap:+.3f} |"
                for bucket in report.buckets
            ),
            "",
        ]
    return lines
