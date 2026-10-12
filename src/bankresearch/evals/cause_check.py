"""How well the guards on a driver's stated cause work, over every cause the live runs drafted."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from ..agent.steps.explain import _SILENCE, CAUSE_WORD_LIMIT

__all__ = [
    "CauseClaim",
    "CauseScore",
    "Passage",
    "load_claims",
    "rejudge",
    "score_claims",
    "todays_guard",
]

LABELS = ("supported", "partial", "unsupported")


@dataclass(frozen=True, slots=True)
class Passage:
    """One of the bank's sentences a claim cites, as the run's evidence card held it."""

    evidence_id: str
    doc_id: str
    page: int
    text: str


@dataclass(frozen=True, slots=True)
class CauseClaim:
    claim_id: str
    bank: str
    metric: str
    driver: str
    claim: str
    outcome: str
    """``shown`` or ``dropped``, as the run did it."""
    label: str
    note: str = ""
    driver_label: str = ""
    passages: tuple[Passage, ...] = ()
    period: str = ""
    contribution: str = ""
    """The driver's size as the run's memo showed it, such as ``+A$190m`` or ``-3.0 bps``."""
    method: str = ""
    """How the driver was worked out: ``computed``, or read from the issuer's walk."""


@dataclass
class CauseScore:
    """What reached the reader, and what was taken out."""

    shown: int = 0
    shown_supported: int = 0
    flagged: int = 0
    """Claims labelled partial or unsupported."""
    flagged_removed: int = 0
    supported: int = 0
    supported_removed: int = 0

    @property
    def shown_precision(self) -> float | None:
        """Of the causes a reader saw, the share the bank's words support."""
        return self.shown_supported / self.shown if self.shown else None

    @property
    def removal_recall(self) -> float | None:
        """Of the causes that shouldn't reach a reader, the share that didn't."""
        return self.flagged_removed / self.flagged if self.flagged else None

    @property
    def false_removal_rate(self) -> float | None:
        """Of the causes the bank's words support, the share taken out anyway."""
        return self.supported_removed / self.supported if self.supported else None


def load_claims(path: Path) -> list[CauseClaim]:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    claims: list[CauseClaim] = []
    for item in raw.get("claims", ()) or ():
        label = str(item["label"])
        if label not in LABELS:
            raise ValueError(f"{item.get('id')}: label must be one of {', '.join(LABELS)}")
        claims.append(
            CauseClaim(
                claim_id=str(item["id"]),
                bank=str(item["bank"]),
                metric=str(item["metric"]),
                driver=str(item["driver"]),
                claim=str(item["claim"]),
                outcome=str(item["outcome"]),
                label=label,
                note=str(item.get("note", "")),
                driver_label=str(item.get("driver_label", "")),
                period=str(item.get("period", "")),
                contribution=str(item.get("contribution", "")),
                method=str(item.get("method", "")),
                passages=tuple(
                    Passage(
                        evidence_id=str(passage["id"]),
                        doc_id=str(passage["doc"]),
                        page=int(passage["page"]),
                        text=str(passage["text"]),
                    )
                    for passage in item.get("passages", ()) or ()
                ),
            )
        )
    return claims


def todays_guard(claim: CauseClaim) -> str | None:
    """Why today's code guards would drop this cause before the checker sees it, if they
    would. Figures and citations are checked against the run's own brief, which isn't kept
    here; the live runs' drafts passed both."""
    if _SILENCE.search(claim.claim):
        return "claims the bank is silent"
    if len(claim.claim.split()) > CAUSE_WORD_LIMIT:
        return f"over {CAUSE_WORD_LIMIT} words"
    return None


def score_claims(
    claims: list[CauseClaim],
    *,
    today: bool = False,
    verdicts: dict[str, tuple[str, str]] | None = None,
) -> CauseScore:
    """Scored as the runs went, with today's code guards applied on top, or with today's
    guards and the checker's verdicts re-taken live (``verdicts``, from :func:`rejudge`)."""
    score = CauseScore()
    for claim in claims:
        removed = claim.outcome == "dropped"
        if verdicts is not None:
            # Both guards as they stand today. A claim the checker gave no verdict on counts
            # as unsupported, as it does in a run.
            verdict = verdicts.get(claim.claim_id, ("unsupported", ""))[0]
            removed = todays_guard(claim) is not None or verdict != "supported"
        elif today:
            # A cause dropped only for running past the old limit would now reach the
            # checker, whose verdict on it isn't known; it's counted as the run had it.
            removed = removed or todays_guard(claim) is not None
        if not removed:
            score.shown += 1
            score.shown_supported += int(claim.label == "supported")
        if claim.label == "supported":
            score.supported += 1
            score.supported_removed += int(removed)
        else:
            score.flagged += 1
            score.flagged_removed += int(removed)
    return score


#: Claims judged in one call. The run's own check sends a memo's few claims at once; twenty
#: keeps a call well inside the verify step's token caps and the benchmark to four calls.
REJUDGE_BATCH = 20

_EVIDENCE_ID = re.compile(r"\bev_\d+\b")


def _with_ids_renamed(text: str, renamed: dict[str, str]) -> str:
    """The claim with each evidence id it cites replaced by its new name."""
    return _EVIDENCE_ID.sub(lambda match: renamed.get(match.group(0), match.group(0)), text)


def rejudge(
    claims: list[CauseClaim],
    client: Any,
    *,
    model: str,
    effort: str,
    batch: int = REJUDGE_BATCH,
) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """The checker's verdict on every claim, re-taken with today's prompt.

    Each claim goes through the run's own check (``check_narrative``) as a driver's stated
    cause, with the passages it cited as its evidence and, as a run gives it, the driver's
    size. Without the size the checker can't see that a passage explains only part of a
    driver, and the first re-judge passed the stretches the runs had caught. Claims from
    different runs share a call, so each one's evidence ids are prefixed with its own id
    (``ev_066`` is a different sentence in each run) and each driver is named with its bank,
    period and metric. Returns the verdicts by claim id, and why any batch went unjudged.
    """
    from ..agent.steps.explain import check_narrative
    from ..domain.models import EvidenceCard, SourceRef
    from ..llm.consent import ConsentDenied

    verdicts: dict[str, tuple[str, str]] = {}
    unjudged: list[str] = []
    for start in range(0, len(claims), batch):
        group = claims[start : start + batch]
        cards: list[EvidenceCard] = []
        causes: dict[str, tuple[str, list[str]]] = {}
        labels: dict[str, str] = {}
        sizes: list[str] = []
        for claim in group:
            renamed = {p.evidence_id: f"{claim.claim_id}_{p.evidence_id}" for p in claim.passages}
            text = _with_ids_renamed(claim.claim, renamed)
            causes[claim.claim_id] = (text, [renamed[p.evidence_id] for p in claim.passages])
            labels[claim.claim_id] = (
                f"{claim.driver_label or claim.driver} "
                f"({claim.bank} {claim.period} {claim.metric.replace('_', ' ')})"
            )
            if claim.contribution:
                how = (
                    "computed exactly from the issuer's published figures"
                    if claim.method == "computed"
                    else "from the issuer's own walk"
                )
                sizes.append(f"{labels[claim.claim_id]}: {claim.contribution}, {how}")
            cards.extend(
                EvidenceCard(
                    evidence_id=renamed[p.evidence_id],
                    component=claim.driver,
                    period="",
                    unit="aud_m",
                    statement=f"The issuer's commentary, cited for {labels[claim.claim_id]}.",
                    passage=p.text,
                    source=SourceRef(
                        doc_id=p.doc_id,
                        doc_type="RESULTS_DOCUMENT",
                        page=p.page,
                        publish_date=date(2026, 1, 1),
                    ),
                    method="text",
                )
                for p in claim.passages
            )
        try:
            check = check_narrative(
                "",
                cards,
                {},
                client,
                model=model,
                effort=effort,
                causes=causes,
                labels=labels,
                computed=sizes,
            )
        except ConsentDenied as refusal:
            # A grant that runs out leaves the rest unjudged, and the judged part is still
            # reported: the suite used to stop with a traceback and report nothing, after
            # three of its four calls had been paid for.
            unjudged.append(f"claims {group[0].claim_id} to {group[-1].claim_id}: {refusal}")
            continue
        if check.skipped:
            unjudged.append(f"claims {group[0].claim_id} to {group[-1].claim_id}: {check.skipped}")
            continue
        verdicts.update(
            {v.driver_id: (v.verdict, v.reason) for v in check.verdicts if v.driver_id is not None}
        )
    return verdicts, unjudged
