"""Layered scoring of one case: its status, figures, drivers, citations and abstention."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from ..domain.units import normalise_numbers, same_published_figure
from ..ingestion.artifacts import read_processed
from ..output.schema import ResearchResult
from ..settings import Settings
from .cases import EvalCase

__all__ = [
    "CaseScore",
    "citation_faithfulness",
    "claims_cite_their_own_figure",
    "score_case",
]

_NUMBER_IN_TEXT = re.compile(r"-?\$?[\d,]+(?:\.\d+)?%?")


@dataclass
class CaseScore:
    """One case's result, with each measure marked as scored or not applicable."""

    case_id: str
    status: str
    expected_status: tuple[str, ...]
    status_correct: bool
    abstain_reason: str | None = None
    abstain_reason_correct: bool | None = None

    drivers: int = 0
    reconciles: bool | None = None
    residual: Decimal | None = None
    tolerance: Decimal | None = None

    cited_numbers: int = 0
    cited_numbers_found: int = 0
    citations_without_a_page: int = 0
    claims_citing_another_figure: int = 0

    overall_confidence: float | None = None
    confidence_level: str | None = None
    headline_confidence: float | None = None
    headline_confidence_level: str | None = None
    """The headline figure's own confidence, which is what ``headline_exact`` measures.

    ``overall_confidence`` is the lower of the headline and the attribution, minus a penalty
    for an unexplained residual, so it answers "how much of this movement is explained", not
    "is this figure right". Calibrating it against ``headline_exact`` scores a composite
    against an outcome that measures only one part of it: tightening the attribution caps, a
    real improvement to the memo, moved ECE from 0.14 to 0.32 without a single figure
    changing.
    """

    latency_ms: int = 0
    llm_calls: int = 0
    cost_usd: Decimal = Decimal(0)

    # Gold-dependent, ``None`` when the case carries no verified gold.
    headline_exact: bool | None = None
    change_error: Decimal | None = None
    change_tolerance: Decimal | None = None
    driver_precision: float | None = None
    driver_recall: float | None = None
    driver_values_checked: int = 0
    driver_values_right: int = 0
    """Gold drivers the run also named, and how many of those it sized right: the same
    sign and within the gold's tolerance. Naming the right driver with the wrong size is
    the error precision and recall can't see."""
    driver_claims: list[tuple[str, float, str, bool]] = field(default_factory=list)
    """(driver id, confidence score, level, correct) for every material driver the run
    named in a case with driver gold, so calibration can be measured per driver."""

    notes: list[str] = field(default_factory=list)

    @property
    def citation_faithfulness(self) -> float | None:
        if not self.cited_numbers:
            return None
        return self.cited_numbers_found / self.cited_numbers

    @property
    def passed(self) -> bool:
        """A case passes when it behaved as asserted and its citations check out."""
        if not self.status_correct:
            return False
        if self.abstain_reason_correct is False:
            return False
        if self.citations_without_a_page:
            return False
        if self.claims_citing_another_figure:
            return False
        faithfulness = self.citation_faithfulness
        return not (faithfulness is not None and faithfulness < 1.0)


def citation_faithfulness(
    result: ResearchResult, settings: Settings
) -> tuple[int, int, int, list[str]]:
    """Check every cited figure against the page it cites.

    Gold-free, and the strongest check there is without labels: a number reported against a
    page that doesn't contain it is a fabricated citation, whoever produced it.

    Matching goes through :func:`normalise_numbers`, so ``1.95%``, ``195 bps`` and ``1,234``
    match their variants. Requiring a literal substring would fail correct citations, and
    then nobody would trust the check.
    """
    checked = 0
    found = 0
    missing_page = 0
    notes: list[str] = []
    page_cache: dict[tuple[str, int], set[Decimal]] = {}

    for card in result.evidence:
        if card.evidence_id not in result.cited_evidence_ids():
            continue
        if card.method == "chart":
            # A bar read off a chart image has no text to find it in. It was checked when it
            # was read: the bars had to bridge the published endpoints, or the whole reading
            # was thrown away.
            continue
        printed = card.published_value if card.published_value is not None else card.value
        if printed is None:
            continue
        if not card.source.doc_id or card.source.page < 1:
            missing_page += 1
            notes.append(f"{card.evidence_id} cites no document page")
            continue

        checked += 1
        key = (card.source.doc_id, card.source.page)
        if key not in page_cache:
            page_cache[key] = _numbers_on_page(key[0], key[1], settings)
        if printed in page_cache[key] or _close_enough(printed, page_cache[key]):
            found += 1
        else:
            notes.append(
                f"{card.evidence_id}: {printed} not found on "
                f"{card.source.doc_id} p.{card.source.page}"
            )
    return checked, found, missing_page, notes


def _numbers_on_page(doc_id: str, page_no: int, settings: Settings) -> set[Decimal]:
    document = read_processed(doc_id, settings.processed_dir)
    if document is None:
        return set()
    page = document.page(page_no)
    if page is None:
        return set()
    numbers = normalise_numbers(page.text)
    for table in page.tables:
        numbers |= normalise_numbers(table.to_markdown())
        # A bare dash in a figure column is how a results table prints nil, so a page that
        # prints one supports a figure of zero. CBA's Corporate Centre charge for 1H26 is
        # printed as a dash.
        if any(_NIL.match((cell or "").strip()) for row in table.rows for cell in row[1:]):
            numbers.add(Decimal(0))
    return numbers


_NIL = re.compile("^[-\u2013\u2014]$")


def _close_enough(value: Decimal, candidates: set[Decimal]) -> bool:
    """Allow for the page printing a figure at coarser precision than it was stored."""
    for candidate in candidates:
        if candidate == 0 and value == 0:
            return True
        if candidate and abs(value - candidate) <= abs(candidate) * Decimal("0.0005"):
            return True
    return False


def claims_cite_their_own_figure(result: ResearchResult) -> list[str]:
    """Claims citing a card that carries a different figure.

    ``citation_faithfulness`` asks whether each card's own number is on the page it names,
    which a card for a rejected candidate passes: Westpac's investor pack really does print
    $3,483m on page 119. What it can't see is that the claim next to it says $3,414m. A
    reader who follows that citation finds the other number with no way to tell which one
    the memo meant, so the claim is wrong even though every card in it is right.
    """
    by_id = {card.evidence_id: card for card in result.evidence}
    wrong: list[str] = []
    for name, claim in (("current", result.current), ("prior", result.prior)):
        if claim is None:
            continue
        for evidence_id in claim.evidence_ids:
            card = by_id.get(evidence_id)
            if card is None or card.value is None:
                continue
            if not same_published_figure(card.value, card.unit, claim.value, claim.unit):
                wrong.append(
                    f"{name} claims {claim.value} but cites {evidence_id} "
                    f"({card.value} on {card.source.doc_id} p.{card.source.page})"
                )
    return wrong


def score_case(
    case: EvalCase, result: ResearchResult, settings: Settings, model_mode: str = "off"
) -> CaseScore:
    """Score one case on everything that can be measured for it, against what the case
    expects in this model mode."""
    checked, found, missing_page, notes = citation_faithfulness(result, settings)
    mismatched = claims_cite_their_own_figure(result)
    notes.extend(mismatched)
    expected_status, expected_reason = case.expectation(model_mode)

    score = CaseScore(
        case_id=case.case_id,
        status=result.status,
        expected_status=expected_status,
        status_correct=result.status in expected_status,
        abstain_reason=result.abstain_reason,
        drivers=len(result.top_level_drivers),
        cited_numbers=checked,
        cited_numbers_found=found,
        citations_without_a_page=missing_page,
        claims_citing_another_figure=len(mismatched),
        overall_confidence=(result.overall_confidence.score if result.overall_confidence else None),
        confidence_level=(result.overall_confidence.level if result.overall_confidence else None),
        headline_confidence=(
            result.headline_confidence.score if result.headline_confidence else None
        ),
        headline_confidence_level=(
            result.headline_confidence.level if result.headline_confidence else None
        ),
        latency_ms=result.diagnostics.latency_ms,
        llm_calls=result.diagnostics.llm_calls,
        cost_usd=result.diagnostics.cost_usd,
        notes=notes,
    )

    if expected_reason is not None and result.status == "ABSTAIN":
        score.abstain_reason_correct = result.abstain_reason == expected_reason
    elif expected_reason is not None and "ABSTAIN" in expected_status:
        score.abstain_reason_correct = False
    elif result.status == "ABSTAIN":
        score.abstain_reason_correct = result.abstain_reason is not None

    if result.reconciliation is not None:
        score.reconciles = result.reconciliation.within_tolerance
        score.residual = result.reconciliation.residual
        score.tolerance = result.reconciliation.tolerance

    # Gold-dependent measures only run against verified gold.
    if case.gold_verified and case.has_gold:
        _score_against_gold(case, result, score)
    elif case.has_gold:
        score.notes.append("gold present but unverified; accuracy measures not scored")

    return score


def _score_against_gold(case: EvalCase, result: ResearchResult, score: CaseScore) -> None:
    if case.current is not None and case.prior is not None:
        current_ok = (
            result.current is not None
            and abs(result.current.value - case.current.value) <= case.current.tolerance
        )
        prior_ok = (
            result.prior is not None
            and abs(result.prior.value - case.prior.value) <= case.prior.tolerance
        )
        score.headline_exact = bool(current_ok and prior_ok)

    if case.change is not None and result.change is not None:
        score.change_error = abs(result.change.absolute - case.change.value)
        score.change_tolerance = case.change.tolerance

    # The gold's materiality threshold has to apply to both sides, or precision punishes a
    # run for naming a driver the gold itself chose to ignore. ANZ's minority interests
    # moved $1m on a $212m half and Westpac's didn't move at all; against a $2m threshold
    # neither is an error to report or a hit to claim.
    def _material(contribution: Decimal | None) -> bool:
        return (
            case.materiality is None
            or contribution is None
            or abs(contribution) >= case.materiality
        )

    gold_drivers = _gold_drivers_for(case, result)
    expected = {driver.driver_id for driver in gold_drivers if _material(driver.contribution)}
    if expected:
        predicted = {
            driver.driver_id
            for driver in result.top_level_drivers
            if _material(driver.contribution)
        }
        hits = len(expected & predicted)
        score.driver_precision = hits / len(predicted) if predicted else 0.0
        score.driver_recall = hits / len(expected)
        _score_driver_values(gold_drivers, result, score, _material)


_WALKS = frozenset({"management_walk", "chained_walk"})


def _gold_drivers_for(case: EvalCase, result: ResearchResult) -> tuple[Any, ...]:
    """The bank's walk where the run used a walk and the case labels one; else the
    identity's drivers."""
    walked = any(driver.method in _WALKS for driver in result.top_level_drivers)
    return case.walk_drivers if walked and case.walk_drivers else case.drivers


def _score_driver_values(
    gold_drivers: tuple[Any, ...],
    result: ResearchResult,
    score: CaseScore,
    material: Callable[[Decimal | None], bool],
) -> None:
    """Whether each driver the run shares with the gold is the right size and sign.

    A driver the run named that the gold doesn't have is a wrong claim; one it sized wrong
    is too. Every material driver the run named becomes one calibration claim, scored
    against its own confidence.
    """
    gold = {
        driver.driver_id: driver
        for driver in gold_drivers
        if driver.contribution is not None and material(driver.contribution)
    }
    for driver in result.top_level_drivers:
        if driver.contribution is None or not material(driver.contribution):
            continue
        expected = gold.get(driver.driver_id)
        correct = False
        if expected is not None and expected.contribution is not None:
            tolerance = expected.tolerance if expected.tolerance is not None else Decimal(1)
            same_sign = (driver.contribution >= 0) == (expected.contribution >= 0)
            correct = same_sign and abs(driver.contribution - expected.contribution) <= tolerance
            score.driver_values_checked += 1
            score.driver_values_right += int(correct)
            if not correct:
                score.notes.append(
                    f"{driver.driver_id} sized {driver.contribution} against gold "
                    f"{expected.contribution} (tolerance {tolerance})"
                )
        score.driver_claims.append(
            (
                driver.driver_id,
                float(driver.confidence.score),
                str(driver.confidence.level),
                correct,
            )
        )
