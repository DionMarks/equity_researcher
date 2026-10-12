"""Step 10: confidence from what stands behind each figure and driver, never from a model's
view of itself."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from ...confidence.features import (
    BY_IDENTITY,
    ConfidenceFeatures,
    DriverEvidence,
    FigureEvidence,
    ReadFrom,
)
from ...confidence.rubric import ScoredConfidence, overall_confidence, score_driver, score_figures
from ...decomposition.taxonomy import STAND_INS
from ...domain.models import (
    Decomposition,
    DriverContribution,
    EvidenceCard,
    Fact,
    RequestSpec,
    ResolvedInputs,
)
from ...domain.registry import get_registry
from ...ingestion.validate import IdentityCheck, validate_facts

__all__ = ["ScoredRun", "driver_evidence", "figure_evidence", "score_run"]

#: Filings that state a bank's results as its accounts: what a figure printed there is
#: printed in, rather than repeated in a presentation or a release.
_PRIMARY_FILINGS: Final = frozenset(
    {"RESULTS_DOCUMENT", "ANNUAL_REPORT", "PILLAR3", "RESTATEMENT_NOTICE"}
)
#: The share of the movement a flat figure's rounding counts as: all of it, and then some.
_FLAT: Final = Decimal(9)
_WALK_CONFIRMED: Final = "the walk adds up to the published movement"


class ScoredRun:
    """The scores for one run: the headline, each driver, and the whole."""

    def __init__(
        self,
        headline: ScoredConfidence,
        drivers: dict[str, ScoredConfidence],
        overall: ScoredConfidence,
    ) -> None:
        self.headline = headline
        self.drivers = drivers
        self.overall = overall

    def removed_driver_ids(self) -> list[str]:
        return [driver_id for driver_id, score in self.drivers.items() if score.removed]


@dataclass
class _Tally:
    """Counts what stands behind each printed figure, one figure at a time."""

    figures: int = 0
    primary: int = 0
    corroborated: int = 0
    confirmed: int = 0
    half: Decimal = Decimal(0)

    def add(self, readings: list[Fact], confirmed: bool) -> None:
        self.figures += 1
        self.primary += any(fact.doc_type in _PRIMARY_FILINGS for fact in readings)
        self.corroborated += len({fact.doc_id for fact in readings}) > 1
        self.confirmed += confirmed
        self.half += min((_half_unit(fact) for fact in readings), default=Decimal(0))

    def evidence(
        self,
        movement: Decimal | None,
        *,
        model_matched: bool = False,
        restated: bool = False,
        footnoted: bool = True,
        confirmed_by: str = BY_IDENTITY,
    ) -> FigureEvidence:
        """The counts as evidence. Rounding is measured against ``movement``, and not at all
        where there is none to measure it against."""
        if movement is None:
            share = Decimal(0)
        else:
            share = min(self.half / abs(movement), _FLAT) if movement else _FLAT
        return FigureEvidence(
            figures=self.figures,
            primary=self.primary,
            corroborated=self.corroborated,
            confirmed=self.confirmed,
            model_matched=model_matched,
            restated=restated,
            restatement_footnoted=footnoted,
            rounding_share=share,
            confirmed_by=confirmed_by,
        )


def figure_evidence(
    spec: RequestSpec,
    facts: list[Fact],
    inputs: ResolvedInputs,
    movement: Decimal,
    checks: list[IdentityCheck] | None = None,
) -> FigureEvidence:
    """What stands behind the two headline figures, counted off the stored figures.

    Every stored reading of the headline that equals the figure chosen for a period counts:
    one in a primary filing makes that figure primary, two in different filings corroborate
    it, and a passed identity check whose total it is confirms it.
    """
    checks = _checks(spec, facts) if checks is None else checks
    components = {spec.metric_id, spec.measure}
    tally = _Tally()
    for side, period in _sides(spec):
        value = inputs.get(spec.measure, side)  # type: ignore[arg-type]
        if value is None:
            value = inputs.get(spec.metric_id, side)  # type: ignore[arg-type]
        if value is None:
            continue
        readings = _readings(facts, period, components, value)
        confirmed = any(
            check.passed
            and check.period == period
            and check.inputs[:1]
            and check.inputs[0] in components
            and check.reported == value
            for check in checks
        )
        tally.add(readings, confirmed)
    return tally.evidence(
        movement,
        model_matched=_model_matched(spec, components),
        restated=bool(components & set(inputs.restated_components)),
        footnoted=inputs.restatement_footnoted,
    )


def driver_evidence(
    driver: DriverContribution,
    spec: RequestSpec,
    decomposition: Decomposition,
    facts: list[Fact],
    inputs: ResolvedInputs,
    cards: dict[str, EvidenceCard],
    checks: list[IdentityCheck],
) -> DriverEvidence:
    """What stands behind one driver: for one computed by the identity, the figures it was
    computed from; for a step of the bank's walk, the step as printed on its page."""
    if driver.method == "computed":
        components = list(dict.fromkeys(driver.from_components))
        tally = _Tally()
        for component in components:
            for side, period in _sides(spec):
                value = inputs.get(component, side)  # type: ignore[arg-type]
                if value is None:
                    continue
                confirmed = any(
                    check.passed and check.period == period and component in check.inputs
                    for check in checks
                )
                tally.add(_readings(facts, period, {component}, value), confirmed)
        # Rounding counts where the driver is one line's own movement, in the unit the line
        # was printed in. A ratio split is exact arithmetic over amounts, whose printing is
        # far finer than the basis points it is measured in.
        own_line = decomposition.change_unit == "aud_m" and len(components) == 1
        return DriverEvidence(
            figures=tally.evidence(
                driver.contribution if own_line else None,
                model_matched=_model_matched(spec, components),
                restated=bool(set(components) & set(inputs.restated_components)),
                footnoted=inputs.restatement_footnoted,
            )
        )

    printed = [
        cards[evidence_id]
        for evidence_id in driver.evidence_ids
        if evidence_id in cards and cards[evidence_id].value is not None
    ]
    tally = _Tally()
    for card in printed:
        tally.figures += 1
        tally.primary += card.source.doc_type in _PRIMARY_FILINGS
        tally.confirmed += "walk_checksum" in card.checks_passed
        tally.half += _half_printed(card.published_value, card.value, card.unit)
    methods = {card.method for card in printed}
    read_from: ReadFrom = (
        "chart" if "chart" in methods else "text" if "text" in methods else "table"
    )
    headline = {spec.metric_id, spec.measure}
    return DriverEvidence(
        figures=tally.evidence(
            driver.contribution if driver.contribution is not None else Decimal(0),
            restated=bool(headline & set(inputs.restated_components)),
            footnoted=inputs.restatement_footnoted,
            confirmed_by=_WALK_CONFIRMED,
        ),
        read_from=read_from,
        # A walk reaches here only after it bridged the published endpoints.
        checksum_ok=True if read_from == "chart" else None,
        chained=driver.method == "chained_walk",
    )


def _sides(spec: RequestSpec) -> tuple[tuple[str, str], tuple[str, str]]:
    return (("prior", spec.comparison_period), ("current", spec.period))


def _readings(
    facts: list[Fact], period: str, components: Collection[str], value: Decimal
) -> list[Fact]:
    """Every stored reading of a figure: the same component and period, the same value."""
    return [
        fact
        for fact in facts
        if fact.period == period and fact.component in components and fact.value == value
    ]


def _checks(spec: RequestSpec, facts: list[Fact]) -> list[IdentityCheck]:
    """The identity checks over this question's two periods, run again on what's held."""
    if not facts:
        return []
    try:
        return list(
            validate_facts(facts, spec.bank_id, (spec.comparison_period, spec.period)).checks
        )
    except Exception:  # a check that can't run confirms nothing, and mustn't stop the run
        return []


def _model_matched(spec: RequestSpec, components: Collection[str]) -> bool:
    """Whether a model named the row of any of these, for a bank nobody configured."""
    profile = get_registry().bank(spec.bank_id)
    return profile.discovered and any(
        component in components for _, component in profile.row_labels
    )


def _half_unit(fact: Fact) -> Decimal:
    return _half_printed(fact.published_value, fact.value, fact.unit)


def _half_printed(printed: Decimal | None, value: Decimal | None, unit: str) -> Decimal:
    """Half the last digit printed, in the unit a movement is measured in: basis points for a
    ratio, A$ millions for an amount."""
    shown = printed if printed is not None else value
    if shown is None:
        return Decimal(0)
    exponent = shown.as_tuple().exponent
    places = -exponent if isinstance(exponent, int) and exponent < 0 else 0
    half = Decimal(5) / (Decimal(10) ** (places + 1))
    if shown and value is not None and value != shown:  # printed in thousands or billions
        half *= abs(value / shown)
    if unit == "pct":
        half *= 100  # percentage points to basis points
    return half


def score_run(
    spec: RequestSpec,
    decomposition: Decomposition,
    inputs: ResolvedInputs,
    facts: list[Fact] | None = None,
    cards: list[EvidenceCard] | None = None,
    validation_failed: bool = False,
    figures: FigureEvidence | None = None,
) -> ScoredRun:
    """Score the headline and every driver from what stands behind them."""
    registry = get_registry()
    facts = facts or []
    by_id = {card.evidence_id: card for card in cards or []}
    checks = _checks(spec, facts)
    definition_break = bool(
        registry.regime_changes_for(
            spec.bank_id, spec.metric_id, spec.comparison_start, spec.period_end
        )
    )
    unexplained = {
        conflict.component
        for conflict in inputs.conflicts
        if conflict.classification == "UNEXPLAINED"
    }
    sign_flipping = {
        conflict.component
        for conflict in inputs.conflicts
        if conflict.classification == "UNEXPLAINED"
        and len({1 if c.value > 0 else -1 if c.value < 0 else 0 for c in conflict.candidates}) > 1
    }

    headline = score_figures(
        figures
        if figures is not None
        else figure_evidence(spec, facts, inputs, decomposition.headline_change, checks),
        ConfidenceFeatures(
            unexplained_conflict=spec.metric_id in unexplained,
            conflict_flips_sign=spec.metric_id in sign_flipping,
            definition_break=definition_break,
            input_failed_validation=validation_failed,
        ),
    )

    residual_beyond = (
        decomposition.residual if abs(decomposition.residual) > decomposition.tolerance else None
    )
    scores: dict[str, ScoredConfidence] = {}
    for driver in decomposition.drivers:
        # A conflict caps only the drivers computed from the component in conflict; one with
        # no recorded inputs is a step of the bank's walk, tested at the metric as the
        # headline is.
        depends_on = set(driver.from_components)
        touched = (depends_on & unexplained) if depends_on else ({spec.metric_id} & unexplained)
        flipped = (depends_on & sign_flipping) if depends_on else ({spec.metric_id} & sign_flipping)
        # A stand-in for a breakdown the run didn't read (CET1 capital as one figure), with
        # nothing beneath it, is a total rather than a cause.
        stand_in = driver.driver_id in STAND_INS and not any(
            other.parent_id == driver.driver_id for other in decomposition.drivers
        )
        indicative = (
            driver.parent_id is not None
            and driver.method == "computed"
            and any("is indicative" in note for note in decomposition.notes)
        )
        scores[driver.driver_id] = score_driver(
            driver_evidence(driver, spec, decomposition, facts, inputs, by_id, checks),
            ConfidenceFeatures(
                path_b_residual=residual_beyond,
                path_b_tolerance=decomposition.tolerance if residual_beyond else None,
                paths_disagree=driver.driver_id in decomposition.disagreements,
                unexplained_conflict=bool(touched),
                conflict_flips_sign=bool(flipped),
                inputs_aggregated=stand_in and driver.method == "computed",
                indicative_split=indicative,
                uncited=not driver.evidence_ids and not driver.from_components,
                definition_break=definition_break,
                input_failed_validation=validation_failed,
            ),
        )

    weighted = [
        (
            scores[driver.driver_id].score,
            abs(driver.contribution) if driver.contribution is not None else Decimal(0),
        )
        for driver in decomposition.drivers
        if driver.parent_id is None and driver.driver_id in scores
    ]
    overall = overall_confidence(
        headline=headline,
        driver_scores=weighted,
        residual=decomposition.residual,
        headline_change=decomposition.headline_change,
    )
    return ScoredRun(headline=headline, drivers=scores, overall=overall)
