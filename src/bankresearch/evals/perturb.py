"""Perturbation suites: take evidence away, move the date before publication or inject a conflict,
and check the answer degrades as it should."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from ..domain.models import Fact
from ..reconcile.conflicts import CLASSIFICATION_ORDER
from .cases import EvalCase

__all__ = [
    "SUITES",
    "UNEXPLAINED_CAP",
    "Perturbation",
    "PerturbationResult",
    "ablations_for",
    "conflicts_for",
    "look_ahead_for",
    "perturbations_for",
]

SUITES = ("ablation", "look_ahead", "conflict", "distractor")


@dataclass(frozen=True, slots=True)
class Perturbation:
    """One case, altered in one way, with what the alteration should cost."""

    case_id: str
    suite: str
    description: str
    drop_components: tuple[str, ...] = ()
    as_of_override: date | None = None
    inject: tuple[Fact, ...] = ()
    expect_answer: bool = True
    """Whether a figure should still be reportable."""
    expect_drivers: bool | None = True
    """Whether an attribution should still be claimed. ``None`` leaves it unasserted."""
    expect_confidence_at_most: Decimal | None = None
    """A ceiling confidence must respect, for a class that caps it."""
    expect_conflict: str = ""
    """The class the reconciler should assign, where the perturbation implies one."""
    expect_other_decomposition: bool = False
    """Whether the attribution should move to another complete decomposition that doesn't
    use the removed input, rather than disappear."""

    def applies_to(self, facts: list[Fact]) -> list[Fact]:
        """The fact set this perturbation presents to the agent."""
        kept = [fact for fact in facts if fact.component not in self.drop_components]
        return [*kept, *self.inject]


@dataclass
class PerturbationResult:
    perturbation: Perturbation
    answered: bool = False
    drivers: int = 0
    confidence: Decimal | None = None
    baseline_confidence: Decimal | None = None
    classification: str = ""
    driver_ids: frozenset[str] = frozenset()
    baseline_driver_ids: frozenset[str] = frozenset()
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether the system degraded the way it should have.

        Each expectation is checked, and a perturbation that improved an outcome fails too:
        an ablation that produced more drivers than the clean run means the drivers weren't
        coming from the removed input.
        """
        if self.answered != self.perturbation.expect_answer:
            return False
        if self.perturbation.expect_other_decomposition:
            # The drivers must come from the other decomposition: none of them may be one
            # the clean run attributed through the input that was removed.
            return self.drivers > 0 and self.driver_ids.isdisjoint(self.baseline_driver_ids)
        if self.perturbation.expect_drivers is not None and self.perturbation.expect_drivers != (
            self.drivers > 0
        ):
            return False
        if self.perturbation.expect_conflict:
            # Detection is the point, since a conflict nobody noticed doesn't cost anything
            # either. A class earlier in the order than the one implied is a more specific
            # cause and a better answer, so it counts.
            if not self.classification:
                return False
            order: list[str] = list(CLASSIFICATION_ORDER)
            if (
                self.classification not in order
                or self.perturbation.expect_conflict not in order
                or order.index(self.classification) > order.index(self.perturbation.expect_conflict)
            ):
                return False
            # The cap belongs to the class, not the perturbation: a conflict resolved as
            # BASIS isn't capped, and asserting the cap anyway failed a run that had found a
            # better explanation than the one asked for.
            ceiling = self.perturbation.expect_confidence_at_most
            if self.classification == "UNEXPLAINED" and ceiling is not None:
                return self.confidence is not None and self.confidence <= ceiling
        return True


def ablations_for(
    case: EvalCase,
    identity_components: tuple[str, ...],
    covered_elsewhere: frozenset[str] = frozenset(),
) -> list[Perturbation]:
    """One ablation per identity input.

    The headline figure is never removed: without it there's no movement, which tests the
    abstention path rather than attribution, and that path has its own cases.

    ``covered_elsewhere`` names inputs whose loss another complete decomposition survives.
    CBA's impairment charge is split both ways in its pack, individual and collective in
    the notes and by division in the segment pages, so removing the individual charge
    should move the answer to the divisions, not end it.
    """
    out: list[Perturbation] = []
    for component in identity_components:
        if component == case.metric:
            continue
        fallback = component in covered_elsewhere
        out.append(
            Perturbation(
                case_id=case.case_id,
                suite="ablation",
                description=f"{component} removed from the store"
                + ("; another complete split is held" if fallback else ""),
                drop_components=(component,),
                # The movement is the issuer's own published figure and survives.
                expect_answer=True,
                # The attribution needs every input, so it must not, unless another
                # decomposition can carry it without the input.
                expect_drivers=None if fallback else False,
                expect_other_decomposition=fallback,
            )
        )
    return out


def look_ahead_for(case: EvalCase, publish_date: date) -> Perturbation:
    """The as-of date moved back before the document's publication.

    The only perturbation whose failure is a correctness bug rather than a confidence bug.
    An answer here means a backtest could read a figure before it existed, in the direction
    that flatters the result.
    """
    return Perturbation(
        case_id=case.case_id,
        suite="look_ahead",
        description=f"as-of moved to the day before {publish_date.isoformat()}",
        as_of_override=publish_date - timedelta(days=1),
        expect_answer=False,
        expect_drivers=False,
    )


# The ``UNEXPLAINED`` cap from config/precedence.yaml. It's a ceiling, not a deduction: a
# run already at or below it isn't lowered further, so the check is "no higher than the cap"
# rather than "lower than before".
UNEXPLAINED_CAP = Decimal("0.75")


def conflicts_for(case: EvalCase, prior: Fact, *, rounding: bool) -> Perturbation:
    """A second figure for the prior period, disagreeing by a known amount.

    The disagreement is sized against the figure's own published precision rather than a
    fixed amount, because the same 0.01 is a rounding difference in one unit and a material
    one in another, and a suite that mislabels its own perturbation measures nothing. Half a
    tolerance is reliably a rounding difference; ten tolerances reliably isn't.

    The expectation follows the class, which is the reason for having classes:

    * A conflict must be detected, and classified no later in the order than the class the
      perturbation implies. Where a metric carries two conflicts, the basis difference
      between its headline and its variant and a disagreement among the headline's own
      readings, the second is the one judged. Injecting a materially different prior into
      CBA's impairment used to come back as ``BASIS`` alone, because the corpus holds two
      bases for it, and that let a same-basis disagreement pass with no cost to
      confidence. The headline's own readings are now classified among themselves.
    * Where ``UNEXPLAINED`` is the class, confidence must respect its cap.

    Two things aren't checked, because both would mix up the class's own cost with ordinary
    downstream effects. Whether an attribution survives depends on the identity. And a
    ``ROUNDING`` conflict is resolved by taking the more precise value, which changes the
    movement and so the confidence features, so confidence can move even though the class
    itself has no penalty.
    """
    # Sized in the figure's own unit. ``value_tolerance`` answers in basis points for a
    # ratio, and adding that to a percentage turns 11.8% into 14.3%: a 250bp change labelled
    # as a rounding difference, which would have made this suite check the opposite of what
    # it was measuring.
    ulp = Decimal(1).scaleb(prior.value.as_tuple().exponent)  # type: ignore[arg-type]
    shift = ulp * Decimal("0.4") if rounding else ulp * Decimal(50)
    expect = "ROUNDING" if rounding else "UNEXPLAINED"
    injected = prior.model_copy(
        update={
            "fact_id": f"{prior.fact_id}-injected",
            "value": prior.value + shift,
            "published_value": prior.value + shift,
            "doc_id": f"{prior.doc_id}-injected",
            "doc_type": "INVESTOR_PRESENTATION",
        }
    )
    return Perturbation(
        case_id=case.case_id,
        suite="conflict",
        description=(
            f"{prior.component} {prior.period} also stated as {injected.value} "
            f"({expect}, {'half a' if rounding else 'ten'} tolerance)"
        ),
        inject=(injected,),
        expect_answer=True,
        # Whether an attribution survives depends on the identity and can legitimately vary:
        # a conflict resolved by precedence may or may not leave the bridge intact. Checking
        # it was overreach, and this suite is about detection and cost.
        expect_drivers=None,
        expect_confidence_at_most=None if rounding else UNEXPLAINED_CAP,
        expect_conflict=expect,
    )


def perturbations_for(
    case: EvalCase,
    identity_components: tuple[str, ...],
    publish_date: date | None = None,
    prior: Fact | None = None,
    covered_elsewhere: frozenset[str] = frozenset(),
) -> list[Perturbation]:
    """Every perturbation this case supports.

    A case that can't support a suite contributes nothing to it, rather than a made-up
    stand-in that would just measure the stand-in.
    """
    out = ablations_for(case, identity_components, covered_elsewhere)
    if publish_date is not None:
        out.append(look_ahead_for(case, publish_date))
    if prior is not None:
        out.append(conflicts_for(case, prior, rounding=True))
        out.append(conflicts_for(case, prior, rounding=False))
    return out


def distractor_note() -> dict[str, Any]:
    """Why the distractor suite doesn't run here.

    Listed rather than left out without comment: the suite's question is whether irrelevant
    pages in a model's context change its answer, and with no model there's no context.
    """
    return {
        "suite": "distractor",
        "skipped": "needs a model: without one there is no context for a distractor to pollute",
    }
