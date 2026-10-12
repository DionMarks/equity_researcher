"""Step 7: settling on one value per component, and recording what it cost."""

from __future__ import annotations

from collections import defaultdict
from typing import Final

from ...decomposition.identities import identity_for
from ...domain.models import (
    Basis,
    CandidateValue,
    EvidenceCard,
    Fact,
    RequestSpec,
    ResolvedInputs,
)
from ...domain.registry import component_name, get_registry
from ...domain.units import normalise_aud, same_published_figure
from ...output.schema import describe_basis
from ...reconcile.conflicts import resolve

__all__ = ["reconcile_inputs"]


def _supports(card_value: CandidateValue, chosen: CandidateValue) -> bool:
    """Whether this candidate's figure is the one that was chosen.

    Westpac prints 1H26 net profit as $3,414m in the interim financial report and as $3,483m
    in the investor pack's net-profit appendix. Both are real, the conflict between them is
    recorded, and the results document wins on precedence. But citing the appendix page next
    to a figure it doesn't contain would be a false citation: an analyst who opens that page
    finds a different number.
    """
    return same_published_figure(card_value.value, card_value.unit, chosen.value, chosen.unit)


#: What comes after profit before tax: none of it is part of a headline struck before tax.
_AFTER_TAX: Final = frozenset(
    {"headline_earnings", "tax", "minority_interests", "discontinued_ops"}
)


def _held_as(component: str, spec: RequestSpec) -> str | None:
    """The component a figure is resolved as for this request, or ``None`` to leave it out.

    Judo has led with profit before tax since its 2026 half year, so its earnings question
    is answered before tax in both periods: profit before tax takes the headline's place,
    and the after-tax profit, tax and the lines below them aren't part of the answer. The
    after-tax figure is still what return on equity rests on, and a return on equity
    request reads it as usual (DESIGN.md ADR-062).
    """
    if not spec.pre_tax:
        return component
    if component == "profit_before_tax":
        return spec.metric_id
    return None if component in _AFTER_TAX else component


def _headline_basis(spec: RequestSpec) -> Basis:
    """The basis the issuer led with, which is what every conflict resolves toward."""
    registry = get_registry()
    profile = registry.bank(spec.bank_id)
    scope = "level2" if spec.metric_id == "cet1_ratio" else "group"
    earnings = "cash" if spec.metric_id in {"headline_earnings", "roe"} else "n/a"
    # Westpac's headline measure is statutory net profit from 1H23, so "cash" would
    # be the wrong target for it.
    rule = registry.metrics[spec.metric_id].label_for(profile, spec.period)
    if rule is not None and rule.label.lower().startswith("net profit"):
        earnings = "statutory"
    return Basis(earnings_basis=earnings, scope=scope)  # type: ignore[arg-type]


def _with_the_metrics_own_treatment(
    headline: Basis,
    candidates: dict[tuple[str, str], list[CandidateValue]],
    spec: RequestSpec,
) -> tuple[Basis, str | None]:
    """The headline basis, excluding notable items where that's the treatment both periods
    can be compared on, with the caveat that says why.

    Westpac publishes its expense-to-income ratio only excluding Notable Items, and also
    publishes the expenses and income that ratio is made of, both ways. Resolving the
    inputs toward the reported figures left a ratio on one basis and its components on
    another, so the identity couldn't run and the movement had no drivers. Where the
    issuer prints the metric both ways for both periods, the reported figure leads and the
    variant is context.

    The choice is made over the two periods together. Westpac's 1H26 pack prints its margin
    only excluding Notable Items, for both halves, while its own 1H25 pack also printed the
    reported 1.88%. Deciding over every reading held let that one reported figure keep the
    headline "reported", and the answer compared 1.89% excluding Notable Items with 1.88%
    including them.
    """
    held: dict[str, set[bool]] = defaultdict(set)
    for (period, component), options in candidates.items():
        if component == spec.metric_id:
            held[period].update(option.basis.ex_notable_items for option in options)
    current, prior = held[spec.period], held[spec.comparison_period]
    excluding = headline.model_copy(update={"ex_notable_items": True})
    if False in current and (False in prior or not prior):
        return headline, None
    if current == {True}:
        return excluding, (
            f"{spec.bank_id} publishes this measure only excluding notable items, so the "
            "measure and the figures it's explained by are all on that basis. The reported "
            "figures, including notable items, are recorded as alternative readings."
        )
    if True in current and True in prior:
        return excluding, (
            f"{spec.bank_id} publishes the {spec.comparison_period} figure only excluding "
            "notable items, so both periods are compared on that basis. The reported "
            f"{spec.period} figure is recorded as an alternative reading."
        )
    return headline, None


def reconcile_inputs(
    cards: list[EvidenceCard],
    facts: list[Fact],
    spec: RequestSpec,
) -> ResolvedInputs:
    """Merge cards and facts into one resolved value per component and period."""
    by_evidence: dict[tuple[str, str], list[str]] = defaultdict(list)
    candidates: dict[tuple[str, str], list[CandidateValue]] = defaultdict(list)

    for card in cards:
        component = _held_as(card.component, spec)
        if card.value is None or component is None:
            continue
        key = (card.period, component)
        by_evidence[key].append(card.evidence_id)
        candidates[key].append(
            CandidateValue(
                value=card.value,
                unit=card.unit,
                basis=card.basis,
                source=card.source,
                vintage=card.source.publish_date,
                is_restated=card.is_restated,
                footnoted=card.is_restated,
                evidence_id=card.evidence_id,
            )
        )

    seen = {(card.period, card.component, card.source.doc_id, card.value) for card in cards}
    for fact in facts:
        component = _held_as(fact.component, spec)
        if component is None or (fact.period, fact.component, fact.doc_id, fact.value) in seen:
            continue
        candidates[(fact.period, component)].append(
            CandidateValue(
                value=fact.value,
                unit=fact.unit,
                basis=fact.basis,
                source=_fact_source(fact),
                vintage=fact.publish_date,
                is_restated=fact.is_restated,
                footnoted=fact.is_restated,
            )
        )

    inputs = ResolvedInputs()
    headline, treatment = _with_the_metrics_own_treatment(_headline_basis(spec), candidates, spec)
    if treatment is not None:
        inputs.caveats.append(treatment)
    restated: set[str] = set()
    # Only a restatement of something this answer uses is worth a caveat, or a lower
    # confidence: a capital memo was told CBA had restated its credit impairment comparative.
    identity = identity_for(spec.metric_id, pre_tax=spec.pre_tax)
    relevant = {
        spec.metric_id,
        *(identity.components if identity else ()),
        *(identity.optional_components if identity else ()),
    }

    for (period, component), options in sorted(candidates.items()):
        chosen, conflicts = resolve(
            component,
            options,
            as_of=spec.as_of,
            headline_basis=headline,
            period_end=spec.period_end if period == spec.period else spec.comparison_end,
        )
        # Converted to millions before anything uses them. A pack gives its income statement
        # in $M and its balance sheet in $B, and an identity that divides one by the other
        # would otherwise be out by a factor of a thousand while still adding up internally.
        value, unit = normalise_aud(chosen.value, chosen.unit)
        target = inputs.current if period == spec.period else inputs.prior
        target[component] = value
        inputs.units[component] = unit
        inputs.bases[f"{period}:{component}"] = chosen.basis
        # Only the cards that carry the chosen figure. Every candidate's card stays in the
        # evidence list (the conflict names them, and a reader has to be able to follow it),
        # but a claim only cites the pages that support it. On its own basis where it can:
        # NAB prints its 1H25 return on equity as 11.7% both with and without large notable
        # items, and citing the variant's row for the headline sends a reader to a
        # different measure that happens to round the same.
        carrying = [
            option
            for option in options
            if option.evidence_id is not None and _supports(option, chosen)
        ]
        same_basis = [option for option in carrying if option.basis == chosen.basis]
        supporting = [
            option.evidence_id
            for option in same_basis or carrying
            if option.evidence_id is not None
        ]
        if supporting:
            inputs.evidence_ids[f"{period}:{component}"] = supporting
        inputs.conflicts.extend(conflicts)
        if chosen.is_restated and component in relevant:
            restated.add(component)
            if not chosen.footnoted:
                inputs.restatement_footnoted = False

    mixed = _across_bases(inputs, spec)
    if mixed is not None:
        inputs.caveats.append(mixed)
    inputs.restated_components = sorted(restated)
    return inputs


def restatement_caveat(
    spec: RequestSpec, restated: list[str], used: set[str], footnoted: bool
) -> str | None:
    """A caveat for the restated comparatives the answer actually uses, named in words.

    It used to be written before the decomposition, so it couldn't know what the answer
    would use: Westpac's 1H26 margin answer, explained wholly by the bank's own walk, warned
    that its "nii" comparative was restated, a figure it never touched, under an id no reader
    knows. Now it names only figures the answer rests on, and says nothing otherwise.
    """
    relevant = [component for component in restated if component in used]
    if not relevant:
        return None
    names = [component_name(component) for component in relevant[:2]]
    others = len(relevant) - len(names)
    listed = " and ".join(names) + (f" and {others} other figure(s)" if others else "")
    why = (
        "The bank's footnote says comparatives were restated without saying which, so each "
        "one read from that table is flagged."
        if footnoted
        else "No footnote says why."
    )
    return f"The {spec.comparison_period} {listed} is the restated comparative. {why}"


def _across_bases(inputs: ResolvedInputs, spec: RequestSpec) -> str | None:
    """A caveat where the two periods' figures for the metric are on different bases.

    It happens only where no one basis is printed for both, as when an issuer starts or
    stops excluding notable items between the two documents. The movement then includes
    the change of basis, and a reader has to be told rather than shown a clean difference.
    """
    now = inputs.bases.get(f"{spec.period}:{spec.metric_id}")
    then = inputs.bases.get(f"{spec.comparison_period}:{spec.metric_id}")
    if now is None or then is None:
        return None
    if now.ex_notable_items == then.ex_notable_items and (
        now.earnings_basis == then.earnings_basis
        or "n/a" in {now.earnings_basis, then.earnings_basis}
    ):
        return None
    return (
        f"The {spec.period} figure is {describe_basis(now) or 'as reported'} and the "
        f"{spec.comparison_period} figure is {describe_basis(then) or 'as reported'}: no one "
        "basis is printed for both, so part of the movement is the change of basis."
    )


def _fact_source(fact: Fact):  # type: ignore[no-untyped-def]
    from ...domain.models import SourceRef

    return SourceRef(
        doc_id=fact.doc_id,
        doc_type=fact.doc_type,
        page=fact.page_no,
        table_id=fact.table_id,
        publish_date=fact.publish_date,
    )
