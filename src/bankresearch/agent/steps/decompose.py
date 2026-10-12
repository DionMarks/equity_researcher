"""Step 8: choosing between the issuer's walk and the accounting identity."""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Final

from ...decomposition.identities import identity_for
from ...decomposition.metrics import decompose_path_b
from ...decomposition.metrics.common import to_change_unit
from ...decomposition.taxonomy import display_label
from ...decomposition.walks import Walk, WalkStep, choose_path_a
from ...domain.calendar import fiscal_period
from ...domain.models import Decomposition, DriverContribution, RequestSpec, ResolvedInputs
from ...domain.registry import component_name, get_registry, other_driver_id
from ...domain.units import Unit, value_tolerance, walk_tolerance

__all__ = ["run_decomposition"]

_DOUBLE = Decimal(2)


def run_decomposition(
    spec: RequestSpec,
    inputs: ResolvedInputs,
    walks: list[Walk],
    walk_evidence: dict[str, list[str]] | None = None,
) -> tuple[Decomposition, Walk | None]:
    """Decompose the reported movement, and say which path produced it."""
    registry = get_registry()
    metric = registry.metrics[spec.metric_id]
    profile = registry.bank(spec.bank_id)
    change_unit = "bps" if metric.change_unit == "bps" else "aud_m"

    reported = inputs.pair(spec.metric_id)
    if reported is None:
        return (
            Decomposition(
                metric_id=spec.metric_id,
                change_unit=change_unit,  # type: ignore[arg-type]
                headline_change=Decimal(0),
                path="none",
                missing_inputs=[spec.metric_id],
                notes=[
                    f"{spec.metric_id} was not established for both {spec.period} and "
                    f"{spec.comparison_period}, so there is no movement to decompose."
                ],
            ),
            None,
        )

    reported_prior, reported_current = reported
    current_period = fiscal_period(spec.period, profile.fy_end_month)
    comparison_period = fiscal_period(spec.comparison_period, profile.fy_end_month)
    days, pcp_days = current_period.days, comparison_period.days

    # A walk's two ends are the published figures, rounded as printed, and its bars owe
    # nothing to that rounding: CBA prints its CET1 ratio to one decimal place.
    reported_unit: Unit = metric.unit  # type: ignore[assignment]
    endpoint_rounding = sum(
        (value_tolerance(reported_unit, reported=value) for value in reported), Decimal(0)
    )

    # A walk whose components aren't on the ratio's own basis isn't a weaker explanation,
    # it's a different quantity. Westpac publishes its expense-to-income ratio excluding
    # Notable Items and its operating income including them, and reconciling one against the
    # other reports a 94bp error in figures that are each exactly as printed. The identity
    # is declined and the difference is stated instead.
    tolerance = walk_tolerance(change_unit, 1)  # type: ignore[arg-type]
    identity = identity_for(spec.metric_id, pre_tax=spec.pre_tax)
    if identity is not None and not inputs.comparable(spec.metric_id, *identity.components):
        # Path B is refused, but Path A isn't. The issuer's walk is read off their own chart
        # and uses none of the identity's components, so whether those are on comparable
        # bases doesn't matter to it, and for NIM, Path A is the main path.
        #
        # This used to return here, so one basis mismatch in Path B quietly turned off the
        # better evidence too. Westpac publishes its margin excluding Notable Items and its
        # net interest income including them, which rightly refuses the identity, and then
        # refused the margin walk as well, for a reason that doesn't apply to it.
        refusal = (
            f"The accounting identity wasn't run as a cross-check: {spec.bank_id} publishes "
            f"its {component_name(spec.metric_id)} and the parts of its identity on "
            "different bases, so the identity would reconcile to neither."
        )
        only_walk = choose_path_a(
            walks,
            period=spec.period,
            pcp=spec.comparison_period,
            tolerance=tolerance,
            endpoint_rounding=endpoint_rounding,
        )
        if only_walk is not None:
            decomposition = _from_walk(
                only_walk,
                spec,
                to_change_unit(
                    reported_current - reported_prior,
                    change_unit,  # type: ignore[arg-type]
                ),
                walk_evidence or {},
                reported_prior,
                reported_current,
                endpoint_rounding,
            )
            decomposition.notes.append(f"{refusal} The issuer's own walk stands alone.")
            return decomposition, only_walk
        return (
            Decomposition(
                metric_id=spec.metric_id,
                change_unit=change_unit,  # type: ignore[arg-type]
                headline_change=to_change_unit(
                    reported_current - reported_prior,
                    change_unit,  # type: ignore[arg-type]
                ),
                path="none",
                notes=[
                    f"{refusal} No published walk was found either, so the movement "
                    "stands on the issuer's own figures with no attribution."
                ],
            ),
            None,
        )

    path_b = decompose_path_b(
        spec.metric_id,
        prior=dict(inputs.prior),
        current=dict(inputs.current),
        reported_prior=reported_prior,
        reported_current=reported_current,
        days=days,
        pcp_days=pcp_days,
        year_days=current_period.year_days,
        pcp_year_days=comparison_period.year_days,
        identity=identity.metric_id if identity is not None else None,
    )
    if profile.discovered and profile.row_labels:
        _in_the_banks_words(path_b, dict((c, label) for label, c in profile.row_labels))

    walk = choose_path_a(
        walks,
        period=spec.period,
        pcp=spec.comparison_period,
        tolerance=tolerance,
        endpoint_rounding=endpoint_rounding,
    )
    path_a = (
        _from_walk(
            walk,
            spec,
            path_b.headline_change,
            walk_evidence or {},
            reported_prior,
            reported_current,
            endpoint_rounding,
        )
        if walk is not None
        else None
    )

    chosen = _choose(metric.primary_path, path_a, path_b)
    if path_a is not None and path_b.drivers:
        chosen.disagreements.extend(_disagreements(path_a, path_b))
        if chosen is path_a:
            chosen.path_b_residual = path_b.residual
            # The figure itself, not a promise of it: the note used to say what the identity
            # left unexplained was "reported alongside", and nothing reported it.
            within = abs(path_b.residual) <= path_b.tolerance
            chosen.notes.append(
                "The accounting identity ran as a cross-check: its own split leaves "
                f"{_amount(path_b.residual, path_b.change_unit)} of the movement unexplained, "
                f"{'within' if within else 'outside'} its tolerance of "
                f"{_amount(path_b.tolerance, path_b.change_unit)}."
            )
    return chosen, (walk if chosen is path_a else None)


def _amount(value: Decimal, change_unit: str) -> str:
    """How much of a movement, as a reader says it: 0.4bps, A$12.5m. A size, so no sign."""
    size = abs(value).quantize(Decimal("0.1"))
    return f"{size}bps" if change_unit == "bps" else f"A${size}m"


def _in_the_banks_words(decomposition: Decomposition, names: dict[str, str]) -> None:
    """A found bank's lines named as it names them. A model matched BNK's "Goodwill
    Impairment" to the bridge's one-off items, and the memo then called its A$3.5m the
    "notable item" it is in the taxonomy rather than the impairment it is."""
    for index, driver in enumerate(decomposition.drivers):
        components = driver.from_components or ()
        if len(components) == 1 and components[0] in names:
            decomposition.drivers[index] = driver.model_copy(update={"label": names[components[0]]})


def _choose(primary: str, path_a: Decomposition | None, path_b: Decomposition) -> Decomposition:
    if primary == "A" and path_a is not None:
        return path_a
    if primary == "A_else_B" and path_a is not None:
        return path_a
    if primary == "B" and path_b.drivers:
        return path_b
    # A metric whose identity could not run still has the issuer's walk, and a
    # metric with no walk still has its identity. Prefer whichever produced drivers.
    if path_b.drivers:
        return path_b
    if path_a is not None:
        return path_a
    return path_b


#: A bar labelled with an abbreviation: Westpac's "WSF" and "T&M", spelled out only in the
#: chart's own footnotes.
_ABBREVIATION = re.compile(r"[A-Z][A-Z0-9&/.]{1,5}")


#: Words a shared label can't end on and still name something.
_DANGLING: Final = frozenset(
    {"a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"}
)


def _shared_label(labels: list[str]) -> str:
    """One name for a driver a chained walk names in both halves.

    The words the labels start with in common, when they still name something: ANZ's
    "Capital floor" and "Capital floor adjustment" are the capital floor, and "Cash profit
    (Level 2)" is shared by both halves' cash profit lines. Two labels that share less than
    that ("Payment of the 2025 interim dividend" and "... final dividend") leave it to the
    taxonomy's own name. They used to be joined with "and", which read as two drivers.
    """
    if len(labels) <= 1:
        return labels[0] if labels else ""
    words = [label.split() for label in labels]
    common: list[str] = []
    for column in zip(*words, strict=False):
        if len({word.lower() for word in column}) != 1:
            break
        common.append(column[0])
    while common and (
        common[-1].lower().strip(",:;-") in _DANGLING
        or re.fullmatch(r"\d+(?:\.\d+)?", common[-1]) is not None
        or " ".join(common).count("(") != " ".join(common).count(")")
    ):
        common.pop()
    return " ".join(common).rstrip(",:;") if len(common) >= 2 else ""


def _spelled_out(label: str, metric_id: str, driver_id: str) -> str:
    """The bank's own label, with what it stands for when it's an abbreviation a reader
    might not know: "WSF (Wholesale and term funding costs)"."""
    if _ABBREVIATION.fullmatch(label.strip()):
        return f"{label.strip()} ({display_label(metric_id, driver_id)})"
    return label


def _from_walk(
    walk: Walk,
    spec: RequestSpec,
    headline_change: Decimal,
    walk_evidence: dict[str, list[str]],
    reported_prior: Decimal,
    reported_current: Decimal,
    endpoint_rounding: Decimal = Decimal(0),
) -> Decomposition:
    """Drivers from the issuer's own walk, measured against the reported movement."""
    evidence_type = (
        "disclosed_quantified_chained" if walk.method == "chained_walk" else "disclosed_quantified"
    )
    drivers: list[DriverContribution] = []
    unmapped_total = Decimal(0)
    unmapped_labels: list[str] = []

    # Bars that map to the same driver are one driver. ANZ draws "Assets pricing" and a
    # competition bar that both land on lending pricing; as two entries with one id, the
    # second overwrote the first when confidence was keyed by driver.
    grouped: dict[str, list[WalkStep]] = {}
    for step in walk.steps:
        if step.driver_id is None:
            unmapped_total += step.value
            unmapped_labels.append(step.label)
            continue
        grouped.setdefault(step.driver_id, []).append(step)

    for driver_id, steps in grouped.items():
        value = sum((step.value for step in steps), Decimal(0))
        # Once each: a chained walk names the same step in both halves.
        labels = list(dict.fromkeys(step.label for step in steps if step.label))
        shared = _shared_label(labels)
        drivers.append(
            DriverContribution(
                driver_id=driver_id,
                label=_spelled_out(shared, spec.metric_id, driver_id)
                if shared
                else display_label(spec.metric_id, driver_id),
                contribution=value,
                direction="+" if value > 0 else "-" if value < 0 else "0",
                method=walk.method,
                evidence_type=evidence_type,  # type: ignore[arg-type]
                evidence_ids=walk_evidence.get(driver_id, []),
            )
        )

    if unmapped_labels:
        # Aggregated rather than dropped: the issuer disclosed a real movement that
        # the taxonomy has no name for, and losing it would overstate the residual.
        # Named by what it holds, because "Other disclosed movements" told a reader
        # nothing that "Timing difference" doesn't.
        other_id = other_driver_id(spec.metric_id)
        drivers.append(
            DriverContribution(
                driver_id=other_id,
                label=(
                    unmapped_labels[0] if len(unmapped_labels) == 1 else "Other disclosed movements"
                ),
                contribution=unmapped_total,
                direction="+" if unmapped_total > 0 else "-" if unmapped_total < 0 else "0",
                method=walk.method,
                evidence_type=evidence_type,  # type: ignore[arg-type]
                evidence_ids=walk_evidence.get(other_id, []),
                note="Disclosed rows with no canonical driver: " + "; ".join(unmapped_labels),
            )
        )

    explained = sum((d.contribution for d in drivers if d.contribution is not None), Decimal(0))
    # The bars' rounding and the published figures' own, as Path B allows: CBA prints its
    # CET1 ratio to one decimal place, and its chained walk, 5bps from a movement of nil,
    # was reported as unexplained against an allowance for its bars alone.
    tolerance = walk_tolerance(walk.unit, max(len(walk.steps), 1)) + endpoint_rounding
    notes: list[str] = []
    if walk.method == "chained_walk":
        notes.append(
            f"No walk spanned {spec.comparison_period} to {spec.period}; two "
            "half-on-half walks were chained, and each driver cites the half or halves it "
            "appears in."
        )
    if abs(walk.declared_change - headline_change) > tolerance:
        notes.append(
            f"The walk declares a movement of {walk.declared_change}{walk.unit} while the "
            f"reported figures moved {headline_change}{walk.unit}; the reported movement is "
            "the target and the difference is in the residual."
        )

    return Decomposition(
        metric_id=spec.metric_id,
        change_unit="bps" if walk.unit in {"bps", "pct", "ratio"} else "aud_m",
        headline_change=headline_change,
        drivers=drivers,
        residual=headline_change - explained,
        tolerance=tolerance,
        path="A",
        path_a_residual=walk.checksum_residual(),
        notes=notes,
    )


def _disagreements(path_a: Decomposition, path_b: Decomposition) -> list[str]:
    """Drivers the two paths both quantify and differ on by more than 2x tolerance."""
    by_id_b = {
        driver.driver_id: driver.contribution
        for driver in path_b.drivers
        if driver.contribution is not None and driver.parent_id is None
    }
    found: list[str] = []
    limit = _DOUBLE * max(path_a.tolerance, path_b.tolerance)
    for driver in path_a.drivers:
        other = by_id_b.get(driver.driver_id)
        if other is None or driver.contribution is None:
            continue
        if abs(driver.contribution - other) > limit:
            found.append(driver.driver_id)
    return found
