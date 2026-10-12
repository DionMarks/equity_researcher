"""Renders ``result.md`` from ``result.json``, by code rather than by a model."""

from __future__ import annotations

from decimal import Decimal

from ..domain.money import aud, aud_amounts
from ..domain.registry import get_registry
from .schema import Driver, ResearchResult, describe_basis, request_names

__all__ = [
    "amount_places",
    "format_change",
    "format_magnitude",
    "format_share",
    "format_value",
    "model_calls_said",
    "render_markdown",
    "result_places",
]

_BPS_DISPLAY = Decimal("0.1")
_AUD_DISPLAY = Decimal("1")


def model_calls_said(result: ResearchResult, noun: str = "model call") -> str:
    """How many model calls the answer made, counted the same way everywhere: every call in
    its trail, naming those the cache answered for nothing. A memo said "4 model calls" in one
    place and "2 model calls" in another, because one count left the replayed calls out."""
    calls = result.trail.model_calls
    total = len(calls) if calls else result.diagnostics.llm_calls
    replayed = sum(1 for call in calls if call.cached)
    said = f"1 {noun}" if total == 1 else f"{total} {noun}s"
    return said + (f" ({replayed} replayed from the cache)" if replayed else "")


def format_value(value: Decimal | None, unit: str) -> str:
    """A reported figure, at the precision the issuer published it."""
    if value is None:
        return "n/a"
    if unit == "pct":
        return f"{value}%"
    if unit == "bps":
        return f"{value} bps"
    # The sign goes before the currency, as in a movement: "-A$10.5m", not "A$-10.5m".
    # Bendigo's charge was a net reversal in both halves, which is when it shows.
    sign = "-" if value < 0 else ""
    if unit == "aud_m":
        return f"{sign}A${abs(value):,}m"
    if unit == "aud_bn":
        return f"{sign}A${abs(value):,}bn"
    return str(value)


def amount_places(*published: Decimal | None) -> int:
    """How many decimal places a movement in A$m is shown with: as many as the bank printed
    the figures it runs between, and no more than three, which is a bank printing thousands
    (BNK's loss of A$3,706 thousand is A$3.706m, and its change of A$4.989m showed as A$4.99m).

    The majors print whole millions. Judo prints its profit to one decimal place, and a rise
    from A$125.6m to A$168.1m shown as "+A$42m" was taken out of a memo by the checker, as
    a figure nothing it was given said.
    """
    places = [
        -int(value.as_tuple().exponent)
        for value in published
        if value is not None and int(value.as_tuple().exponent) < 0
    ]
    return min(max(places, default=0), 3)


def format_change(value: Decimal | None, change_unit: str, places: int = 0) -> str:
    """A movement, with an explicit sign and display-rounded.

    Nothing has no sign. "+0.0 bps" next to ANZ's wholesale funding read as a small rise
    when the bank's own words were "largely flat", and a residual that rounds to zero is
    no more positive than negative. ``places`` is for an amount: see :func:`amount_places`.
    """
    if value is None:
        return "n/a"
    if change_unit == "bps":
        shown = value.quantize(_BPS_DISPLAY)
        return "0.0 bps" if shown == 0 else f"{shown:+} bps"
    shown = value.quantize(_AUD_DISPLAY.scaleb(-places))
    if shown == 0:
        return "A$0m"
    return f"{'+' if shown > 0 else '-'}A${abs(shown):,}m"


def format_magnitude(value: Decimal | None, change_unit: str, places: int = 0) -> str:
    """A tolerance or other size, which is a magnitude rather than a movement."""
    if value is None:
        return "n/a"
    if change_unit == "bps":
        return f"{abs(value).quantize(_BPS_DISPLAY)} bps"
    return f"A${abs(value).quantize(_AUD_DISPLAY.scaleb(-places)):,}m"


def format_share(share: Decimal | None) -> str:
    """A driver's share of the movement, or n/a where it's withheld."""
    if share is None:
        return "n/a"
    # A share quantised from -0.01% is Decimal("-0.0"), which keeps its sign and reads as
    # a driver that pushed the other way when it moved nothing.
    return f"{abs(share) if share == 0 else share}%"


def result_places(result: ResearchResult) -> int:
    """The decimal places this result's amounts are shown with: see :func:`amount_places`."""
    return amount_places(
        result.prior.value if result.prior else None,
        result.current.value if result.current else None,
    )


def _confidence_cell(driver: Driver) -> str:
    return f"{driver.confidence.level} ({driver.confidence.score:.2f})"


# A computed driver legitimately cites every figure its arithmetic touched, which
# for a ratio effect means both sides of the ratio in both periods. Showing all of
# them makes the table unreadable, so the cell shows the first few and the full list
# stays in result.json, where an auditor reads it.
_MAX_CITATIONS_SHOWN = 3


def _evidence_cell(result: ResearchResult, ids: list[str]) -> str:
    if not ids:
        return "n/a"
    cards = result.evidence_by_id()
    parts: list[str] = []
    for evidence_id in ids[:_MAX_CITATIONS_SHOWN]:
        card = cards.get(evidence_id)
        if card is None:
            parts.append(f"[{evidence_id}]")
            continue
        parts.append(f"[{evidence_id}] p.{card.source.page}")
    remaining = len(ids) - _MAX_CITATIONS_SHOWN
    if remaining > 0:
        parts.append(f"+{remaining} more in result.json")
    return "; ".join(parts)


def render_markdown(result: ResearchResult) -> str:
    """The analyst-facing memo for one run."""
    registry = get_registry()
    metric = registry.metrics[result.request.metric_id]
    bank_name, metric_name = request_names(result)

    lines: list[str] = [
        f"# {bank_name} · {metric_name} · {result.request.period}"
        + (
            ""
            if result.request.asked_as is not None
            else f" vs {result.request.comparison_period} (same period last financial year)"
        ),
        "",
    ]

    basis = (
        result.current.label_at_bank
        if result.current
        else (result.request.headline_label or "(not established)")
    )
    qualified = describe_basis(result.current.basis) if result.current else ""
    if qualified:
        basis = f"{basis} ({qualified})"
    confidence = result.overall_confidence
    lines.append(
        f"Basis: {basis} · As of {result.as_of.isoformat()} · Status: {result.status}"
        + (
            f" · Overall confidence: {confidence.level} ({confidence.score:.2f})"
            if confidence
            else ""
        )
    )
    lines.append("")

    if result.status == "ABSTAIN":
        lines += _render_abstention(result)
    else:
        lines += _render_movement(result, metric)
        lines += _render_drivers(result)
        lines += _render_causes(result)
        lines += _render_narrative(result)
        lines += _render_commentary(result)

    lines += _render_caveats(result)
    lines += _render_sources(result)
    lines += _render_trail(result)
    lines += _render_diagnostics(result)
    return "\n".join(lines).rstrip() + "\n"


def _render_abstention(result: ResearchResult) -> list[str]:
    return [
        "## No answer given",
        "",
        f"**{result.abstain_reason}**",
        "",
        *([caveat for caveat in result.caveats] or ["No further detail was recorded."]),
        "",
    ]


def _render_movement(result: ResearchResult, metric) -> list[str]:  # type: ignore[no-untyped-def]
    request = result.request
    current = format_value(result.current.value if result.current else None, metric.unit)
    prior = format_value(result.prior.value if result.prior else None, metric.unit)
    change = (
        format_change(result.change.absolute, result.change.change_unit, result_places(result))
        if result.change
        else "n/a"
    )
    if result.change and result.change.pct is not None:
        change += f" ({result.change.pct:+}%)"

    rows = [
        "| Metric | " + request.comparison_period + " | " + request.period + " | Change |",
        "| --- | ---: | ---: | ---: |",
        f"| {result.current.label_at_bank if result.current else metric.display_name} "
        f"| {prior} | {current} | {change} |",
        "",
    ]
    if result.change and result.change.pct_suppressed_reason:
        rows += [
            f"_Percentage change not shown: {result.change.pct_suppressed_reason}._",
            "",
        ]
    if result.prior and result.prior.restated:
        rows += [
            f"_The {request.comparison_period} figure is the restated comparative"
            + (
                "; as originally reported it was "
                f"{format_value(result.prior.original_value, metric.unit)}._"
                if result.prior.original_value is not None
                else "._"
            ),
            "",
        ]
    return rows


def _render_drivers(result: ResearchResult) -> list[str]:
    if not result.drivers and result.residual is None:
        return ["## Drivers", "", "No drivers were established from the evidence.", ""]

    lines = [
        "## Drivers",
        "",
        "| Driver | Contribution | Share | Confidence | Method | Evidence |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    unit = result.reconciliation.headline_change if result.reconciliation else None
    change_unit = result.change.change_unit if result.change else "bps"

    for driver in result.top_level_drivers:
        lines.append(
            f"| {driver.label or driver.driver_id} "
            f"| {format_change(driver.contribution, change_unit, result_places(result))} "
            f"| {format_share(driver.share_of_change) if result.shares_mean_something else 'n/m'} "
            f"| {_confidence_cell(driver)} "
            f"| {driver.method} "
            f"| {_evidence_cell(result, driver.evidence_ids)} |"
        )
        for child in result.children_of(driver.driver_id):
            lines.append(
                f"| ⤷ {child.label or child.driver_id} "
                f"| {format_change(child.contribution, change_unit, result_places(result))} "
                f"| n/a "
                f"| {_confidence_cell(child)} "
                f"| {child.method} "
                f"| {_evidence_cell(result, child.evidence_ids)} |"
            )

    if result.residual is not None:
        lines.append(
            f"| Residual | {format_change(result.residual.contribution, change_unit, result_places(result))} "
            f"| n/a | n/a | n/a | n/a |"
        )
    lines.append("")
    if not result.shares_mean_something and len(result.top_level_drivers) > 1:
        lines += [
            "_Shares are n/m (not meaningful): the drivers pull in opposite directions, so "
            "each one's share of the net movement would run past 100% or below zero._",
            "",
        ]

    reconciliation = result.reconciliation
    if reconciliation is not None:
        verdict = (
            "within tolerance"
            if reconciliation.within_tolerance
            else "**beyond tolerance, and reported as unexplained**"
        )
        lines += [
            "### Reconciliation",
            "",
            f"- Movement to explain: **{format_change(reconciliation.headline_change, change_unit, result_places(result))}**",
            f"- Explained by drivers: **{format_change(reconciliation.sum_of_drivers, change_unit, result_places(result))}**",
            f"- Residual: **{format_change(reconciliation.residual, change_unit, result_places(result))}** "
            f"against a tolerance of "
            f"{format_magnitude(reconciliation.tolerance, change_unit, result_places(result))}: {verdict}",
            "",
        ]
    _ = unit
    return lines


def _render_narrative(result: ResearchResult) -> list[str]:
    if not result.narrative:
        return []
    return ["## What moved", "", aud_amounts(result.narrative), ""]


def _render_causes(result: ResearchResult) -> list[str]:
    """Each driver's cause as the bank states it, checked against the bank's own words."""
    explained = [driver for driver in result.drivers if driver.cause]
    if not explained:
        return []
    change_unit = result.change.change_unit if result.change else "bps"
    lines = ["### Why, as the bank explains it", ""]
    for driver in explained:
        lines.append(
            f"- **{driver.label or driver.driver_id}** "
            f"({format_change(driver.contribution, change_unit, result_places(result))}): "
            f"{aud_amounts(driver.cause)}"
        )
    lines.append("")
    return lines


#: How many of the issuer's sentences the memo shows when there's no narrative to use them.
_COMMENTARY_SHOWN = 6


def _render_commentary(result: ResearchResult) -> list[str]:
    """The issuer's own explanation, shown as it stands when no model wrote a narrative.

    Found by code, so it needs no model: the sentences where the bank explains this
    movement against the same period a year earlier. Without a model nothing matches them
    to drivers, and the memo says so rather than implying it did.
    """
    if result.narrative or not result.issuer_commentary:
        return []
    cards = result.evidence_by_id()
    shown = [cards[i] for i in result.issuer_commentary if i in cards][:_COMMENTARY_SHOWN]
    if not shown:
        return []
    lines = [
        "## In the bank's words",
        "",
        "The bank's own sentences on this movement against the same period a year earlier, "
        "found in its results documents. They are quoted, not matched to drivers.",
        "",
    ]
    for card in shown:
        lines.append(f'- "{card.passage or card.quote}" [{card.evidence_id}] p.{card.source.page}')
    lines.append("")
    return lines


def _render_caveats(result: ResearchResult) -> list[str]:
    if not (result.caveats or result.data_conflicts or result.alternative_bases):
        return []
    lines = ["## Caveats, data conflicts, alternative bases", ""]
    if result.status != "ABSTAIN":
        for caveat in result.caveats:
            lines.append(f"- {caveat}")
    for conflict in result.data_conflicts:
        lines.append(
            f"- **{conflict.classification}** on `{conflict.component}`: "
            + " vs ".join(
                f"{candidate.value}{candidate.unit} ({candidate.source.doc_id} "
                f"p.{candidate.source.page})"
                for candidate in conflict.candidates
            )
            + f". {conflict.rule_applied}."
            + (f" {conflict.impact}." if conflict.impact else "")
        )
    for alternative in result.alternative_bases:
        lines.append(
            f"- Alternative basis, {alternative.label}: "
            f"{format_value(alternative.prior, alternative.unit)} to "
            f"{format_value(alternative.current, alternative.unit)}"
            + (f" ({alternative.note})" if alternative.note else "")
        )
    lines.append("")
    return lines


def _render_sources(result: ResearchResult) -> list[str]:
    cited = result.cited_evidence_ids()
    cards = [card for card in result.evidence if card.evidence_id in cited]
    if not cards:
        return []
    lines = ["## Sources", ""]
    for card in sorted(cards, key=lambda c: (c.source.doc_id, c.source.page, c.evidence_id)):
        table = f", table {card.source.table_id}" if card.source.table_id else ""
        quote = f', "{card.quote}"' if card.quote else ""
        # Named by the document it was read from. A comparative read from this period's
        # pack used to be listed as "CBA 1H25 RESULTS_DOCUMENT ... table
        # cba-1h26-results-document_p16_L0", naming a document nobody opened.
        what = (
            "the bank's commentary"
            if card.value is None and card.passage
            else f"{card.period} {'step' if '.walk.' in card.component else 'figure'}"
        )
        lines.append(
            f"- [{card.evidence_id}] {card.source.doc_id} p.{card.source.page}{table} "
            f"(published {card.source.publish_date.isoformat()}): {what}{quote}"
        )
    lines.append("")
    return lines


def _render_trail(result: ResearchResult) -> list[str]:
    """How the answer was reached, beyond the sources it cites: each model call in order, and
    each filing found and read during the run."""
    from .progress import PURPOSES

    lines: list[str] = []
    calls = result.trail.model_calls
    if calls:
        spent = sum((call.cost_usd or Decimal(0) for call in calls), Decimal(0))
        lines += [
            "## How it was answered",
            "",
            f"{model_calls_said(result)}, {aud(spent)} in all.",
            "",
            "| # | What it was for | Model | Tokens in | Tokens out | Cost |",
            "| ---: | --- | --- | ---: | ---: | ---: |",
        ]
        for number, call in enumerate(calls, start=1):
            purpose = PURPOSES.get(call.step, call.step.replace("_", " "))
            tools = [*call.tool_calls] + (["web search"] * call.web_searches)
            if tools:
                purpose += f" (used: {', '.join(dict.fromkeys(tools))})"
            cost = (
                "replayed, A$0"
                if call.cached
                else aud(call.cost_usd)
                if call.cost_usd is not None
                else "unpriced"
            )
            lines.append(
                f"| {number} | {purpose} | {call.model} | {call.input_tokens:,} | "
                f"{call.output_tokens:,} | {cost} |"
            )
        lines.append("")
    if result.trail.fetched:
        if not lines:
            lines += ["## How it was answered", ""]
        lines += ["Found and read during the run:", ""]
        lines += [f"- {note}" for note in result.trail.fetched]
        lines.append("")
    return lines


def _format_latency(milliseconds: int) -> str:
    if milliseconds >= 1000:
        return f"{milliseconds / 1000:.1f}s"
    return f"{milliseconds}ms"


def _render_diagnostics(result: ResearchResult) -> list[str]:
    diagnostics = result.diagnostics
    parts = [
        f"run `{result.run_id}`",
        f"config `{diagnostics.config_hash or 'unrecorded'}`",
        model_calls_said(result),
        aud(diagnostics.cost_usd),
        _format_latency(diagnostics.latency_ms),
    ]
    lines = ["---", "", "_" + " · ".join(parts) + "._", ""]
    if diagnostics.warnings:
        lines += ["", "**Warnings**", ""]
        lines += [f"- {warning}" for warning in diagnostics.warnings]
    return lines
