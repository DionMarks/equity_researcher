"""Reads a walk out of a chart, and rejects the reading if it doesn't balance."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from ..domain.models import SourceRef
from ..domain.units import AUD_UNITS, RATIO_UNITS, Unit, to_aud_m, to_bps, value_tolerance
from ..llm.budget import Budget
from ..llm.client import LLMClient, LLMRequest, LLMUnavailable, ModelUnreachable
from ..llm.json_schema import strict_subset
from ..llm.tracing import PromptLibrary, Tracer, default_library

__all__ = [
    "ChartReading",
    "checksum",
    "read_walk_chart",
    "walk_schema",
]

# A step the issuer drew but did not label. Kept as a step with no value.
UNLABELLED = None

#: The units a chart can be printed in. Anything else is a reading that can't be checked.
CHART_UNITS: tuple[Unit, ...] = ("bps", "pct", "aud_m", "aud_bn")


def walk_schema() -> dict[str, Any]:
    """The structured output the reader must return.

    ``additionalProperties: false`` throughout, so a field the reader invents is rejected
    rather than ignored. An ignored field is how a reading picks up a number nobody asked
    for.

    Put through ``strict_subset`` for the same reason as the tool schemas: structured
    outputs only accept a subset of JSON Schema, and the ``minLength`` below made every
    chart call fail with a 400. The bound moves into the description, and empty labels are
    still dropped later when the reading is parsed.

    The endpoints and the bars get a unit each. There used to be one ``unit`` field, and
    the first Opus 5 reading of Westpac's margin walk filled it with ``pct`` because the
    walk runs from 1.92% to 1.89%, while every bar on it is printed in basis points. The
    bars were right and summed to the published -3bps exactly, but read as percentages
    they missed by 297bps and the whole reading was quarantined.
    """
    return strict_subset(
        {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "chart_matches",
                "reads_as",
                "opening",
                "closing",
                "endpoint_unit",
                "step_unit",
                "steps",
            ],
            "properties": {
                # Asked before the steps, so the reader commits to what the page is
                # before it starts reading bars off it.
                "chart_matches": {"type": "boolean"},
                "reads_as": {"type": "string"},
                "opening": {"type": ["number", "null"]},
                "closing": {"type": ["number", "null"]},
                "endpoint_unit": {
                    "type": "string",
                    "enum": list(CHART_UNITS),
                    "description": "The unit the opening and closing values are printed in. "
                    "1.92% is pct.",
                },
                "step_unit": {
                    "type": "string",
                    "enum": list(CHART_UNITS),
                    "description": "The unit the bars' own labels are printed in. A margin "
                    "or capital walk between two percentages usually labels its bars in "
                    "basis points, so this is often not the endpoints' unit.",
                },
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["label", "value"],
                        "properties": {
                            "label": {"type": "string", "minLength": 1},
                            "value": {"type": ["number", "null"]},
                        },
                    },
                },
            },
        }
    )


@dataclass
class ChartReading:
    """One chart, read and checked."""

    steps: list[tuple[str, Decimal | None]] = field(default_factory=list)
    unit: Unit | None = None
    """The unit the bars are printed in."""
    endpoint_unit: Unit | None = None
    """The unit the chart prints its opening and closing values in."""
    opening_read: Decimal | None = None
    closing_read: Decimal | None = None
    checksum_passed: bool = False
    residual: Decimal | None = None
    tolerance: Decimal | None = None
    source: SourceRef | None = None
    note: str = ""

    @property
    def usable(self) -> bool:
        """Whether the reading can be used for attribution.

        A reading that didn't balance is kept for diagnostics and never used. There's no
        partial credit, because the steps only mean something together.
        """
        return self.checksum_passed and bool(self.labelled)

    @property
    def labelled(self) -> list[tuple[str, Decimal]]:
        return [(label, value) for label, value in self.steps if value is not None]


def checksum(
    steps: list[tuple[str, Decimal | None]],
    reported_opening: Decimal,
    reported_closing: Decimal,
    unit: Unit,
    step_unit: Unit | None = None,
) -> tuple[bool, Decimal, Decimal]:
    """Whether the steps bridge the issuer's own published endpoints.

    The endpoints are the ones in the fact store, read from the text of the filing, so this
    checks against independent evidence rather than the chart's own arithmetic. The
    tolerance is the rounding the published figures carry, plus one step's worth per
    labelled bar, which is the precision the bars are printed to.

    ``step_unit`` is the unit the chart is drawn in, which often isn't the unit the metric
    is published in: ANZ states its margin as 1.53% and draws the walk in basis points.
    Subtracting one from the other without converting made every real walk miss by about a
    hundred times (the quarantine note read "miss the published movement by -61903.00 pct"),
    so every chart was thrown away and the vision path produced nothing while looking like
    it worked.
    """
    step_unit = step_unit or unit
    movement = reported_closing - reported_opening
    raw = sum((value for _, value in steps if value is not None), Decimal(0))

    # Both sides go into one unit, the one ``value_tolerance`` uses: basis points for a
    # ratio, millions for an amount. Comparing a percentage residual with a basis-point
    # tolerance lets everything through by a factor of a hundred, which is the other half of
    # the same mistake.
    if unit in RATIO_UNITS and step_unit in RATIO_UNITS:
        residual = to_bps(movement, unit) - to_bps(raw, step_unit)
    elif unit in AUD_UNITS and step_unit in AUD_UNITS:
        residual = to_aud_m(movement, unit) - to_aud_m(raw, step_unit)
    else:
        # A margin walk drawn in dollars isn't a conversion problem, it's a misreading. It's
        # rejected rather than reconciled, and flagged with a NaN residual so the note says
        # that instead of claiming a zero miss against a zero tolerance, which is what
        # returning zeroes made it print.
        return False, Decimal("NaN"), Decimal(0)

    per_figure = value_tolerance(unit, reported=reported_closing)
    bars = sum(1 for _, value in steps if value is not None)
    tolerance = per_figure * Decimal(2 + bars)
    return abs(residual) <= tolerance, residual, tolerance


def _periods(opening: str, closing: str, ends: tuple[date, date] | None) -> str:
    """The two periods as the reader is told them. A leg of a chained walk gives its end
    dates too: "FY25 to 1H26" is June to December 2025 for CBA, and the chart says Jun-25."""
    if ends is None:
        return f"{opening} to {closing}"
    return f"{opening} (ending {ends[0]:%d %B %Y}) to {closing} (ending {ends[1]:%d %B %Y})"


def read_walk_chart(
    *,
    page_image: bytes,
    media_type: str,
    source: SourceRef,
    reported_opening: Decimal,
    reported_closing: Decimal,
    unit: Unit,
    client: LLMClient,
    budget: Budget | None = None,
    metric_id: str,
    opening_period: str = "",
    closing_period: str = "",
    period_ends: tuple[date, date] | None = None,
    # Overridden from config/models.yaml through ``Registry.route("chart_read")``. A default
    # that differs from the routing is how that config ended up unused for the whole build.
    model: str = "claude-opus-5",
    effort: str = "medium",
    tracer: Tracer | None = None,
    prompts: PromptLibrary | None = None,
) -> ChartReading:
    """Read one walk chart, then refuse it unless it balances."""
    reading = ChartReading(source=source)
    if not client.available:
        reading.note = "no model configured: a charted walk needs vision"
        return reading

    library = prompts or default_library()
    prompt = library.get("chart_read")
    request = LLMRequest(
        step="chart_read",
        model=model,
        effort=effort,
        system=prompt.text,
        messages=(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": _b64(page_image),
                        },
                    },
                    # The endpoints are left out on purpose. If the reader is told what the
                    # steps must add up to, it can adjust a bar until they do, and the
                    # checksum then just confirms its own input.
                    # No unit is suggested. "Expected unit: pct" was in this message, and
                    # the reader echoed it back for bars printed in basis points.
                    {
                        "type": "text",
                        "text": (
                            f"Metric: {metric_id}. Periods: "
                            + _periods(opening_period, closing_period, period_ends)
                            + ".\n\n"
                            "Say first whether this page charts a walk of that metric "
                            "between those two periods. If it does, return the labelled "
                            "steps as printed, with their signs, and give the unit of the "
                            "endpoints and the unit of the bars separately; use null for a "
                            "bar with no printed value. If it charts something else, say "
                            "what."
                        ),
                    },
                ],
            },
        ),
        output_schema=walk_schema(),
    )

    # Counted like every other step. Chart reading used to be the one model call nothing
    # recorded, so a run that spent real money on it reported ``llm_calls: 0`` and zero
    # cost, and the per-run limit had nothing to check against.
    if budget is not None and not budget.may_call("chart_read"):
        reading.note = (
            f"chart reading skipped: {budget.warnings[-1] if budget.warnings else 'budget'}"
        )
        return reading
    try:
        response = client.complete(request)
    except LLMUnavailable as error:
        if isinstance(error, ModelUnreachable):  # the run stops; see llm.client
            raise
        reading.note = f"chart reading unavailable: {error}"
        return reading
    if budget is not None:
        budget.record("chart_read", request.model, response.usage, cached=response.cached)
    if tracer is not None:
        tracer.model_call(request, response, prompt=prompt)

    payload = response.structured
    if not isinstance(payload, dict):
        reading.note = "the chart reader returned no structured output"
        return reading

    reading.steps = [
        (str(step.get("label", "")).strip(), _decimal(step.get("value")))
        for step in payload.get("steps") or ()
        if str(step.get("label", "")).strip()
    ]
    if payload.get("chart_matches") is False:
        # It identifies the chart first, before the checksum. A page that turns out to be
        # the wrong chart costs one call; a confident reading of it costs the call and a
        # quarantine, and still doesn't say where the walk actually is.
        reading.note = (
            "not a walk of this metric on this page: the reader reads it as "
            f"{str(payload.get('reads_as') or 'something else')[:120]!r}"
        )
        return reading
    reading.unit = _unit(payload.get("step_unit"))
    reading.endpoint_unit = _unit(payload.get("endpoint_unit"))
    reading.opening_read = _decimal(payload.get("opening"))
    reading.closing_read = _decimal(payload.get("closing"))

    if not reading.steps:
        reading.note = "no labelled steps were found on the page"
        return reading
    if reading.unit is None:
        reading.note = (
            "quarantined: the reader gave no unit for the bars it read, so there is "
            "nothing to check them against"
        )
        return reading

    mismatch = endpoint_mismatch(
        reading.opening_read,
        reading.closing_read,
        reading.endpoint_unit,
        reported_opening,
        reported_closing,
        unit,
    )
    if mismatch:
        reading.note = f"quarantined: {mismatch}"
        return reading

    passed, residual, tolerance = checksum(
        reading.steps, reported_opening, reported_closing, unit, reading.unit
    )
    reading.checksum_passed = passed
    reading.residual = residual
    reading.tolerance = tolerance
    if not passed:
        if residual.is_nan():
            reading.note = (
                f"quarantined: the steps were read in {reading.unit}, which cannot measure "
                f"a movement in {unit}. That is a misreading of the chart rather than a "
                "near-miss, so there is nothing to reconcile"
            )
        else:
            residual_unit = "bps" if unit in RATIO_UNITS else "aud_m"
            reading.note = (
                f"quarantined: the steps (read in {reading.unit}) miss the published "
                f"movement by {residual} {residual_unit} against a tolerance of "
                f"{tolerance}. A chart reading is kept whole or not at all, because one "
                "wrong bar misattributes the whole movement"
            )
    return reading


def endpoint_mismatch(
    opening_read: Decimal | None,
    closing_read: Decimal | None,
    endpoint_unit: Unit | None,
    reported_opening: Decimal,
    reported_closing: Decimal,
    unit: Unit,
) -> str:
    """Why the chart's own endpoints aren't the published figures, or "" if they are.

    The checksum alone can't tell a walk of the right movement from a walk of a different
    basis that happens to move by about as much. On a three basis point movement the
    rounding allowance is wide enough for that to matter. A walk drawn from the reported
    opening to the reported closing is the issuer's walk of this figure; one drawn between
    other values is a walk of something else (another basis, a division, the other half)
    however well its bars add up.

    An endpoint the reader couldn't see is skipped rather than failed: an unlabelled end
    bar is a gap in the chart, not evidence against it.
    """
    if endpoint_unit is None:
        return ""
    ratio = unit in RATIO_UNITS and endpoint_unit in RATIO_UNITS
    amount = unit in AUD_UNITS and endpoint_unit in AUD_UNITS
    if not (ratio or amount):
        if opening_read is None and closing_read is None:
            return ""
        return (
            f"the chart's endpoints are printed in {endpoint_unit}, which can't be the "
            f"{unit} figures the issuer reports for this metric"
        )
    convert = to_bps if ratio else to_aud_m
    for which, read, reported in (
        ("opening", opening_read, reported_opening),
        ("closing", closing_read, reported_closing),
    ):
        if read is None:
            continue
        # The looser of the two precisions: a chart often prints fewer digits than the text.
        allowed = max(
            value_tolerance(endpoint_unit, reported=read),
            value_tolerance(unit, reported=reported),
        )
        gap = convert(read, endpoint_unit) - convert(reported, unit)
        if abs(gap) > allowed:
            return (
                f"the chart's {which} value of {read} {endpoint_unit} is not the published "
                f"{reported} {unit}, so it walks a different figure (another basis, a "
                "division or another period) however well its bars add up"
            )
    return ""


def _unit(value: Any) -> Unit | None:
    """A unit the reader returned, if it's one a chart can be printed in."""
    return value if value in CHART_UNITS else None


def _decimal(value: Any) -> Decimal | None:
    """A chart value as an exact Decimal, or ``None``.

    Converted through ``str`` because the reader returns JSON numbers: ``Decimal(0.1)``
    isn't 0.1, and a tenth of a basis point of float error in every bar adds up to a
    residual that looks like a real unexplained remainder.
    """
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _b64(data: bytes) -> str:
    import base64

    return base64.standard_b64encode(data).decode("ascii")
