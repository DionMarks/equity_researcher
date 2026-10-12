"""Walks the issuer wrote out in prose, read by a model and checked by code."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Final

from ..decomposition.taxonomy import map_label
from ..decomposition.walks import Walk, WalkStep
from ..domain.calendar import PeriodKind, fiscal_period
from ..domain.models import DocMeta, RequestSpec, SourceRef
from ..domain.registry import get_registry
from ..domain.units import RATIO_UNITS, Unit, to_aud_m, to_bps, value_tolerance, walk_tolerance
from ..llm.budget import Budget
from ..llm.client import LLMClient, LLMRequest, LLMUnavailable, ModelUnreachable
from ..llm.json_schema import strict_subset
from ..llm.tracing import PromptLibrary, Tracer, default_library
from ..retrieval.commentary import (
    _ends,
    _plain,
    _words,
    comparison_of,
    page_sentences,
    placed_sentences,
)
from .chart_walks import _METRIC_WORDS, _printed
from .parse_pdf import ParsedDocument, ParsedPage

__all__ = [
    "TextWalkOutcome",
    "WalkLeg",
    "check_text_walk",
    "half_year_legs",
    "rank_text_walk_pages",
    "read_text_walk",
    "text_walk_schema",
]

#: How much of each page the reader sees. A results page is 4,000 to 7,000 characters.
_PAGE_CHARS: Final = 7000
_MAX_PAGES: Final = 2
_UNITS: Final = ("bps", "pct", "aud_m", "aud_bn")
#: Where a walk is written out. A presentation's text layer is chart labels.
_DOC_TYPES: Final = ("RESULTS_DOCUMENT", "RESULTS_RELEASE")
_BASIS_POINTS = re.compile(r"\b(?:basis points?|bps?|bpts?)\b", re.IGNORECASE)

_DOWN: Final = (
    "decrease",
    "decreased",
    "decreases",
    "decreasing",
    "lower",
    "down",
    "narrower",
    "reduced",
    "reduction",
    "decline",
    "declined",
    "fell",
    "drag",
    "negative",
    "headwind",
    "tighter",
    "weaker",
    "less",
)
_UP: Final = (
    "increase",
    "increased",
    "increases",
    "increasing",
    "higher",
    "up",
    "wider",
    "improved",
    "improvement",
    "benefit",
    "rose",
    "positive",
    "tailwind",
    "added",
    "favourable",
)
_FLAT: Final = ("flat", "unchanged", "neutral", "nil", "stable")
#: A hyphen, a minus sign or an en dash, any of which a PDF may print before a negative figure.
_MINUS_SIGNS: Final = "-\u2212\u2013"
#: A figure in a quote. Not a footnote marker stuck to a word ("Timing difference1:").
_NUMBER = re.compile(
    r"(?P<open>\()?\s*(?P<sign>[-+\u2212\u2013])?\s*(?<![A-Za-z\d.,])(?P<num>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>%|basis points?|bpts?|bps?)?",
    re.IGNORECASE,
)


@dataclass
class TextWalkOutcome:
    """What the text pass produced, and why it didn't, when it didn't."""

    walks: list[Walk] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    pages: list[tuple[str, int]] = field(default_factory=list)
    refused: str = ""


@dataclass(frozen=True, slots=True)
class WalkLeg:
    """One half of a year-on-year movement in a balance, read as a walk of its own.

    Capital walks are written over the half: ANZ's 1H26 pack walks its CET1 ratio from
    September 2025 to March 2026 (p.40), and its 2025 full-year pack from March to September
    2025 (p.43). Neither answers a question against March 2025 alone, and the two together
    do. ``request`` is labelled by the halves, which is what places the commentary;
    ``opening`` and ``closing`` are the labels the chained walk carries, which match the
    question's own at its two ends.
    """

    opening: str
    closing: str
    request: RequestSpec


def leg_request(spec: RequestSpec, opening: str, closing: str, fy_end_month: int) -> RequestSpec:
    """The request for the movement from ``opening`` to ``closing``, otherwise as asked."""
    start, end = fiscal_period(opening, fy_end_month), fiscal_period(closing, fy_end_month)
    return spec.model_copy(
        update={
            "period": closing,
            "comparison_period": opening,
            "period_start": end.start,
            "period_end": end.end,
            "comparison_start": start.start,
            "comparison_end": start.end,
        }
    )


def half_year_legs(spec: RequestSpec, fy_end_month: int) -> list[WalkLeg]:
    """The two halves a year-on-year movement runs through, or none if it doesn't.

    The halves end where the question's periods end, so a full year's closing half is its
    second half: CBA's FY26 against FY25 runs from June 2025 to December 2025 and on to June
    2026.
    """
    period = fiscal_period(spec.period, fy_end_month)
    closing = (
        fiscal_period(f"2H{period.fy % 100:02d}", fy_end_month)
        if period.kind is PeriodKind.FY
        else period
    )
    middle = closing.hoh()
    opening = middle.hoh()
    if opening.end != spec.comparison_end or closing.end != spec.period_end:
        return []
    return [
        WalkLeg(
            spec.comparison_period,
            middle.label,
            leg_request(spec, opening.label, middle.label, fy_end_month),
        ),
        WalkLeg(
            middle.label,
            spec.period,
            leg_request(spec, middle.label, closing.label, fy_end_month),
        ),
    ]


def _covers(document_period: str, period: str) -> bool:
    """Whether a document reports this period: its own, or a full year's second half."""
    document_period, period = document_period.upper(), period.upper()
    return document_period == period or (
        period.startswith("2H") and document_period == f"FY{period[2:]}"
    )


def text_walk_schema() -> dict[str, Any]:
    """The structured answer. ``additionalProperties: false`` throughout, so a field the
    reader invents is rejected rather than ignored."""
    return strict_subset(
        {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "found",
                "reads_as",
                "page",
                "opening",
                "closing",
                "endpoint_unit",
                "step_unit",
                "steps",
            ],
            "properties": {
                "found": {"type": "boolean"},
                "reads_as": {"type": "string"},
                "page": {"type": "integer"},
                "opening": {"type": ["number", "null"]},
                "closing": {"type": ["number", "null"]},
                "endpoint_unit": {"type": "string", "enum": list(_UNITS)},
                "step_unit": {"type": "string", "enum": list(_UNITS)},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["label", "value", "quote"],
                        "properties": {
                            "label": {"type": "string"},
                            "value": {"type": "number"},
                            "quote": {"type": "string"},
                        },
                    },
                },
            },
        }
    )


def _condensed(text: str) -> str:
    """Letters and digits only, lowercased.

    Comparing on this makes a quote match its page whatever the PDF did to the spacing:
    lines broken mid-sentence, a word hyphenated across a line ("compet-" / "ition"), curly
    quotes. It's still the page's own words in the page's own order.
    """
    return re.sub(r"[^a-z0-9]", "", _plain(text))


def _on_page(quote: str, page_text: str) -> bool:
    needle = _condensed(quote)
    return len(needle) >= 15 and needle in _condensed(page_text)


def _in_canonical(value: Decimal, unit: Unit, ratio: bool) -> Decimal:
    return to_bps(value, unit) if ratio else to_aud_m(value, unit)


def _printed_values(quote: str, ratio: bool) -> list[tuple[Decimal, int, bool]]:
    """Each figure in a quote, in the walk's canonical unit, with its position and whether
    it was bracketed or signed negative."""
    found: list[tuple[Decimal, int, bool]] = []
    for match in _NUMBER.finditer(quote):
        try:
            number = Decimal(match.group("num"))
        except InvalidOperation:
            continue
        unit = (match.group("unit") or "").lower()
        # An explicit sign wins; brackets alone mean negative. ANZ writes both "(-4 bps)"
        # and "(+4 bps)", so a bracket can't be read as a minus on its own.
        sign = match.group("sign") or ""
        negative = sign in _MINUS_SIGNS if sign else bool(match.group("open"))
        canonical = number * 100 if ratio and unit == "%" else number
        found.append((canonical, match.start(), negative))
    return found


#: Words that compare a step's figure for the period with the one before it.
_COMPARED = re.compile(r"\bcompared (?:to|with)\b|\bversus\b|\bvs\.?(?=\s)|\bagainst\b", re.I)


def _derived_from_two(quote: str, value: Decimal, ratio: bool) -> tuple[Decimal, int, bool] | None:
    """The first of two figures the quote compares, when their difference is ``value``.

    Westpac writes its notable items step as "detracted 5 basis points compared to negative
    2 basis points in the prior corresponding period": a step of 3bps that isn't printed.
    Only a quote that says it's comparing counts, and the step's sign still has to come from
    the quote's own words, so the reading is held to the bank's sentence either way.
    """
    if not _COMPARED.search(quote):
        return None
    values = _printed_values(quote, ratio)
    for index, first in enumerate(values):
        for second in values[index + 1 :]:
            signed_first = -first[0] if first[2] else first[0]
            signed_second = -second[0] if second[2] else second[0]
            if value in (abs(first[0] - second[0]), abs(signed_first - signed_second)):
                return first
    return None


def _stated_sign(quote: str, at: int) -> int:
    """The sign the quote's own words give the figure at ``at``: the nearest direction word
    before it, else just after it. 0 when the words don't say."""
    before = re.findall(r"[a-z]+", _plain(quote[:at]))[-8:]
    after = re.findall(r"[a-z]+", _plain(quote[at:]))[:4]
    for word in reversed(before):
        if word in _DOWN:
            return -1
        if word in _UP:
            return 1
    for word in after:
        if word in _DOWN:
            return -1
        if word in _UP:
            return 1
    return 0


def _page_places_against_comparison(page: ParsedPage, spec: RequestSpec, fy_end_month: int) -> bool:
    """Whether some sentence on the page compares against the request's comparison period:
    the same period a year earlier, or for one leg of a chained walk the half before."""
    words = _words()
    ends = _ends(spec, fy_end_month)
    return any(
        comparison_of(sentence, ends, fy_end_month, words) == "pcp"
        for sentence, _ in page_sentences(page.text or "")
    )


def check_text_walk(
    answer: dict[str, Any],
    pages: dict[int, ParsedPage],
    *,
    spec: RequestSpec,
    fy_end_month: int,
    reported_opening: Decimal,
    reported_closing: Decimal,
    unit: Unit,
) -> tuple[list[tuple[str, Decimal, str]], int | None, str]:
    """The checked steps (label, value in the walk's canonical unit, quote), the page they
    are on, and why the reading was refused when it was. Steps are empty on a refusal."""
    if not answer.get("found"):
        return [], None, f"no walk between these periods in the text: {answer.get('reads_as', '')}"
    page_no = answer.get("page")
    page = pages.get(page_no) if isinstance(page_no, int) else None
    if page is None:
        return [], None, f"the reading names page {page_no}, which it wasn't given"
    # A walk can run over a page break, so a step's words may be on any page offered.
    offered = "\n".join(candidate.text or "" for candidate in pages.values())
    ratio = unit in RATIO_UNITS
    step_unit = answer.get("step_unit") if answer.get("step_unit") in _UNITS else None
    endpoint_unit = answer.get("endpoint_unit") if answer.get("endpoint_unit") in _UNITS else None
    if step_unit is None:
        return [], None, "the reading gives no unit for its steps"

    steps: list[tuple[str, Decimal, str]] = []
    for raw in answer.get("steps") or ():
        label = str(raw.get("label") or "").strip()
        quote = " ".join(str(raw.get("quote") or "").split())
        try:
            value = _in_canonical(Decimal(str(raw.get("value"))), step_unit, ratio)
        except (InvalidOperation, ValueError):
            return [], None, f"step {label!r} has no value"
        if not label:
            return [], None, "a step has no label"
        if not _on_page(quote, offered):
            if value == 0:
                # A step worth nothing can't change the walk. Westpac's 1H24 margin walk has
                # "Liquid Assets 0bps" only in its chart, and quoting just the label cost the
                # whole written walk, and with it the bank's cause for every other step.
                continue
            return [], None, f"step {label!r}: its quote is not on the pages it was given"
        if value == 0:
            if not any(word in _plain(quote).split() for word in _FLAT) and "0" not in quote:
                return [], None, f"step {label!r} is zero but its quote doesn't say flat"
            steps.append((label, value, quote))
            continue
        printed = [entry for entry in _printed_values(quote, ratio) if entry[0] == abs(value)]
        if not printed:
            derived = _derived_from_two(quote, abs(value), ratio)
            if derived is None:
                return [], None, f"step {label!r}: {abs(value)} is not printed in its quote"
            printed = [derived]
        _, at, bracketed = printed[0]
        stated = -1 if bracketed else _stated_sign(quote, at)
        if stated and (stated < 0) != (value < 0):
            return [], None, f"step {label!r}: the quote's wording gives it the other sign"
        steps.append((label, value, quote))
    if not steps:
        return [], None, "the reading has no steps"

    opening = _in_canonical(reported_opening, unit, ratio)
    closing = _in_canonical(reported_closing, unit, ratio)
    movement = closing - opening
    total = sum((value for _, value, _ in steps), Decimal(0))
    # The published endpoints are rounded too. ANZ prints its CET1 ratio to one decimal
    # place in its tables (11.8%, 12.0%) and walks it to two (11.78% to 12.03%, +25bps), so
    # the published movement of 20bps is only known to within ten.
    rounding = sum(
        (
            value_tolerance(unit, reported=reported)
            if ratio
            else to_aud_m(value_tolerance(unit, reported=reported), unit)
            for reported in (reported_opening, reported_closing)
        ),
        Decimal(0),
    )
    tolerance = walk_tolerance("bps" if ratio else "aud_m", len(steps)) + rounding
    if abs(movement - total) > tolerance:
        return (
            [],
            None,
            (
                f"the steps add to {total} against a published movement of {movement} "
                f"(tolerance {tolerance}): a different walk, period or basis"
            ),
        )

    read_opening, read_closing = answer.get("opening"), answer.get("closing")
    if read_opening is not None and read_closing is not None and endpoint_unit is not None:
        for read, published, as_published in (
            (read_opening, opening, reported_opening),
            (read_closing, closing, reported_closing),
        ):
            points = value_tolerance(unit, reported=as_published)
            if not ratio:
                points = to_aud_m(points, unit)
            if abs(_in_canonical(Decimal(str(read)), endpoint_unit, ratio) - published) > points:
                return (
                    [],
                    None,
                    (
                        f"the walk runs from {read_opening} to {read_closing}, not the published "
                        "figures: it walks another basis or period"
                    ),
                )
    elif not _page_places_against_comparison(page, spec, fy_end_month):
        return (
            [],
            None,
            (
                "the reading gives no endpoints and the page doesn't place its commentary "
                f"against {spec.comparison_period}"
            ),
        )
    return steps, page_no, ""


def rank_text_walk_pages(
    documents: list[tuple[ParsedDocument, DocMeta]],
    spec: RequestSpec,
    *,
    fy_end_month: int,
    endpoints: tuple[Decimal, Decimal, Unit],
) -> list[tuple[float, ParsedDocument, DocMeta, ParsedPage]]:
    """Pages of the current period's own documents that write out this metric's year-on-year
    walk, best first.

    A walk in words names the metric, says "basis points" several times over, gives causes,
    and often prints both published endpoints. Only pages where some sentence about the
    metric is placed against the same period a year earlier are offered at all: capital
    walks are almost always against the last balance date (NAB, ANZ and CBA all walk their
    CET1 ratio over the half), and a call to read a walk the page doesn't hold costs as much
    as one that finds it.
    """
    # The words prose uses for the metric, which are looser than a chart title's: CBA's
    # last margin step says "Decreased margin by 3 basis points", without "net interest".
    metric_words = (
        *_METRIC_WORDS.get(spec.metric_id, ()),
        *_words()["topics"].get(spec.metric_id, ()),
    )
    ranked: list[tuple[float, ParsedDocument, DocMeta, ParsedPage]] = []
    for document, meta in documents:
        if not _covers(meta.period, spec.period) or meta.doc_type not in _DOC_TYPES:
            continue
        year_on_year: set[int] = set()
        for item in placed_sentences(
            document, spec, fy_end_month=fy_end_month, topics=(spec.metric_id,)
        ):
            if item.placed != "pcp" or len(item.sentence.split()) < 6:
                continue
            lowered = _plain(item.sentence)
            if spec.metric_id in item.section or any(word in lowered for word in metric_words):
                year_on_year.add(item.page.page_no)
        for page in document.pages:
            if page.page_no not in year_on_year:
                continue
            text = page.text or ""
            lowered = _plain(text)
            mentions = len(_BASIS_POINTS.findall(text))
            if endpoints[2] in RATIO_UNITS and mentions < 3:
                continue
            score = min(mentions, 12) * 0.5
            shown = sum(1 for value in endpoints[:2] if _printed(value, endpoints[2], text))
            score += {2: 4.0, 1: 1.5}.get(shown, 0.0)
            score += 0.5 * min(len(re.findall(r"driven by|due to|reflecting", lowered)), 4)
            ranked.append((score, document, meta, page))
    ranked.sort(key=lambda item: (-item[0], item[2].doc_id, item[3].page_no))
    # A walk runs on to the next page as often as not: CBA's year-on-year margin walk ends
    # on p.29 with its Treasury and Markets step. The page after the best one goes second
    # when it holds year-on-year commentary too.
    if ranked:
        best = ranked[0]
        following = next(
            (
                item
                for item in ranked[1:]
                if item[2].doc_id == best[2].doc_id and item[3].page_no == best[3].page_no + 1
            ),
            None,
        )
        if following is not None:
            ranked = [best, following, *(item for item in ranked[1:] if item is not following)]
    return ranked


def read_text_walk(
    documents: list[tuple[ParsedDocument, DocMeta]],
    spec: RequestSpec,
    *,
    fy_end_month: int,
    reported_opening: Decimal,
    reported_closing: Decimal,
    unit: Unit,
    client: LLMClient,
    budget: Budget | None = None,
    model: str = "claude-fable-5-1",
    effort: str = "high",
    tracer: Tracer | None = None,
    prompts: PromptLibrary | None = None,
    labels: tuple[str, str] | None = None,
) -> TextWalkOutcome:
    """Read the issuer's written walk for this metric between the two periods, or nothing.

    One call, on the best two pages, which is cheaper and faster than a vision call per
    page and reads the causes as well as the values. ``spec`` may be one leg of a chained
    walk (see :func:`half_year_legs`), and ``labels`` are then the periods the walk is
    filed under.
    """
    outcome = TextWalkOutcome()
    if not client.available:
        outcome.notes.append("a written walk needs a model to read it: none configured")
        return outcome
    ranked = rank_text_walk_pages(
        documents,
        spec,
        fy_end_month=fy_end_month,
        endpoints=(reported_opening, reported_closing, unit),
    )[:_MAX_PAGES]
    if not ranked:
        outcome.notes.append(
            f"no page of a {spec.period} results document writes out a {spec.metric_id} walk "
            f"against {spec.comparison_period}"
        )
        return outcome
    if budget is not None and not budget.may_call("text_walk"):
        outcome.notes.append("reading the written walk was skipped on the run's budget")
        return outcome

    metric = get_registry().metrics[spec.metric_id]
    blocks = []
    pages: dict[int, ParsedPage] = {}
    metas: dict[int, DocMeta] = {}
    for _, _document, meta, page in ranked:
        pages[page.page_no] = page
        metas[page.page_no] = meta
        outcome.pages.append((meta.doc_id, page.page_no))
        blocks.append(
            f'<document_content doc="{meta.doc_id}" page="{page.page_no}">\n'
            f"{(page.text or '')[:_PAGE_CHARS]}\n</document_content>"
        )
    library = prompts or default_library()
    prompt = library.get("text_walk")
    request = LLMRequest(
        step="text_walk",
        model=model,
        effort=effort,
        system=prompt.text,
        messages=(
            {
                "role": "user",
                "content": (
                    f"Metric: {metric.display_name}\n"
                    f"Bank: {spec.bank_id}\n"
                    f"Walk wanted: from {spec.comparison_period} (ending "
                    f"{spec.comparison_end:%d %B %Y}) to {spec.period} (ending "
                    f"{spec.period_end:%d %B %Y}), "
                    + (
                        "the half immediately before."
                        if _ends(spec, fy_end_month).half_on_half
                        else "the same period a year earlier."
                    )
                    + "\n\n"
                    + "\n\n".join(blocks)
                ),
            },
        ),
        output_schema=text_walk_schema(),
    )
    try:
        response = client.complete(request)
    except LLMUnavailable as error:
        if isinstance(error, ModelUnreachable):  # the run stops; see llm.client
            raise
        outcome.notes.append(f"the written walk wasn't read: {error}")
        return outcome
    if budget is not None:
        budget.record("text_walk", model, response.usage, cached=response.cached)
    if tracer is not None:
        tracer.model_call(request, response, prompt=prompt)
    if response.stop_reason == "max_tokens" or not isinstance(response.structured, dict):
        outcome.notes.append("the written walk's reading came back incomplete and wasn't used")
        return outcome

    steps, page_no, refused = check_text_walk(
        response.structured,
        pages,
        spec=spec,
        fy_end_month=fy_end_month,
        reported_opening=reported_opening,
        reported_closing=reported_closing,
        unit=unit,
    )
    if refused or page_no is None:
        outcome.notes.append(f"written walk not used: {refused}")
        return outcome

    meta = metas[page_no]
    source = SourceRef(
        doc_id=meta.doc_id, doc_type=meta.doc_type, page=page_no, publish_date=meta.publish_date
    )
    ratio = unit in RATIO_UNITS
    opening_label, closing_label = labels or (spec.comparison_period, spec.period)
    outcome.walks.append(
        Walk(
            opening_period=opening_label,
            closing_period=closing_label,
            opening_value=_in_canonical(reported_opening, unit, ratio),
            closing_value=_in_canonical(reported_closing, unit, ratio),
            steps=[
                WalkStep(
                    label=label,
                    driver_id=map_label(spec.metric_id, label),
                    value=value,
                    quote=quote,
                )
                for label, value, quote in steps
            ],
            unit="bps" if ratio else "aud_m",
            source=source,
            method="management_walk",
            sources=(source,),
            read_from="text",
        )
    )
    outcome.notes.append(
        f"{meta.doc_id} p.{page_no}: the issuer's written {spec.metric_id} walk from "
        f"{opening_label} to {closing_label}, every step found on the page and reconciled to "
        "the published movement"
    )
    return outcome
