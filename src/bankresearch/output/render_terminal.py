"""The research result, laid out for a terminal."""

from __future__ import annotations

import re
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.segment import Segment
from rich.table import Table
from rich.text import Text

from ..domain.money import aud, aud_amounts
from ..domain.registry import get_registry
from .render_md import (
    format_change,
    format_magnitude,
    format_share,
    format_value,
    model_calls_said,
    result_places,
)
from .schema import Driver, ResearchResult, describe_basis, request_names

__all__ = ["render_terminal"]

_LEVEL_STYLE = {"High": "bold green", "Medium": "bold yellow", "Low": "bold red"}
_STATUS_STYLE = {
    "OK": "bold white on green",
    "PARTIAL": "bold black on yellow",
    "ABSTAIN": "bold white on red",
}

#: What each status means, in the terms someone deciding whether to use it needs.
_STATUS_MEANING = {
    "OK": "Both figures established and the drivers reconcile to the movement.",
    "PARTIAL": "The movement is established and cited; the attribution is incomplete.",
    "ABSTAIN": "No answer was given. The reason below says what was missing.",
}

#: Abstention reasons, and what someone can do about each.
_NEXT_STEP = {
    "DOCS_UNAVAILABLE": "Ingest the filing for that period: `make ingest`, or "
    "`bankresearch ingest --path <pdf> --bank ... --period ...`.",
    "PERIOD_NOT_YET_REPORTED": "The period had not been published by the as-of date. "
    "If it has been released since, ask again with a later --as-of; otherwise ask about "
    "an earlier period.",
    "BANK_UNKNOWN": "Not a configured issuer. `bankresearch coverage` lists the ones that are.",
    "METRIC_OUT_OF_SCOPE": "Not one of the six metrics in scope. "
    "`bankresearch coverage` lists them.",
    "AMBIGUOUS_INPUT": "Name the period in the bank's own terms (1H26, 2H25 or FY26) "
    "rather than as a calendar half or a month.",
    "METRIC_NOT_REPORTED": "This issuer does not report that metric.",
    "INSUFFICIENT_EVIDENCE": "The figures could not both be established from the "
    "documents held. Ingesting the issuer's investor presentation often helps.",
}

_BAR_WIDTH = 14

#: A citation as the narrative writes it: one or more evidence ids in brackets.
# Square brackets or parentheses: the prompt asks for brackets, but a model that writes
# "(ev_017)" shouldn't leave a reader with an evidence id instead of a page.
_CITATION = re.compile(r"[\[(](ev_\w+(?:\s*,\s*ev_\w+)*)[\])]")


def _bar(contribution: Decimal | None, largest: Decimal) -> Text:
    """A sign and a length: which way the driver pushed, and how hard."""
    # A zero contribution draws nothing. Rounding it up to one block would show a
    # driver as having moved the figure when it did not.
    if contribution is None or largest == 0 or contribution == 0:
        return Text("")
    filled = max(1, round(float(abs(contribution) / largest) * _BAR_WIDTH))
    style = "green" if contribution > 0 else "red" if contribution < 0 else "dim"
    return Text("█" * filled, style=style)


def _confidence(level: str | None, score: object = None) -> Text:
    if not level:
        return Text("n/a", style="dim")
    shown = f"{level}" + (f" {float(score):.2f}" if score is not None else "")  # type: ignore[arg-type]
    return Text(shown, style=_LEVEL_STYLE.get(level, ""))


def _pages_of(result: ResearchResult, ids: list[str]) -> str:
    """The pages a driver rests on, compactly: "p.11, p.22" rather than every card."""
    by_id = result.evidence_by_id()
    pages = sorted({by_id[i].source.page for i in ids if i in by_id})
    if not pages:
        return "n/a"
    shown = ", ".join(f"p.{page}" for page in pages[:3])
    return shown + (f" +{len(pages) - 3}" if len(pages) > 3 else "")


def _header(result: ResearchResult) -> Panel:
    request = result.request
    bank_name, metric_name = request_names(result)
    title = Text.assemble(
        (f" {result.status} ", _STATUS_STYLE.get(result.status, "bold")),
        "  ",
        (bank_name, "bold"),
    )
    asked = (
        Text.assemble((metric_name, "bold cyan"), "   ", (request.period, "bold"))
        if request.asked_as is not None
        else Text.assemble(
            (metric_name, "bold cyan"),
            "   ",
            (request.period, "bold"),
            " against ",
            (request.comparison_period, "bold"),
            ("  (same period last financial year)", "dim"),
        )
    )
    lines = [
        asked,
        Text(f"As of {request.as_of.isoformat()}", style="dim"),
        Text(_STATUS_MEANING.get(result.status, ""), style="italic"),
    ]
    current = result.current
    qualified = describe_basis(current.basis) if current is not None else ""
    if current is not None and qualified:
        # Said up front, because it changes what the figures mean: Westpac's
        # expense-to-income ratio is published only excluding Notable Items.
        lines.insert(2, Text(f"Basis: {current.label_at_bank}, {qualified}", style="dim"))
    return Panel(
        Group(*lines), title=title, title_align="left", box=box.ROUNDED, border_style="cyan"
    )


def _movement(result: ResearchResult) -> Table:
    metric = get_registry().metric(result.request.metric_id)
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold", expand=False)
    table.add_column("", style="bold")
    table.add_column(result.request.comparison_period, justify="right")
    table.add_column(result.request.period, justify="right")
    table.add_column("Change", justify="right")
    table.add_column("Figure confidence", justify="left")

    change = Text("n/a")
    if result.change is not None:
        shown = format_change(
            result.change.absolute, result.change.change_unit, result_places(result)
        )
        if result.change.pct is not None:
            shown += f"  ({result.change.pct:+}%)"
        style = (
            "green" if result.change.absolute > 0 else "red" if result.change.absolute < 0 else ""
        )
        change = Text(shown, style=f"bold {style}")

    headline = result.headline_confidence
    # The figures are named as the bank names them. Judo's from 1H26 are its profit before
    # tax, and a row headed "Cash earnings" would say they were something else.
    table.add_row(
        result.current.label_at_bank if result.current else metric.display_name,
        format_value(result.prior.value if result.prior else None, metric.unit),
        Text(
            format_value(result.current.value if result.current else None, metric.unit),
            style="bold",
        ),
        change,
        _confidence(headline.level if headline else None, headline.score if headline else None),
    )
    return table


def _driver_table(result: ResearchResult, *, bar: bool, pages: bool, label_width: int) -> Table:
    unit = result.change.change_unit if result.change else "bps"
    largest = max(
        (abs(d.contribution) for d in result.drivers if d.contribution is not None),
        default=Decimal(0),
    )

    table = Table(box=box.SIMPLE, header_style="bold", expand=False, pad_edge=False)
    table.add_column("Driver", min_width=label_width, max_width=56, overflow="fold")
    table.add_column("Contribution", justify="right", no_wrap=True)
    if bar:
        table.add_column("", min_width=_BAR_WIDTH, no_wrap=True)
    table.add_column("Share", justify="right", style="dim", no_wrap=True)
    table.add_column("Confidence", no_wrap=True)
    if pages:
        table.add_column("Source", style="dim", no_wrap=True)

    def row(driver: Driver, nested: bool = False) -> None:
        label = Text(("  ↳ " if nested else "") + driver.label, style="dim" if nested else "")
        cells: list[Text | str] = [
            label,
            format_change(driver.contribution, unit, result_places(result)),
        ]
        if bar:
            cells.append(_bar(driver.contribution, largest))
        cells.append(
            format_share(driver.share_of_change) if result.shares_mean_something else "n/m"
        )
        cells.append(
            _confidence(driver.confidence.level, driver.confidence.score)
            if driver.confidence
            else Text("n/a", style="dim")
        )
        if pages:
            cells.append(_pages_of(result, driver.evidence_ids))
        table.add_row(*cells)

    for driver in sorted(
        result.top_level_drivers, key=lambda d: -abs(d.contribution or Decimal(0))
    ):
        row(driver)
        for child in result.children_of(driver.driver_id):
            row(child, nested=True)
    return table


def _drivers(result: ResearchResult, console: Console) -> Table | Text:
    """The drivers, in the widest layout the terminal can show without cropping.

    A table wider than the terminal isn't wrapped, it's cut off at the edge, and the first
    column to go was the page reference: "p.22" printed as "p.2", a citation to the wrong
    page. So optional columns are dropped instead (the bar first, then the pages), and the
    pages are still listed under Sources.
    """
    if not result.top_level_drivers:
        return Text(
            "No drivers established. The movement above is cited and correct; what it is "
            "missing is the attribution.",
            style="yellow",
        )
    # The label is what a reader needs most, so the bar only stays while every label fits on
    # one line (up to a point). On a terminal too narrow even for that, labels wrap rather
    # than anything being cut off.
    longest = max(len(d.label) + (4 if d.parent_id else 0) for d in result.drivers)
    comfortable = min(longest, 30)
    layouts = ((True, True, comfortable), (False, True, comfortable), (False, False, 16))
    for bar, pages, label_width in layouts:
        table = _driver_table(result, bar=bar, pages=pages, label_width=label_width)
        if _rendered_width(table, console) <= console.width and _labels_shown(
            table, console, result
        ):
            break
    return table


def _labels_shown(renderable: Table, console: Console, result: ResearchResult) -> bool:
    """Whether every driver's label is visible at all.

    Rich fits a table to the terminal by squeezing columns, and it will squeeze the label
    column to nothing before giving up a no-wrap one. With short shares ("n/m") the table
    fit at 60 columns by printing no driver names.
    """
    lines = Segment.split_lines(console.render(renderable, console.options))
    printed = "\n".join("".join(segment.text for segment in line) for line in lines)
    return all(d.label[:6] in printed for d in result.top_level_drivers if d.label)


def _rendered_width(renderable: Table, console: Console) -> int:
    """How wide it actually prints, before the console crops it.

    Not ``console.measure(...).minimum``: rich shrinks a table to fit and then widens each
    no-wrap column back to its content, so a table that measures as fitting can still print
    wider than the terminal.
    """
    lines = Segment.split_lines(console.render(renderable, console.options))
    return max((Segment.get_line_length(line) for line in lines), default=0)


def _reconciliation(result: ResearchResult) -> Text | None:
    rec = result.reconciliation
    if rec is None or result.change is None:
        return None
    unit = result.change.change_unit
    tick = ("✓", "green") if rec.within_tolerance else ("✗", "red")
    return Text.assemble(
        (f" {tick[0]} ", f"bold {tick[1]}"),
        "Drivers explain ",
        (format_change(rec.sum_of_drivers, unit, result_places(result)), "bold"),
        " of ",
        (format_change(rec.headline_change, unit, result_places(result)), "bold"),
        "   residual ",
        (format_change(rec.residual, unit, result_places(result)), "bold"),
        f"  (tolerance {format_magnitude(rec.tolerance, unit, result_places(result))})",
        style="",
    )


def _overall(result: ResearchResult) -> Text | None:
    overall = result.overall_confidence
    if overall is None:
        return None
    return Text.assemble(
        "Explanation confidence  ",
        _confidence(overall.level, overall.score),
        ("   how much of the movement is attributed, and how well", "dim"),
    )


def _narrative(result: ResearchResult) -> Text:
    """The narrative, with each citation turned into the page it points to.

    A ``Text``, never a plain string: rich treats a string as markup, and "[ev_003]" is
    valid markup, so every citation was being deleted from the screen. A page is also what
    someone checking a claim will open, whereas an evidence id means nothing without the
    result file next to it. The document is only named where a page number alone would be
    ambiguous among the pages this answer shows. Judged over every card a run collected, it
    named the document on one page of a memo and not the next: Westpac's page 13 shared its
    number with a page of the investor pack the answer never cited.
    """
    by_id = result.evidence_by_id()
    shown = result.cited_evidence_ids() | {
        part.strip()
        for match in _CITATION.finditer(result.narrative)
        for part in match.group(1).split(",")
    }
    documents_by_page: dict[int, set[str]] = defaultdict(set)
    for card in result.evidence:
        if card.evidence_id in shown:
            documents_by_page[card.source.page].add(card.source.doc_id)

    def page(evidence_id: str) -> str:
        card = by_id.get(evidence_id)
        if card is None:
            return evidence_id
        if len(documents_by_page.get(card.source.page, set())) > 1:
            return f"{card.source.doc_id} p.{card.source.page}"
        return f"p.{card.source.page}"

    def replace(match: re.Match[str]) -> str:
        ids = [part.strip() for part in match.group(1).split(",")]
        return "[" + ", ".join(dict.fromkeys(page(i) for i in ids)) + "]"

    return Text(_CITATION.sub(replace, _adjacent_joined(aud_amounts(result.narrative))))


def _with_pages(result: ResearchResult, text: str) -> Text:
    """Text with each citation turned into the page it points to (see ``_narrative``)."""
    by_id = result.evidence_by_id()

    def replace(match: re.Match[str]) -> str:
        ids = [part.strip() for part in match.group(1).split(",")]
        shown = [f"p.{by_id[i].source.page}" if i in by_id else i for i in ids]
        return "[" + ", ".join(dict.fromkeys(shown)) + "]"

    return Text(_CITATION.sub(replace, _adjacent_joined(text)))


def _adjacent_joined(text: str) -> str:
    """Citations written side by side as one: "[ev_030][ev_008]" is "[ev_030, ev_008]".

    Two pieces of evidence on one page then show as one page. CBA's FY24 memo printed
    "[p.141][p.141][p.26]" and "[p.18][p.18][p.18][p.18]".
    """
    return re.sub(r"\]\s*\[(?=ev_\d)", ", ", text)


def _causes(result: ResearchResult) -> Panel | None:
    """Each driver's cause as the bank states it, checked against its own words."""
    explained = [driver for driver in result.drivers if driver.cause]
    if not explained:
        return None
    rows = []
    for driver in explained:
        line = Text()
        line.append(f"{driver.label or driver.driver_id}  ", style="bold")
        line.append_text(_with_pages(result, aud_amounts(driver.cause)))
        rows.append(line)
    return Panel(
        Text("\n").join(rows),
        title="Why, as the bank explains it",
        title_align="left",
        border_style="green",
        box=box.ROUNDED,
    )


def _commentary(result: ResearchResult) -> Panel | None:
    """The bank's own sentences on the movement, when no model wrote a narrative."""
    if result.narrative or not result.issuer_commentary:
        return None
    cards = result.evidence_by_id()
    shown = [cards[i] for i in result.issuer_commentary if i in cards][:6]
    if not shown:
        return None
    body = Text("\n").join(
        Text.assemble(f'"{card.passage or card.quote}" ', (f"p.{card.source.page}", "dim"))
        for card in shown
    )
    return Panel(
        body,
        title="In the bank's words (quoted, not matched to drivers)",
        title_align="left",
        border_style="cyan",
        box=box.ROUNDED,
    )


def _caveats(result: ResearchResult) -> Panel | None:
    lines = [c for c in result.caveats if c.strip()]
    if not lines:
        return None
    shown = [
        Text.assemble((f"{number}. ", "dim"), caveat)
        for number, caveat in enumerate(lines[:8], start=1)
    ]
    if len(lines) > 8:
        shown.append(Text(f"… {len(lines) - 8} more in result.json", style="dim"))
    body = Text("\n").join(shown)
    return Panel(
        body,
        title="Read before relying on this",
        title_align="left",
        border_style="yellow",
        box=box.ROUNDED,
    )


def _sources(result: ResearchResult) -> Text | None:
    cited = result.cited_evidence_ids()
    pages: dict[str, set[int]] = defaultdict(set)
    for card in result.evidence:
        if card.evidence_id in cited:
            pages[card.source.doc_id].add(card.source.page)
    if not pages:
        return None
    body = Text("Sources\n", style="bold")
    for doc_id in sorted(pages):
        listed = ", ".join(f"p.{p}" for p in sorted(pages[doc_id]))
        body.append(f"  {doc_id}", style="cyan")
        body.append(f"   {listed}\n", style="dim")
    return body


#: Rows of what was read shown in the terminal; the rest are in result.json and the memo.
_READ_ROWS = 14


def _clipped(text: str, limit: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _what_was_read(result: ResearchResult) -> Table | None:
    """Every figure and sentence the answer rests on: where it was read, what it is, and the
    cell or the words it was read from."""
    cited = result.cited_evidence_ids()
    cards = sorted(
        (card for card in result.evidence if card.evidence_id in cited),
        key=lambda c: (c.source.doc_id, c.source.page, c.evidence_id),
    )
    if not cards:
        return None
    several = len({card.source.doc_id for card in cards}) > 1
    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="dim",
        title="What was read",
        title_style="bold",
        title_justify="left",
        expand=False,
    )
    table.add_column("Where", style="dim", no_wrap=True)
    table.add_column("What")
    table.add_column("Value", justify="right", no_wrap=True)
    table.add_column("Read from", style="dim", overflow="fold")
    for card in cards[:_READ_ROWS]:
        where = f"{card.source.doc_id} p.{card.source.page}" if several else f"p.{card.source.page}"
        if card.value is None and card.passage:
            what, value, read = "the bank's words", "", f'"{_clipped(card.passage)}"'
        else:
            step = ".walk." in card.component or card.method == "chart"
            what = f"{'walk step' if step else _plain(card.component)}, {card.period}"
            value = format_value(card.published_value or card.value, str(card.unit))
            read = _clipped(card.quote or card.statement)
        table.add_row(where, what, value, read)
    if len(cards) > _READ_ROWS:
        table.add_row("", Text(f"… {len(cards) - _READ_ROWS} more in result.json", "dim"), "", "")
    return table


def _plain(component: str) -> str:
    from ..domain.registry import component_name

    try:
        return component_name(component)
    except Exception:  # an id the registry doesn't know is shown as it is
        return component.replace("_", " ")


def _model_calls(result: ResearchResult) -> Table | None:
    """Each model call in order: what it was for, the model, its tokens and what it cost."""
    from .progress import PURPOSES

    calls = result.trail.model_calls
    if not calls:
        return None
    spent = sum((call.cost_usd or Decimal(0) for call in calls), Decimal(0))
    count = model_calls_said(result, noun="call")
    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="dim",
        title=f"Model calls   {count}, {aud(spent)}",
        title_style="bold",
        title_justify="left",
        expand=False,
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("What it was for")
    table.add_column("Model", style="dim", no_wrap=True)
    table.add_column("Tokens in · out", justify="right", no_wrap=True)
    table.add_column("Cost", justify="right", no_wrap=True)
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
        table.add_row(
            str(number),
            purpose,
            call.model,
            f"{call.input_tokens:,} · {call.output_tokens:,}",
            cost,
        )
    return table


def _fetched(result: ResearchResult) -> Text | None:
    if not result.trail.fetched:
        return None
    body = Text("Found and read during the run\n", style="bold")
    for note in result.trail.fetched:
        body.append(f"  {_clipped(note, 160)}\n", style="dim")
    return body


def _relative(path: object) -> str:
    """A path under the working directory, shown from there; any other path in full."""
    try:
        return Path(str(path)).resolve().relative_to(Path.cwd().resolve()).as_posix()
    except (OSError, ValueError):
        return Path(str(path)).as_posix()


def _footer(result: ResearchResult, artefacts: object = None) -> Text:
    d = result.diagnostics
    parts = [f"run {result.run_id}"]
    if d.llm_calls or result.trail.model_calls:
        parts.append(f"{model_calls_said(result)}, {aud(d.cost_usd)}")
    else:
        parts.append("no model calls")
    parts.append(f"{d.latency_ms / 1000:.1f}s")
    if artefacts is not None:
        parts.append(f"saved to {_relative(artefacts)}")
    return Text("  ·  ".join(parts), style="dim")


def _abstention(result: ResearchResult) -> Panel:
    reason = result.abstain_reason or ""
    body = Text()
    body.append(f"{reason}\n\n", style="bold red")
    detail = next((c for c in result.caveats if c.strip()), "")
    if detail:
        body.append(detail + "\n\n")
    hint = _NEXT_STEP.get(reason)
    if hint:
        body.append("What to do  ", style="bold")
        body.append(hint)
    return Panel(body, title="No answer", title_align="left", border_style="red", box=box.ROUNDED)


def render_terminal(result: ResearchResult, console: Console, artefacts: object = None) -> None:
    """Print one result."""
    console.print()
    console.print(_header(result))
    if result.status == "ABSTAIN":
        console.print(_abstention(result))
        for part in (_model_calls(result), _fetched(result)):
            if part is not None:
                console.print(part)
        console.print(_footer(result, artefacts))
        return

    console.print(_movement(result))
    console.print(Text("Drivers", style="bold"), Text("  largest first", style="dim"))
    console.print(_drivers(result, console))
    for part in (_reconciliation(result), _overall(result)):
        if part is not None:
            console.print(part)

    causes = _causes(result)
    if causes is not None:
        console.print(causes)
    if result.narrative:
        console.print()
        console.print(
            Panel(
                _narrative(result),
                title="What moved",
                title_align="left",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )
    commentary = _commentary(result)
    if commentary is not None:
        console.print(commentary)
    caveats = _caveats(result)
    if caveats is not None:
        console.print(caveats)
    sources = _sources(result)
    if sources is not None:
        console.print(sources)
    # How the answer was reached: what was read to reach it, each model call, and anything
    # fetched on the way.
    for trail in (_what_was_read(result), _model_calls(result), _fetched(result)):
        if trail is not None:
            console.print(trail)
    console.print(_footer(result, artefacts))
