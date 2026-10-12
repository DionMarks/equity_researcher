"""The locate loop: a bounded tool loop that names the pages to read, and never returns a figure."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ...domain.models import ReadingList, RequestSpec
from ...llm.budget import Budget
from ...llm.client import LLMClient, LLMRequest, LLMUnavailable, ModelUnreachable
from ...llm.tracing import PromptLibrary, Tracer, default_library
from ...retrieval.playbooks import playbook_for
from ...tools.doc_tools import Toolkit
from ...tools.registry import LLM_CALLABLE, SUBMIT_READING_LIST, tool_definitions

__all__ = ["LocateOutcome", "run_locate"]

_MAX_TURNS = 8


@dataclass
class LocateOutcome:
    """What the loop produced, and how it stopped."""

    reading_list: ReadingList | None = None
    turns: int = 0
    stopped: str = "submitted"
    tool_calls: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    charts_read: list[tuple[str, int]] = field(default_factory=list)
    """Pages the loop looked at as charts, in the order it chose them."""

    @property
    def succeeded(self) -> bool:
        return self.reading_list is not None

    def pages(self) -> list[tuple[str, int]]:
        """Every page the loop pointed at: the charts it read, then the reading list's pages."""
        found = list(self.charts_read)
        for item in self.reading_list.items if self.reading_list is not None else ():
            for page in item.pages:
                if (item.doc_id, page) not in found:
                    found.append((item.doc_id, page))
        return found


def run_locate(
    spec: RequestSpec,
    toolkit: Toolkit,
    client: LLMClient,
    *,
    budget: Budget | None = None,
    # Overridden from config/models.yaml through ``Registry.route("locate")``. A default
    # that differs from the routing is how that config ended up unused for the whole build.
    model: str = "claude-opus-5",
    effort: str = "medium",
    tracer: Tracer | None = None,
    prompts: PromptLibrary | None = None,
    max_turns: int = _MAX_TURNS,
) -> LocateOutcome:
    """Choose the pages to read. Returns a reading list, or says why there is none."""
    outcome = LocateOutcome(stopped="submitted")
    if not client.available:
        outcome.stopped = "no_model"
        outcome.notes.append("no model configured: the deterministic page selection stands")
        return outcome

    library = prompts or default_library()
    prompt = library.get("locate")
    tools = tuple(tool for tool in tool_definitions() if tool.get("name") in LLM_CALLABLE)
    messages: list[dict[str, Any]] = [{"role": "user", "content": _opening(spec)}]

    for turn in range(1, max_turns + 1):
        outcome.turns = turn
        if budget is not None and not budget.may_call("locate"):
            outcome.stopped = "budget"
            outcome.notes.append(f"locate stopped on its budget after {turn - 1} turn(s)")
            break

        request = LLMRequest(
            step="locate",
            model=model,
            effort=effort,
            system=prompt.text,
            messages=tuple(messages),
            tools=tools,
        )
        try:
            response = client.complete(request)
        except LLMUnavailable as error:
            if isinstance(error, ModelUnreachable):  # the run stops; see llm.client
                raise
            outcome.stopped = "model_error"
            outcome.notes.append(f"locate stopped: {error}")
            break

        if budget is not None:
            budget.record("locate", model, response.usage, cached=response.cached)
        if tracer is not None:
            tracer.model_call(request, response, prompt=prompt)

        if not response.wants_tools:
            outcome.stopped = "no_tool_call"
            outcome.notes.append("the loop ended without submitting a reading list")
            break

        # Append-only: the assistant's blocks go back as they came.
        messages.append({"role": "assistant", "content": list(response.raw_blocks)})
        results: list[dict[str, Any]] = []
        submitted: ReadingList | None = None

        for call in response.tool_calls:
            outcome.tool_calls.append(call.name)
            if call.name == "read_chart":
                page = call.arguments.get("page")
                doc_id = call.arguments.get("doc_id")
                if isinstance(doc_id, str) and isinstance(page, int):
                    outcome.charts_read.append((doc_id, page))
            result = toolkit.dispatch(call.name, call.arguments)
            if call.name == SUBMIT_READING_LIST:
                # Read off the toolkit, not from the return value. The handler returns the
                # acknowledgement the model sees (``{"accepted": true, ...}``) and records
                # the reading list on itself. Checking ``isinstance(result, ReadingList)``
                # was therefore always false: the model submitted a list, the toolkit stored
                # it, and the loop carried on until the model gave up, reporting no pages for
                # a run that had found them.
                submitted = getattr(toolkit, "reading_list", None)
                if submitted is None and isinstance(result, ReadingList):
                    submitted = result  # a toolkit that returns it directly
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": _render(result),
                }
            )

        messages.append({"role": "user", "content": results})
        if submitted is not None:
            outcome.reading_list = submitted
            outcome.stopped = "submitted"
            return outcome
    else:
        outcome.stopped = "max_turns"
        outcome.notes.append(f"locate used all {max_turns} turns without submitting a reading list")

    return outcome


def _opening(spec: RequestSpec) -> str:
    """The request, stated plainly, with the metric's playbook. No evidence is loaded.

    Loading pages up front was tried and is worse: it spends the context on pages nobody
    chose, and anchors the loop on whatever happened to come first. The playbook is
    different. It's a few lines saying where this metric is usually found and which pages
    only look right, and it costs far less than the search turns it saves.
    """
    checklist = playbook_for(spec.metric_id).checklist(spec.bank_id, spec.period)
    return (
        f"Bank: {spec.bank_id}\n"
        f"Metric: {spec.metric_id}\n"
        f"Period: {spec.period}\n"
        f"Comparison period: {spec.comparison_period} (the same period last financial year)\n"
        f"As of: {spec.as_of.isoformat()}. No document published after this date exists.\n\n"
        "Find the pages that carry the headline figure for both periods, the components of "
        "its accounting identity, and any walk the issuer published. Start with "
        "`get_outline`, then `search_docs`. Call `submit_reading_list` when you have them."
        + (f"\n\nWhere this metric is usually found:\n{checklist}" if checklist else "")
    )


def _render(result: Any) -> str:
    """A tool result as the model sees it.

    Serialised rather than stringified so the shape is stable between turns, and bounded
    by the toolkit before it reaches here.
    """
    dump = getattr(result, "model_dump", None)
    if callable(dump):
        return json.dumps(dump(), default=str)
    if isinstance(result, dict | list):
        return json.dumps(result, default=str)
    return str(result)
