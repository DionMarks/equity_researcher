"""The financial year of a listed bank nobody configured, found from its ASX announcements."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Final

from ..llm.budget import Budget
from ..llm.client import LLMClient, LLMRequest, LLMUnavailable, ModelUnreachable
from ..llm.tracing import PromptLibrary, Tracer, default_library
from .discover import SEARCH_ROUND_USD, search_tool, urls_from_search_results
from .fetch import ASX_DOMAIN

__all__ = ["YearEnd", "find_year_end"]

#: Where the search may look: the ASX's own site and its announcements platform.
ASX_SITES: Final = ("www.asx.com.au", "asx.com.au", ASX_DOMAIN)
_DOMAIN: Final = re.compile(r"^(?:[a-z0-9-]+\.)+[a-z]{2,}$")


@dataclass
class YearEnd:
    """What the search found, and why it settled or didn't."""

    fy_end_month: int | None = None
    website: str | None = None
    notes: list[str] = field(default_factory=list)


def find_year_end(
    name: str,
    code: str,
    client: LLMClient,
    *,
    model: str,
    effort: str,
    budget: Budget | None = None,
    tracer: Tracer | None = None,
    prompts: PromptLibrary | None = None,
) -> YearEnd:
    """The month ``name``'s financial year ends, from a search of the ASX, or why not."""
    found = YearEnd()
    if not client.available:
        found.notes.append("finding a bank's financial year needs a model, and none is set")
        return found
    if budget is not None and not budget.may_call("identify_bank", projected_usd=SEARCH_ROUND_USD):
        found.notes.append("identifying the bank stopped on its budget")
        return found
    prompt = (prompts or default_library()).get("identify_bank")
    request = LLMRequest(
        step="identify_bank",
        model=model,
        effort=effort,
        system=prompt.text,
        messages=(
            {
                "role": "user",
                "content": f"The bank: {name}, ASX code {code}. When does its financial year end?",
            },
        ),
        tools=(search_tool(ASX_SITES),),
    )
    try:
        response = client.complete(request)
    except LLMUnavailable as error:
        if isinstance(error, ModelUnreachable):  # the run stops; see llm.client
            raise
        found.notes.append(f"identifying the bank stopped: {error}")
        return found
    if budget is not None:
        budget.record("identify_bank", model, response.usage, cached=response.cached)
    if tracer is not None:
        tracer.model_call(request, response, prompt=prompt)

    results = urls_from_search_results(response.raw_blocks, ASX_SITES)
    if not any(not result.rejected_because for result in results):
        found.notes.append("the search of the ASX returned nothing to go on")
        return found
    answer = _json_in(response.text)
    month = answer.get("fy_end_month") if answer.get("found") is True else None
    if not isinstance(month, int) or not 1 <= month <= 12:
        found.notes.append(
            "the ASX's announcements didn't settle its financial year: "
            + str(answer.get("why") or "no reason was given")
        )
        return found
    found.fy_end_month = month
    found.website = _own_domain(answer.get("website"), name, code)
    found.notes.append(
        f"its financial year ends in month {month}: {answer.get('why') or ''}".strip()
    )
    return found


def _json_in(text: str) -> dict[str, Any]:
    """The JSON object in a reply, or an empty one if there isn't one."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _own_domain(website: Any, name: str, code: str) -> str | None:
    """The company's own domain, only if it reads as one: a plain domain sharing a word
    with its name or code. Anything else stays off the list a download may come from."""
    if not isinstance(website, str):
        return None
    domain = website.strip().lower().removeprefix("https://").removeprefix("http://")
    domain = domain.removeprefix("www.").split("/")[0]
    if not _DOMAIN.match(domain):
        return None
    words = {word for word in re.findall(r"[a-z0-9]+", f"{name} {code}".lower()) if len(word) >= 3}
    words -= {"limited", "group", "holdings", "corporation", "bank", "banking", "the"}
    return domain if any(word in domain for word in words) else None
