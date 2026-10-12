"""Cold-path discovery: finding a document for an issuer the corpus doesn't have."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Final

from ..domain.calendar import fiscal_period
from ..domain.registry import get_registry
from ..llm.budget import Budget
from ..llm.client import LLMClient, LLMRequest, LLMUnavailable, ModelUnreachable
from ..llm.tracing import PromptLibrary, Tracer, default_library
from .fetch import ASX_DOMAIN, host_allowed

__all__ = [
    "DOC_TYPES",
    "MAX_ROUNDS",
    "Candidate",
    "DiscoveryOutcome",
    "classification_agrees",
    "classification_schema",
    "confirm_document",
    "discover_documents",
    "search_errors",
    "search_tool",
    "urls_from_search_results",
]

MAX_ROUNDS = 3
#: What a results document is called, by the kind of period, for the rounds after the first.
#: The first round uses the issuer's own name for it from config. When that finds nothing, the
#: next rounds ask for the names every ASX bank's filing goes by, so a hint that's wrong for
#: the period (Bank of Queensland's "Appendix 4D", for a full year) costs one round, not all.
GENERIC_NAMES: dict[str, tuple[str, ...]] = {
    "full year": ("Appendix 4E and annual report", "full year results announcement"),
    "half year": ("Appendix 4D and half year financial report", "half year results announcement"),
}
#: What a bank's slide pack goes by, for the rounds after the first.
DECK_NAMES: tuple[str, ...] = ("results investor presentation", "results presentation slides")
_DECK: Final = "INVESTOR_PRESENTATION"
#: Candidates downloaded and classified per round. Each costs a download and a call, and a
#: search for a results PDF can easily return eight of them.
MAX_CANDIDATES = 4
# The basic version. A later one must set allowed_callers to `direct`.
WEB_SEARCH_TOOL = "web_search_20250305"
_MAX_USES = 5
#: What one search round costs, in US dollars like the bill it's checked against. Judo's three
#: rounds cost A$0.56, A$1.03 and A$1.05, nearly all of it search results written to the cache.
SEARCH_ROUND_USD = Decimal("0.75")

_HALF = re.compile(
    r"\b(?:half[- ]year|six months|interim)\b[^.]{0,40}?\b(?:ended|ending)\b[^.]{0,30}?"
    r"(\d{1,2}\s+\w+\s+(\d{4}))",
    re.IGNORECASE,
)
# "year ended" mustn't match the "year ended" inside "half year ended", which it does unless
# the half is excluded explicitly. Accepting a half-year document for a full-year request
# means a whole period of wrong figures.
_FULL = re.compile(
    r"(?<!half\s)(?<!half-)\b(?:full[- ]year|twelve months|annual|year)\b"
    r"[^.]{0,40}?\b(?:ended|ending)\b[^.]{0,30}?(\d{1,2}\s+\w+\s+(\d{4}))",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One document the search proposed, after validation."""

    url: str
    title: str
    bank_id: str | None = None
    doc_type: str | None = None
    period: str | None = None
    publish_date: date | None = None
    confirmed: bool = False
    rejected_because: str = ""


@dataclass
class DiscoveryOutcome:
    accepted: list[Candidate] = field(default_factory=list)
    rejected: list[Candidate] = field(default_factory=list)
    rounds: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.accepted)


def search_tool(allowed_domains: tuple[str, ...]) -> dict[str, Any]:
    """The provider's server-side search tool, restricted to the issuer's own sites.

    Restricting the search isn't the same as filtering its results: a result that never
    arrives can't be followed by mistake, and can't use up context arguing for itself.
    """
    return {
        "type": WEB_SEARCH_TOOL,
        "name": "web_search",
        "max_uses": _MAX_USES,
        "allowed_domains": list(dict.fromkeys((*allowed_domains, ASX_DOMAIN))),
    }


def search_errors(blocks: tuple[dict[str, Any], ...]) -> list[str]:
    """Error codes from any search that failed.

    A failed search and a search that found nothing both give no candidates, but they need
    different responses: one should be retried or reported, the other means the issuer
    really has nothing on its site. The provider tells them apart. A failure returns
    ``content`` as a single ``web_search_tool_result_error`` object instead of a list of
    results, and a search with no matches returns an empty list. Reading that shape is the
    difference between "no documents" and "we didn't look".
    """
    codes: list[str] = []
    for block in blocks:
        if block.get("type") != "web_search_tool_result":
            continue
        content = block.get("content")
        if isinstance(content, dict) and content.get("type") == "web_search_tool_result_error":
            code = str(content.get("error_code") or "unknown")
            if code not in codes:
                codes.append(code)
    return codes


def urls_from_search_results(
    blocks: tuple[dict[str, Any], ...], allowed_domains: tuple[str, ...]
) -> list[Candidate]:
    """Candidates from the search-result blocks, and from nowhere else.

    Reads the structured ``url`` field of each ``web_search_result``, which the provider
    nests in the result block's ``content`` list. The model's text isn't parsed for links at
    all, not even as a fallback. Free text can be manipulated, and a result block can't.
    """
    found: list[Candidate] = []
    seen: set[str] = set()
    for block in blocks:
        if block.get("type") not in {"web_search_tool_result", "web_search_result"}:
            continue
        results = block.get("content")
        # On a failure `content` is one error object, not a list; `search_errors` reads
        # those. Falling back to the block itself covers a bare `web_search_result`.
        if isinstance(results, dict):
            continue
        items = results if isinstance(results, list) else [block]
        for item in items:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            title = str(item.get("title") or "").strip()
            if not host_allowed(url, allowed_domains):
                found.append(
                    Candidate(
                        url=url,
                        title=title,
                        rejected_because="not on the issuer's allowlist",
                    )
                )
                continue
            found.append(Candidate(url=url, title=title))
    return found


#: What a document can be. "OTHER" is a real answer: a presentation or a cover letter.
DOC_TYPES = (
    "RESULTS_DOCUMENT",
    "RESULTS_RELEASE",
    "INVESTOR_PRESENTATION",
    "ANNUAL_REPORT",
    "PILLAR3",
    "OTHER",
)


def classification_schema() -> dict[str, Any]:
    """What the classifier returns: what the page states, never a ticker or a fiscal label.

    It used to be asked for ``bank_id`` and ``period`` and compared against "BEN" and
    "1H26", which nothing told it to write. A model reading "Bendigo and Adelaide Bank
    Limited ... half year ended 31 December 2025" would rightly say so, and every real
    document would have been rejected. It reads, and code maps what it read.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["issuer", "doc_type", "period_kind", "period_end", "publish_date"],
        "properties": {
            "issuer": {
                "type": ["string", "null"],
                "description": "The issuer's name as the document prints it.",
            },
            "doc_type": {"type": "string", "enum": list(DOC_TYPES)},
            "period_kind": {"type": "string", "enum": ["half_year", "full_year", "unknown"]},
            "period_end": {
                "type": ["string", "null"],
                "description": "The last day of the period the document reports, YYYY-MM-DD.",
            },
            "publish_date": {
                "type": ["string", "null"],
                "description": "The date the document was released, YYYY-MM-DD, if it says.",
            },
        },
    }


def classification_agrees(
    claimed: dict[str, Any],
    text: str,
    *,
    bank_id: str,
    period: str,
    fy_end_month: int,
    doc_type: str = "RESULTS_DOCUMENT",
) -> tuple[bool, str]:
    """Whether the model's reading of a document and a pattern over its text agree.

    Both have to match what was asked for. The model may be confidently wrong and the
    pattern may not find anything, so a candidate is only accepted when two independent
    readings agree, and a wrong one costs nothing.

    The period is compared as a date, against the issuer's own calendar. Comparing the year
    on the page with the label's year was wrong for a June year end: CBA's and Bendigo's
    1H26 ends on 31 December 2025, so every genuine half-year report would have failed.
    """
    profile = get_registry().bank(bank_id)
    issuer = _plain(str(claimed.get("issuer") or ""))
    names = {_plain(profile.name), _plain(re.sub(r"(?i)\b(limited|ltd)\b", "", profile.name))}
    names.discard("")
    if not issuer or not any(name in issuer or issuer in name for name in names):
        return False, f"the document reads as issued by {claimed.get('issuer')!r}, not {bank_id}"

    try:
        wanted = fiscal_period(period, fy_end_month)
    except ValueError as error:
        return False, f"the requested period can't be resolved: {error}"
    half = not period.upper().startswith("FY")
    kind = "half_year" if half else "full_year"
    if doc_type.upper() == _DECK and _cover_names(text, period):
        # A slide pack names its period on the cover ("FY26 Financial Results") and rarely
        # says "year ended" anywhere. BNK Banking Corporation's FY26 deck never does. The
        # classifier may then not commit to a period, which is right, but if it does it has
        # to agree.
        stated_kind = claimed.get("period_kind")
        if stated_kind not in (None, "unknown", kind):
            return False, _reads_as(stated_kind, kind)
        stated_end = _as_date(claimed.get("period_end"))
        if stated_end is not None and stated_end != wanted.end:
            return False, f"the document reads as ending {stated_end}, not {wanted.end}"
        return True, ""
    if claimed.get("period_kind") != kind:
        return False, _reads_as(claimed.get("period_kind"), kind)
    if _as_date(claimed.get("period_end")) != wanted.end:
        return False, (
            f"the document reads as ending {claimed.get('period_end')}, "
            f"not {wanted.end.isoformat()}"
        )

    match = (_HALF if half else _FULL).search(text)
    if match is None:
        what = "half year" if half else "full year"
        return False, f"no '{what} ended ...' statement found in the document's own text"
    stated = _printed_date(match.group(1))
    if stated != wanted.end:
        return False, f"the document's text says {match.group(1)}, not {wanted.end.isoformat()}"
    return True, ""


#: A period named the way a slide pack's cover names it: "FY26", "1H26", "H1 FY26".
_COVER_LABEL = re.compile(
    r"\b(FY|1H|2H)\s?'?(?:20)?(\d{2})\b|\bH([12])\s?FY\s?(?:20)?(\d{2})\b", re.I
)
#: How much of the text counts as the cover.
_COVER_CHARS = 600


def _reads_as(stated: object, wanted: str) -> str:
    """Why a document of the wrong kind was turned down, in words: "the document reads as an
    unknown report, not a full-year one", where it used to read "a unknown report, not
    full_year"."""
    words = {"half_year": "half-year", "full_year": "full-year"}
    said = words.get(str(stated), "unknown" if stated in (None, "unknown") else str(stated))
    article = "an" if said[:1] in "aeiou" else "a"
    return f"the document reads as {article} {said} report, not a {words.get(wanted, wanted)} one"


def _cover_names(text: str, period: str) -> bool:
    """Whether the first period label on the cover is the period asked for."""
    match = _COVER_LABEL.search(text[:_COVER_CHARS])
    if match is None:
        return False
    if match.group(1):
        label = f"{match.group(1).upper()}{match.group(2)}"
    else:
        label = f"{match.group(3)}H{match.group(4)}"
    return label == period.upper()


def states_its_period(text: str) -> bool:
    """Whether the text says which half or year it reports ("year ended 30 June 2025")."""
    return bool(_HALF.search(text) or _FULL.search(text))


def _plain(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _printed_date(text: str) -> date | None:
    """A date as a document prints it: "31 December 2025" or "31 Dec 2025"."""
    for pattern in ("%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(" ".join(text.split()), pattern).date()
        except ValueError:
            continue
    return None


def _name_for(hints: dict[str, Any], doc_type: str, kind: str) -> str:
    """The issuer's own name for a document, for this kind of period.

    A hint can be one name, or one per kind of period (``{FY: ..., H: ...}``), since a bank
    calls its full-year and half-year filings different things.
    """
    hint = hints.get(doc_type.upper())
    if isinstance(hint, dict):
        hint = hint.get("FY" if kind == "full year" else "H")
    return str(hint) if hint else doc_type.replace("_", " ").lower()


def discover_documents(
    bank_id: str,
    period: str,
    doc_type: str,
    client: LLMClient,
    *,
    page_text_of: Any,
    # Overridden from config/models.yaml through ``Registry.route("discover_classify")``. A
    # default that differs from the routing is how that config ended up unused for the whole
    # build.
    model: str = "claude-opus-5",
    effort: str = "low",
    tracer: Tracer | None = None,
    prompts: PromptLibrary | None = None,
    max_rounds: int = MAX_ROUNDS,
    budget: Budget | None = None,
) -> DiscoveryOutcome:
    """Find and confirm a document, or say why none was accepted.

    ``page_text_of`` takes a candidate URL and returns the text of its first pages. It's
    passed in so downloading stays in ``Fetcher``; discovery decides what to ask about and
    never makes a request itself.
    """
    outcome = DiscoveryOutcome()
    profile = get_registry().bank(bank_id)
    allowed = profile.ir_domains

    if not client.available:
        outcome.notes.append(
            "cold-path discovery needs a model: no ANTHROPIC_API_KEY is set, so the "
            "request abstains with DOCS_UNAVAILABLE rather than guessing a URL"
        )
        return outcome

    library = prompts or default_library()
    prompt = library.get("discover_classify")
    ends = fiscal_period(period, profile.fy_end_month).end
    kind = "full year" if period.upper().startswith("FY") else "half year"
    hint = _name_for(profile.doc_name_hints or {}, doc_type, kind)
    # Each round asks a different question. Three identical requests are three answers to
    # one question: the cache returns the first every time, and the rounds after it only
    # re-check what was already rejected.
    generic = DECK_NAMES if doc_type.upper() == _DECK else GENERIC_NAMES[kind]
    names = list(dict.fromkeys((hint, *generic)))
    seen: set[str] = set()

    for round_number in range(1, max_rounds + 1):
        outcome.rounds = round_number
        if budget is not None and not budget.may_call(
            "discover_classify", projected_usd=SEARCH_ROUND_USD
        ):
            outcome.notes.append("discovery stopped on its budget")
            return outcome
        name = names[(round_number - 1) % len(names)]
        already = sorted(seen)[:8]
        request = LLMRequest(
            step="discover_classify",
            model=model,
            effort=effort,
            system=prompt.text,
            messages=(
                {
                    "role": "user",
                    "content": (
                        f"Find {profile.name}'s {name} for the {kind} ended "
                        f"{ends.day} {ends:%B %Y} ({bank_id} {period}), as a PDF. Search the "
                        "issuer's own site. Do not write URLs in your reply; the search "
                        "results are read directly."
                        + (
                            " These were checked already and aren't it, so look past "
                            "them: " + "; ".join(already) + "."
                            if already
                            else ""
                        )
                    ),
                },
            ),
            tools=(search_tool(allowed),),
        )
        try:
            response = client.complete(request)
        except LLMUnavailable as error:
            if isinstance(error, ModelUnreachable):  # the run stops; see llm.client
                raise
            outcome.notes.append(f"discovery stopped: {error}")
            return outcome
        if budget is not None:
            budget.record("discover_classify", model, response.usage, cached=response.cached)
        if tracer is not None:
            tracer.model_call(request, response, prompt=prompt)

        candidates = urls_from_search_results(response.raw_blocks, allowed)
        if not candidates:
            failed = search_errors(response.raw_blocks)
            outcome.notes.append(
                f"round {round_number}: the search failed ({', '.join(failed)}), "
                "which is not the same as the issuer having nothing"
                if failed
                else f"round {round_number}: the search returned no results"
            )
            continue

        considered = 0
        for candidate in _most_likely_first(candidates, period=period, hint=name):
            if candidate.url in seen:
                continue  # rejected in an earlier round; checking it again changes nothing
            seen.add(candidate.url)
            if candidate.rejected_because:
                outcome.rejected.append(candidate)
                continue
            if considered >= MAX_CANDIDATES:
                break
            considered += 1
            if budget is not None and not budget.may_call("discover_classify"):
                outcome.notes.append("discovery stopped on its budget")
                return outcome
            confirmed = _confirm(
                candidate,
                bank_id=bank_id,
                period=period,
                doc_type=doc_type,
                fy_end_month=profile.fy_end_month,
                client=client,
                page_text_of=page_text_of,
                prompt=prompt,
                model=model,
                effort=effort,
                tracer=tracer,
                budget=budget,
            )
            (outcome.accepted if confirmed.confirmed else outcome.rejected).append(confirmed)
            if confirmed.confirmed:
                # One confirmed document is the answer; checking the rest costs a download
                # and a call each.
                return outcome

    outcome.notes.append(
        f"no candidate was confirmed in {outcome.rounds} round(s); "
        f"{len(outcome.rejected)} were rejected"
    )
    return outcome


def _most_likely_first(candidates: list[Candidate], *, period: str, hint: str) -> list[Candidate]:
    """Candidates in the order worth checking: PDFs naming the period or the document first.

    Only the order changes. Every candidate still has to pass the same checks, so a wrong
    guess here costs one extra download, never a wrong document.
    """
    year = f"20{period[-2:]}" if period[-2:].isdigit() else ""
    words = {w for w in re.split(r"[^a-z0-9]+", hint.lower()) if len(w) > 1}

    def score(candidate: Candidate) -> tuple[int, int]:
        url = candidate.url.lower()
        text = f"{url} {candidate.title.lower()}"
        points = 0
        points += 4 if url.split("?")[0].endswith(".pdf") else 0
        points += 2 if year and year in text else 0
        points += sum(1 for word in words if word in text)
        return (-points, len(url))

    return sorted(candidates, key=score)


def confirm_document(
    candidate: Candidate,
    *,
    bank_id: str,
    period: str,
    doc_type: str,
    client: LLMClient,
    page_text_of: Any,
    model: str,
    effort: str,
    tracer: Tracer | None = None,
    budget: Budget | None = None,
    prompts: PromptLibrary | None = None,
) -> Candidate:
    """Classify one document and cross-check the classification, however it was found: from
    a search, or from the ASX's own list of a company's announcements."""
    return _confirm(
        candidate,
        bank_id=bank_id,
        period=period,
        doc_type=doc_type,
        fy_end_month=get_registry().bank(bank_id).fy_end_month,
        client=client,
        page_text_of=page_text_of,
        prompt=(prompts or default_library()).get("discover_classify"),
        model=model,
        effort=effort,
        tracer=tracer,
        budget=budget,
    )


def _confirm(
    candidate: Candidate,
    *,
    bank_id: str,
    period: str,
    doc_type: str,
    fy_end_month: int,
    client: LLMClient,
    page_text_of: Any,
    prompt: Any,
    model: str,
    effort: str,
    tracer: Tracer | None,
    budget: Budget | None = None,
) -> Candidate:
    """Classify a candidate's own first pages, then cross-check the classification."""
    try:
        text = page_text_of(candidate.url)
    except Exception as error:  # a candidate that cannot be read is simply not accepted
        return Candidate(
            url=candidate.url, title=candidate.title, rejected_because=f"unreadable: {error}"
        )

    request = LLMRequest(
        step="discover_classify",
        model=model,
        effort=effort,
        system=prompt.text,
        messages=(
            {
                "role": "user",
                # Wrapped, and the prompt says document text is evidence, not instructions.
                # The real protection isn't the wrapper, though: this call has no tools, so
                # whatever the pages say, all it can produce is a verdict.
                "content": (
                    "Classify this document from its first pages.\n\n"
                    f"<document_content>\n{text[:12000]}\n</document_content>"
                ),
            },
        ),
        output_schema=classification_schema(),
    )
    try:
        response = client.complete(request)
    except LLMUnavailable as error:
        if isinstance(error, ModelUnreachable):  # the run stops; see llm.client
            raise
        return Candidate(
            url=candidate.url, title=candidate.title, rejected_because=f"not classified: {error}"
        )
    if budget is not None:
        budget.record("discover_classify", model, response.usage, cached=response.cached)
    if tracer is not None:
        tracer.model_call(request, response, prompt=prompt)

    claimed = response.structured or {}
    agrees, why = classification_agrees(
        claimed, text, bank_id=bank_id, period=period, fy_end_month=fy_end_month, doc_type=doc_type
    )
    if not agrees:
        return Candidate(url=candidate.url, title=candidate.title, rejected_because=why)
    if str(claimed.get("doc_type") or "").upper() not in _ACCEPTED_AS.get(
        (doc_type.upper(), period.upper().startswith("FY")), {doc_type.upper()}
    ):
        return Candidate(
            url=candidate.url,
            title=candidate.title,
            rejected_because=f"reads as {claimed.get('doc_type')!r}, not {doc_type}",
        )

    return Candidate(
        url=candidate.url,
        title=candidate.title,
        bank_id=bank_id,
        doc_type=doc_type.upper(),
        period=period,
        publish_date=_as_date(claimed.get("publish_date")),
        confirmed=True,
    )


#: What may stand in for a requested document type, by (type, full year). A bank without a
#: separate profit announcement publishes its full-year results in its annual report, with
#: the Appendix 4E: Bank of Queensland's FY25 filing is "25-annual-report.pdf", and asking
#: for a "results document" turned it down. For a half year, only the results document will
#: do.
_ACCEPTED_AS: dict[tuple[str, bool], set[str]] = {
    ("RESULTS_DOCUMENT", True): {"RESULTS_DOCUMENT", "ANNUAL_REPORT"},
}


def _as_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None
