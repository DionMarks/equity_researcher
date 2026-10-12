"""Step 3: Discover. The cold path, for a period the store holds no document for."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Final

from ...acquisition.asx_announcements import AsxListing, likely_filings, years_to_read
from ...acquisition.discover import Candidate, confirm_document, discover_documents
from ...acquisition.fetch import ASX_DOMAIN, Fetcher, FetchResult
from ...domain.calendar import fiscal_period
from ...domain.models import DocMeta, DocType
from ...domain.registry import get_registry
from ...ingestion.pipeline import ingest_document
from ...llm.budget import Budget
from ...llm.client import LLMClient
from ...llm.tracing import Tracer
from ...memory.fact_store import FactStore
from ...settings import Settings

__all__ = ["Acquired", "acquire_documents", "first_pages_text", "release_date_from_signoff"]

#: The document a run needs: the one with the income statement and the key metrics.
_DOC_TYPE: Final[DocType] = "RESULTS_DOCUMENT"
_FIRST_PAGES = 3
#: How far into a document its period statement is looked for, when the first pages lack it.
_PERIOD_PAGES = 60
_ALL_PAGES = 10_000


_MONTH_NAMES = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)
#: The board's sign-off and the date under it. An Appendix 4D often prints no release date
#: on its cover, but the directors approve the accounts the day the bank releases them and
#: the date is printed beneath their names.
_SIGNOFF = re.compile(
    r"signed in accordance with a resolution of the (?:board of )?directors"
    rf".{{0,400}}?(\d{{1,2}}\s+(?:{_MONTH_NAMES})\s+20\d\d)",
    re.IGNORECASE | re.DOTALL,
)
#: How long after the period end a sign-off can be and still be this period's.
_SIGNOFF_WINDOW_DAYS = 150


def release_date_from_signoff(text: str, period_end: date) -> date | None:
    """The date under the directors' sign-off, if one falls after the period ended."""
    for match in _SIGNOFF.finditer(text):
        try:
            signed = datetime.strptime(" ".join(match.group(1).split()), "%d %B %Y").date()
        except ValueError:
            continue
        if period_end < signed <= period_end + timedelta(days=_SIGNOFF_WINDOW_DAYS):
            return signed
    return None


@dataclass
class Acquired:
    """What the cold path found and ingested, and what it says about the rest."""

    doc_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    not_on_the_asx: bool = False
    """The ASX's list of the bank's announcements was read and holds nothing for the
    period: for a result due this month, that means it isn't out yet."""


def first_pages_text(path: Path, pages: int = _FIRST_PAGES) -> str:
    """The text a classifier needs to see: the first page, the page that says which period
    the document reports, and the rest of its first pages.

    An annual report opens with a cover, a contents page and letters. Judo's 2025 report
    says "for the year ended 30 June 2025" in its directors' report, and with only its first
    three pages to read, the classifier called it an unknown report and the pattern found
    no period: the right document was rejected. The period page goes second so the
    classifier's limit on how much it reads can't cut it off.
    """
    import pymupdf

    from ...acquisition.discover import states_its_period

    with pymupdf.open(path) as document:  # type: ignore[no-untyped-call]
        texts = [document[i].get_text() for i in range(min(pages, document.page_count))]
        if pages < _ALL_PAGES and not any(states_its_period(text) for text in texts):
            for index in range(len(texts), min(_PERIOD_PAGES, document.page_count)):
                text = document[index].get_text()
                if states_its_period(text):
                    texts.insert(1, text)
                    break
        return "\n".join(texts)


def all_pages_text(path: Path) -> str:
    """The text of every page, for finding the directors' sign-off."""
    return first_pages_text(path, pages=_ALL_PAGES)


#: Filings from the ASX's list worth downloading and classifying, most likely first.
_ASX_CANDIDATES: Final = 3


def _from_the_asx_list(
    asx: AsxListing,
    bank_id: str,
    period: str,
    period_end: date,
    as_of: date,
    doc_type: str,
    *,
    page_text_of: Callable[[str], str],
    client: LLMClient,
    budget: Budget | None,
    tracer: Tracer | None,
    model: str,
    effort: str,
    notes: list[str],
) -> tuple[Candidate | None, bool]:
    """The filing, found in the ASX's list of the bank's announcements and confirmed, dated
    the day the ASX released it, and whether the list could be read at all. No candidate
    sends the caller to the search."""
    code = get_registry().bank(bank_id).ticker
    try:
        listed = [
            item
            for year in years_to_read(period_end, as_of)
            for item in asx.announcements(code, year)
        ]
    except Exception as error:  # the list is the shortcut; the search is still there
        notes.append(
            f"the ASX's list of {code}'s announcements couldn't be read "
            f"({type(error).__name__}), so the search was used"
        )
        return None, False
    candidates = likely_filings(
        listed,
        doc_type,
        full_year=period.upper().startswith("FY"),
        period_end=period_end,
        as_of=as_of,
        period=period.upper(),
    )
    if not candidates:
        notes.append(
            f"nothing in the ASX's list of {code}'s announcements reads as its {period} "
            f"{doc_type.replace('_', ' ').lower()}"
        )
        return None, True
    for item in candidates[:_ASX_CANDIDATES]:
        if budget is not None and not budget.may_call("discover_classify"):
            notes.append("checking the ASX's list stopped on its budget")
            return None, True
        try:
            url = asx.pdf_url(item)
        except Exception as error:  # one unreadable page shouldn't stop the next
            notes.append(f"the ASX's page for {item.headline!r} couldn't be read: {error}")
            continue
        if url is None:
            notes.append(f"the ASX's page for {item.headline!r} gives no PDF to download")
            continue
        confirmed = confirm_document(
            Candidate(url=url, title=item.headline),
            bank_id=bank_id,
            period=period,
            doc_type=doc_type,
            client=client,
            page_text_of=page_text_of,
            model=model,
            effort=effort,
            tracer=tracer,
            budget=budget,
        )
        if confirmed.confirmed:
            notes.append(
                f'found "{item.headline}", released {item.released.isoformat()}, in the '
                f"ASX's list of {code}'s announcements"
            )
            return replace(confirmed, publish_date=item.released), True
        notes.append(f"rejected {url} ({item.headline}): {confirmed.rejected_because}")
    return None, True


def acquire_documents(
    bank_id: str,
    period: str,
    as_of: date,
    client: LLMClient,
    store: FactStore,
    settings: Settings,
    *,
    budget: Budget | None = None,
    tracer: Tracer | None = None,
    model: str = "claude-opus-5",
    effort: str = "low",
    fetch: Callable[[str], FetchResult] | None = None,
    read_first_pages: Callable[[Path], str] = first_pages_text,
    read_all_pages: Callable[[Path], str] = all_pages_text,
    doc_type: DocType = _DOC_TYPE,
    asx: AsxListing | None = None,
    doing: Callable[[str], None] | None = None,
    search: bool = True,
) -> Acquired:
    """Find, check, fetch and ingest the period's results document, or say why not.

    ``doc_type`` asks for another of the period's documents instead: the slide pack, for a
    bank that draws its walk only there. With ``asx``, the ASX's own list of the bank's
    announcements is read first, and the search is only used if that finds nothing.
    ``doing`` is told when a download or a read starts, the slow parts with no model call,
    so a terminal can say what the run is waiting on. ``search=False`` stops at the ASX's
    list when it could be read, for a result that may not be out yet.
    """
    say = doing or (lambda activity: None)
    profile = get_registry().bank(bank_id)
    acquired = Acquired()
    if not profile.ir_domains:
        acquired.notes.append(
            f"{bank_id} has no investor-relations site configured, so there is nowhere to look"
        )
        return acquired

    fetcher = Fetcher(raw_root=settings.raw_dir)
    # The ASX's announcements site holds every listed company's filings, so it's allowed
    # for any bank, configured or found.
    allowed = tuple(dict.fromkeys((*profile.ir_domains, ASX_DOMAIN)))
    get = fetch or (lambda url: fetcher.fetch(url, allowed))
    fetched: dict[str, FetchResult] = {}

    def page_text_of(url: str) -> str:
        say("Downloading a candidate filing")
        result = get(url)
        fetched[url] = result
        return read_first_pages(result.path)

    period_end = fiscal_period(period, profile.fy_end_month).end
    accepted: list[Candidate] = []
    list_read = False
    if asx is not None and profile.ticker:
        listed, list_read = _from_the_asx_list(
            asx,
            bank_id,
            period,
            period_end,
            as_of,
            doc_type,
            page_text_of=page_text_of,
            client=client,
            budget=budget,
            tracer=tracer,
            model=model,
            effort=effort,
            notes=acquired.notes,
        )
        accepted = [listed] if listed is not None else []
    if not accepted and list_read and not search:
        acquired.not_on_the_asx = True
        return acquired
    if not accepted:
        if list_read:
            acquired.notes.append("the bank's site and the ASX were searched instead")
        outcome = discover_documents(
            bank_id,
            period,
            doc_type,
            client,
            page_text_of=page_text_of,
            model=model,
            effort=effort,
            tracer=tracer,
            budget=budget,
        )
        acquired.notes.extend(outcome.notes)
        acquired.notes.extend(
            f"rejected {candidate.url}: {candidate.rejected_because}"
            for candidate in outcome.rejected[:5]
            if candidate.rejected_because
        )
        accepted = outcome.accepted

    for candidate in accepted:
        result = fetched.get(candidate.url)
        if result is None:
            continue
        published = candidate.publish_date
        if published is None:
            # Bendigo's 1H26 Appendix 4D prints no release date on its first pages, and
            # rejecting it threw away the right document. The board's sign-off inside it
            # gives the date the accounts were approved and released.
            published = release_date_from_signoff(read_all_pages(result.path), period_end)
            if published is not None:
                acquired.notes.append(
                    f"{candidate.url} prints no release date on its cover; the directors "
                    f"signed it off on {published.isoformat()}, which is taken as its release"
                )
        if published is None:
            acquired.notes.append(
                f"{candidate.url} states no release date and has no dated sign-off, so it "
                "can't be placed in time and isn't used"
            )
            continue
        if not period_end < published <= as_of:
            acquired.notes.append(
                f"{candidate.url} was released {published.isoformat()}, outside "
                f"{period_end.isoformat()} to {as_of.isoformat()}, so it isn't used"
            )
            continue
        meta = DocMeta(
            doc_id=f"{bank_id}-{period}-{doc_type.replace('_', '-')}".lower(),
            bank_id=bank_id,
            period=period,
            doc_type=doc_type,
            publish_date=published,
            url=candidate.url,
            sha256=result.sha256,
        )
        say("Reading the filing's tables and checking every figure")
        report = ingest_document(result.path, meta, store, processed_root=settings.processed_dir)
        acquired.doc_ids.append(meta.doc_id)
        passed = sum(1 for check in report.checks if check.passed)
        acquired.notes.append(
            f"found and ingested {meta.doc_id} from {candidate.url}: "
            f"{report.facts_stored} figure(s) stored, {report.facts_quarantined} quarantined, "
            f"{passed}/{len(report.checks)} identity checks passed"
        )
        break
    return acquired
