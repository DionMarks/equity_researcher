"""A listed company's filings, found in the ASX's own list of its announcements."""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Final
from urllib.parse import urlsplit

from .fetch import ASX_DOMAIN

__all__ = ["ASX_SITE", "Announcement", "AsxListing", "likely_filings", "years_to_read"]

ASX_SITE: Final = "www.asx.com.au"
_LIST_URL: Final = (
    "https://www.asx.com.au/asx/v2/statistics/announcements.do"
    "?by=asxCode&asxCode={code}&timeframe=Y&year={year}"
)
#: Fewer pages than this, and a "results" filing is the announcement, not the report.
_SHORT: Final = 10
#: How long after a period ends its filings can still be released and be its own.
RELEASE_WINDOW: Final = timedelta(days=150)

_ROW: Final = re.compile(r"<tr>(.*?)</tr>", re.S)
_DATE: Final = re.compile(r"(\d{2})/(\d{2})/(\d{4})")
_LINK: Final = re.compile(
    r'href="(/asx/v2/statistics/displayAnnouncement\.do\?[^"]+)"[^>]*>(.*?)<br>', re.S
)
_PAGES: Final = re.compile(r'class="page">\s*(\d+)')
_PDF_URL: Final = re.compile(r'name="pdfURL"\s+value="([^"]+)"')

#: What a headline says, by the document wanted and the kind of period.
_HEADLINES: Final[dict[tuple[str, bool], re.Pattern[str]]] = {
    ("RESULTS_DOCUMENT", True): re.compile(
        r"appendix\s*4e|annual (?:financial )?report|full[- ]year (?:financial )?"
        r"(?:report|results?)|preliminary final report|profit announcement",
        re.I,
    ),
    # Westpac files its half as an "Interim Financial Results Announcement", and MyState as
    # a "Half Yearly Report and Accounts".
    ("RESULTS_DOCUMENT", False): re.compile(
        r"appendix\s*4d|half[- ]year(?:ly)? (?:financial )?(?:report|results?)"
        r"|interim (?:financial )?(?:report|results)|\bhy\s?\d{2} results|profit announcement",
        re.I,
    ),
    # Bank of Queensland calls its slide pack "Investor Materials".
    ("INVESTOR_PRESENTATION", True): re.compile(
        r"presentation|investor (?:materials|discussion pack|pack)", re.I
    ),
    ("INVESTOR_PRESENTATION", False): re.compile(
        r"presentation|investor (?:materials|discussion pack|pack)", re.I
    ),
}
#: Headlines that name the wrong document however close they look. Westpac lodges its US
#: filings on the ASX too ("US Interim Financial Results Announcement on Form 6-K"), the same
#: length as the real one.
_NOT: Final[dict[str, re.Pattern[str]]] = {
    "RESULTS_DOCUMENT": re.compile(
        r"presentation|corporate governance|appendix\s*4g|sustainability|notice|briefing"
        r"|webcast|teleconference|call\b|form\s*(?:6-k|20-f)|pillar\s*3|disclosure statement"
        r"|email to shareholders|investor (?:materials|discussion pack|pack)",
        re.I,
    ),
    "INVESTOR_PRESENTATION": re.compile(
        r"\bagm\b|annual general meeting|conference|microcap", re.I
    ),
}
_RESULTS: Final = re.compile(
    r"results|\bFY\s?\d{2}\b|\b[12]H\s?\d{2}\b|\bH[12]\b|half|full[- ]year", re.I
)


@dataclass(frozen=True, slots=True)
class Announcement:
    """One line of the ASX's list."""

    released: date
    headline: str
    pages: int | None
    link: str
    """The ASX page that holds the PDF's address."""


@dataclass
class AsxListing:
    """Reads the ASX's list of a company's announcements. ``get`` is the only network
    access, so a test passes its own."""

    get: Callable[[str], str] | None = None

    def announcements(self, code: str, year: int) -> list[Announcement]:
        """Everything ``code`` announced in ``year``, newest first, as the ASX lists it."""
        page = self._get(_LIST_URL.format(code=code.upper(), year=year))
        found: list[Announcement] = []
        for row in _ROW.findall(page):
            when, link = _DATE.search(row), _LINK.search(row)
            if when is None or link is None:
                continue
            day, month, year_ = (int(part) for part in when.groups())
            pages = _PAGES.search(row)
            found.append(
                Announcement(
                    released=date(year_, month, day),
                    headline=" ".join(html.unescape(_strip_tags(link.group(2))).split()),
                    pages=int(pages.group(1)) if pages else None,
                    link=f"https://{ASX_SITE}{html.unescape(link.group(1))}",
                )
            )
        return found

    def pdf_url(self, announcement: Announcement) -> str | None:
        """The PDF behind an announcement, only if the ASX's own page gives one on its
        announcements domain."""
        match = _PDF_URL.search(self._get(announcement.link))
        if match is None:
            return None
        url = html.unescape(match.group(1))
        parts = urlsplit(url)
        return url if parts.scheme == "https" and parts.hostname == ASX_DOMAIN else None

    def _get(self, url: str) -> str:
        if urlsplit(url).hostname != ASX_SITE:
            raise ValueError(f"{url} is not on {ASX_SITE}")
        return (self.get or network_get)(url)


def network_get(url: str) -> str:
    """One page from the ASX's site. The test suite replaces this, so no test reaches it."""
    import httpx

    response = httpx.get(url, timeout=15.0, follow_redirects=False)
    response.raise_for_status()
    return response.text


def likely_filings(
    announcements: list[Announcement],
    doc_type: str,
    *,
    full_year: bool,
    period_end: date,
    as_of: date,
    period: str,
) -> list[Announcement]:
    """The announcements that could be the filing wanted, most likely first.

    Released after the period ended, by the as-of date and within the window a period's
    filings come out in, with a headline that names the document and doesn't name another.
    A results filing of a few pages goes last: MyState's four-page "Full Year Results
    Announcement" is the release, and its 87-page "Preliminary Final Report" is the filing.
    Then one naming the period comes first, then the earliest, which is the first full
    disclosure the market had, then the longer.
    """
    wanted = _HEADLINES.get((doc_type, full_year))
    if wanted is None:
        return []
    refused = _NOT.get(doc_type)
    latest = min(as_of, period_end + RELEASE_WINDOW)
    label = re.compile(
        rf"\b{period[:2]}\s?{period[2:]}\b|\bH{period[0]}\s?FY\s?{period[2:]}\b"
        if not full_year
        else rf"\bFY\s?(?:20)?{period[2:]}\b|\b20{period[2:]}\b",
        re.I,
    )
    candidates = [
        item
        for item in announcements
        if period_end < item.released <= latest
        and wanted.search(item.headline)
        and not (refused is not None and refused.search(item.headline))
    ]
    return sorted(
        candidates,
        key=lambda item: (
            doc_type == "RESULTS_DOCUMENT" and (item.pages or 0) < _SHORT,
            not label.search(item.headline),
            item.released,
            not _RESULTS.search(item.headline),
            -(item.pages or 0),
        ),
    )


def years_to_read(period_end: date, as_of: date) -> list[int]:
    """The calendar years a period's filings can be listed under."""
    latest = min(as_of, period_end + RELEASE_WINDOW)
    return sorted({period_end.year, latest.year})


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)
