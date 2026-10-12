"""Finding a listed bank's filing in the ASX's own list of its announcements."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from bankresearch.acquisition.asx_announcements import (
    Announcement,
    AsxListing,
    likely_filings,
    years_to_read,
)


def _row(day: str, headline: str, pages: int, ids: str) -> str:
    return f"""
    <tr>
        <td>{day}<br><span class="dates-time">8:34 am</span></td>
        <td class="pricesens"></td>
        <td>
            <a target="_blank" href="/asx/v2/statistics/displayAnnouncement.do?display=pdf&amp;idsId={ids}">
                {headline}<br>
                <img src="/asx/v2/markets/image/pdf_icon.png">
                <span class="page">{pages}
                    page</span>
                <span class="filesize">3.4MB</span>
            </a>
        </td>
    </tr>"""


_LIST = (
    '<table summary="Most recent company announcements"><tbody>'
    + _row("06/10/2026", "Date of 2026 Annual General Meeting", 1, "03148864")
    + _row("27/08/2026", "Appendix 4G &amp; FY26 Corporate Governance Statement", 26, "03130971")
    + _row("27/08/2026", "FY26 Financial Results Investor Presentation", 30, "03130969")
    + _row("27/08/2026", "FY26 Financial Results Release", 3, "03130967")
    + _row("27/08/2026", "Appendix 4E &amp; FY26 Annual Financial Report", 77, "03130963")
    + _row("05/03/2026", "Coffee Microcaps Investor Presentation", 21, "03068727")
    + _row("26/02/2026", "Appendix 4D &amp; Interim Financial Report for H1 FY26", 23, "03065356")
    + "</tbody></table>"
)
_PDF = "https://announcements.asx.com.au/asxpdf/20260827/pdf/073995g4tyv3gt.pdf"


def _pages(url: str) -> str:
    if "announcements.do" in url:
        return _LIST
    return f'<form><input name="pdfURL" value="{_PDF}" type="hidden"></form>'


def test_the_list_is_read_with_each_release_date_headline_and_length():
    listed = AsxListing(get=_pages).announcements("BBC", 2026)
    report = next(item for item in listed if item.headline.startswith("Appendix 4E"))
    assert report == Announcement(
        released=date(2026, 8, 27),
        headline="Appendix 4E & FY26 Annual Financial Report",
        pages=77,
        link="https://www.asx.com.au/asx/v2/statistics/displayAnnouncement.do?display=pdf&idsId=03130963",
    )


@pytest.mark.parametrize(
    ("doc_type", "period", "end", "full_year", "expected"),
    [
        (
            "RESULTS_DOCUMENT",
            "FY26",
            date(2026, 6, 30),
            True,
            "Appendix 4E & FY26 Annual Financial Report",
        ),
        (
            "INVESTOR_PRESENTATION",
            "FY26",
            date(2026, 6, 30),
            True,
            "FY26 Financial Results Investor Presentation",
        ),
        (
            "RESULTS_DOCUMENT",
            "1H26",
            date(2025, 12, 31),
            False,
            "Appendix 4D & Interim Financial Report for H1 FY26",
        ),
    ],
)
def test_the_filing_wanted_comes_first(doc_type, period, end, full_year, expected):
    listed = AsxListing(get=_pages).announcements("BBC", 2026)
    found = likely_filings(
        listed,
        doc_type,
        full_year=full_year,
        period_end=end,
        as_of=date(2026, 10, 11),
        period=period,
    )
    assert found[0].headline == expected
    # Never the governance statement, the meeting notice or a conference deck.
    assert not any("4G" in item.headline or "Microcaps" in item.headline for item in found)


def test_a_filing_released_after_the_as_of_date_is_never_offered():
    listed = AsxListing(get=_pages).announcements("BBC", 2026)
    found = likely_filings(
        listed,
        "RESULTS_DOCUMENT",
        full_year=True,
        period_end=date(2026, 6, 30),
        as_of=date(2026, 8, 26),
        period="FY26",
    )
    assert found == []


def test_a_short_results_announcement_goes_after_the_report():
    """MyState's four-page "Full Year Results Announcement" is the release; its 87-page
    "Preliminary Final Report" is the filing."""
    listed = [
        Announcement(date(2025, 8, 19), "MYS FY25 Full Year Results Announcement", 4, "x"),
        Announcement(date(2025, 9, 19), "Annual Report to shareholders", 124, "y"),
        Announcement(date(2025, 8, 19), "Preliminary Final Report", 87, "z"),
    ]
    found = likely_filings(
        listed,
        "RESULTS_DOCUMENT",
        full_year=True,
        period_end=date(2025, 6, 30),
        as_of=date(2026, 1, 1),
        period="FY25",
    )
    assert [item.headline for item in found] == [
        "Preliminary Final Report",
        "Annual Report to shareholders",
        "MYS FY25 Full Year Results Announcement",
    ]


def test_only_the_asx_is_read_and_only_its_announcements_site_downloaded():
    listing = AsxListing(get=_pages)
    with pytest.raises(ValueError, match=r"not on www\.asx\.com\.au"):
        listing._get("https://evil.example/announcements.do")
    item = Announcement(date(2026, 8, 27), "Appendix 4E", 77, "https://www.asx.com.au/x")
    assert listing.pdf_url(item) == _PDF
    elsewhere = AsxListing(
        get=lambda url: '<input name="pdfURL" value="https://evil.example/a.pdf">'
    )
    assert elsewhere.pdf_url(item) is None


def test_a_half_year_ending_in_december_is_listed_the_next_year():
    assert years_to_read(date(2025, 12, 31), date(2026, 10, 11)) == [2025, 2026]
    assert years_to_read(date(2026, 6, 30), date(2026, 10, 11)) == [2026]


def test_the_tests_never_reach_the_asx():
    with pytest.raises(OSError, match="don't reach"):
        AsxListing().announcements("BBC", 2026)


def test_a_filing_on_the_list_is_found_without_a_search_and_dated_by_the_asx(tmp_path, monkeypatch):
    from bankresearch.acquisition.fetch import FetchResult
    from bankresearch.agent.orchestrator import _found_profile, _register
    from bankresearch.agent.steps.acquire import acquire_documents
    from bankresearch.domain.registry import get_registry
    from bankresearch.llm.client import LLMResponse, ScriptedClient
    from bankresearch.memory.db import connect
    from bankresearch.memory.fact_store import FactStore
    from bankresearch.settings import Settings

    _register(_found_profile("BBC", "BNK Banking Corporation Limited", 6, (), ()))
    try:
        settings = Settings(
            config_dir=Path(__file__).parents[2] / "config",
            data_dir=tmp_path / "data",
            runs_dir=tmp_path / "runs",
            evals_dir=tmp_path / "evals",
        )
        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(b"%PDF-")
        downloaded: list[str] = []

        def fetch(url: str) -> FetchResult:
            downloaded.append(url)
            return FetchResult(url, url, "abc", pdf, 5, False)

        ingested: list[str] = []
        monkeypatch.setattr(
            "bankresearch.agent.steps.acquire.ingest_document",
            lambda path, meta, store, processed_root=None: (
                ingested.append(meta.doc_id)
                or type("Report", (), {"facts_stored": 1, "facts_quarantined": 0, "checks": []})()
            ),
        )
        client = ScriptedClient(
            responses=[
                LLMResponse(
                    structured={
                        "issuer": "BNK Banking Corporation Limited",
                        "doc_type": "RESULTS_DOCUMENT",
                        "period_kind": "full_year",
                        "period_end": "2026-06-30",
                        "publish_date": None,
                    }
                )
            ]
        )
        acquired = acquire_documents(
            "BBC",
            "FY26",
            date(2026, 10, 11),
            client,
            FactStore(connect(tmp_path / "db.sqlite")),
            settings,
            fetch=fetch,
            read_first_pages=lambda path: (
                "BNK Banking Corporation Limited Appendix 4E for the year ended 30 June 2026"
            ),
            asx=AsxListing(get=_pages),
        )
        assert downloaded == [_PDF] and ingested == ["bbc-fy26-results-document"]
        assert len(client.requests) == 1 and not client.requests[0].tools  # no search at all
        assert any("released 2026-08-27, in the ASX's list" in note for note in acquired.notes)
    finally:
        get_registry().banks.pop("BBC", None)


@pytest.mark.parametrize(
    ("doc_type", "period", "end", "listed", "expected"),
    [
        # Westpac's half is an "Interim Financial Results Announcement", and its US copy on
        # Form 6-K is lodged the same day at the same length.
        (
            "RESULTS_DOCUMENT",
            "1H24",
            date(2024, 3, 31),
            [
                ("US Interim Financial Results Announcement on Form 6-K", 141),
                ("Westpac 2024 Interim Financial Results Announcement", 141),
                ("Westpac 2024 Half Year Result", 6),
            ],
            "Westpac 2024 Interim Financial Results Announcement",
        ),
        (
            "RESULTS_DOCUMENT",
            "1H26",
            date(2025, 12, 31),
            [
                ("MyState Limited HY26 Results Announcement", 5),
                ("Half Yearly Report and Accounts", 29),
            ],
            "Half Yearly Report and Accounts",
        ),
        (
            "INVESTOR_PRESENTATION",
            "1H25",
            date(2025, 2, 28),
            [("Half Year Results Announcement", 4), ("1H25 Investor Materials", 67)],
            "1H25 Investor Materials",
        ),
    ],
)
def test_each_banks_own_name_for_its_filing_is_found(doc_type, period, end, listed, expected):
    released = end + timedelta(days=40)
    announcements = [Announcement(released, headline, pages, "x") for headline, pages in listed]
    found = likely_filings(
        announcements,
        doc_type,
        full_year=period.startswith("FY"),
        period_end=end,
        as_of=date(2026, 10, 12),
        period=period,
    )
    assert found[0].headline == expected
    assert not any("Form 6-K" in item.headline for item in found)
