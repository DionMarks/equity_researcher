"""The cold path: a bank or period the corpus doesn't hold, found, checked and ingested."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from bankresearch.llm.budget import Budget
from bankresearch.llm.client import LLMResponse, ScriptedClient, Usage

BEN_PDF_URL = "https://www.bendigoadelaide.com.au/investor-centre/2026-appendix-4d.pdf"
BEN_TEXT = (
    "Bendigo and Adelaide Bank Limited\nAppendix 4D\n"
    "Results for announcement to the market for the half year ended 31 December 2025\n"
)


def _search(*urls: str) -> LLMResponse:
    """A search response, with candidates only in the provider's result blocks."""
    return LLMResponse(
        text="I found the half-year report.",
        raw_blocks=(
            {
                "type": "web_search_tool_result",
                "content": [
                    {"type": "web_search_result", "url": url, "title": "Appendix 4D"}
                    for url in urls
                ],
            },
        ),
        usage=Usage(100, 50, web_searches=2),
    )


def _classified(publish_date: str | None = "2026-02-16") -> LLMResponse:
    return LLMResponse(
        structured={
            "issuer": "Bendigo and Adelaide Bank Limited",
            "doc_type": "RESULTS_DOCUMENT",
            "period_kind": "half_year",
            "period_end": "2025-12-31",
            "publish_date": publish_date,
        },
        usage=Usage(500, 40),
    )


def test_discovery_confirms_a_document_found_only_in_the_search_results():
    from bankresearch.acquisition.discover import discover_documents

    client = ScriptedClient(responses=[_search(BEN_PDF_URL), _classified()])
    budget = Budget()
    outcome = discover_documents(
        "BEN",
        "1H26",
        "RESULTS_DOCUMENT",
        client,
        page_text_of=lambda url: BEN_TEXT,
        budget=budget,
    )
    assert [c.url for c in outcome.accepted] == [BEN_PDF_URL]
    assert outcome.accepted[0].publish_date == date(2026, 2, 16)
    # The search says what to look for in the issuer's own terms, and for the exact period.
    asked = str(client.requests[0].messages)
    assert "half year ended 31 December 2025" in asked
    assert "Appendix 4D and half year financial report" in asked
    # Both calls count against the run, and so do the two searches.
    assert budget.calls == 2 and budget.usage.web_searches == 2


def test_a_url_off_the_allowlist_is_never_downloaded():
    from bankresearch.acquisition.discover import discover_documents

    def must_not_fetch(url: str) -> str:
        raise AssertionError(f"fetched {url}")

    client = ScriptedClient(responses=[_search("https://evil.example/report.pdf")] * 3)
    outcome = discover_documents(
        "BEN", "1H26", "RESULTS_DOCUMENT", client, page_text_of=must_not_fetch
    )
    assert not outcome.accepted
    assert outcome.rejected and "allowlist" in outcome.rejected[0].rejected_because


def test_a_document_that_names_another_period_is_rejected_whatever_the_model_says():
    from bankresearch.acquisition.discover import discover_documents

    client = ScriptedClient(responses=[_search(BEN_PDF_URL), _classified()] * 3)
    outcome = discover_documents(
        "BEN",
        "1H26",
        "RESULTS_DOCUMENT",
        client,
        page_text_of=lambda url: BEN_TEXT.replace("31 December 2025", "31 December 2024"),
    )
    assert not outcome.accepted
    assert "31 December 2024" in outcome.rejected[0].rejected_because


@pytest.fixture
def cold(tmp_path):
    """Empty settings and store, and a small real PDF standing in for Bendigo's report."""
    import pymupdf

    from bankresearch.memory.db import connect
    from bankresearch.memory.fact_store import FactStore
    from bankresearch.settings import Settings

    settings = Settings(
        config_dir=Path(__file__).parents[2] / "config",
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        evals_dir=tmp_path / "evals",
    )
    store = FactStore(connect(settings.db_path))
    pdf = tmp_path / "ben.pdf"
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), BEN_TEXT, fontsize=11)
    document.save(pdf)
    return settings, store, pdf


def _fetch_as(pdf: Path):
    import hashlib

    from bankresearch.acquisition.fetch import FetchResult

    content = pdf.read_bytes()

    def fetch(url: str) -> FetchResult:
        return FetchResult(
            url=url,
            final_url=url,
            sha256=hashlib.sha256(content).hexdigest(),
            path=pdf,
            bytes=len(content),
            from_cache=False,
        )

    return fetch


def _spec(as_of: date = date(2026, 5, 31)):
    from bankresearch.agent.steps.resolve import resolve_request

    return resolve_request("BEN", "1H26", "nim", as_of=as_of)


def test_the_cold_path_ingests_a_confirmed_document_for_the_run_to_use(cold):
    from bankresearch.agent.steps.acquire import acquire_documents

    settings, store, pdf = cold
    spec = _spec()
    client = ScriptedClient(responses=[_search(BEN_PDF_URL), _classified()])
    acquired = acquire_documents(
        spec.bank_id, spec.period, spec.as_of, client, store, settings, fetch=_fetch_as(pdf)
    )
    assert acquired.doc_ids == ["ben-1h26-results-document"]
    assert any(note.startswith("found and ingested") for note in acquired.notes)
    held = store.documents_for("BEN", ["1H26"], spec.as_of)
    assert [doc.doc_id for doc in held] == ["ben-1h26-results-document"]
    assert held[0].url == BEN_PDF_URL and held[0].publish_date == date(2026, 2, 16)


def test_a_document_with_no_release_date_is_not_used(cold):
    """It can't be placed in time, so it could answer an as-of question with a document
    published afterwards."""
    from bankresearch.agent.steps.acquire import acquire_documents

    settings, store, pdf = cold
    client = ScriptedClient(responses=[_search(BEN_PDF_URL), _classified(publish_date=None)])
    spec = _spec()
    acquired = acquire_documents(
        spec.bank_id, spec.period, spec.as_of, client, store, settings, fetch=_fetch_as(pdf)
    )
    assert not acquired.doc_ids
    assert any("no release date" in note for note in acquired.notes)


def test_a_document_released_after_the_as_of_date_is_not_used(cold):
    from bankresearch.agent.steps.acquire import acquire_documents

    settings, store, pdf = cold
    client = ScriptedClient(responses=[_search(BEN_PDF_URL), _classified("2026-02-16")])
    acquired = acquire_documents(
        "BEN",
        "1H26",
        date(2026, 2, 10),
        client,
        store,
        settings,
        fetch=_fetch_as(pdf),
    )
    assert not acquired.doc_ids
    assert any("isn't used" in note for note in acquired.notes)


# --- dating a filing from its sign-off ---------------------------------------------
#
# Bendigo's half-year Appendix 4D prints no release date on its cover, and the first cold
# start refused it for that. The date comes from the directors' sign-off instead, and the
# wording below follows that page, with the names left out.

_SIGNED = (
    "This report is signed in accordance with a resolution of the Board of Directors.\n"
    "A. Director\nChair\n\nB. Director\nManaging Director\n\n16 February 2026"
)


def test_the_release_date_is_read_from_the_directors_sign_off():
    from bankresearch.agent.steps.acquire import release_date_from_signoff

    assert release_date_from_signoff(_SIGNED, date(2025, 12, 31)) == date(2026, 2, 16)


def test_a_sign_off_written_without_the_board_still_counts():
    from bankresearch.agent.steps.acquire import release_date_from_signoff

    text = "Signed in accordance with a resolution of the directors.\n\n3 May 2026"
    assert release_date_from_signoff(text, date(2026, 3, 31)) == date(2026, 5, 3)


def test_a_date_before_the_period_ended_is_not_this_periods_sign_off():
    """A sign-off dated inside the period belongs to an earlier report, such as a prior
    year's accounts quoted in the notes."""
    from bankresearch.agent.steps.acquire import release_date_from_signoff

    assert release_date_from_signoff(_SIGNED, date(2026, 6, 30)) is None


def test_a_sign_off_long_after_the_period_is_not_trusted():
    from bankresearch.agent.steps.acquire import release_date_from_signoff

    assert release_date_from_signoff(_SIGNED, date(2025, 6, 30)) is None


def test_text_with_no_sign_off_gives_no_date():
    from bankresearch.agent.steps.acquire import release_date_from_signoff

    assert (
        release_date_from_signoff("Directors' report\n16 February 2026", date(2025, 12, 31)) is None
    )


def _classified_as(doc_type: str) -> LLMResponse:
    return LLMResponse(
        structured={
            "issuer": "Bank of Queensland Limited",
            "doc_type": doc_type,
            "period_kind": "full_year",
            "period_end": "2025-08-31",
            "publish_date": None,
        },
        usage=Usage(500, 40),
    )


def test_each_round_asks_a_different_question_and_skips_what_was_rejected():
    """Bank of Queensland's first live cold start sent one search three times: the cache
    answered rounds two and three with round one's results, and the same four documents
    were turned down again. Later rounds now ask for the names every filing goes by, and
    say what was already checked."""
    from bankresearch.acquisition.discover import discover_documents

    governance = "https://www.boq.com.au/content/dam/boq/files/governance-2025.pdf"
    client = ScriptedClient(
        responses=[
            _search(governance),
            _classified_as(doc_type="OTHER"),
            _search(governance),
            _search(governance),
        ]
    )
    outcome = discover_documents(
        "BOQ",
        "FY25",
        "RESULTS_DOCUMENT",
        client,
        page_text_of=lambda url: "Corporate governance statement",
    )
    asked = [str(request.messages) for request in client.requests if request.tools]
    assert len(asked) == 3
    assert "Appendix 4E and annual report" in asked[0]  # Bank of Queensland's own name
    assert len(set(asked)) == 3  # three different questions
    assert governance in asked[1]  # told what was already turned down
    # The rejected document is classified once, not once a round.
    classified = [request for request in client.requests if not request.tools]
    assert len(classified) == 1
    assert not outcome.accepted


def test_a_full_year_annual_report_stands_in_for_the_results_document():
    """Bank of Queensland publishes its full-year results in its annual report, with the
    Appendix 4E. The search found it, and it was turned down for not being a "results
    document"."""
    from bankresearch.acquisition.discover import discover_documents

    report = (
        "https://www.boq.com.au/content/dam/boq/files/shareholder-centre/2025/25-annual-report.pdf"
    )
    client = ScriptedClient(responses=[_search(report), _classified_as(doc_type="ANNUAL_REPORT")])
    outcome = discover_documents(
        "BOQ",
        "FY25",
        "RESULTS_DOCUMENT",
        client,
        page_text_of=lambda url: "Annual Report for the year ended 31 August 2025",
    )
    assert [c.url for c in outcome.accepted] == [report]
    assert outcome.accepted[0].doc_type == "RESULTS_DOCUMENT"


def test_a_half_year_still_needs_the_results_document_itself():
    from bankresearch.acquisition.discover import discover_documents

    client = ScriptedClient(
        responses=[
            _search(BEN_PDF_URL),
            LLMResponse(
                structured={
                    "issuer": "Bendigo and Adelaide Bank Limited",
                    "doc_type": "ANNUAL_REPORT",
                    "period_kind": "half_year",
                    "period_end": "2025-12-31",
                    "publish_date": "2026-02-16",
                },
                usage=Usage(500, 40),
            ),
        ]
    )
    outcome = discover_documents(
        "BEN", "1H26", "RESULTS_DOCUMENT", client, page_text_of=lambda url: BEN_TEXT, max_rounds=1
    )
    assert not outcome.accepted


def test_a_document_of_the_wrong_kind_is_turned_down_in_words():
    """The warning read "the document reads as a unknown report, not full_year"."""
    from bankresearch.acquisition.discover import _reads_as

    assert _reads_as("unknown", "full_year") == (
        "the document reads as an unknown report, not a full-year one"
    )
    assert _reads_as("half_year", "full_year") == (
        "the document reads as a half-year report, not a full-year one"
    )
    assert _reads_as(None, "half_year").startswith("the document reads as an unknown report")
