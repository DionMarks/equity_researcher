"""When a result counts as published: a filing held settles it, a question is never dated
after today, and a result due this month is looked for on the ASX's own list before anything
is paid for."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from bankresearch.acquisition.asx_announcements import Announcement
from bankresearch.agent.estimate import PATHS
from bankresearch.agent.orchestrator import Agent
from bankresearch.agent.steps import resolve as resolve_step
from bankresearch.agent.steps.availability import check_availability
from bankresearch.agent.steps.resolve import resolve_request
from bankresearch.domain.models import DocMeta
from bankresearch.llm.client import ScriptedClient
from bankresearch.memory.db import connect
from bankresearch.memory.fact_store import FactStore


@pytest.fixture
def store(tmp_path) -> FactStore:
    return FactStore(connect(tmp_path / "research.sqlite"))


def _hold(store: FactStore, bank: str, period: str, published: date) -> None:
    store.put_document(
        DocMeta(
            doc_id=f"{bank.lower()}-{period.lower()}-results-document",
            bank_id=bank,
            period=period,
            doc_type="RESULTS_DOCUMENT",
            publish_date=published,
            url="https://example.com/results.pdf",
        )
    )


def test_a_filing_held_settles_it_whatever_the_usual_date_says(store):
    """NAB released its 1H26 results on 4 May. Asked as at 8 May, the run said nothing had
    been published, because the usual release is "by 31 May"."""
    _hold(store, "NAB", "1H26", date(2026, 5, 4))
    _hold(store, "NAB", "1H25", date(2025, 5, 7))
    spec = resolve_request("NAB", "1H26", "cost_to_income", as_of=date(2026, 5, 8))
    assert check_availability(spec, store).reportable


def test_before_the_release_month_nothing_is_out(store):
    spec = resolve_request("NAB", "1H26", "cost_to_income", as_of=date(2026, 4, 15))
    availability = check_availability(spec, store)
    assert availability.reason == "PERIOD_NOT_YET_REPORTED"
    assert not availability.release_pending
    assert "usually released in May 2026" in availability.detail


def test_in_the_release_month_it_may_be_out(store):
    spec = resolve_request("NAB", "1H26", "cost_to_income", as_of=date(2026, 5, 3))
    availability = check_availability(spec, store)
    assert availability.reason == "PERIOD_NOT_YET_REPORTED"
    assert availability.release_pending
    assert "may not be out yet" in availability.detail


def test_a_question_is_never_dated_after_today(monkeypatch):
    """Bank of Queensland's FY26 question, asked on 12 October, was dated 31 October, the end
    of its usual release month, and paid to search for a result that wasn't out."""
    monkeypatch.setattr(resolve_step, "today", lambda: date(2026, 10, 12))
    assert resolve_request("BOQ", "FY26", "nim").as_of == date(2026, 10, 12)
    # A result whose month has passed keeps the end of that month.
    assert resolve_request("NAB", "1H26", "nim").as_of == date(2026, 5, 31)


class _Listing:
    """The ASX's list of a company's announcements, with nothing for the period asked."""

    def __init__(self, announcements: list[Announcement]) -> None:
        self.items = announcements
        self.asked: list[tuple[str, int]] = []

    def announcements(self, code: str, year: int) -> list[Announcement]:
        self.asked.append((code, year))
        return self.items

    def pdf_url(self, announcement: Announcement) -> str | None:
        return None


def test_a_result_due_this_month_and_not_listed_isnt_searched_for(store, tmp_path):
    """Read off the ASX's own list, for nothing: no model call and no web search."""
    from bankresearch.settings import Settings

    settings = Settings(data_dir=tmp_path / "data", runs_dir=tmp_path / "runs")
    listing = _Listing(
        [Announcement(date(2026, 8, 10), "Change of Director's Interest Notice", 2, "x")]
    )
    client = ScriptedClient()  # any call raises: this run must not make one
    agent = Agent(store=store, settings=settings, client=client, persist=False, asx=listing)  # type: ignore[arg-type]
    result = agent.run("BOQ", "FY26", "nim", as_of=date(2026, 10, 12)).result
    assert result.status == "ABSTAIN"
    assert result.abstain_reason == "PERIOD_NOT_YET_REPORTED"
    assert "hasn't been released yet" in result.caveats[0]
    assert "2026-10-12" in result.caveats[0]
    assert listing.asked and client.requests == []


def test_the_estimate_says_a_pending_result_may_cost_nothing():
    pending = PATHS["pending"]
    assert pending.usd[0] == Decimal("0.00") and pending.calls[0] == 0
    # The saved paths allow for the memo's one rewrite: Westpac's cost A$1.15 in 2 minutes.
    assert PATHS["saved"].usd[1] >= Decimal("0.80")
    assert PATHS["saved"].seconds[1] >= 122
