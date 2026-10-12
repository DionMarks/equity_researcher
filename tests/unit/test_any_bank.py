"""A question about any Australian bank: configured, a brand, listed but unread, or none.

Before this, a bank outside config/banks.yaml was refused, and "UP" (Up, a brand of
Bendigo and Adelaide Bank) crashed the renderer looking it up.
"""

from __future__ import annotations

from datetime import date

import pytest
from typer.testing import CliRunner

from bankresearch.acquisition.bank_directory import find_bank
from bankresearch.acquisition.identify import find_year_end
from bankresearch.domain.registry import get_registry
from bankresearch.llm.client import LLMResponse, Usage
from bankresearch.settings import get_settings


def _find(asked: str, **kwargs):  # type: ignore[no-untyped-def]
    known = {b: p.name for b, p in get_registry().banks.items() if not p.synthetic}
    return find_bank(asked, config_dir=get_settings().config_dir, known=known, **kwargs)


@pytest.mark.parametrize(
    ("asked", "kind", "code"),
    [
        ("Up", "brand", "BEN"),
        ("Bankwest", "brand", "CBA"),
        ("ubank", "brand", "NAB"),
        ("Suncorp Bank", "brand", "ANZ"),
        ("BNK", "listed_bank", "BBC"),
        ("Macquarie", "listed_bank", "MQG"),
        ("BSP", "overseas", "BFL"),
        ("Telstra", "listed_not_bank", "TLS"),
        ("Great Southern Bank", "unlisted", None),
        # A bank's own name before an ASX code that spells it: Inghams' code is ING.
        ("ING", "unlisted", None),
    ],
)
def test_a_name_is_placed_among_australias_banks(asked, kind, code):
    match = _find(asked)
    assert (match.kind, match.code) == (kind, code)
    assert match.answerable is (kind in {"brand", "listed_bank"})
    assert match.note or match.message or kind == "listed_bank"


def test_a_name_that_is_no_bank_says_so_and_suggests_the_closest():
    match = _find("Bank of Sidney")
    assert match.kind == "unknown"
    assert "Bank of Sydney" in match.message


def test_a_name_missing_from_the_copy_held_is_looked_up_on_the_live_list(tmp_path):
    fresh = 'ASX listed companies\n\nCompany name,ASX code,GICS industry group\n"NEWLY LISTED BANK LIMITED","NLB","Banks"\n'
    assert _find("Newly Listed Bank").kind == "unknown"
    assert _find("Newly Listed Bank", data_dir=tmp_path, fetch=lambda: fresh).code == "NLB"


def test_a_list_that_cant_be_fetched_falls_back_without_failing(tmp_path):
    def no_network() -> str:
        raise OSError("no network")

    assert _find("BNK", data_dir=tmp_path, fetch=no_network).code == "BBC"
    assert _find("Not A Bank At All", data_dir=tmp_path, fetch=no_network).kind == "unknown"


class _Searches:
    """A model that searched the ASX and replied with ``reply``."""

    available = True

    def __init__(self, reply: str, url: str = "https://www.asx.com.au/markets/company/BBC"):
        self.reply, self.url = reply, url

    def complete(self, request):  # type: ignore[no-untyped-def]
        blocks = (
            {
                "type": "web_search_tool_result",
                "content": [{"type": "web_search_result", "url": self.url, "title": "BNK"}],
            },
        )
        return LLMResponse(text=self.reply, raw_blocks=blocks, usage=Usage(input_tokens=10))


def test_a_banks_year_end_comes_from_its_asx_announcements():
    reply = '{"found": true, "fy_end_month": 6, "website": "www.bnk.com.au", "why": "Appendix 4E for the year ended 30 June 2025"}'
    year = find_year_end(
        "BNK Banking Corporation Limited", "BBC", _Searches(reply), model="m", effort="medium"
    )
    assert (year.fy_end_month, year.website) == (6, "bnk.com.au")


def test_a_year_end_the_search_didnt_settle_is_not_guessed():
    unsettled = '{"found": false, "why": "no annual report in the results"}'
    year = find_year_end(
        "BNK Banking Corporation Limited", "BBC", _Searches(unsettled), model="m", effort="medium"
    )
    assert year.fy_end_month is None
    off_site = '{"found": true, "fy_end_month": 6, "website": null, "why": "x"}'
    year = find_year_end(
        "BNK Banking Corporation Limited",
        "BBC",
        _Searches(off_site, url="https://www.marketindex.com.au/asx/bbc"),
        model="m",
        effort="medium",
    )
    assert year.fy_end_month is None, "a year end with no ASX page behind it isn't taken"


def test_a_website_that_isnt_the_banks_own_is_not_trusted():
    reply = '{"found": true, "fy_end_month": 6, "website": "marketindex.com.au", "why": "x"}'
    year = find_year_end(
        "BNK Banking Corporation Limited", "BBC", _Searches(reply), model="m", effort="medium"
    )
    assert year.fy_end_month == 6 and year.website is None


# --- what a run says before it starts -----------------------------------------------


@pytest.fixture
def agent():  # type: ignore[no-untyped-def]
    """An agent on the ingested corpus that remembers no bank found in a live run.

    The corpus is needed for the saved-data paths. A bank a live run found and kept is not:
    once BNK Banking Corporation had been read for real, "a bank not read before" no longer
    meant BNK on that machine, and these tests failed there and nowhere else.
    """
    from bankresearch.agent.orchestrator import Agent
    from bankresearch.memory.db import connect
    from bankresearch.memory.fact_store import FactStore

    class Forgetful(FactStore):
        def discovered_banks(self) -> list[dict[str, object]]:
            return []

    settings = get_settings()
    if not settings.db_path.exists():
        pytest.skip("no ingested corpus: run `make ingest` first")
    settings = settings.model_copy(update={"anthropic_api_key": None})
    banks = get_registry().banks
    for bank_id in [bank_id for bank_id, profile in banks.items() if profile.discovered]:
        banks.pop(bank_id)
    return Agent(store=Forgetful(connect(settings.db_path)), settings=settings, persist=False)


def test_the_estimate_says_whether_the_question_is_in_the_saved_data(agent):
    saved = agent.estimate("NAB", "1H26", "nim")
    assert saved.path == "saved" and not saved.outside_saved_data
    assert saved.cost is not None and saved.cost.ceiling_usd > 0
    outside = agent.estimate("CBA", "1H24", "cet1")
    assert outside.path == "fetch" and outside.outside_saved_data
    assert any("external calls" in line for line in outside.lines())


def test_a_brand_is_estimated_as_its_owner_and_says_so(agent):
    estimate = agent.estimate("UP", "FY25", "cash earnings")
    assert estimate.heading.startswith("Bendigo and Adelaide Bank")
    assert any("brand of Bendigo" in note for note in estimate.notes)


def test_an_unread_listed_bank_is_estimated_and_anything_else_stops(agent):
    assert agent.estimate("BNK", "FY25", "nim").path == "new_bank"
    telstra = agent.estimate("Telstra", "FY25", "nim")
    assert telstra.path is None and "not as a bank" in (telstra.stop or "")


def test_a_bank_met_in_an_earlier_run_says_why_nothing_is_looked_up(tmp_path):
    """BNK, once found and remembered, is estimated like a configured bank. The estimate
    says so, or the missing search of the ASX reads like a mistake."""
    from pathlib import Path

    from bankresearch.agent.orchestrator import Agent, _found_profile, _register
    from bankresearch.domain.registry import get_registry
    from bankresearch.memory.db import connect
    from bankresearch.memory.fact_store import FactStore
    from bankresearch.settings import Settings

    settings = Settings(
        config_dir=Path(__file__).parents[2] / "config",
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        evals_dir=tmp_path / "evals",
    )
    _register(_found_profile("BBC", "BNK Banking Corporation Limited", 6, (), ()))
    try:
        agent = Agent(store=FactStore(connect(settings.db_path)), settings=settings, persist=False)
        estimate = agent.estimate("BBC", "FY26", "nim")
        assert estimate.path != "new_bank"
        assert any("found on the ASX in an earlier run" in note for note in estimate.notes)
        configured = agent.estimate("NAB", "1H26", "nim")
        assert not any("found on the ASX" in note for note in configured.notes)
    finally:
        get_registry().banks.pop("BBC", None)


def test_an_unread_listed_bank_without_a_model_is_told_what_it_needs(agent):
    result = agent.run("BNK", "FY25", "nim").result
    assert result.status == "ABSTAIN" and result.abstain_reason == "BANK_UNKNOWN"
    assert "needs the model" in result.caveats[0]


def test_an_unread_listed_bank_is_found_registered_and_resolved(agent, monkeypatch):
    from bankresearch.acquisition.identify import YearEnd
    from bankresearch.agent import orchestrator
    from bankresearch.llm.budget import Budget

    class Present:
        available = True

    monkeypatch.setattr(agent, "_model", lambda: Present())
    monkeypatch.setattr(orchestrator, "find_year_end", lambda *a, **k: YearEnd(fy_end_month=6))
    registry = get_registry()
    try:
        spec, signal, notes = agent._resolve("BNK", "FY25", "nim", None, budget=Budget())
        assert signal is None and spec is not None
        assert (spec.bank_id, spec.period_end) == ("BBC", date(2025, 6, 30))
        assert registry.banks["BBC"].discovered
        assert any("found on the ASX" in note for note in notes)
    finally:
        registry.banks.pop("BBC", None)


# --- asking before spending ---------------------------------------------------------


def _run_cli(monkeypatch, *arguments: str, key: bool):  # type: ignore[no-untyped-def]
    from pydantic import SecretStr

    from bankresearch import cli

    if not get_settings().db_path.exists():
        pytest.skip("no ingested corpus: run `make ingest` first")
    monkeypatch.setattr(
        cli,
        "_with_the_key",
        lambda settings: settings.model_copy(
            update={"anthropic_api_key": SecretStr("test-key") if key else None}
        ),
    )
    return CliRunner().invoke(cli.app, ["run", *arguments])


def test_a_run_with_nobody_to_ask_spends_nothing(monkeypatch):
    result = _run_cli(monkeypatch, "--bank", "NAB", "--period", "1H26", "--metric", "nim", key=True)
    assert result.exit_code == 1
    assert "In the saved data" in result.output
    assert "needs approval" in result.output


def test_a_run_with_no_key_says_how_to_give_it_one(monkeypatch):
    result = _run_cli(
        monkeypatch, "--bank", "NAB", "--period", "1H26", "--metric", "nim", key=False
    )
    assert result.exit_code == 1
    assert "no API key was found" in result.output and "make key-set" in result.output
    assert "--offline" not in result.output


def test_a_question_that_cant_be_answered_stops_before_any_cost(monkeypatch):
    result = _run_cli(
        monkeypatch, "--bank", "Telstra", "--period", "FY25", "--metric", "nim", key=True
    )
    assert result.exit_code == 2
    assert "not as a bank" in result.output and "Estimated cost" not in result.output
