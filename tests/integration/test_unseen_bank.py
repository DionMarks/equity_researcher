"""An issuer outside the seed corpus, read from its own filing as the cold path would."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from bankresearch.settings import get_settings

JUDO_FY25 = "5e0447a913f3"


def _report() -> Path:
    found = sorted(get_settings().raw_dir.glob(f"{JUDO_FY25}*.pdf"))
    if not found:
        pytest.skip("Judo's 2025 annual report isn't on disk; a cold-path run downloads it")
    return found[0]


@pytest.fixture(scope="module")
def prepared():
    from bankresearch.domain.models import DocMeta
    from bankresearch.ingestion.pipeline import prepare_document

    meta = DocMeta(
        doc_id="jdo-fy25-annual-report",
        bank_id="JDO",
        period="FY25",
        doc_type="ANNUAL_REPORT",
        publish_date=date(2025, 8, 19),
        url="https://www.judo.bank/wp-content/uploads/2025/11/Judo-Bank-FY25-Annual-Report.pdf",
    )
    return prepare_document(_report(), meta)


def test_every_identity_holds_in_both_years(prepared):
    assert not prepared.quarantined
    passed = {(check.name, check.period) for check in prepared.checks if check.passed}
    assert passed == {
        (name, period)
        for name in (
            "nim",
            "cet1_ratio",
            "cost_to_income",
            "headline_earnings",
            "profit_before_tax",
        )
        for period in ("FY24", "FY25")
    }


def test_the_figures_are_the_ones_judo_published(prepared):
    values = {(f.component, f.period): f.value for f in prepared.validated}
    assert values[("nim", "FY25")] == Decimal("2.93")
    assert values[("nim", "FY24")] == Decimal("2.94")
    assert values[("cet1_ratio", "FY25")] == Decimal("13.1")
    assert values[("headline_earnings", "FY25")] == Decimal("86.4")
    assert values[("headline_earnings", "FY24")] == Decimal("69.9")
    # Statutory, not the underlying $110.1m printed two rows above it.
    assert values[("profit_before_tax", "FY25")] == Decimal("125.6")
    assert values[("profit_before_tax", "FY24")] == Decimal("104.3")
    # FY24's CEO transition and restructuring costs, below underlying profit, and a dash
    # for FY25: the bridge adds up in both years only with both.
    assert values[("notable_items", "FY24")] == Decimal("-5.8")
    assert values[("notable_items", "FY25")] == Decimal("0")


def test_the_cold_path_classifier_is_shown_where_the_report_states_its_period():
    from bankresearch.acquisition.discover import classification_agrees, states_its_period
    from bankresearch.agent.steps.acquire import first_pages_text

    text = first_pages_text(_report())
    assert states_its_period(text[:12000])
    claimed = {
        "issuer": "Judo Capital Holdings Limited",
        "doc_type": "ANNUAL_REPORT",
        "period_kind": "full_year",
        "period_end": "2025-06-30",
        "publish_date": None,
    }
    assert classification_agrees(claimed, text, bank_id="JDO", period="FY25", fy_end_month=6) == (
        True,
        "",
    )


# --- the 2026 report, which leads with profit before tax ------------------------------

JUDO_FY26 = "266a598a137b"
FY26_PUBLISHED = date(2026, 8, 18)


def _fy26_report() -> Path:
    found = sorted(get_settings().raw_dir.glob(f"{JUDO_FY26}*.pdf"))
    if not found:
        pytest.skip("Judo's 2026 annual report isn't on disk; a cold-path run downloads it")
    return found[0]


def _fy26_meta():  # type: ignore[no-untyped-def]
    from bankresearch.domain.models import DocMeta

    return DocMeta(
        doc_id="jdo-fy26-annual-report",
        bank_id="JDO",
        period="FY26",
        doc_type="ANNUAL_REPORT",
        publish_date=FY26_PUBLISHED,
        url="https://www.judo.bank/example.pdf",
    )


def test_the_2026_report_gives_both_profits_and_both_bridges_hold():
    """Its ASX copy is stamped "For personal use only" down the margin, which left a stray
    "use" over "Profit before tax" on page 33, and the row was refused with it."""
    from bankresearch.ingestion.pipeline import prepare_document

    prepared = prepare_document(_fy26_report(), _fy26_meta())
    assert not prepared.quarantined
    values = {(f.component, f.period): f.value for f in prepared.validated}
    assert values[("profit_before_tax", "FY26")] == Decimal("168.1")
    assert values[("profit_before_tax", "FY25")] == Decimal("125.6")
    assert values[("headline_earnings", "FY26")] == Decimal("111.1")
    assert values[("headline_earnings", "FY25")] == Decimal("86.4")
    assert all(check.passed for check in prepared.checks)
    passed = {(check.name, check.period) for check in prepared.checks}
    assert {("profit_before_tax", "FY26"), ("headline_earnings", "FY26")} <= passed


def test_the_2026_earnings_question_is_answered_before_tax(tmp_path):
    from bankresearch.agent.orchestrator import Agent
    from bankresearch.ingestion.pipeline import ingest_document
    from bankresearch.memory.db import connect
    from bankresearch.memory.fact_store import FactStore
    from bankresearch.settings import Settings

    report = _fy26_report()
    settings = Settings(
        config_dir=get_settings().config_dir,
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        evals_dir=tmp_path / "evals",
    ).model_copy(update={"anthropic_api_key": None})
    store = FactStore(connect(settings.db_path))
    ingest_document(report, _fy26_meta(), store, processed_root=settings.processed_dir)

    agent = Agent(store=store, settings=settings, persist=False)
    earnings = agent.run("JDO", "FY26", "headline_earnings", as_of=date(2026, 9, 30)).result
    assert earnings.status == "OK"
    assert earnings.current is not None and earnings.prior is not None
    assert (earnings.prior.value, earnings.current.value) == (Decimal("125.6"), Decimal("168.1"))
    assert earnings.current.label_at_bank == "Profit before tax"
    drivers = {d.driver_id for d in earnings.drivers}
    assert "earn.nii" in drivers and "earn.tax" not in drivers
    assert any("compared before tax" in caveat for caveat in earnings.caveats)

    # Return on equity, asked of the same report, keeps the after-tax profit.
    roe = agent.run("JDO", "FY26", "roe", as_of=date(2026, 9, 30)).result
    assert roe.current is not None and roe.current.value == Decimal("6.4")
