"""A headline struck before tax: Judo's profit before tax, which it has led with since 1H26."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from bankresearch.domain.models import DocMeta, Fact
from bankresearch.ingestion.facts import extract_facts
from bankresearch.ingestion.parse_pdf import ParsedDocument, ParsedPage, ParsedTable
from bankresearch.ingestion.validate import validate_facts

PUB = date(2026, 8, 18)

#: Judo's 2026 annual report p.33, "2.1 Income statement", the Jun-26 and Jun-25 columns.
INCOME_STATEMENT = (
    ("Interest income", "1,179.5", "1,059.9"),
    ("Interest expense", "(678.9)", "(652.6)"),
    ("Net interest income", "500.6", "407.3"),
    ("Other operating income", "21.8", "15.6"),
    ("Net banking income", "522.4", "422.9"),
    ("Employee benefits expense", "(144.4)", "(132.8)"),
    ("Other expenses", "(92.2)", "(89.0)"),
    ("only", "", ""),
    ("Total operating expenses", "(236.6)", "(221.8)"),
    ("Profit before impairment", "285.8", "201.1"),
    ("Impairment expense", "(117.7)", "(75.5)"),
    ("use", "", ""),
    ("Profit before tax", "168.1", "125.6"),
    ("Tax expense", "(57.0)", "(39.2)"),
    ("Profit after tax", "111.1", "86.4"),
    ("personal", "", ""),
    ("For", "", ""),
)


@pytest.fixture(scope="module")
def facts() -> list[Fact]:
    meta = DocMeta(
        doc_id="jdo-fy26-annual-report",
        bank_id="JDO",
        period="FY26",
        doc_type="ANNUAL_REPORT",
        publish_date=PUB,
        url="https://www.judo.bank/example.pdf",
        page_count=174,
    )
    table = ParsedTable(
        table_id="jdo_p33_L0",
        page_no=33,
        caption="2.1 Income statement",
        header=("$m", "Jun-26", "Jun-25"),
        rows=INCOME_STATEMENT,
        units="aud_m",
    )
    document = ParsedDocument(
        doc_id=meta.doc_id,
        sha256="0" * 64,
        page_count=174,
        pages=(ParsedPage(page_no=33, text="2. Analyst Pack", kind="text", tables=(table,)),),
    )
    outcome = validate_facts(
        extract_facts(document, meta, ("FY26", "FY25")), "JDO", ("FY26", "FY25")
    )
    assert not outcome.quarantined
    return outcome.validated


def _value(facts: list[Fact], component: str, period: str) -> Decimal | None:
    return next((f.value for f in facts if f.component == component and f.period == period), None)


# --- reading it -------------------------------------------------------------------------


def test_both_profits_are_read_and_each_bridge_adds_up(facts):
    assert _value(facts, "profit_before_tax", "FY26") == Decimal("168.1")
    assert _value(facts, "profit_before_tax", "FY25") == Decimal("125.6")
    assert _value(facts, "headline_earnings", "FY26") == Decimal("111.1")
    assert _value(facts, "headline_earnings", "FY25") == Decimal("86.4")


def test_the_pre_tax_bridge_is_checked_on_its_own():
    """Not just stored: a profit before tax that the lines above it don't add up to is
    refused, as an after-tax one is."""
    meta = DocMeta(
        doc_id="jdo-fy26-annual-report",
        bank_id="JDO",
        period="FY26",
        doc_type="ANNUAL_REPORT",
        publish_date=PUB,
        url="https://www.judo.bank/example.pdf",
        page_count=174,
    )
    rows = tuple(
        (label, "178.1", jun25) if label == "Profit before tax" else (label, jun26, jun25)
        for label, jun26, jun25 in INCOME_STATEMENT
    )
    table = ParsedTable(
        table_id="jdo_p33_L0",
        page_no=33,
        caption="",
        header=("$m", "Jun-26", "Jun-25"),
        rows=rows,
        units="aud_m",
    )
    document = ParsedDocument(
        doc_id=meta.doc_id,
        sha256="0" * 64,
        page_count=174,
        pages=(ParsedPage(page_no=33, text="", kind="text", tables=(table,)),),
    )
    outcome = validate_facts(
        extract_facts(document, meta, ("FY26", "FY25")), "JDO", ("FY26", "FY25")
    )
    failed = {(c.name, c.period) for c in outcome.failed_checks}
    assert ("profit_before_tax", "FY26") in failed
    assert ("profit_before_tax", "FY25") not in failed


def test_another_banks_pre_tax_line_is_still_refused():
    """ "Before tax" stays an exclusion for every bank that doesn't lead with it: Westpac's
    "Profit before income tax expense" is a line of its bridge, not its headline."""
    from bankresearch.ingestion.facts import line_item_index

    assert line_item_index("WBC").components_for("Profit before income tax expense") == ()
    assert line_item_index("WBC").components_for("Profit before tax") == ()
    assert line_item_index("JDO").components_for("Profit before tax") == ("profit_before_tax",)
    assert line_item_index("JDO").components_for("Underlying profit before tax") == ()


# --- asking about it --------------------------------------------------------------------


def _spec(period: str, metric: str = "headline_earnings"):  # type: ignore[no-untyped-def]
    from bankresearch.agent.steps.resolve import resolve_request

    return resolve_request("JDO", period, metric, as_of=date(2026, 9, 30))


def test_judos_earnings_are_asked_before_tax_from_its_2026_half_year():
    for period in ("1H26", "FY26"):
        spec = _spec(period)
        assert spec.pre_tax and spec.measure == "profit_before_tax"
        assert spec.headline_label == "Profit before tax"
    earlier = _spec("FY25")
    assert not earlier.pre_tax and earlier.measure == "headline_earnings"
    # Return on equity still rests on the after-tax profit.
    assert not _spec("FY26", "roe").pre_tax


def test_the_reader_is_told_both_periods_are_compared_before_tax():
    from bankresearch.agent.steps.resolve import caveats_for

    caveats = caveats_for(_spec("FY26"))
    assert any("both periods are compared before tax" in caveat for caveat in caveats)
    # FY25's own headline was after tax, but this comparison isn't across that change.
    assert not any("spans a change in the headline measure" in caveat for caveat in caveats)


def test_the_headline_is_profit_before_tax_in_both_periods(facts):
    from bankresearch.agent.steps.extract import cards_from_facts
    from bankresearch.agent.steps.reconcile import reconcile_inputs

    spec = _spec("FY26")
    cards = cards_from_facts(facts, spec)
    inputs = reconcile_inputs(cards, facts, spec)
    assert inputs.pair("headline_earnings") == (Decimal("125.6"), Decimal("168.1"))
    assert "tax" not in inputs.current and "tax" not in inputs.prior
    cited = {card.evidence_id: card for card in cards}
    for period in ("FY26", "FY25"):
        sources = [cited[e] for e in inputs.citations_for("headline_earnings", period)]
        assert sources and all(card.component == "profit_before_tax" for card in sources)


def test_return_on_equity_still_reads_the_after_tax_profit(facts):
    from bankresearch.agent.steps.extract import cards_from_facts
    from bankresearch.agent.steps.reconcile import reconcile_inputs

    spec = _spec("FY26", "roe")
    inputs = reconcile_inputs(cards_from_facts(facts, spec), facts, spec)
    assert inputs.pair("headline_earnings") == (Decimal("86.4"), Decimal("111.1"))


def test_the_bridge_stops_at_profit_before_tax(facts):
    from bankresearch.agent.steps.decompose import run_decomposition
    from bankresearch.agent.steps.extract import cards_from_facts
    from bankresearch.agent.steps.reconcile import reconcile_inputs

    spec = _spec("FY26")
    inputs = reconcile_inputs(cards_from_facts(facts, spec), facts, spec)
    decomposition, _ = run_decomposition(spec, inputs, [])
    top = {d.driver_id: d.contribution for d in decomposition.drivers if d.parent_id is None}
    assert top == {
        "earn.nii": Decimal("93.3"),
        "earn.other_income": Decimal("6.2"),
        "earn.operating_expenses": Decimal("-14.8"),
        "earn.credit_impairment": Decimal("-42.2"),
    }
    assert decomposition.headline_change == Decimal("42.5")
    assert decomposition.residual == 0


def test_only_an_earnings_measure_can_be_struck_before_tax(tmp_path):
    import shutil

    import yaml

    from bankresearch.domain.registry import _build
    from bankresearch.settings import get_settings

    config = tmp_path / "config"
    shutil.copytree(get_settings().config_dir, config)
    metrics = yaml.safe_load((config / "metrics.yaml").read_text(encoding="utf-8"))
    metrics["metrics"]["nim"]["bank_labels"]["JDO"] = [{"label": "NIM", "pre_tax": True}]
    (config / "metrics.yaml").write_text(yaml.safe_dump(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="pre_tax is only meaningful"):
        _build(config)


def test_a_quote_names_the_row_the_figure_came_from_when_two_rows_print_it():
    """Judo's FY25 underlying and statutory profit before tax were both $125.6m (2025
    annual report p.44). The statutory figure was quoted under the underlying row's name."""
    from bankresearch.agent.steps.extract import quote_for

    meta = DocMeta(
        doc_id="jdo-fy25-results-document",
        bank_id="JDO",
        period="FY25",
        doc_type="ANNUAL_REPORT",
        publish_date=date(2025, 8, 19),
        url="https://www.judo.bank/example.pdf",
        page_count=156,
    )
    table = ParsedTable(
        table_id="jdo_p44_L0",
        page_no=44,
        caption="",
        header=("$m", "Jun-25", "Jun-24"),
        rows=(
            ("Underlying profit before tax1", "125.6", "110.1"),
            ("Non-recurring costs", "\u2013", "(5.8)"),
            ("Statutory profit before tax", "125.6", "104.3"),
        ),
        units="aud_m",
    )
    document = ParsedDocument(
        doc_id=meta.doc_id,
        sha256="0" * 64,
        page_count=156,
        pages=(ParsedPage(page_no=44, text="", kind="text", tables=(table,)),),
    )
    found = [
        f
        for f in extract_facts(document, meta, ("FY25", "FY24"))
        if f.component == "profit_before_tax" and f.period == "FY25"
    ]
    assert len(found) == 1
    assert quote_for(found[0], {meta.doc_id: document}) == (
        "Statutory profit before tax | Jun-25 | 125.6"
    )
