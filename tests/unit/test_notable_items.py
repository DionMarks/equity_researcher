"""A figure held twice: as the issuer printed its headline, and excluding notable items."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from bankresearch.domain.models import (
    Basis,
    CandidateValue,
    DocMeta,
    Fact,
    SourceRef,
)
from bankresearch.ingestion.facts import _prefer, extract_facts, line_item_index
from bankresearch.ingestion.layout_tables import Word, _label_header, header_columns
from bankresearch.ingestion.parse_pdf import ParsedDocument, ParsedPage, ParsedTable
from bankresearch.ingestion.validate import validate_facts

PUB = date(2026, 5, 4)
PERIODS = ("1H26", "1H25")
HEADER = ("March 2026", "Sept 2025", "March 2025")

#: Page 9, as reported.
REPORTED = (
    ("Net interest income", "9,771", "9,904", "9,351"),
    ("Non-interest income", "1,522", "1,567", "1,442"),
    ("Net operating income", "11,293", "11,471", "10,793"),
    ("Operating expenses", "(5,937)", "(6,491)", "(5,698)"),
    ("Impairment (charges)/benefits", "(443)", "(174)", "(250)"),
    ("Income tax expense", "(1,491)", "(1,198)", "(1,520)"),
    ("Profit attributable to non-controlling interests (NCI)", "(8)", "(9)", "(8)"),
    ("Net profit attributable to owners of WBC", "3,414", "3,599", "3,317"),
)

#: Page 10, headed "Excluding Notable Items, $m" on the line that names the periods, and
#: reconciling to the reported net profit after a "Notable Items (post tax)" heading.
EXCLUDING = (
    ("Net interest income", "9,763", "9,904", "9,569"),
    ("Non-interest income", "1,521", "1,567", "1,424"),
    ("Net operating income", "11,284", "11,471", "10,993"),
    ("Operating expenses", "(5,830)", "(6,218)", "(5,698)"),
    ("Pre-provision profit", "5,454", "5,253", "5,295"),
    ("Impairment (charges)/benefits", "(443)", "(174)", "(250)"),
    ("Income tax expense", "(1,520)", "(1,555)", "(1,580)"),
    ("Profit attributable to non-controlling interests (NCI)", "(8)", "(9)", "(8)"),
    ("Net profit excluding Notable Items", "3,483", "3,515", "3,457"),
    ("Notable Items (post tax)", "", "", ""),
    ("Hedging items", "6", "84", "(140)"),
    ("Large items", "(75)", "-", "-"),
    ("Net profit attributable to owners of WBC", "3,414", "3,599", "3,317"),
)

#: Page 11: the ratio, published only excluding Notable Items, and balances under the same
#: heading, which have no notable items to exclude.
KEY_METRICS = (
    ("Business performance - excluding Notable Items", "", "", ""),
    ("Expense to income ratio", "51.67%", "54.21%", "51.83%"),
    ("Average interest-earning assets ($m)", "1,035,226", "1,008,977", "996,701"),
)


def _table(page: int, label_header: str, rows) -> ParsedTable:  # type: ignore[no-untyped-def]
    return ParsedTable(
        table_id=f"wbc_p{page}_L0",
        page_no=page,
        caption="Half Year Half Year Half Year % Mov't",
        header=(label_header, *HEADER),
        rows=tuple(rows),
        units="aud_m",
    )


def _document(*tables: ParsedTable) -> ParsedDocument:
    pages = tuple(
        ParsedPage(page_no=t.page_no, text="GROUP PERFORMANCE", kind="text", tables=(t,))
        for t in tables
    )
    return ParsedDocument(
        doc_id="wbc-1h26-results-document", sha256="0" * 64, page_count=120, pages=pages
    )


@pytest.fixture(scope="module")
def meta() -> DocMeta:
    return DocMeta(
        doc_id="wbc-1h26-results-document",
        bank_id="WBC",
        period="1H26",
        doc_type="RESULTS_DOCUMENT",
        publish_date=PUB,
        url="https://www.westpac.com.au/example.pdf",
        page_count=120,
    )


@pytest.fixture(scope="module")
def facts(meta) -> list[Fact]:  # type: ignore[no-untyped-def]
    document = _document(
        _table(9, "$m", REPORTED),
        _table(10, "Excluding Notable Items, $m", EXCLUDING),
        _table(11, "", KEY_METRICS),
    )
    return extract_facts(document, meta, PERIODS)


def _reading(facts, component: str, period: str, *, excluding: bool) -> Fact | None:  # type: ignore[no-untyped-def]
    return next(
        (
            f
            for f in facts
            if f.component == component
            and f.period == period
            and f.basis.ex_notable_items is excluding
        ),
        None,
    )


# --- reading the words over the label column ------------------------------------------


def _line(*words: tuple[float, float, str]) -> list[Word]:
    return [Word(x0, x1, text) for x0, x1, text in words]


def test_the_words_over_the_label_column_are_kept():
    """Westpac's header line, with its word positions from page 10."""
    line = _line(
        (31.2, 60.9, "Excluding"),
        (62.5, 87.1, "Notable"),
        (88.8, 107.7, "Items,"),
        (109.3, 118.7, "$m"),
        (253.7, 272.7, "March"),
        (274.4, 288.5, "2026"),
        (319.6, 334.0, "Sept"),
        (335.6, 349.7, "2025"),
        (376.2, 395.2, "March"),
        (396.8, 411.0, "2025"),
    )
    columns = header_columns(line)
    assert [column.label for column in columns] == ["March 2026", "Sept 2025", "March 2025"]
    assert _label_header([line], columns) == "Excluding Notable Items, $m"


def test_label_text_that_names_a_period_is_left_out():
    """It would claim that period's column, and the leftmost claim wins, so the label
    column would take the real column's place."""
    periods = _line(
        (253.7, 272.7, "March"),
        (274.4, 288.5, "2026"),
        (319.6, 334.0, "Sept"),
        (335.6, 349.7, "2025"),
    )
    line = _line(
        (31.2, 60.0, "Six"),
        (62.0, 90.0, "months"),
        (92.0, 104.0, "to"),
        (106.0, 120.0, "1H26"),
        *[(w.x0, w.x1, w.text) for w in periods],
    )
    assert _label_header([line], header_columns(periods)) == ""


# --- extraction: both readings, each from its own table --------------------------------


def test_the_variant_is_kept_beside_the_headline(facts):
    reported = _reading(facts, "operating_expenses", "1H26", excluding=False)
    variant = _reading(facts, "operating_expenses", "1H26", excluding=True)
    assert reported is not None and reported.value == Decimal("5937") and reported.page_no == 9
    assert variant is not None and variant.value == Decimal("5830") and variant.page_no == 10


def test_each_bridge_comes_whole_from_one_table(facts):
    """The headline's lines all from page 9 and the variant's all from page 10: a bridge
    assembled across the two would reconcile to neither."""
    bridge = ("nii", "other_income", "operating_income", "operating_expenses", "tax")
    for excluding, page in ((False, 9), (True, 10)):
        for component in bridge:
            for period in PERIODS:
                reading = _reading(facts, component, period, excluding=excluding)
                assert reading is not None, (component, period, excluding)
                assert reading.page_no == page, (component, period, excluding)


def test_the_reported_total_after_the_notable_items_is_the_headline(facts):
    """The heading "Notable Items (post tax)" ends the excluding section: the rows after it
    are the items, then the reported net profit they reconcile to. The row above them,
    "Net profit excluding Notable Items", is held too, as the variant it names: refusing it
    left the investor pack's identical $3,483m looking like an unexplained disagreement."""
    headline = _reading(facts, "headline_earnings", "1H26", excluding=False)
    assert headline is not None and headline.value == Decimal("3414")
    variant = _reading(facts, "headline_earnings", "1H26", excluding=True)
    assert variant is not None and variant.value == Decimal("3483")


def test_the_ratio_published_only_excluding_notable_items_is_held_that_way(facts):
    ratio = _reading(facts, "cost_to_income", "1H26", excluding=True)
    assert ratio is not None and ratio.value == Decimal("51.67")
    assert _reading(facts, "cost_to_income", "1H26", excluding=False) is None


def test_a_balance_under_an_excluding_heading_is_just_the_balance(facts):
    """Average assets have no notable items to exclude, so there's one reading of them."""
    readings = [f for f in facts if f.component == "aiea" and f.period == "1H26"]
    assert len(readings) == 1
    assert readings[0].basis.ex_notable_items is False


def test_a_label_header_naming_only_an_excluded_variant_rules_out_the_table(meta):
    """Rows under "Excluding Markets" are a variant this tool doesn't store at all."""
    document = _document(_table(12, "Excluding Markets, $m", REPORTED[:3]))
    assert extract_facts(document, meta, PERIODS) == []


# --- validation: each reading checked on its own ---------------------------------------


def test_the_ratio_is_checked_against_inputs_on_its_own_basis(facts):
    outcome = validate_facts(facts, "WBC", PERIODS)
    ratio_checks = [c for c in outcome.checks if c.name == "cost_to_income"]
    assert ratio_checks, "the variant identity wasn't run"
    assert all(check.passed and not check.skipped for check in ratio_checks)
    assert all("excluding notable items" in check.detail for check in ratio_checks)
    assert not outcome.quarantined


def test_a_variant_that_does_not_add_up_costs_only_the_variant(facts):
    """A misread variant is quarantined. The reported reading of the same line, on the same
    period, isn't touched by a check it was never part of."""
    broken = [
        f.model_copy(update={"value": Decimal("5900")})
        if f.component == "operating_expenses" and f.period == "1H26" and f.basis.ex_notable_items
        else f
        for f in facts
    ]
    outcome = validate_facts(broken, "WBC", PERIODS)
    quarantined = {
        (f.component, f.period, f.basis.ex_notable_items) for f, _reason in outcome.quarantined
    }
    assert ("operating_expenses", "1H26", True) in quarantined
    assert ("operating_expenses", "1H26", False) not in quarantined
    reasons = [reason for _fact, reason in outcome.quarantined]
    assert all("excluding notable items" in reason for reason in reasons)


# --- choosing between two readings that claim the same treatment -----------------------


def _fact(scope: str, page: int) -> Fact:
    return Fact(
        fact_id="",
        bank_id="WBC",
        period="1H26",
        component="cet1_ratio",
        value=Decimal("12.42"),
        unit="pct",
        basis=Basis(scope=scope),  # type: ignore[arg-type]
        doc_id="wbc-1h26-results-document",
        doc_type="RESULTS_DOCUMENT",
        page_no=page,
        publish_date=PUB,
    )


def test_a_reading_that_states_its_regulatory_level_beats_a_silent_one():
    """Westpac's key-metrics table on page 11 prints the CET1 ratio with no Level 2
    heading; the capital page, 29, says Level 2. Page order alone took page 11."""
    index = line_item_index("WBC")
    silent, stated = _fact("group", 11), _fact("level2", 29)
    assert _prefer(stated, silent, "statutory", index)
    assert not _prefer(silent, stated, "statutory", index)


# --- storing both ----------------------------------------------------------------------


def test_two_readings_on_one_page_are_two_rows(tmp_path):
    from bankresearch.memory.db import connect
    from bankresearch.memory.fact_store import FactStore

    store = FactStore(connect(tmp_path / "db.sqlite"))
    store.put_document(
        DocMeta(
            doc_id="wbc-1h26-results-document",
            bank_id="WBC",
            period="1H26",
            doc_type="RESULTS_DOCUMENT",
            publish_date=PUB,
            url="https://www.westpac.com.au/example.pdf",
        )
    )
    base = Fact(
        fact_id="",
        bank_id="WBC",
        period="1H26",
        component="operating_expenses",
        value=Decimal("5937"),
        unit="aud_m",
        doc_id="wbc-1h26-results-document",
        doc_type="RESULTS_DOCUMENT",
        page_no=10,
        publish_date=PUB,
        validated=True,
    )
    variant = base.model_copy(
        update={"value": Decimal("5830"), "basis": Basis(ex_notable_items=True)}
    )
    assert store.put_fact(base) != store.put_fact(variant)


# --- reconciliation: the inputs follow the ratio's own basis ---------------------------


def _candidate(value: str, *, excluding: bool, page: int, earnings: str = "n/a") -> CandidateValue:
    return CandidateValue(
        value=Decimal(value),
        unit="aud_m",
        basis=Basis(earnings_basis=earnings, ex_notable_items=excluding),  # type: ignore[arg-type]
        source=SourceRef(
            doc_id="wbc-1h26-results-document",
            doc_type="RESULTS_DOCUMENT",
            page=page,
            publish_date=PUB,
        ),
        vintage=PUB,
    )


def test_inputs_follow_a_ratio_published_only_excluding_notable_items(facts):
    from bankresearch.agent.steps.reconcile import reconcile_inputs
    from bankresearch.agent.steps.resolve import resolve_request

    spec = resolve_request("WBC", "1H26", "cost_to_income", as_of=date(2026, 5, 31))
    inputs = reconcile_inputs([], [f.model_copy(update={"validated": True}) for f in facts], spec)
    assert inputs.get("operating_expenses") == Decimal("5830")
    assert inputs.get("operating_income") == Decimal("11284")
    assert inputs.comparable("cost_to_income", "operating_expenses", "operating_income")
    assert any("only excluding notable items" in caveat for caveat in inputs.caveats)


def test_a_metric_printed_both_ways_still_leads_with_the_reported_figure(facts):
    """Westpac's net profit is printed both ways; the reported figure is its headline."""
    from bankresearch.agent.steps.reconcile import reconcile_inputs
    from bankresearch.agent.steps.resolve import resolve_request

    spec = resolve_request("WBC", "1H26", "headline_earnings", as_of=date(2026, 5, 31))
    inputs = reconcile_inputs([], [f.model_copy(update={"validated": True}) for f in facts], spec)
    assert inputs.get("operating_expenses") == Decimal("5937")
    assert not any("only excluding notable items" in caveat for caveat in inputs.caveats)


def _older_reading(facts, component: str, value: str, *, excluding: bool) -> Fact:  # type: ignore[no-untyped-def]
    """A 1H25 reading from Westpac's own 1H25 pack, which printed its ratios both ways."""
    template = _reading(facts, component, "1H25", excluding=True)
    assert template is not None
    return template.model_copy(
        update={
            "value": Decimal(value),
            "published_value": Decimal(value),
            "basis": template.basis.model_copy(update={"ex_notable_items": excluding}),
            "doc_id": "wbc-1h25-results-document",
            "page_no": 14,
            "publish_date": date(2025, 5, 5),
            "validated": True,
        }
    )


def test_the_prior_half_is_read_on_the_basis_the_current_pack_publishes(facts):
    """Westpac's 1H26 pack prints its expense-to-income ratio only excluding Notable Items,
    for both halves. Its 1H25 pack also printed the reported 52.79%, and holding that one
    reading kept the headline "reported", so the answer compared 51.67% excluding notable
    items with 52.79% including them."""
    from bankresearch.agent.steps.reconcile import reconcile_inputs
    from bankresearch.agent.steps.resolve import resolve_request

    spec = resolve_request("WBC", "1H26", "cost_to_income", as_of=date(2026, 5, 31))
    held = [f.model_copy(update={"validated": True}) for f in facts]
    held.append(_older_reading(facts, "cost_to_income", "52.79", excluding=False))
    inputs = reconcile_inputs([], held, spec)
    assert inputs.current["cost_to_income"] == Decimal("51.67")
    assert inputs.prior["cost_to_income"] == Decimal("51.83")
    assert inputs.comparable("cost_to_income", "operating_expenses", "operating_income")
    assert not any("change of basis" in caveat for caveat in inputs.caveats)


def test_a_movement_with_no_shared_basis_says_so(facts):
    """With the prior half held only as reported, there's no one basis for both. The
    movement is still given, with the change of basis named."""
    from bankresearch.agent.steps.reconcile import reconcile_inputs
    from bankresearch.agent.steps.resolve import resolve_request

    spec = resolve_request("WBC", "1H26", "cost_to_income", as_of=date(2026, 5, 31))
    held = [
        f.model_copy(update={"validated": True})
        for f in facts
        if not (f.component == "cost_to_income" and f.period == "1H25")
    ]
    held.append(_older_reading(facts, "cost_to_income", "52.79", excluding=False))
    inputs = reconcile_inputs([], held, spec)
    assert inputs.prior["cost_to_income"] == Decimal("52.79")
    assert any(
        "1H26 figure is excluding notable items and the 1H25 figure is as reported" in caveat
        for caveat in inputs.caveats
    )


def test_agreeing_readings_carry_the_headline_basis_forward():
    """NAB prints its 1H25 return on equity as 11.7% both with and without large notable
    items. Carrying the variant forward made the ratio incomparable with the earnings it's
    built from, and the drivers disappeared."""
    from bankresearch.reconcile.conflicts import resolve

    headline = Basis(earnings_basis="cash")
    variant = _candidate("11.7", excluding=True, page=12, earnings="cash")
    reported = _candidate("11.7", excluding=False, page=12, earnings="cash")
    for order in ([variant, reported], [reported, variant]):
        chosen, conflicts = resolve("roe", order, headline_basis=headline)
        assert conflicts == []
        assert chosen.basis.ex_notable_items is False


def test_a_basis_conflict_narrows_on_every_dimension_the_headline_states():
    """Matching on the first dimension that differs and then taking document precedence
    could choose the variant among the readings on the headline's earnings basis."""
    from bankresearch.reconcile.conflicts import classify

    headline = Basis(earnings_basis="cash")
    candidates = [
        _candidate("2639", excluding=False, page=6, earnings="statutory"),
        _candidate("3588", excluding=True, page=5, earnings="cash"),
        _candidate("2711", excluding=False, page=7, earnings="cash"),
    ]
    conflict = classify("headline_earnings", candidates, headline_basis=headline)
    assert conflict is not None and conflict.classification == "BASIS"
    assert conflict.chosen.value == Decimal("2711")


# --- the result says which basis it's on -----------------------------------------------


@pytest.mark.parametrize(
    ("basis", "words"),
    [
        (Basis(), ""),
        (Basis(ex_notable_items=True), "excluding notable items"),
        (Basis(earnings_basis="cash", continuing_ops=True), "cash basis, continuing operations"),
        (Basis(scope="level2"), "APRA Level 2"),
    ],
)
def test_a_basis_is_described_in_a_readers_words(basis, words):
    from bankresearch.output.schema import describe_basis

    assert describe_basis(basis) == words


# --- the headline's own readings are settled among themselves -------------------------


def test_a_variant_cannot_vouch_for_a_disagreement_among_the_headline_readings():
    """Two statements of the reported figure that disagree materially are unexplained,
    whether or not a variant is also held. Classifying all three at once called it a basis
    difference, at no cost to confidence."""
    from bankresearch.reconcile.conflicts import resolve

    headline = Basis()
    candidates = [
        _candidate("5937", excluding=False, page=9),
        _candidate("6100", excluding=False, page=40),
        _candidate("5830", excluding=True, page=10),
    ]
    chosen, conflicts = resolve("operating_expenses", candidates, headline_basis=headline)
    classes = [conflict.classification for conflict in conflicts]
    assert classes == ["BASIS", "UNEXPLAINED"]
    assert chosen.basis.ex_notable_items is False


def test_a_rounding_difference_across_bases_never_carries_the_variant_forward():
    """The variant printed to one more decimal place used to win as "the more precise
    reading", and took its basis with it."""
    from bankresearch.reconcile.conflicts import resolve

    headline = Basis(earnings_basis="cash")
    candidates = [
        _candidate("11.7", excluding=False, page=12, earnings="cash"),
        _candidate("11.74", excluding=True, page=12, earnings="cash"),
    ]
    chosen, conflicts = resolve("roe", candidates, headline_basis=headline)
    assert chosen.basis.ex_notable_items is False
    assert conflicts == []  # the same figure to the precision either was printed to


def test_a_variant_that_differs_is_recorded_as_a_basis_difference_only():
    from bankresearch.reconcile.conflicts import resolve

    headline = Basis()
    candidates = [
        _candidate("5937", excluding=False, page=9),
        _candidate("5830", excluding=True, page=10),
    ]
    chosen, conflicts = resolve("operating_expenses", candidates, headline_basis=headline)
    assert chosen.value == Decimal("5937")
    assert [c.classification for c in conflicts] == ["BASIS"]
    assert "notable-items treatment" in conflicts[0].rule_applied


def test_a_figure_cites_the_rows_on_its_own_basis():
    """NAB's 1H25 return on equity is 11.7% both ways. The headline cites the headline's
    row, not the variant's, which only rounds the same."""
    from bankresearch.agent.steps.reconcile import reconcile_inputs
    from bankresearch.agent.steps.resolve import resolve_request
    from bankresearch.domain.models import EvidenceCard

    spec = resolve_request("NAB", "1H26", "roe", as_of=date(2026, 5, 31))

    def card(evidence_id: str, value: str, excluding: bool) -> EvidenceCard:
        return EvidenceCard(
            evidence_id=evidence_id,
            component="roe",
            period="1H25",
            value=Decimal(value),
            unit="pct",
            basis=Basis(earnings_basis="cash", ex_notable_items=excluding),
            source=SourceRef(
                doc_id="nab-1h26-results-document",
                doc_type="RESULTS_DOCUMENT",
                page=12,
                publish_date=PUB,
            ),
            statement=f"Return on equity was {value}% for 1H25.",
            quote=f"Return on equity | Mar 25 | {value}",
            method="table",
        )

    inputs = reconcile_inputs([card("ev_1", "11.7", False), card("ev_2", "11.7", True)], [], spec)
    assert inputs.citations_for("roe", "1H25") == ["ev_1"]


def test_an_unlabelled_figure_equal_to_the_variant_is_that_variant_not_a_mystery():
    """Westpac's investor pack prints net profit of $3,483m under a bare "Net profit"
    heading; its results document prints $3,483m as net profit excluding notable items and
    $3,414m as the headline. That's a basis difference, not an unexplained conflict."""
    from datetime import date as _date

    from bankresearch.domain.models import Basis, CandidateValue, SourceRef
    from bankresearch.reconcile.conflicts import resolve

    def candidate(value: str, doc: str, page: int, excluding: bool = False) -> CandidateValue:
        return CandidateValue(
            value=Decimal(value),
            unit="aud_m",
            basis=Basis(ex_notable_items=excluding),
            source=SourceRef(
                doc_id=doc,
                doc_type="RESULTS_DOCUMENT" if "results" in doc else "INVESTOR_PRESENTATION",
                page=page,
                publish_date=_date(2026, 5, 4),
            ),
            vintage=_date(2026, 5, 4),
        )

    chosen, conflicts = resolve(
        "headline_earnings",
        [
            candidate("3414", "wbc-1h26-results-document", 9),
            candidate("3483", "wbc-1h26-investor-presentation", 119),
            candidate("3483", "wbc-1h26-results-document", 10, excluding=True),
        ],
        headline_basis=Basis(),
    )
    assert chosen.value == Decimal("3414")
    assert [c.classification for c in conflicts] == ["BASIS"]
    assert "states no basis but prints exactly" in conflicts[0].rule_applied


# --- notable items under the headline are a memo, above it a line ---------------------

#: Westpac's 1H25 summary (p.13): the reported bridge, then "Notable Items" printed under
#: net profit as a memo of what the lines above already include.
SUMMARY_1H25 = (
    ("Net interest income", "9,351", "9,626", "9,127"),
    ("Non-interest income", "1,442", "1,372", "1,463"),
    ("Net operating income", "10,793", "10,998", "10,590"),
    ("Operating expenses", "(5,698)", "(5,549)", "(5,395)"),
    ("Impairment (charges)/benefits", "(250)", "(175)", "(362)"),
    ("Income tax expense", "(1,520)", "(1,626)", "(1,491)"),
    ("Profit attributable to non-controlling interests (NCI)", "(8)", "-", "-"),
    ("Net profit attributable to owners of WBC", "3,317", "3,648", "3,342"),
    ("Notable Items", "(140)", "41", "(164)"),
)

#: Judo's income statement (2025 annual report p.44), where the non-recurring costs are a
#: line above statutory profit, and a dash in FY25 means there were none.
JUDO_FY25 = (
    ("Net interest income", "407.3", "386.0"),
    ("Other operating income", "15.6", "10.7"),
    ("Net banking income", "422.9", "396.7"),
    ("Total underlying operating expenses", "(221.8)", "(216.5)"),
    ("Underlying profit before impairment", "201.1", "180.2"),
    ("Impairment expense", "(75.5)", "(70.1)"),
    ("Underlying profit before tax", "125.6", "110.1"),
    ("Non-recurring costs", "\u2013", "(5.8)"),
    ("Statutory profit before tax", "125.6", "104.3"),
    ("Tax expense", "(39.2)", "(34.4)"),
    ("Statutory profit after tax", "86.4", "69.9"),
)


def _one_table(meta: DocMeta, page: int, header, rows) -> ParsedDocument:  # type: ignore[no-untyped-def]
    table = ParsedTable(
        table_id=f"{meta.bank_id.lower()}_p{page}_L0",
        page_no=page,
        caption="",
        header=header,
        rows=rows,
        units="aud_m",
    )
    return ParsedDocument(
        doc_id=meta.doc_id,
        sha256="0" * 64,
        page_count=page + 1,
        pages=(ParsedPage(page_no=page, text="GROUP PERFORMANCE", kind="text", tables=(table,)),),
    )


def _earnings_checks(facts, bank: str, periods: tuple[str, ...]):  # type: ignore[no-untyped-def]
    outcome = validate_facts(facts, bank, periods)
    return [c for c in outcome.checks if c.name == "headline_earnings" and not c.skipped]


def test_notable_items_under_the_headline_are_a_memo_not_a_line():
    """Counted as a line of the bridge, they put Westpac's 1H25 earnings identity out by
    $140m and its 1H24 one by $164m."""
    meta = DocMeta(
        doc_id="wbc-1h25-results-document",
        bank_id="WBC",
        period="1H25",
        doc_type="RESULTS_DOCUMENT",
        publish_date=date(2025, 5, 5),
        url="https://www.westpac.com.au/example.pdf",
        page_count=134,
    )
    document = _one_table(meta, 13, ("$m", "March 2025", "Sept 2024", "March 2024"), SUMMARY_1H25)
    found = extract_facts(document, meta, ("1H25", "1H24"))
    assert not [f for f in found if f.component == "notable_items"]
    checks = _earnings_checks(found, "WBC", ("1H25", "1H24"))
    assert {c.period for c in checks} == {"1H25", "1H24"}
    assert all(c.passed for c in checks)


def test_notable_items_above_the_headline_are_a_line_of_the_bridge():
    meta = DocMeta(
        doc_id="jdo-fy25-annual-report",
        bank_id="JDO",
        period="FY25",
        doc_type="ANNUAL_REPORT",
        publish_date=date(2025, 8, 19),
        url="https://www.judo.bank/example.pdf",
        page_count=156,
    )
    document = _one_table(meta, 44, ("$m", "Jun-25", "Jun-24"), JUDO_FY25)
    found = extract_facts(document, meta, ("FY25", "FY24"))
    notable = {f.period: f.value for f in found if f.component == "notable_items"}
    assert notable == {"FY25": Decimal("0"), "FY24": Decimal("-5.8")}
    checks = _earnings_checks(found, "JDO", ("FY25", "FY24"))
    assert {c.period for c in checks} == {"FY25", "FY24"}
    assert all(c.passed for c in checks)
