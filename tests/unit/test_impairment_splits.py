"""How a credit impairment charge is split, and which way its table prints it."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from bankresearch.domain.models import DocMeta
from bankresearch.ingestion.facts import (
    _DivisionSplit,
    _stated_convention,
    label_candidates,
    line_item_index,
    mirrored,
)
from bankresearch.ingestion.parse_pdf import ParsedPage, ParsedTable

PUB = date(2026, 8, 12)
#: How the packs print nil: an en dash, held by code point so the source stays plain.
NIL = chr(0x2013)
FY = ("FY26", "FY25")

#: CBA's division names, as configured.
DIVISIONS = {
    "Retail Banking Services": "retail",
    "Business Banking": "business",
    "Institutional Banking and Markets": "institutional",
    "New Zealand": "new_zealand",
    "Corporate Centre and Other": "other",
}

#: Each division's page as CBA sets it: the section title, the division's name, then its
#: profit and loss with expenses in brackets. (Name, label, FY26, FY25.)
PAGES = (
    ("Retail Banking Services", "Loan impairment expense", "(378)", "(272)"),
    ("Business Banking", "Loan impairment expense", "(310)", "(355)"),
    ("Institutional Banking and Markets", "Loan impairment (expense)/benefit", "(33)", "(49)"),
    ("New Zealand", "Loan impairment expense", "(66)", "(55)"),
    ("Corporate Centre and Other", "Loan impairment (expense)/benefit", "(1)", "5"),
)


@pytest.fixture(scope="module")
def meta() -> DocMeta:
    return DocMeta(
        doc_id="cba-fy26-results-document",
        bank_id="CBA",
        period="FY26",
        doc_type="RESULTS_DOCUMENT",
        publish_date=PUB,
        url="https://www.commbank.com.au/example.pdf",
    )


def _division_page(page_no: int, name: str, label: str, current: str, prior: str) -> ParsedPage:
    table = ParsedTable(
        table_id=f"cba_p{page_no}_L0",
        page_no=page_no,
        caption="Full Year Ended Half Year Ended",
        header=("", "Jun 26", "Jun 25"),
        rows=(
            ("Net interest income", "12,017", "11,382"),
            ("Operating expenses", "(5,417)", "(5,117)"),
            (label, current, prior),
            ("Corporate tax expense", "(2,401)", "(2,307)"),
        ),
        units="aud_m",
    )
    text = f"Divisional Performance (continued)\n{page_no - 22} Commonwealth Bank\n{name}\n"
    return ParsedPage(page_no=page_no, text=text, kind="text", tables=(table,))


def _read(pages, meta) -> list:  # type: ignore[no-untyped-def]
    index = line_item_index("CBA")
    split = _DivisionSplit(DIVISIONS)
    for page in pages:
        split.read(page, index, meta, 6, set(FY), False)
    return split.complete(set(FY))


def _pages() -> list[ParsedPage]:
    return [
        _division_page(62 + 5 * i, name, label, current, prior)
        for i, (name, label, current, prior) in enumerate(PAGES)
    ]


def test_each_division_is_read_from_its_own_page_and_adds_up(meta):
    facts = _read(_pages(), meta)
    by = {(f.component, f.period): f.value for f in facts}
    assert by[("impairment_div_retail", "FY26")] == Decimal("378")
    assert by[("impairment_div_other", "FY25")] == Decimal("-5")  # a benefit, as printed
    for period, total in (("FY26", Decimal("788")), ("FY25", Decimal("726"))):
        assert sum(v for (_c, p), v in by.items() if p == period) == total
    assert all(f.basis.scope == "division" for f in facts)


def test_a_split_with_a_division_missing_is_not_kept(meta):
    """The missing division would sit in the residual and still look like an answer."""
    assert _read(_pages()[:-1], meta) == []


def test_a_continued_page_is_never_a_divisions_first(meta):
    """CBA's New Zealand pages after the first restate the division in NZ$."""
    pages = _pages()
    restated = _division_page(
        76, "New Zealand (continued)", "Loan impairment expense", "(78)", "(60)"
    )
    facts = _read([*pages[:3], restated, *pages[3:]], meta)
    by = {(f.component, f.period): f.value for f in facts}
    assert by[("impairment_div_new_zealand", "FY26")] == Decimal("66")


def test_a_dash_in_the_impairment_row_is_a_nil_charge(meta):
    """CBA's Corporate Centre printed a dash for its 1H26 charge."""
    pages = _pages()
    pages[-1] = _division_page(
        82, "Corporate Centre and Other", "Loan impairment (expense)/benefit", NIL, "5"
    )
    by = {(f.component, f.period): f.value for f in _read(pages, meta)}
    assert by[("impairment_div_other", "FY26")] == Decimal("0")
    assert by[("impairment_div_other", "FY25")] == Decimal("-5")


def test_a_bank_without_a_division_map_reads_no_division(meta):
    split = _DivisionSplit({})
    split.read(_pages()[0], line_item_index("CBA"), meta, 6, set(FY), False)
    assert split.complete(set(FY)) == []


def test_a_division_map_must_name_every_segment_once():
    from bankresearch.domain.registry import _impairment_divisions

    whole = {"bank_id": "X", "impairment_divisions": DIVISIONS}
    assert _impairment_divisions(whole) == DIVISIONS
    short = dict(list(DIVISIONS.items())[:4])
    with pytest.raises(ValueError, match="exactly once"):
        _impairment_divisions({"bank_id": "X", "impairment_divisions": short})
    doubled = {**short, "Wealth": "retail"}
    with pytest.raises(ValueError, match="exactly once"):
        _impairment_divisions({"bank_id": "X", "impairment_divisions": doubled})


# --- decomposition and validation ------------------------------------------------------


def _by_division(values: dict[str, str]) -> dict[str, Decimal]:
    return {f"impairment_div_{k}": Decimal(v) for k, v in values.items()}


CURRENT = _by_division(
    {"retail": "378", "business": "310", "institutional": "33", "new_zealand": "66", "other": "1"}
)
PRIOR = _by_division(
    {"retail": "272", "business": "355", "institutional": "49", "new_zealand": "55", "other": "-5"}
)


def test_the_charge_is_split_by_division_when_that_is_all_there_is():
    from bankresearch.decomposition.metrics import decompose_path_b

    result = decompose_path_b(
        "credit_impairment_charge", PRIOR, CURRENT, Decimal("726"), Decimal("788")
    )
    contributions = {d.driver_id: d.contribution for d in result.drivers}
    assert contributions == {
        "cic.div_retail": Decimal("106"),
        "cic.div_business": Decimal("-45"),
        "cic.div_institutional": Decimal("-16"),
        "cic.div_new_zealand": Decimal("11"),
        "cic.div_other": Decimal("6"),
    }
    assert result.residual == 0
    assert any("where the charge rose, not why" in note for note in result.notes)


def test_a_division_split_with_a_part_missing_explains_nothing():
    from bankresearch.decomposition.metrics import decompose_path_b

    prior = {k: v for k, v in PRIOR.items() if k != "impairment_div_other"}
    current = {k: v for k, v in CURRENT.items() if k != "impairment_div_other"}
    result = decompose_path_b(
        "credit_impairment_charge", prior, current, Decimal("726"), Decimal("788")
    )
    assert result.drivers == [] and result.path == "none"


def test_the_individual_and_collective_split_is_preferred_where_it_exists():
    from bankresearch.decomposition.metrics import decompose_path_b

    prior = {
        **PRIOR,
        "impairment_individual": Decimal("390"),
        "impairment_collective": Decimal("-42"),
    }
    current = {
        **CURRENT,
        "impairment_individual": Decimal("541"),
        "impairment_collective": Decimal("165"),
    }
    result = decompose_path_b(
        "credit_impairment_charge", prior, current, Decimal("348"), Decimal("706")
    )
    assert {d.driver_id for d in result.drivers} == {"cic.individual", "cic.collective"}


def test_recoveries_on_their_own_line_count_with_the_individual_charge():
    """Westpac: (173 - 118) - (162 - 115) = 8, and 388 - 203 = 185."""
    from bankresearch.decomposition.metrics import decompose_path_b

    prior = {
        "impairment_individual": Decimal("162"),
        "impairment_collective": Decimal("203"),
        "impairment_recoveries": Decimal("-115"),
    }
    current = {
        "impairment_individual": Decimal("173"),
        "impairment_collective": Decimal("388"),
        "impairment_recoveries": Decimal("-118"),
    }
    result = decompose_path_b(
        "credit_impairment_charge", prior, current, Decimal("250"), Decimal("443")
    )
    contributions = {d.driver_id: d.contribution for d in result.drivers}
    assert contributions == {"cic.individual": Decimal("8"), "cic.collective": Decimal("185")}
    assert result.residual == 0


def _fact(component: str, period: str, value: str):  # type: ignore[no-untyped-def]
    from bankresearch.domain.models import Fact

    return Fact(
        fact_id="",
        bank_id="CBA",
        period=period,
        component=component,
        value=Decimal(value),
        unit="aud_m",
        doc_id="cba-fy26-results-document",
        doc_type="RESULTS_DOCUMENT",
        page_no=27,
        publish_date=PUB,
    )


def test_the_divisions_must_add_up_to_the_group_charge():
    from bankresearch.ingestion.validate import validate_facts

    facts = [_fact("credit_impairment_charge", "FY26", "788")] + [
        _fact(component, "FY26", str(value)) for component, value in CURRENT.items()
    ]
    outcome = validate_facts(facts, "CBA", ("FY26",))
    check = next(c for c in outcome.checks if c.name == "impairment_by_division")
    assert check.passed and check.gap == 0

    wrong = [
        f.model_copy(update={"value": Decimal("400")})
        if f.component == "impairment_div_retail"
        else f
        for f in facts
    ]
    outcome = validate_facts(wrong, "CBA", ("FY26",))
    quarantined = {f.component for f, _ in outcome.quarantined}
    assert "impairment_div_retail" in quarantined
    assert "credit_impairment_charge" not in quarantined  # the published figure stands


# --- which way a table prints its charges ----------------------------------------------


def _table(*rows: tuple[str, ...]) -> ParsedTable:
    return ParsedTable(
        table_id="t",
        page_no=21,
        caption="",
        header=("$m", "March 2026", "March 2025"),
        rows=rows,
        units="aud_m",
    )


def test_a_label_that_brackets_its_charges_says_they_print_negative():
    """Westpac's page 21, which has no expense or tax line to go by."""
    table = _table(
        ("Net IAP charges", "(173)", "(162)"),
        ("Recoveries", "118", "115"),
        ("Total impairment (charges)/benefits", "(443)", "(250)"),
    )
    assert _stated_convention(table, line_item_index("WBC")) is True


def test_a_label_that_brackets_its_write_backs_says_charges_print_positive():
    """NAB's page 35: "Collective credit impairment charge / (write-back) 165 (42)"."""
    table = _table(
        ("Total individually assessed credit impairment charge", "541", "390"),
        ("Collective credit impairment charge / (write-back)", "165", "(42)"),
    )
    assert _stated_convention(table, line_item_index("NAB")) is False


def test_a_cost_line_no_component_stores_still_settles_the_convention():
    """Westpac's investor pack appendix heads its cost line just "Expenses"."""
    table = _table(
        ("Net operating income", "11,284", "10,993"),
        ("Expenses", "(5,830)", "(5,698)"),
        ("Impairment charges", "(443)", "(250)"),
    )
    assert _stated_convention(table, line_item_index("WBC")) is True


def test_a_ratio_row_says_nothing_about_the_amounts():
    """ "Impairment charges/(benefits) to average loans" describes a ratio, not a deduction."""
    table = _table(("Impairment charges/(benefits) to average loans", "10", "6"))
    assert _stated_convention(table, line_item_index("WBC")) is None


def test_a_dash_on_the_cited_page_supports_a_nil_figure(tmp_path, monkeypatch):
    from bankresearch.evals import scorers
    from bankresearch.ingestion.parse_pdf import ParsedDocument

    page = ParsedPage(
        page_no=80,
        text="Corporate Centre and Other",
        kind="text",
        tables=(_table(("Loan impairment (expense)/benefit", NIL, "1")),),
    )
    document = ParsedDocument(doc_id="d", sha256="0" * 64, page_count=200, pages=(page,))
    from types import SimpleNamespace

    monkeypatch.setattr(scorers, "read_processed", lambda doc_id, root: document)
    settings = SimpleNamespace(processed_dir=tmp_path)
    assert Decimal(0) in scorers._numbers_on_page("d", 80, settings)  # type: ignore[arg-type]


def test_a_two_sided_label_matches_the_other_way_round():
    """Bendigo's FY25 results print "Credit reversals/(expenses)" and its FY26 annual report
    "Credit Expenses/(Reversals)". The run abstained on FY26 because no label matched."""
    assert mirrored("Credit Expenses/(Reversals)") == "Credit Reversals/(Expenses)"
    assert mirrored("Total impairment (charges)/benefits") == "Total impairment (benefits)/charges"
    assert mirrored("Net interest income") is None
    index = line_item_index("BEN")
    assert "credit_impairment_charge" in index.components_for("Credit Expenses/(Reversals)")
    # Tried after every other reading, so a label that matches as printed still does first.
    assert label_candidates("Credit Expenses/(Reversals)")[-1] == "Credit Reversals/(Expenses)"


def test_a_label_that_brackets_its_reversals_says_charges_print_positive():
    """The sign still comes from the label as printed: brackets round the reversals put the
    expenses positive, whichever way round the configured label is."""
    table = _table(
        ("Credit Expenses/(Reversals)", "13.3", "(14.7)"),
    )
    assert _stated_convention(table, line_item_index("BEN")) is False
