"""The extraction rules that only real filings exposed."""

from __future__ import annotations

from decimal import Decimal

import pytest

from bankresearch.domain.units import normalise_aud
from bankresearch.ingestion.facts import (
    clean_label,
    label_candidates,
    line_item_index,
    unit_in_label,
)
from bankresearch.ingestion.layout_tables import Word, group_lines, header_columns
from bankresearch.ingestion.tables import detect_units

# --- units ----------------------------------------------------------------


@pytest.mark.parametrize("text", ["$B", "$ B", "Average balance sheet ($B)", "$bn", "A$bn"])
def test_billions_are_recognised(text):
    """ "$B" is how the majors write billions. Missing it left average balances a
    thousand times too small, which the NIM identity caught as a 15,000bp gap."""
    assert detect_units(text) == "aud_bn"


def test_an_aud_figure_normalises_to_millions():
    assert normalise_aud(Decimal("1167.4"), "aud_bn") == (Decimal("1167400.0"), "aud_m")
    assert normalise_aud(Decimal("8888"), "aud_m") == (Decimal("8888"), "aud_m")


def test_a_ratio_is_left_alone_by_aud_normalisation():
    assert normalise_aud(Decimal("1.53"), "pct") == (Decimal("1.53"), "pct")


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Average interest earning assets ($M)", "aud_m"),
        ("Average interest earning assets ($M) 3", "aud_m"),
        ("Risk weighted assets ($B)", "aud_bn"),
        ("Average ordinary equity ($m)", "aud_m"),
        ("Common Equity Tier 1 (APRA) (%)", "pct"),
        ("Net interest income", None),
    ],
)
def test_a_row_names_its_own_unit(label, expected):
    """Searched for as a parenthetical, not a suffix, because a footnote follows it.
    A balance of 448,489 read as a percentage fails the margin identity by 4,000bp."""
    assert unit_in_label(label) == expected


def test_a_cell_in_per_cent_beats_the_tables_caption():
    index = line_item_index("ANZ")
    assert (
        index.unit_for("roe", "Return on average ordinary shareholders' equity", "aud_m", "10.3%")
        == "pct"
    )


def test_the_tables_caption_beats_the_components_default():
    """Only the table knows whether a balance sheet is in $M or $B."""
    index = line_item_index("ANZ")
    assert index.unit_for("rwa", "Risk weighted assets", "aud_bn", "469.0") == "aud_bn"


# --- label cleaning -------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "component"),
    [
        ("Average interest earning assets ($M)", "aiea"),
        ("Average interest earning assets4", "aiea"),
        ("Average interest earning assetsb", "aiea"),
        ("Expense to income ratioa", "cost_to_income"),
        ("Operating expenses to operating income", "cost_to_income"),
        ("Return on average ordinary shareholders' equity5", "roe"),
        ("Income tax expense 6", "tax"),
        ("Common Equity Tier 1 (APRA) (%)", "cet1_ratio"),
        ("Credit Risk Weighted Assets (RWA) ($M)", "rwa_credit"),
        ("Cash earnings 2,639 3,508 3,583", "headline_earnings"),
    ],
)
def test_real_row_labels_map_to_their_component(label, component):
    assert component in line_item_index("NAB").components_for(label)


def test_a_name_that_legitimately_ends_in_a_digit_survives():
    """ "Common Equity Tier 1" must not be cleaned into "Common Equity Tier"."""
    assert "cet1_ratio" in line_item_index("CBA").components_for("Common Equity Tier 1")


def test_the_raw_label_is_always_the_first_reading_tried():
    readings = label_candidates("Common Equity Tier 1 (APRA) (%)")
    assert readings[0] == "Common Equity Tier 1 (APRA) (%)"


def test_figures_that_leaked_into_the_label_are_stripped():
    assert clean_label("Collectively assessed provisions 4,580 4,439 4,451") == (
        "Collectively assessed provisions"
    )


# --- exclusions: non-headline variants ------------------------------------


@pytest.mark.parametrize(
    "label",
    [
        "Average interest earning assets ex markets",
        "Average interest earning assets (excl. Markets) ($M)",
        "Core net interest income",
        "Cash profit before income tax",
        "Cash profit/(loss) before credit impairment and income tax",
        "Average interest earning assets (NZ$M)",
        "Average interest earning assets (ASB) (NZ$M)",
        "Common Equity Tier 1 Capital before regulatory adjustments",
        "Net interest margin (excl. Markets)",
    ],
)
def test_a_non_headline_variant_is_not_extracted(label):
    """These bridge differently from the group headline figure, so
    extracting one as that figure is a real misstatement."""
    assert line_item_index("ANZ").components_for(label) == ()


def test_a_label_naming_its_notable_items_treatment_is_read_as_that_variant():
    """ "Expense to income ratio excluding Notable Items" is the ratio on another basis, not
    a different measure (ADR-037), so it's held as the variant and never as the headline.
    A label with any other excluded qualifier is still refused outright."""
    from bankresearch.ingestion.facts import line_item_index as index_for

    label = "Expense to income ratio excluding Notable Items"
    assert index_for("ANZ").components_for(label) == ("cost_to_income",)
    assert index_for("ANZ").sections.basis_dimensions(label).get("ex_notable_items") is True
    assert index_for("ANZ").components_for("Net interest margin (excl. Markets)") == ()


def test_the_headline_row_itself_is_still_extracted():
    assert "nim" in line_item_index("ANZ").components_for("Net interest margin")
    assert "aiea" in line_item_index("ANZ").components_for("Average interest earning assets")


def test_a_provision_balance_is_not_the_period_charge():
    """A stock is not a flow; treating the provision balance as the charge is out by
    an order of magnitude while reconciling against nothing."""
    index = line_item_index("ANZ")
    assert index.components_for("Collectively assessed provisions") == ()
    assert "impairment_collective" in index.components_for(
        "Collectively assessed credit impairment charge/(release)"
    )


# --- plausibility ---------------------------------------------------------


@pytest.mark.parametrize(
    ("component", "value", "unit", "ok"),
    [
        ("nim", "1.53", "pct", True),
        ("nim", "48.9", "pct", False),
        ("nii", "8888", "aud_m", True),
        ("nii", "8.5", "aud_m", False),
        ("aiea", "1167.4", "aud_bn", True),
        ("aiea", "1167.4", "aud_m", False),
        ("headline_earnings", "3780", "aud_m", True),
        ("headline_earnings", "-26.3", "aud_m", False),
        ("cet1_ratio", "12.4", "pct", True),
        ("cet1_ratio", "123.9", "pct", False),
    ],
)
def test_a_figure_outside_its_plausible_range_is_refused(component, value, unit, ok):
    """A label match is not enough: packs repeat the same label on divisional pages in
    different units and magnitudes."""
    assert line_item_index("ANZ").is_plausible(component, Decimal(value), unit) is ok


def test_a_component_with_no_declared_range_is_allowed():
    """Only the components a range is declared for are bounded."""
    assert line_item_index("ANZ").is_plausible("gross_loans_nz", Decimal("999999"), "aud_m")


def test_a_ratio_matched_to_an_amount_component_is_refused():
    """Westpac's adjusted dividend payout ratio of 75.56% was being read as notable
    items, because nothing bounded that component."""
    index = line_item_index("WBC")
    assert not index.is_plausible("notable_items", Decimal("75.56"), "pct")
    assert index.is_plausible("notable_items", Decimal("-69"), "aud_m")


# --- sign conventions -----------------------------------------------------


def test_a_table_that_brackets_its_deductions_is_detected():
    policy = line_item_index("ANZ").signs
    assert policy.flip_needed({"operating_expenses": Decimal("-5595")})
    assert not policy.flip_needed({"operating_expenses": Decimal("5595")})


def test_the_anchors_are_lines_that_are_never_a_credit():
    assert line_item_index("ANZ").signs.anchors == ("operating_expenses", "tax")


def test_the_second_anchor_settles_a_table_with_no_expense_line():
    """Westpac's presentation heads its cost line simply "Expenses", which no synonym
    matches. With no anchor at all the impairment charge kept the sign the page printed,
    and a $193m increase in the charge entered the bridge as a $193m benefit."""
    policy = line_item_index("WBC").signs
    assert policy.flip_needed({"tax": Decimal("-1491"), "credit_impairment": Decimal("-443")})
    assert not policy.flip_needed({"tax": Decimal("1491")})
    assert not policy.flip_needed({"credit_impairment": Decimal("-443")})


def test_the_deduction_set_covers_the_bridge_lines():
    deductions = line_item_index("ANZ").signs.deductions
    assert {"operating_expenses", "credit_impairment", "tax", "minority_interests"} <= deductions


# --- layout-based table reconstruction ------------------------------------


def test_words_group_into_lines_by_vertical_position():
    words = [
        (45.0, 100.0, 80.0, 110.0, "Net"),
        (82.0, 100.0, 120.0, 110.0, "interest"),
        (45.0, 130.0, 70.0, 140.0, "Other"),
    ]
    lines = group_lines(words)
    assert [[w.text for w in line] for line in lines] == [["Net", "interest"], ["Other"]]


def test_period_columns_are_found_from_a_header_line():
    line = [
        Word(353, 383, "Mar"),
        Word(384, 392, "26"),
        Word(394, 424, "Sep"),
        Word(425, 433, "25"),
        Word(434, 464, "Mar"),
        Word(465, 473, "25"),
    ]
    assert [c.label for c in header_columns(line)] == ["Mar 26", "Sep 25", "Mar 25"]


def test_a_movement_column_on_the_same_line_is_excluded():
    """ "Mar 26 v. Sep 25" is a half-on-half movement, which is never copied as the answer."""
    line = [
        Word(353, 383, "Mar"),
        Word(384, 392, "26"),
        Word(481, 487, "v."),
        Word(489, 519, "Sep"),
        Word(520, 528, "25"),
    ]
    assert [c.label for c in header_columns(line)] == ["Mar 26"]


def test_a_movement_column_written_over_two_lines_is_excluded():
    """The packs also write the period on one line and "v. Sep 25" beneath it."""
    header = [
        Word(353, 383, "Mar"),
        Word(384, 392, "26"),
        Word(489, 519, "Mar"),
        Word(520, 528, "26"),
    ]
    below = [Word(489, 495, "v."), Word(497, 527, "Sep"), Word(528, 536, "25")]
    assert [c.label for c in header_columns(header, below)] == ["Mar 26"]


#: Westpac's 1H25 summary (p.13) sets its months over their years, and writes each
#: movement as "Mar 25" over "- Sept 24", a dash standing for "versus".
_WBC_1H25_MONTHS = [
    Word(298, 317, "March"),
    Word(364, 378, "Sept"),
    Word(420, 439, "March"),
    Word(480, 492, "Mar"),
    Word(493, 501, "25"),
    Word(541, 553, "Mar"),
    Word(555, 562, "25"),
]
_WBC_1H25_YEARS = [
    Word(60, 69, "$m"),
    Word(303, 317, "2025"),
    Word(364, 378, "2024"),
    Word(425, 439, "2024"),
    Word(473, 476, "-"),
    Word(477, 492, "Sept"),
    Word(493, 501, "24"),
    Word(537, 539, "-"),
    Word(541, 553, "Mar"),
    Word(555, 562, "24"),
]


def test_a_month_takes_the_year_printed_under_it():
    """Before this, the months alone named no period, the movement headers did, and their
    percentage changes were stored as Westpac's margin and return on equity."""
    columns = header_columns(_WBC_1H25_MONTHS, _WBC_1H25_YEARS)
    assert [c.label for c in columns] == ["March 2025", "Sept 2024", "March 2024"]


def test_the_line_of_years_is_part_of_the_header_not_a_row():
    from bankresearch.ingestion.layout_tables import extract_layout_tables

    def placed(y: float, line: list[Word]) -> list[tuple[float, float, float, float, str]]:
        return [(w.x0, y, w.x1, y + 8.0, w.text) for w in line]

    words = placed(100, _WBC_1H25_MONTHS) + placed(110, _WBC_1H25_YEARS)
    for y, label, figures in (
        (125, ("Net", "interest", "income"), ("9,351", "9,626", "9,127", "(3)", "2")),
        (137, ("Non-interest", "income"), ("1,442", "1,372", "1,463", "5", "(1)")),
    ):
        words += [(60.0 + 50 * i, y, 100.0 + 50 * i, y + 8.0, w) for i, w in enumerate(label)]
        words += [
            (x - 20.0, y, x, y + 8.0, figure)
            for x, figure in zip((317, 378, 439, 503, 562), figures, strict=True)
        ]
    (table,) = extract_layout_tables(words, page_no=13, doc_id="wbc-1h25-results-document")
    assert table.header[1:] == ("March 2025", "Sept 2024", "March 2024")
    assert [row[0] for row in table.rows] == ["Net interest income", "Non-interest income"]
    assert table.rows[0][1:] == ("9,351", "9,626", "9,127")


def test_a_day_before_the_next_month_is_not_this_months_year():
    """Westpac's 1H25 capital table (p.34) puts the day before each month: "31 March | 30
    Sept | 31 March" over "2025 | 2024 | 2024". Read as month and year, "March 30" took the
    first column to 2030 and its CET1 ratio was never stored."""
    line = [
        Word(242.7, 249.8, "31"),
        Word(251.4, 270.4, "March"),
        Word(313.7, 320.8, "30"),
        Word(322.4, 336.8, "Sept"),
        Word(375.3, 382.4, "31"),
        Word(384.1, 403.1, "March"),
        Word(448.6, 460.7, "Mar"),
        Word(462.3, 469.4, "25"),
        Word(512.6, 524.7, "Mar"),
        Word(526.3, 533.4, "25"),
    ]
    below = [
        Word(256.3, 270.4, "2025"),
        Word(322.6, 336.8, "2024"),
        Word(388.9, 403.1, "2024"),
        Word(442.1, 444.7, "-"),
        Word(446.3, 460.7, "Sept"),
        Word(462.3, 469.4, "24"),
        Word(508.4, 511.0, "-"),
        Word(512.6, 524.7, "Mar"),
        Word(526.3, 533.4, "24"),
    ]
    assert [c.label for c in header_columns(line, below)] == [
        "March 2025",
        "Sept 2024",
        "March 2024",
    ]
    # A month with its own year beside it keeps it: "Mar 26 Sep 25" is two periods.
    tight = [
        Word(353, 383, "Mar"),
        Word(384, 392, "26"),
        Word(394, 424, "Sep"),
        Word(425, 433, "25"),
    ]
    assert [c.label for c in header_columns(tight)] == ["Mar 26", "Sep 25"]


def test_a_repeated_period_keeps_only_its_leftmost_column():
    """A period cannot head two columns of one table; the leftmost is the period
    column and the other is a movement."""
    line = [
        Word(353, 383, "Mar"),
        Word(384, 392, "26"),
        Word(489, 519, "Mar"),
        Word(520, 528, "26"),
    ]
    assert [c.label for c in header_columns(line)] == ["Mar 26"]


def test_a_fiscal_label_header_is_recognised():
    line = [Word(300, 330, "1H26"), Word(350, 380, "1H25")]
    assert [c.label for c in header_columns(line)] == ["1H26", "1H25"]


def test_a_second_blocks_labels_stop_at_the_tables_label_column():
    """ANZ's full-year summary sets a half-year block and a full-year block side by side,
    each with its own "Sep 25". The full-year block was labelled "Net interest margin 1.54%
    1.56%", with the half-year figures glued on. Its labels end where the table's do.
    """
    from bankresearch.ingestion.layout_tables import extract_layout_tables

    def at(x0, y, text):
        return (x0, y, x0 + 6.0 * len(text), y + 8.0, text)

    header = [at(300, 100, "Sep"), at(320, 100, "25"), at(360, 100, "Mar"), at(380, 100, "25")]
    header += [at(480, 100, "Sep"), at(500, 100, "25"), at(540, 100, "Sep"), at(560, 100, "24")]
    rows = []
    for y, label, figures in (
        (120, "Net interest margin", ("1.54%", "1.56%", "-2", "1.55%", "1.57%")),
        (135, "Return on average equity", ("6.2%", "10.4%", "-", "8.2%", "9.4%")),
    ):
        x = 45.0
        for word in label.split():
            rows.append(at(x, y, word))
            x += 6.0 * len(word) + 4.0
        rows += [
            at(x, y, figure) for x, figure in zip((300, 360, 420, 480, 540), figures, strict=True)
        ]
    tables = extract_layout_tables(header + rows, page_no=11, doc_id="doc")
    by_header = {table.header[1:]: table for table in tables}
    full_year = by_header[("Sep 25", "Sep 24")]
    assert [row[0] for row in full_year.rows] == ["Net interest margin", "Return on average equity"]
    assert full_year.rows[0][1:] == ("1.55%", "1.57%")
    assert by_header[("Sep 25", "Mar 25")].rows[0][1:] == ("1.54%", "1.56%")


def test_a_ruled_table_that_lost_its_row_labels_is_not_readable():
    """Line detection reduced ANZ's 2025 summary to rows like ``('9,092', '', '17,961')``,
    and because the header still named a period, the rebuild that could read it was skipped.
    """
    from bankresearch.ingestion.parse_pdf import has_row_labels

    assert not has_row_labels([("9,092", "", "17,961"), ("1,915", "", "4,225"), ("$M", "", "$M")])
    assert has_row_labels([("Net interest income", "9,092", "8,869"), ("", "", "")])
    assert not has_row_labels([])


def test_one_period_is_not_enough_to_start_a_table():
    """`header_columns` reports what it finds; the extractor requires at least two,
    because a single period column cannot be a comparison."""
    from bankresearch.ingestion.layout_tables import _MIN_PERIOD_COLUMNS

    found = header_columns([Word(300, 330, "Mar"), Word(331, 340, "26")])
    assert len(found) == 1
    assert len(found) < _MIN_PERIOD_COLUMNS


# --- Bendigo, the unseen bank --------------------------------------------------------
#
# Bendigo's group results table (ben-1h26-results-document p.10), row for row. None of its
# headline, income total or impairment labels mapped, so an earnings question about Bendigo
# abstained even after the cold path had found and ingested the filing.

_BEN_ROWS = (
    ("Net interest income", "871.1", "844.1", "834.7"),
    ("Other operating income", "138.8", "130.1", "137.7"),
    ("Total income", "1,009.9", "974.2", "972.4"),
    ("Operating expenses", "(636.5)", "(611.0)", "(598.4)"),
    ("Operating performance", "373.4", "363.2", "374.0"),
    ("Credit reversals/(expenses)", "2.4", "4.2", "10.5"),
    ("Cash earnings before tax", "375.8", "367.4", "384.5"),
    ("Income tax expense", "(119.4)", "(118.0)", "(119.3)"),
    ("Cash earnings after tax", "256.4", "249.4", "265.2"),
    ("Non-cash items after tax", "(25.8)", "(563.3)", "(48.4)"),
    ("Statutory earnings/(loss) after tax", "230.6", "(313.9)", "216.8"),
)


def _bendigo_facts():  # type: ignore[no-untyped-def]
    from datetime import date

    from bankresearch.domain.models import DocMeta
    from bankresearch.ingestion.facts import extract_facts
    from bankresearch.ingestion.parse_pdf import ParsedDocument, ParsedPage, ParsedTable

    table = ParsedTable(
        table_id="ben_p10_L0",
        page_no=10,
        caption="Half year ended Change",
        header=("", "Dec 25", "Jun 25", "Dec 24"),
        rows=_BEN_ROWS,
        units="aud_m",
    )
    page = ParsedPage(
        page_no=10, text="Bendigo Bank\n2.3 GROUP FINANCIAL RESULTS", kind="text", tables=(table,)
    )
    document = ParsedDocument(doc_id="ben-1h26", sha256="0" * 64, page_count=74, pages=(page,))
    meta = DocMeta(
        doc_id="ben-1h26",
        bank_id="BEN",
        period="1H26",
        doc_type="RESULTS_DOCUMENT",
        publish_date=date(2026, 2, 16),
        url="https://www.bendigoadelaide.com.au/example.pdf",
    )
    return extract_facts(document, meta, ("1H26", "1H25"))


def test_bendigos_headline_income_and_impairment_lines_map():
    by = {(f.component, f.period): f.value for f in _bendigo_facts()}
    assert by[("headline_earnings", "1H26")] == Decimal("256.4")
    assert by[("headline_earnings", "1H25")] == Decimal("265.2")
    assert by[("operating_income", "1H26")] == Decimal("1009.9")
    # Net reversals in both halves: a negative charge, as the label's brackets say.
    assert by[("credit_impairment_charge", "1H26")] == Decimal("-2.4")
    assert by[("credit_impairment_charge", "1H25")] == Decimal("-10.5")


def test_bendigos_earnings_bridge_adds_up_to_the_cent():
    from bankresearch.ingestion.validate import validate_facts

    outcome = validate_facts(_bendigo_facts(), "BEN", ("1H26", "1H25"))
    bridge = [c for c in outcome.checks if c.name == "headline_earnings"]
    assert len(bridge) == 2
    assert all(check.passed and check.gap == 0 for check in bridge)
    assert not outcome.quarantined


def test_a_negative_amount_puts_its_sign_before_the_currency():
    from bankresearch.output.render_md import format_value

    assert format_value(Decimal("-10.5"), "aud_m") == "-A$10.5m"
    assert format_value(Decimal("1009.9"), "aud_m") == "A$1,009.9m"


# --- Bank of Queensland, the second unseen bank ----------------------------------------
#
# Its income statement (boq-fy25-results-document p.74) in the same words as Bendigo's.
# With "Total income" unmapped, its cost-to-income identity couldn't run.

_BOQ_ROWS = (
    ("Net interest income (1)", "1,515", "1,463"),
    ("Non-interest income (1)", "142", "137"),
    ("Total income", "1,657", "1,600"),
    ("Operating expenses (1)", "(1,072)", "(1,069)"),
    ("Underlying profit", "585", "531"),
    ("Loan impairment expense (1)", "(21)", "(20)"),
    ("Profit before tax", "564", "511"),
    ("Income tax expense (1)", "(181)", "(168)"),
    ("Cash earnings after tax", "383", "343"),
)


def test_bank_of_queenslands_income_statement_maps_and_adds_up():
    from datetime import date

    from bankresearch.domain.models import DocMeta
    from bankresearch.ingestion.facts import extract_facts
    from bankresearch.ingestion.parse_pdf import ParsedDocument, ParsedPage, ParsedTable
    from bankresearch.ingestion.validate import validate_facts

    table = ParsedTable(
        table_id="boq_p74_L0",
        page_no=74,
        caption="Year end performance",
        header=("", "Aug 25", "Aug 24"),
        rows=_BOQ_ROWS,
        units="aud_m",
    )
    page = ParsedPage(
        page_no=74,
        text="2. Group performance analysis.\n2.1 Income statement and key metrics",
        kind="text",
        tables=(table,),
    )
    document = ParsedDocument(doc_id="boq-fy25", sha256="0" * 64, page_count=228, pages=(page,))
    meta = DocMeta(
        doc_id="boq-fy25",
        bank_id="BOQ",
        period="FY25",
        doc_type="RESULTS_DOCUMENT",
        publish_date=date(2025, 10, 14),
        url="https://www.boq.com.au/example.pdf",
    )
    facts = extract_facts(document, meta, ("FY25", "FY24"))
    by = {(f.component, f.period): f.value for f in facts}
    assert by[("operating_income", "FY25")] == Decimal("1657")
    assert by[("headline_earnings", "FY25")] == Decimal("383")
    assert by[("credit_impairment_charge", "FY25")] == Decimal("21")
    outcome = validate_facts(facts, "BOQ", ("FY25", "FY24"))
    bridge = [c for c in outcome.checks if c.name == "headline_earnings"]
    assert len(bridge) == 2 and all(c.passed and c.gap == 0 for c in bridge)
