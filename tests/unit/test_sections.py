"""Section markers: the heading rows that decide what a figure means.

Every case here is a row taken from one of the four majors' 2026 filings, where the
figure is extracted correctly and means something other than what the row label says.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from bankresearch.domain.models import Basis
from bankresearch.ingestion.facts import line_item_index
from bankresearch.ingestion.layout_tables import Word, group_lines, header_columns
from bankresearch.ingestion.sections import Section, SectionRules, is_marker_row


@pytest.fixture(scope="module")
def rules() -> SectionRules:
    return line_item_index("NAB").sections


# --- which rows may move the section --------------------------------------


def test_a_row_with_any_figure_is_not_a_marker():
    assert is_marker_row(["", "", ""])
    assert is_marker_row([])
    assert not is_marker_row(["2.04", "", ""])
    assert not is_marker_row(["", "", "(13.1)"])


def test_a_figure_is_never_relabelled_by_its_neighbours(rules):
    """The check that makes the whole mechanism safe.

    Only a row with no figures at all can change the reading, so a data row can never be
    mistaken for a heading, however much its label looks like one.
    """
    section = Section()
    assert rules.advance(section, "Retail Banking Services", None).scope == "division"
    # The same label on a row that carries a figure is simply not offered to advance().
    assert not is_marker_row(["2,727", "2,640", "2,690"])


# --- divisions ------------------------------------------------------------


@pytest.mark.parametrize(
    "heading",
    [
        "Retail Banking Services",
        "Retail Banking Services (continued)",
        "Business and Private Banking",
        "Institutional Banking and Markets",
        "New Zealand",
    ],
)
def test_a_division_heading_stops_the_rows_under_it_being_read(rules, heading):
    """CBA's Retail margin is 2.50% against a group margin of 2.04%, under one header."""
    section = rules.advance(Section(), heading, None)
    assert section.scope == "division"
    assert not section.readable


def test_a_group_heading_returns_the_reading_to_the_group(rules):
    section = rules.advance(Section(scope="division"), "Group Performance", None)
    assert section.scope == "group"
    assert section.readable


def test_a_division_heading_clears_what_the_previous_section_said(rules):
    """A division is a top-level section, so the basis in force does not leak into it."""
    section = Section(earnings_basis="cash", ex_notable_items=True, component="roe")
    moved = rules.advance(section, "New Zealand", None)
    assert moved == Section(scope="division")


# --- basis ----------------------------------------------------------------


def test_one_heading_may_state_several_dimensions(rules):
    """NAB writes "Group performance - cash earnings basis excluding large notable items"."""
    section = rules.advance(
        Section(), "Group performance - cash earnings basis excluding large notable items", None
    )
    assert section.earnings_basis == "cash"
    assert section.ex_notable_items is True
    assert section.scope == "group"


def test_the_more_specific_phrase_wins(rules):
    """ "excluding large notable items" must not be read as the bare "notable items"."""
    dimensions = rules.basis_dimensions("Cash earnings excluding large notable items")
    assert dimensions == {"earnings_basis": "cash", "ex_notable_items": True}


def test_the_statutory_and_cash_sections_are_distinguished(rules):
    """NAB prints cost to income three times in one table: 59.1%, 58.6% and 46.2%."""
    statutory = rules.advance(Section(), "Group performance - statutory basis", None)
    cash = rules.advance(Section(), "Group performance - cash earnings basis", None)
    assert statutory.earnings_basis == "statutory"
    assert cash.earnings_basis == "cash"
    assert statutory.applied_to(Basis()).differs_from(cash.applied_to(Basis()))


def test_continuing_operations_is_read_off_the_heading(rules):
    assert (
        rules.advance(
            Section(), "Group Performance from continuing operations", None
        ).continuing_ops
        is True
    )
    assert (
        rules.advance(
            Section(), "Group Performance including discontinued operations", None
        ).continuing_ops
        is False
    )


# --- regulatory level -----------------------------------------------------


def test_level_1_capital_is_not_read_as_the_group_figure(rules):
    """Westpac prints Level 2 capital of $56,936m and Level 1 of $53,722m under one
    header. Taking Level 1 capital over Level 2 RWA gives a CET1 ratio 70bp below the
    one the bank published, from two figures both read correctly."""
    level2 = rules.advance(Section(), "Level 2 regulatory capital structure", None)
    level1 = rules.advance(Section(), "Level 1 regulatory capital structure", None)
    assert level2.readable and level2.scope == "level2"
    assert not level1.readable and level1.scope == "level1"


# --- a label split over two rows ------------------------------------------


def test_a_heading_lends_its_name_to_the_bare_rows_beneath_it(rules):
    """CBA writes "Return on equity (ROE) (%)" as a heading and labels the rows that
    carry the figures only "Statutory basis" and "Cash basis"."""
    section = rules.advance(Section(), "Return on equity (ROE) (%)", "roe")
    assert section.component == "roe"
    assert rules.basis_dimensions("Cash basis") == {"earnings_basis": "cash"}


def test_a_heading_that_names_nothing_ends_the_block(rules):
    """So "Personnel" under an "Operating expenses" heading never inherits its name."""
    section = rules.advance(Section(component="roe"), "Shareholder Returns", None)
    assert section.component is None


# --- the section's own unit -----------------------------------------------


def test_a_heading_may_state_the_unit_for_the_rows_under_it(rules):
    """NAB sets "Volumes ($bn)" over balances that are otherwise bare numbers."""
    section = rules.advance(Section(scope="division"), "Volumes ($bn)", None)
    assert section.unit == "aud_bn"
    # And it does not end the division it sits inside.
    assert section.scope == "division"


# --- label ranking --------------------------------------------------------


def test_the_ratios_own_denominator_is_preferred():
    """NAB's ROE note gives "Average equity" of $62,937m and, two rows below, "Total
    average equity (attributable to owners of the Company)" of $62,227m. Only the
    second is the denominator of the ratio the bank published."""
    index = line_item_index("NAB")
    specific = index.rank_for(
        "average_equity", "Total average equity (attributable to owners of the Company)"
    )
    generic = index.rank_for("average_equity", "Average equity")
    assert specific < generic


def test_ranking_is_inert_for_a_component_that_declares_no_preference():
    index = line_item_index("NAB")
    assert index.rank_for("nim", "Net interest margin") == 0
    assert index.rank_for("nim", "anything at all") == 0


# --- the layout defects these exposed -------------------------------------


def test_a_row_is_not_torn_in_half_by_a_baseline_wobble():
    """Words a point apart used to land either side of a rounding band, which split "Other
    operating income 526 517 515" into a lone 526 and a row missing its current period.
    That's a dropped figure that leaves a well-formed row behind.
    """
    words = [
        (62.0, 100.0, 120.0, 108.0, "Other"),
        (124.0, 100.4, 180.0, 108.4, "operating"),
        (184.0, 101.6, 220.0, 109.6, "income"),
        (328.0, 100.2, 350.0, 108.2, "526"),
        (379.0, 101.1, 401.0, 109.1, "517"),
        (429.0, 100.9, 451.0, 108.9, "515"),
    ]
    lines = group_lines(words)
    assert len(lines) == 1
    assert [w.text for w in lines[0]] == ["Other", "operating", "income", "526", "517", "515"]


def test_a_two_line_header_keeps_the_period_columns_not_the_movement_ones():
    """NAB sets its divisional headers as

        Mar 26 | Sep 25 | Mar 25 | Mar 26 v | Mar 26 v
        $m     | $m     | $m     | Sep 25 % | Mar 25 %

    The second line names two periods that are the right-hand halves of the two
    movement columns. Choosing the richest single line took them and read every row's
    movement as its figure.
    """
    first = [
        Word(316, 341, "Mar"),
        Word(332, 341, "26"),
        Word(366, 391, "Sep"),
        Word(382, 391, "25"),
        Word(416, 440, "Mar"),
        Word(431, 440, "25"),
        Word(464, 489, "Mar"),
        Word(480, 489, "26"),
        Word(491, 495, "v"),
        Word(514, 539, "Mar"),
        Word(530, 539, "26"),
        Word(540, 544, "v"),
    ]
    assert [c.label for c in header_columns(first)] == ["Mar 26", "Sep 25", "Mar 25"]


def test_a_repeated_period_keeps_its_leftmost_column():
    line = [
        Word(316, 341, "Mar"),
        Word(332, 341, "26"),
        Word(464, 489, "Mar"),
        Word(480, 489, "26"),
        Word(366, 391, "Sep"),
        Word(382, 391, "25"),
    ]
    columns = header_columns(sorted(line, key=lambda w: w.x0))
    assert [(c.label, c.x0) for c in columns] == [("Mar 26", 316), ("Sep 25", 366)]


def test_a_period_set_against_a_trailing_marker_is_a_movement_column():
    """CBA's full-year summary prints three blocks side by side (statutory full year, cash
    full year, cash half year), each with its own comparison column headed "Jun 26 vs" over
    "Jun 25 %". Read as a period, that column won on position and every row's figure became
    its percentage movement: net interest income of $24,023m was read as 7.
    """
    line = [
        Word(225, 257, "30"),
        Word(235, 247, "Jun"),
        Word(249, 257, "26"),
        Word(263, 276, "Jun"),
        Word(277, 285, "26"),
        Word(287, 295, "vs"),
    ]
    assert [c.label for c in header_columns(line)] == ["Jun 26"]


def test_a_trailing_percent_marks_the_continuation_of_a_movement_header():
    line = [Word(387, 399, "Jun"), Word(401, 409, "25"), Word(411, 417, "%")]
    assert header_columns(line) == []


def test_a_period_column_followed_by_a_separate_movement_column_survives():
    """The marker has to be set right against the period. "Mar 26" as its own column with
    "v. Sep 25" heading the next one is 89pt away, not 2pt.
    """
    line = [
        Word(353, 383, "Mar"),
        Word(384, 392, "26"),
        Word(481, 487, "v."),
        Word(489, 519, "Sep"),
        Word(520, 528, "25"),
    ]
    assert [c.label for c in header_columns(line)] == ["Mar 26"]


# --- what may be held against what ----------------------------------------


def test_a_figures_precision_is_read_in_the_unit_it_was_published_in():
    """NAB publishes risk-weighted assets as "$426.4bn", which is known to $50m. Measured
    after conversion the figure is 426,400.0 and its precision reads as $0.05m, so a
    capital ratio right to the digit the bank printed fails by a fraction of a bp."""
    from bankresearch.domain.units import value_tolerance

    assert value_tolerance("aud_bn", reported=Decimal("426.4")) == Decimal("50.00")
    assert value_tolerance("aud_m", reported=Decimal("8445")) == Decimal("0.5")
    assert value_tolerance("pct", reported=Decimal("12.01")) == Decimal("0.500")
    assert value_tolerance("pct", reported=Decimal("1.8")) == Decimal("5.00")


def test_figures_on_different_bases_are_not_held_against_each_other():
    """Westpac publishes its expense-to-income ratio excluding notable items and its
    operating income including them. Holding one against the other reports a 90bp error
    in two figures that are both exactly as the bank printed them."""
    from bankresearch.ingestion.validate import _comparable

    assert _comparable([Basis(ex_notable_items=True), Basis(ex_notable_items=True)])
    assert not _comparable([Basis(ex_notable_items=True), Basis(ex_notable_items=False)])


def test_a_figure_with_no_earnings_basis_is_comparable_with_either():
    """A balance or a capital ratio has no cash-versus-statutory reading to disagree
    about, so it must not block a check against figures that do."""
    from bankresearch.ingestion.validate import _comparable

    assert _comparable([Basis(earnings_basis="cash"), Basis(earnings_basis="n/a")])
    assert not _comparable([Basis(earnings_basis="cash"), Basis(earnings_basis="statutory")])


# --- several tables typeset as one ----------------------------------------


def test_a_header_naming_a_period_twice_is_cut_into_blocks():
    """CBA's full-year summary sets the statutory full year, the cash full year and the cash
    half year side by side, so "30 Jun 26" heads three columns meaning three different
    things. Keeping only the leftmost read the statutory current year against the cash prior
    year, a movement that never happened.
    """
    from bankresearch.ingestion.layout_tables import Column, split_into_blocks

    blocks = split_into_blocks(
        [
            Column("Jun 26", 225, 257),
            Column("Jun 26", 308, 340),
            Column("Jun 25", 347, 379),
            Column("Jun 26", 430, 462),
            Column("Dec 25", 470, 500),
        ]
    )
    assert [[column.label for column in block] for block in blocks] == [
        ["Jun 26"],
        ["Jun 26", "Jun 25"],
        ["Jun 26", "Dec 25"],
    ]


def test_a_header_naming_each_period_once_is_a_single_block():
    from bankresearch.ingestion.layout_tables import Column, split_into_blocks

    columns = [Column("Mar 26", 316, 341), Column("Sep 25", 366, 391), Column("Mar 25", 416, 440)]
    assert split_into_blocks(columns) == [columns]


def test_a_wrapped_label_is_joined_to_the_row_that_carries_the_figures():
    """Westpac sets "Common Equity Tier 1 (CET1) capital after / deductions ($m)" over two
    lines, so the row carrying $56,936m is labelled only "deductions ($m)"."""
    index = line_item_index("WBC")
    assert index.components_for("deductions ($m)") == ()
    assert "cet1_capital" in index.components_for(
        "Common Equity Tier 1 (CET1) capital after deductions ($m)"
    )


def test_a_heading_that_states_a_basis_is_not_a_wrapped_name(rules):
    """Westpac sets "Business performance - excluding Notable Items" above the margin it
    leads with. Read as the first line of the margin's own name, the joined label is an
    excluded variant and the margin gets thrown away, so a heading that says something about
    the rows below it is never treated as a wrap.
    """
    assert rules.states("Business performance - excluding Notable Items")
    assert rules.states("Level 2 regulatory capital structure")
    assert rules.states("Retail Banking Services")
    assert rules.states("Volumes ($bn)")


def test_a_line_that_says_nothing_is_available_as_a_wrapped_name(rules):
    assert not rules.states("Common Equity Tier 1 (CET1) capital after")
    assert not rules.states("Performance indicators")


def test_a_wrap_candidate_is_not_bounded_like_a_heading():
    """The prefix that matters is 91 characters long: "divided by: Net profit attributable
    to owners of WBC (adjusted for RSP dividends) excluding". A heading must be short
    because it changes every row beneath it; a wrap only offers one more reading."""
    from bankresearch.ingestion.sections import is_heading_label, is_wrap_candidate

    long_prefix = (
        "divided by: Net profit attributable to owners of WBC "
        "(adjusted for RSP dividends) excluding"
    )
    assert len(long_prefix) > 90
    assert not is_heading_label(long_prefix)
    assert is_wrap_candidate(long_prefix)


def test_a_footnote_is_not_a_wrapped_name():
    """A footnote mentioning notable items must not suppress the row that follows it."""
    from bankresearch.ingestion.sections import is_wrap_candidate

    assert not is_wrap_candidate("a. Pre-provision profit is a subtotal and is defined on page 92.")
    assert not is_wrap_candidate("2 Presented on a “cash basis”.")


def test_a_movement_heading_is_not_the_tables_unit():
    """Westpac captions its statutory income statement "Half Year Half Year Half Year
    % Mov't". Reading that "%" as the unit made the income tax expense of $1,491m into
    1,491%, which was discarded as implausible and took the earnings bridge with it."""
    from bankresearch.ingestion.tables import detect_units

    assert detect_units("Half Year Half Year Half Year % Mov't") is None
    assert detect_units("% change") is None
    assert detect_units("Ratios (%)") == "pct"
    assert detect_units("Mov't % $m") == "aud_m"


def test_group_is_max_needs_a_material_difference():
    """ANZ gives average interest earning assets as "$1,142.1bn" on its average balance
    sheet and as 1,142,128 in an appendix. Letting the appendix win for one period alone
    split the component across two tables, after which the coherence rule dropped the
    period the margin identity needed."""
    from datetime import date

    from bankresearch.domain.models import Fact
    from bankresearch.ingestion.facts import _prefer

    def fact(value: str, unit: str, page: int) -> Fact:
        return Fact(
            fact_id="",
            bank_id="ANZ",
            period="1H25",
            component="aiea",
            value=Decimal(value),
            published_value=Decimal(value),
            unit=unit,
            basis=Basis(),
            doc_id="d",
            doc_type="RESULTS_DOCUMENT",
            page_no=page,
            publish_date=date(2026, 5, 1),
        )

    index = line_item_index("ANZ")
    rounded = fact("1142.1", "aud_bn", 11)
    exact = fact("1142128", "aud_m", 17)
    assert not _prefer(exact, rounded, "cash", index)
    # A divisional repeat differs by far more than the rounding and still loses.
    assert _prefer(fact("1142128", "aud_m", 17), fact("448489", "aud_m", 20), "cash", index)
