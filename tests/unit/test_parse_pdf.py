"""Parsing, page classification and the table registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from bankresearch.ingestion.parse_pdf import (
    PyMuPdfParser,
    classify_page,
    looks_like_a_table,
    page_features,
)
from bankresearch.ingestion.tables import detect_restated, map_columns

FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthetic_results_document.pdf"
ANZ_LIKE = 9  # the fixture reports a half ended 31 March, so a September year end


@pytest.fixture(scope="module")
def document():
    assert FIXTURE.exists(), "run python tests/fixtures/make_fixture_pdf.py"
    return PyMuPdfParser().parse(FIXTURE, "fixture-1h26-rd")


# --- the document as a whole ----------------------------------------------


def test_the_page_count_matches_the_pdf(document):
    assert document.page_count == 5
    assert len(document.pages) == 5


def test_the_document_is_addressed_by_the_hash_of_its_bytes(document):
    assert len(document.sha256) == 64
    again = PyMuPdfParser().parse(FIXTURE, "fixture-1h26-rd")
    assert again.sha256 == document.sha256


def test_the_outline_comes_from_the_pdfs_own_table_of_contents(document):
    assert document.outline == (
        ("Cover", 1),
        ("Key metrics", 2),
        ("Capital", 3),
        ("Net interest margin movement", 4),
        ("Disclaimer", 5),
    )


def test_each_page_carries_its_heading(document):
    assert document.page(3).heading_path == ("Capital",)
    assert document.page(4).heading_path == ("Net interest margin movement",)


# --- page classification --------------------------------------------------


def test_the_disclaimer_page_is_boilerplate(document):
    assert document.page(5).kind == "boilerplate"


def test_table_pages_are_not_classified_as_text(document):
    assert document.page(2).kind in {"table", "mixed"}
    assert document.page(3).kind in {"table", "mixed"}


def test_boilerplate_wins_even_when_the_page_is_dense_with_numbers():
    """A disclaimer is dense, confident and about nothing."""
    assert classify_page("Important notice: 12.39 1.86 5016 43.33", 1, 0.0) == "boilerplate"


def test_features_are_measurable_not_guessed():
    features = page_features("abc 123", table_count=2, image_area_share=0.4)
    assert features["characters"] == 7
    assert features["table_count"] == 2
    assert features["image_area_share"] == 0.4
    assert 0 < features["numeric_density"] < 1


# --- the table registry ---------------------------------------------------


def test_only_real_tables_are_registered(document):
    """The text-alignment fallback must not carve a banner into columns."""
    assert len(document.tables()) == 3
    assert document.page(1).tables == ()
    assert document.page(5).tables == ()


def test_each_table_has_a_stable_id_carrying_its_page(document):
    for table in document.tables():
        assert table.table_id.startswith("fixture-1h26-rd_p")
        assert f"_p{table.page_no}_" in table.table_id


def test_captions_are_taken_from_the_line_above_the_header(document):
    captions = {t.page_no: t.caption for t in document.tables()}
    assert "Group performance" in captions[2]
    assert "Capital adequacy" in captions[3]


def test_units_are_read_from_the_caption(document):
    units = {t.page_no: t.units for t in document.tables()}
    assert units[2] == "aud_m"
    assert units[3] == "aud_m"
    assert units[4] == "bps"


def test_a_restatement_footnote_flags_the_table_on_that_page(document):
    flags = {t.page_no: t.restated for t in document.tables()}
    assert flags[2] is True
    assert flags[3] is False


def test_a_disclaimers_representation_wording_is_not_a_restatement():
    """ "no representation or warranty" must not read as "re-presented"."""
    assert not detect_restated("no representation or warranty of any kind")
    assert detect_restated("Prior periods have been re-presented")


def test_cells_are_addressable_by_row_label_and_column(document):
    key_metrics = next(t for t in document.tables() if t.page_no == 2)
    assert key_metrics.cell("Cash profit", "Mar 26") == "4450"
    assert key_metrics.cell("Cash profit", "Mar 25") == "4300"
    assert key_metrics.cell("CET1 ratio (%)", "Mar 26") == "12.39"


def test_a_missing_row_or_column_returns_none_rather_than_a_neighbour(document):
    key_metrics = next(t for t in document.tables() if t.page_no == 2)
    assert key_metrics.cell("Dividend per share", "Mar 26") is None
    assert key_metrics.cell("Cash profit", "Mar 24") is None


def test_a_table_renders_as_markdown_for_the_locate_loop(document):
    markdown = next(t for t in document.tables() if t.page_no == 3).to_markdown()
    assert markdown.startswith("| ")
    assert "Total risk weighted assets" in markdown
    assert markdown.count("\n") >= 8


# --- the column-to-period map, end to end on a parsed table --------------


def test_the_parsed_header_row_maps_to_periods_including_the_pcp(document):
    key_metrics = next(t for t in document.tables() if t.page_no == 2)
    mapping = map_columns(
        [h for h in key_metrics.header if h],
        fy_end_month=ANZ_LIKE,
        doc_period="1H26",
    )
    assert mapping == {"Mar 26": "1H26", "Sep 25": "2H25", "Mar 25": "1H25"}


def test_the_pcp_column_is_the_one_a_year_back_not_the_adjacent_half(document):
    capital = next(t for t in document.tables() if t.page_no == 3)
    mapping = map_columns(
        [h for h in capital.header if h], fy_end_month=ANZ_LIKE, doc_period="1H26"
    )
    pcp_header = next(h for h, label in mapping.items() if label == "1H25")
    assert capital.cell("Common Equity Tier 1 capital", pcp_header) == "48840"
    assert capital.cell("Common Equity Tier 1 capital", "Sep 25") == "49803"


# --- walk pages ----------------------------------------------------------


def test_pages_likely_to_hold_a_walk_are_flagged(document):
    """Chart vision is expensive, so it runs only where a walk might be."""
    assert document.walk_pages() == (4,)


def test_the_walk_table_is_extracted_with_its_steps(document):
    walk = next(t for t in document.tables() if t.page_no == 4)
    assert walk.cell("Net interest margin 1H25", "bps") == "180.0"
    assert walk.cell("Net interest margin 1H26", "bps") == "186.0"
    assert walk.cell("Replicating portfolio and capital hedge", "bps") == "11.0"


# --- the table-shape filter ----------------------------------------------


def test_a_single_column_extraction_is_not_a_table():
    assert not looks_like_a_table([["Heading"], ["More text"], ["Even more"]])


def test_a_paragraph_carved_into_columns_is_not_a_table():
    assert not looks_like_a_table([["Important", "notice", "this"], ["document", "is", "prepared"]])


def test_a_two_column_financial_table_is_a_table():
    assert looks_like_a_table(
        [["", "bps"], ["Opening", "180.0"], ["Pricing", "-7.0"], ["Closing", "186.0"]]
    )


def test_the_fixture_generator_is_byte_reproducible(tmp_path):
    """CI regenerates the fixture and fails if it differs from the committed one. That
    catches a generator change that was never committed, which would otherwise leave the
    tests checking a file the code no longer produces. PyMuPDF stamps a creation date and a
    document id by default, so both are pinned.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1] / "fixtures"))
    from make_fixture_pdf import build

    first = build(tmp_path / "a.pdf").read_bytes()
    second = build(tmp_path / "b.pdf").read_bytes()
    assert first == second
    assert first == FIXTURE.read_bytes(), (
        "the committed fixture differs from what the generator now produces; "
        "run python tests/fixtures/make_fixture_pdf.py and commit the result"
    )
