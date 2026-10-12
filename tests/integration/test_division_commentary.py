"""A division's driver is explained on the division's own pages.

The group's commentary leaves divisional pages out on purpose, so a driver attributed to a
division (CBA's loan impairment expense by division, DESIGN.md ADR-038) had no sentence of
the issuer's to cite for its cause.
"""

from __future__ import annotations

from datetime import date

import pytest

from bankresearch.settings import get_settings


def _held(bank: str, period: str, as_of: date):
    from bankresearch.agent.steps.resolve import resolve_request
    from bankresearch.ingestion.artifacts import read_processed
    from bankresearch.memory.db import connect
    from bankresearch.memory.fact_store import FactStore

    settings = get_settings()
    if not settings.db_path.exists():
        pytest.skip("no ingested corpus: run `make ingest` first")
    spec = resolve_request(bank, period, "credit_impairment_charge", as_of=as_of)
    with connect(settings.db_path) as connection:
        metas = FactStore(connection).documents_for(bank, [period], as_of)
    held = [
        (document, meta)
        for meta in metas
        if (document := read_processed(meta.doc_id, settings.processed_dir)) is not None
    ]
    if not held:
        pytest.skip("no ingested corpus: run `make ingest` first")
    return spec, held


def test_each_division_is_explained_from_its_own_pages():
    from bankresearch.domain.registry import get_registry
    from bankresearch.retrieval.commentary import find_division_commentary

    spec, held = _held("CBA", "FY26", date(2026, 8, 31))
    divisions = get_registry().banks["CBA"].impairment_divisions
    wanted = {key: f"impairment_div_{key}" for key in divisions.values()}
    passages = find_division_commentary(spec, held, divisions, wanted, fy_end_month=6)
    by_topic = {passage.topic: passage for passage in passages}
    business = by_topic["impairment_div_business"]
    assert business.page_no == 69
    assert "driven by lower individually assessed provision charges" in business.text
    assert "$378 million" in by_topic["impairment_div_retail"].text
    # Nothing about a loss rate, and nothing from a presentation's navigation bar.
    assert not any("as a percentage of" in passage.text for passage in passages)
    assert {passage.doc_id for passage in passages} == {"cba-fy26-results-document"}


def test_a_navigation_bar_naming_every_division_starts_none():
    from bankresearch.ingestion.parse_pdf import ParsedDocument, ParsedPage
    from bankresearch.retrieval.commentary import division_pages

    def page(number: int, text: str) -> ParsedPage:
        return ParsedPage(page_no=number, text=text, kind="text", heading_path=(), tables=())

    document = ParsedDocument(
        doc_id="x",
        sha256="0",
        page_count=3,
        pages=(
            page(1, "Divisions\nRetail Banking Services\nBusiness Banking\nNew Zealand\n"),
            page(2, "Divisional Performance\nBusiness Banking\nLoan impairment expense fell."),
            page(3, "Divisional Performance (continued)\nBusiness Banking (continued)\nMore."),
        ),
        outline=(),
    )
    divisions = {
        "Retail Banking Services": "retail",
        "Business Banking": "business",
        "New Zealand": "new_zealand",
    }
    assert division_pages(document, divisions) == {2: "business", 3: "business"}
