"""The issuer's own commentary, found by code on the real filings.

What matters is what is never shown: a sentence explaining the movement against the prior
half, a division's result in the group's vocabulary, and anything not printed on its page.
"""

from __future__ import annotations

import re
from datetime import date

import pytest

from bankresearch.settings import get_settings


def _passages(bank: str, period: str, metric: str, as_of: date, topics: list[str]):
    from bankresearch.agent.steps.resolve import resolve_request
    from bankresearch.domain.registry import get_registry
    from bankresearch.ingestion.artifacts import read_processed
    from bankresearch.memory.db import connect
    from bankresearch.memory.fact_store import FactStore
    from bankresearch.retrieval.commentary import find_commentary

    settings = get_settings()
    if not settings.db_path.exists():
        pytest.skip("no ingested corpus: run `make ingest` first")
    with connect(settings.db_path) as connection:
        metas = FactStore(connection).documents_for(bank, [period], as_of)
    docs = [
        (document, meta)
        for meta in metas
        if (document := read_processed(meta.doc_id, settings.processed_dir)) is not None
    ]
    if not docs:
        pytest.skip("no ingested corpus: run `make ingest` first")
    spec = resolve_request(bank, period, metric, as_of=as_of)
    found = find_commentary(
        spec, docs, topics, fy_end_month=get_registry().banks[bank].fy_end_month
    )
    return found, {meta.doc_id: document for document, meta in docs}


def _condensed(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower().replace(chr(0x2019), "'"))


CASES = [
    (
        "CBA",
        "1H26",
        "headline_earnings",
        date(2026, 3, 31),
        ["headline_earnings", "operating_expenses", "credit_impairment"],
    ),
    ("NAB", "1H26", "nim", date(2026, 5, 31), ["nim", "nii", "aiea"]),
    ("WBC", "1H26", "nim", date(2026, 5, 31), ["nim", "nii", "aiea"]),
    (
        "ANZ",
        "1H26",
        "cost_to_income",
        date(2026, 5, 31),
        ["cost_to_income", "operating_expenses", "other_income"],
    ),
]


@pytest.mark.parametrize(("bank", "period", "metric", "as_of", "topics"), CASES)
def test_every_passage_is_printed_on_its_page(bank, period, metric, as_of, topics):
    passages, documents = _passages(bank, period, metric, as_of, topics)
    assert passages, f"no commentary found for {bank} {metric}"
    for passage in passages:
        page = documents[passage.doc_id].page(passage.page_no)
        assert _condensed(passage.text) in _condensed(page.text), passage.text


def test_cbas_cost_growth_comes_with_the_banks_reason():
    passages, _ = _passages(*CASES[0][:4], CASES[0][4])
    expenses = [p.text for p in passages if p.topic == "operating_expenses"]
    assert any("inflation" in text for text in expenses)


def test_nabs_half_on_half_commentary_is_never_shown():
    """NAB compares with the September half unless a heading says otherwise, and explains
    both movements on the same page: net interest income rose $718m year on year and $210m
    on the half."""
    passages, _ = _passages(*CASES[1][:4], CASES[1][4])
    texts = " ".join(p.text for p in passages)
    assert "$210 million" not in texts and "3 basis points." not in texts
    assert "11 basis points" in texts or "$718 million" in texts or "$616 million" in texts


def test_westpacs_prior_half_section_is_left_out():
    passages, _ = _passages(*CASES[2][:4], CASES[2][4])
    texts = " ".join(p.text for p in passages)
    assert "decreased by 1%" not in texts, "that is First Half 2026 against Second Half 2025"
    assert not any(p.page_no in (50, 51) for p in passages), "the directors' report restates"


def test_a_divisions_result_is_not_the_groups():
    passages, _ = _passages(
        "CBA", "1H26", "headline_earnings", date(2026, 3, 31), ["headline_earnings"]
    )
    assert not any("Retail Banking Services" in p.text for p in passages)


def test_a_sentence_split_from_its_subject_explains_the_line_before_it():
    """NAB's collective write-back "was driven by a net release of forward-looking provisions,
    combined with the release of provisions held for customers that transferred to
    individually assessed" (FY25 p.26). The PDF split it from its subject. It was dropped for
    stating no movement, and the sentence after it was filed under the individual charge
    because the fragment mentions it in passing."""
    topics = ["credit_impairment_charge", "impairment_individual", "impairment_collective"]
    passages, _ = _passages("NAB", "FY25", "credit_impairment_charge", date(2025, 11, 30), topics)
    texts = {passage.text: passage.topic for passage in passages}
    fragment = next(text for text in texts if text.startswith("was driven by a net release"))
    assert texts[fragment] in {"credit_impairment_charge", "impairment_collective"}
    offset = next(
        text
        for text in texts
        if text.startswith("This was partially offset by the impact of volume")
    )
    assert texts[offset] == "impairment_collective"
