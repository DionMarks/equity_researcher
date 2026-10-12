"""The tool surface, checked for what it refuses as much as what it returns, and the model
layer, checked for the constraints the API sets."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from bankresearch.agent.steps.resolve import resolve_request
from bankresearch.domain.models import DocMeta
from bankresearch.ingestion.parse_pdf import PyMuPdfParser
from bankresearch.llm.budget import Budget
from bankresearch.llm.cache import CachedClient, ResponseCache
from bankresearch.llm.client import (
    CacheMiss,
    LLMRequest,
    LLMResponse,
    LLMUnavailable,
    ScriptedClient,
    ToolCall,
    Usage,
    blocks_by_type,
    cost_usd,
    request_hash,
)
from bankresearch.memory.db import connect
from bankresearch.retrieval.chunks import chunk_document
from bankresearch.retrieval.hybrid import Retriever, tokenize
from bankresearch.tools.doc_tools import DOCUMENT_WRAPPER, Toolkit
from bankresearch.tools.registry import (
    LLM_CALLABLE,
    ORCHESTRATOR_ONLY,
    SUBMIT_READING_LIST,
    tool_definitions,
)
from bankresearch.tools.schemas import OutlineResult, PageResult, TableResult, ToolError

FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthetic_results_document.pdf"
PUB = date(2026, 5, 1)


@pytest.fixture(scope="module")
def toolkit():
    document = PyMuPdfParser().parse(FIXTURE, "xmpl-1h26-rd")
    meta = DocMeta(
        doc_id="xmpl-1h26-rd",
        bank_id="XMPL",
        period="1H26",
        doc_type="RESULTS_DOCUMENT",
        publish_date=PUB,
        url=FIXTURE.resolve().as_uri(),
        page_count=document.page_count,
    )
    spec = resolve_request("XMPL", "1H26", "nim", as_of=date(2026, 5, 31))
    return Toolkit(spec=spec, documents={meta.doc_id: document}, metas={meta.doc_id: meta})


# --- the tool surface -----------------------------------------------------


def test_only_navigation_and_reading_tools_are_exposed():
    assert set(LLM_CALLABLE) == {
        "get_outline",
        "search_docs",
        "read_pages",
        "get_table",
        "read_chart",
        SUBMIT_READING_LIST,
    }


def test_nothing_deterministic_or_security_relevant_is_exposed():
    """Arithmetic, conflict rules, confidence, verification and URL fetching are code
    the model cannot call."""
    assert not set(LLM_CALLABLE) & set(ORCHESTRATOR_ONLY)
    for forbidden in ("decompose", "score_confidence", "verify", "fetch_url", "get_facts"):
        assert forbidden in ORCHESTRATOR_ONLY
        assert forbidden not in LLM_CALLABLE


def test_every_tool_is_registered_strict_and_closed():
    for definition in tool_definitions():
        assert definition["strict"] is True
        assert definition["input_schema"]["additionalProperties"] is False
        assert definition["description"]


def test_definitions_are_byte_identical_across_calls():
    """They form the prompt-cache prefix, so they must not vary."""
    assert tool_definitions() == tool_definitions()


def test_an_argument_the_schema_does_not_define_is_rejected(toolkit):
    """Ignoring an unknown argument would let a model believe it narrowed a search
    that was in fact unfiltered."""
    with pytest.raises(ValidationError):
        toolkit.dispatch("get_outline", {"doc_id": "xmpl-1h26-rd", "bank_id": "ANZ"})


def test_an_unknown_tool_returns_an_actionable_error(toolkit):
    result = toolkit.dispatch("compute_movement", {})
    assert isinstance(result, ToolError)
    assert "get_outline" in result.suggestion


# --- scope ----------------------------------------------------------------


def test_a_document_outside_the_request_cannot_be_read(toolkit):
    """The distractor case: the corpus holds other issuers, this request may not."""
    result = toolkit.dispatch("get_outline", {"doc_id": "anz-1h26-results-document"})
    assert isinstance(result, ToolError)
    assert result.error == "document_not_in_scope"
    assert "xmpl-1h26-rd" in result.suggestion


def test_search_is_scoped_to_the_requests_periods(toolkit):
    result = toolkit.dispatch("search_docs", {"query": "net interest margin"})
    assert set(result.searched_periods) == {"1H26", "1H25"}
    assert "cannot reach another bank" in result.note


def test_a_tool_takes_no_bank_or_period_argument():
    """Scope comes from the RequestSpec inside the tool, so it cannot be widened."""
    for definition in tool_definitions():
        properties = set(definition["input_schema"].get("properties", {}))
        assert not properties & {"bank", "bank_id", "period", "as_of", "periods"}


# --- results carry provenance --------------------------------------------


def test_the_outline_lists_tables_with_their_periods(toolkit):
    result = toolkit.dispatch("get_outline", {"doc_id": "xmpl-1h26-rd"})
    assert isinstance(result, OutlineResult)
    assert result.page_count == 5
    assert any("Key metrics" == entry.heading for entry in result.sections)
    assert any("1H26" in table.periods for table in result.tables)


def test_a_table_comes_back_with_its_column_to_period_map(toolkit):
    outline = toolkit.dispatch("get_outline", {"doc_id": "xmpl-1h26-rd"})
    table_id = next(
        t.table_id
        for t in outline.tables
        if "Key metrics" in t.caption or "Group performance" in t.caption
    )
    result = toolkit.dispatch("get_table", {"doc_id": "xmpl-1h26-rd", "table_id": table_id})
    assert isinstance(result, TableResult)
    assert result.column_periods.get("Mar 26") == "1H26"
    assert result.column_periods.get("Mar 25") == "1H25"
    assert result.source.table_id == table_id


def test_a_page_read_carries_its_document_and_page(toolkit):
    result = toolkit.dispatch("read_pages", {"doc_id": "xmpl-1h26-rd", "pages": [2]})
    assert isinstance(result, PageResult)
    assert result.pages[0].source.doc_id == "xmpl-1h26-rd"
    assert result.pages[0].source.page == 2
    assert result.pages[0].source.publish_date == PUB


def test_document_text_is_wrapped_as_data(toolkit):
    """Instructions inside a document are ignored; wrapping is what makes that enforceable
    rather than just hoped for.
    """
    result = toolkit.dispatch("read_pages", {"doc_id": "xmpl-1h26-rd", "pages": [2]})
    assert result.pages[0].text.startswith("<document_content>")
    assert DOCUMENT_WRAPPER.format(content="x") == "<document_content>\nx\n</document_content>"


# --- errors a model can act on -------------------------------------------


def test_a_page_out_of_range_says_how_many_pages_there_are(toolkit):
    result = toolkit.dispatch("read_pages", {"doc_id": "xmpl-1h26-rd", "pages": [240]})
    assert "out of range" in result.note
    assert "5 pages" in result.note


def test_a_missing_table_points_at_the_outline(toolkit):
    result = toolkit.dispatch("get_table", {"doc_id": "xmpl-1h26-rd", "table_id": "nope"})
    assert isinstance(result, ToolError)
    assert "get_outline" in result.suggestion


def test_reading_more_than_three_pages_is_refused(toolkit):
    with pytest.raises(ValidationError):
        toolkit.dispatch("read_pages", {"doc_id": "xmpl-1h26-rd", "pages": [1, 2, 3, 4]})


# --- walks ----------------------------------------------------------------


def test_a_tabulated_walk_is_returned_with_its_checksum(toolkit):
    result = toolkit.dispatch("read_chart", {"doc_id": "xmpl-1h26-rd", "page": 4})
    assert result.checksum_passed is True
    assert result.start == 180.0
    assert result.end == 186.0
    assert len(result.steps) == 4


def test_a_page_with_no_walk_says_so_rather_than_inventing_one(toolkit):
    result = toolkit.dispatch("read_chart", {"doc_id": "xmpl-1h26-rd", "page": 2})
    assert result.steps == []
    assert "No tabulated walk" in result.note


# --- the reading list ends the loop --------------------------------------


def test_submitting_a_reading_list_records_it_and_ends_the_loop(toolkit):
    result = toolkit.dispatch(
        SUBMIT_READING_LIST,
        {
            "items": [
                {
                    "component": "nim.walk",
                    "doc_id": "xmpl-1h26-rd",
                    "pages": [4],
                    "table_ids": [],
                    "rationale": "the margin walk",
                }
            ]
        },
    )
    assert result["accepted"] is True
    assert toolkit.reading_list is not None
    assert toolkit.reading_list.items[0].component == "nim.walk"


def test_calls_are_cached_within_a_run(toolkit):
    before = len(toolkit.call_log)
    toolkit.dispatch("get_outline", {"doc_id": "xmpl-1h26-rd"})
    toolkit.dispatch("get_outline", {"doc_id": "xmpl-1h26-rd"})
    assert toolkit.call_log[-1]["cached"] is True
    assert len(toolkit.call_log) > before


# --- retrieval ------------------------------------------------------------


def test_financial_tokens_survive_tokenisation():
    tokens = tokenize("CET1 of 12.39% on RWA of 11,371 in 1H26")
    assert "cet1" in tokens and "1h26" in tokens and "11371" in tokens


def test_an_abbreviation_expands_to_its_token_sequence():
    """Expanding NIM to one glued token would match neither spelling."""
    tokens = tokenize("NIM")
    assert tokens[:4] == ["nim", "net", "interest", "margin"]


def test_a_table_outranks_prose_at_equal_relevance():
    document = PyMuPdfParser().parse(FIXTURE, "xmpl-1h26-rd")
    meta = DocMeta(
        doc_id="xmpl-1h26-rd",
        bank_id="XMPL",
        period="1H26",
        doc_type="RESULTS_DOCUMENT",
        publish_date=PUB,
        url="file:///x",
    )
    retriever = Retriever()
    retriever.add(chunk_document(document, meta))
    hits = retriever.search(
        "net interest margin movement",
        bank_id="XMPL",
        periods=("1H26", "1H25"),
        as_of=date(2026, 5, 31),
        limit=5,
    )
    assert hits
    assert hits[0].chunk.kind == "table"


def test_another_banks_chunks_are_unreachable():
    meta = DocMeta(
        doc_id="d",
        bank_id="XMPL",
        period="1H26",
        doc_type="RESULTS_DOCUMENT",
        publish_date=PUB,
        url="file:///x",
    )
    document = PyMuPdfParser().parse(FIXTURE, "d")
    retriever = Retriever()
    retriever.add(chunk_document(document, meta))
    assert (
        retriever.search("margin", bank_id="ANZ", periods=("1H26",), as_of=date(2026, 5, 31)) == []
    )


def test_a_chunk_published_after_as_of_is_unreachable():
    meta = DocMeta(
        doc_id="d",
        bank_id="XMPL",
        period="1H26",
        doc_type="RESULTS_DOCUMENT",
        publish_date=date(2026, 11, 1),
        url="file:///x",
    )
    retriever = Retriever()
    retriever.add(chunk_document(PyMuPdfParser().parse(FIXTURE, "d"), meta))
    assert (
        retriever.search("margin", bank_id="XMPL", periods=("1H26",), as_of=date(2026, 5, 31)) == []
    )


# --- the LLM layer --------------------------------------------------------


def test_the_cache_key_covers_everything_that_could_change_the_answer():
    base = dict(step="explain", system="s", messages=({"role": "user", "content": "x"},))
    a = LLMRequest(model="claude-opus-5-5", effort="high", **base)
    for changed in (
        LLMRequest(model="claude-sonnet-5-5", effort="high", **base),
        LLMRequest(model="claude-opus-5-5", effort="low", **base),
        LLMRequest(
            model="claude-opus-5-5",
            effort="high",
            step="verify",
            system="s",
            messages=base["messages"],
        ),
        LLMRequest(
            model="claude-opus-5-5",
            effort="high",
            step="explain",
            system="different",
            messages=base["messages"],
        ),
    ):
        assert request_hash(a) != request_hash(changed)


def test_replay_raises_on_a_miss_rather_than_going_live():
    """A replay that quietly went live would make "deterministic and free" untrue
    in exactly the case that matters, and the run would still succeed."""
    client = CachedClient(cache=ResponseCache(connect(":memory:")), replay_only=True)
    request = LLMRequest(
        step="explain",
        model="claude-opus-5-5",
        effort="high",
        system="s",
        messages=({"role": "user", "content": "x"},),
    )
    with pytest.raises(CacheMiss):
        client.complete(request)


def test_a_recorded_response_round_trips():
    cache = ResponseCache(connect(":memory:"))
    request = LLMRequest(
        step="locate",
        model="claude-sonnet-5-5",
        effort="medium",
        system="s",
        messages=({"role": "user", "content": "x"},),
    )
    response = LLMResponse(
        text="ok",
        tool_calls=(ToolCall(id="t1", name="search_docs", arguments={"query": "nim"}),),
        usage=Usage(input_tokens=100, output_tokens=20),
    )
    cache.put(request, response)
    back = CachedClient(cache=cache, replay_only=True).complete(request)
    assert back.text == "ok"
    assert back.tool_calls[0].name == "search_docs"
    assert back.usage.input_tokens == 100


def test_content_blocks_are_read_by_type_not_position():
    """A response begins with thinking, so the first block is not the answer."""
    blocks = [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "answer"}]
    assert blocks_by_type(blocks, "text")[0]["text"] == "answer"
    assert blocks_by_type(blocks, "thinking")


def test_cost_is_computed_from_tokens_and_the_pricing_table():
    cost = cost_usd(
        Usage(input_tokens=1_000_000, output_tokens=500_000),
        {"input": "4.00", "output": "20.00"},
    )
    assert cost == Decimal("14.00")


# --- the budget guard -----------------------------------------------------


def test_the_guard_refuses_further_calls_rather_than_raising():
    budget = Budget(max_calls=2)
    for _ in range(2):
        assert budget.may_call("locate")
        budget.record("locate", "claude-sonnet-5-5", Usage(input_tokens=10, output_tokens=5))
    assert budget.may_call("locate") is False
    assert budget.breached is True
    assert any("confidence lowered" in warning for warning in budget.warnings)


def test_a_per_step_cap_is_enforced_separately():
    from bankresearch.llm.budget import StepBudget

    budget = Budget(max_calls=50, step_caps={"locate": StepBudget(calls=1)})
    assert budget.may_call("locate")
    budget.record("locate", "m", Usage())
    assert budget.may_call("locate") is False
    assert budget.may_call("explain") is True


def test_the_cost_guardrail_stops_exploration():
    # The other caps are raised so that only the cost guardrail can bind.
    budget = Budget(
        max_calls=99,
        max_input_tokens=10_000_000,
        max_output_tokens=10_000_000,
        cost_guardrail_usd=Decimal("0.01"),
        pricing={"claude-opus-5-5": {"input": "4.00", "output": "20.00"}},
    )
    budget.record("explain", "claude-opus-5-5", Usage(input_tokens=1_000_000))
    assert budget.may_call("explain") is False
    assert "cost guardrail" in " ".join(budget.warnings)


def test_the_budget_is_built_from_config():
    from bankresearch.domain.registry import get_registry

    registry = get_registry()
    budget = Budget.from_config(registry.budgets, registry.models)
    assert budget.max_calls == 25
    assert budget.max_input_tokens == 300_000
    assert budget.cost_guardrail_usd == Decimal("1.50")
    assert budget.step_caps["locate"].calls == 12


def test_a_scripted_client_drives_the_real_dispatch_path():
    client = ScriptedClient(responses=[LLMResponse(text="first")])
    request = LLMRequest(step="locate", model="m", effort="low", system="s", messages=())
    assert client.complete(request).text == "first"
    with pytest.raises(LLMUnavailable):
        client.complete(request)
