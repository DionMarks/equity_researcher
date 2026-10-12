"""A full model-backed run against the real Westpac filing, with a scripted model."""

from __future__ import annotations

from datetime import date

import pytest

from bankresearch.llm.client import LLMResponse, ScriptedClient, Usage

WALK = {
    "chart_matches": True,
    "reads_as": "Net interest margin movement, Excluding Notable Items, 1H25 to 1H26",
    "opening": 1.92,
    "closing": 1.89,
    "endpoint_unit": "pct",
    "step_unit": "bps",
    "steps": [
        {"label": "Loans", "value": -2},
        {"label": "Deposits", "value": -1},
        {"label": "Timing difference", "value": -1},
        {"label": "Liquid assets", "value": 5},
        {"label": "WSF (wholesale funding costs)", "value": 0},
        {"label": "Capital & other", "value": -3},
        {"label": "T&M (Treasury & Markets contribution)", "value": -1},
    ],
}
FIRST_DRAFT = (
    "Westpac margin fell -3.0 bps to 1.89% [ev_017], from 1.92% [ev_043], because of deposit "
    "competition. Liquid assets added +5.0 bps [ev_056], while capital and other cost -3.0 "
    "bps [ev_058]."
)
SUPPORTED = (
    "Westpac margin fell -3.0 bps to 1.89% [ev_017], from 1.92% [ev_043]. Liquid assets "
    "added +5.0 bps [ev_056], while capital and other cost -3.0 bps [ev_058]."
)


@pytest.fixture(scope="module")
def outcome():
    from bankresearch.agent.orchestrator import Agent
    from bankresearch.memory.db import connect
    from bankresearch.memory.fact_store import FactStore
    from bankresearch.settings import get_settings

    settings = get_settings()
    if not settings.db_path.exists():
        pytest.skip("no ingested corpus: run `make ingest` first")
    client = ScriptedClient(
        responses=[
            LLMResponse(
                structured={
                    "found": False,
                    "reads_as": "no written walk on these pages",
                    "page": 13,
                    "opening": None,
                    "closing": None,
                    "endpoint_unit": "pct",
                    "step_unit": "bps",
                    "steps": [],
                },
                usage=Usage(3000, 200),
            ),
            LLMResponse(structured=WALK, usage=Usage(3000, 800)),
            LLMResponse(
                structured={"mappings": [{"label": "Timing difference", "driver_id": None}]},
                usage=Usage(500, 50),
            ),
            LLMResponse(text=FIRST_DRAFT, usage=Usage(2500, 300)),
            LLMResponse(
                structured={
                    "verdicts": [
                        {"claim": 1, "verdict": "partial", "reason": "the cause isn't stated"},
                        {"claim": 2, "verdict": "supported", "reason": "both bars"},
                    ]
                },
                usage=Usage(2000, 200),
            ),
            LLMResponse(text=SUPPORTED, usage=Usage(2600, 300)),
            LLMResponse(
                structured={
                    "verdicts": [
                        {"claim": 1, "verdict": "supported", "reason": "both figures"},
                        {"claim": 2, "verdict": "supported", "reason": "both bars"},
                    ]
                },
                usage=Usage(2000, 200),
            ),
        ]
    )
    with connect(settings.db_path) as connection:
        agent = Agent(store=FactStore(connection), settings=settings, persist=False, client=client)
        run = agent.run("WBC", "1H26", "nim", as_of=date(2026, 5, 31))
    return run, client


def test_the_issuers_walk_is_read_from_the_first_page_and_used(outcome):
    run, client = outcome
    result = run.result
    assert result.status == "OK"
    assert [request.step for request in client.requests] == [
        "text_walk",
        "chart_read",
        "taxonomy_classify",
        "explain",
        "verify",
        "explain",
        "verify",
    ], "the written walk first, then one chart read, and no page search"
    pages = next(e for e in run.trace if e.get("charted_walks") == 1)["pages"]
    assert pages == [
        "wbc-1h26-results-document p.13: a nim walk read and reconciled against the "
        "published endpoints"
    ]
    assert {d.driver_id: str(d.contribution) for d in result.drivers} == {
        "nim.lending_pricing": "-2",
        "nim.deposit_pricing": "-1",
        "nim.liquids": "5",
        "nim.funding_costs": "0",
        "nim.hedging": "-3",
        "nim.markets_treasury": "-1",
        "nim.other": "-1",
    }
    assert result.residual is not None and result.residual.contribution == 0


def test_chart_read_drivers_pay_for_the_chart_and_for_their_own_rounding(outcome):
    """Every bar was read off the chart and printed to whole basis points, so each pays for
    the chart, and a small bar pays more for its rounding than a big one."""
    run, _ = outcome
    for driver in run.result.drivers:
        assert any("read off a chart" in reason for reason in driver.confidence.reasons)
    score = {driver.driver_id: driver.confidence.score for driver in run.result.drivers}
    assert score["nim.liquids"] == pytest.approx(0.898)  # +5bps
    assert score["nim.lending_pricing"] == pytest.approx(0.88)  # -2bps
    assert score["nim.deposit_pricing"] == pytest.approx(0.87)  # -1bp


def test_a_partly_supported_draft_is_rewritten_once_and_the_rewrite_kept(outcome):
    run, _ = outcome
    result = run.result
    # The rewrite as checked, then the one material driver it doesn't name, stated by its
    # figure: Westpac's "Loans" bar, two points of a three-point fall.
    assert result.narrative.startswith(SUPPORTED)
    assert result.narrative[len(SUPPORTED) :].strip().startswith("Loans took 2.0 bps off [ev_")
    assert (
        "Stated by figure alone, with no cause, because the checked narrative no longer "
        "named Loans." in result.caveats
    )
    # The rewrite says it correctly, so the reader isn't told the draft's version was left
    # out; the trace keeps the record.
    assert not any(c.startswith("Left out of the narrative") for c in result.caveats)
    assert any(
        w.startswith("Left out of the narrative") and w.endswith("The rewrite replaced it.")
        for w in result.diagnostics.warnings
    )


def test_the_run_records_which_model_and_prompt_answered_each_step(outcome):
    run, _ = outcome
    diagnostics = run.result.diagnostics
    assert diagnostics.model_ids == {
        "text_walk": "claude-fable-5-1",
        "chart_read": "claude-fable-5-1",
        "taxonomy_classify": "claude-fable-5-1",
        "explain": "claude-fable-5-1",
        "verify": "claude-fable-5-1",
    }
    assert set(diagnostics.prompt_hashes) == {
        "text_walk",
        "chart_read",
        "taxonomy_classify",
        "explain",
        "verify",
    }
    assert sum(1 for e in run.trace if e.get("event") == "model_call") == 7


def test_a_metric_whose_identity_can_run_reads_no_charts():
    """NAB's ROE has every line of its identity, and leads with the identity. Reading
    charts for it would cost vision calls whose result the decomposition can't use."""
    from bankresearch.agent.orchestrator import Agent
    from bankresearch.memory.db import connect
    from bankresearch.memory.fact_store import FactStore
    from bankresearch.settings import get_settings

    settings = get_settings()
    if not settings.db_path.exists():
        pytest.skip("no ingested corpus: run `make ingest` first")
    client = ScriptedClient(
        responses=[
            LLMResponse(text="Return on equity fell on lower earnings.", usage=Usage(10, 10))
        ]
    )
    with connect(settings.db_path) as connection:
        agent = Agent(store=FactStore(connection), settings=settings, persist=False, client=client)
        run = agent.run("NAB", "1H26", "roe", as_of=date(2026, 5, 31))
    assert run.result.status == "OK"
    assert [request.step for request in client.requests] == ["explain"]
    locate = next(e for e in run.trace if e.get("step") == "locate")
    assert locate["outcome"] == "skipped: the identity explains this metric and can run"


# --- the written walk ------------------------------------------------------

WRITTEN = {
    "found": True,
    "reads_as": "NIM excluding Notable Items, First Half 2026 against First Half 2025",
    "page": 13,
    "opening": 1.92,
    "closing": 1.89,
    "endpoint_unit": "pct",
    "step_unit": "bps",
    "steps": [
        {
            "label": "Loan interest spread",
            "value": -2,
            "quote": "Loan interest spread: 2 basis point decrease.",
        },
        {
            "label": "Deposit interest spread",
            "value": -1,
            "quote": "Deposit interest spread: 1 basis point decrease driven by a higher proportion of customers qualifying for the savings bonus rate",
        },
        {
            "label": "Timing difference",
            "value": -1,
            "quote": "Timing difference1: 1 basis point decrease due to the timing differences of interest rate changes;",
        },
        {
            "label": "Liquid assets",
            "value": 5,
            "quote": "Liquid assets: 5 basis point increase as trading assets reduced, average liquid assets rose by less than average lending assets and spreads narrowed;",
        },
        {
            "label": "Capital and other",
            "value": -3,
            "quote": "Capital and other: 3 basis point decrease due to a remediation provision in the current period and the non repeat of items in the prior corresponding period.",
        },
        {
            "label": "Treasury and Markets",
            "value": -1,
            "quote": "Treasury and Markets contribution of 11 basis points, which was down 1 basis point due to lower Treasury income.",
        },
    ],
}


def _responder(request):
    """Answers each step as a careful model would, citing the ids the brief actually holds."""
    import re

    if request.step == "text_walk":
        return LLMResponse(structured=WRITTEN, usage=Usage(4000, 900))
    if request.step == "taxonomy_classify":
        return LLMResponse(structured={"mappings": []}, usage=Usage(300, 30))
    if request.step == "explain":
        brief = request.messages[0]["content"]
        liquid = re.search(r"\[(ev_\d+)\] p\.13: \"Liquid assets", brief).group(1)
        capital = re.search(r"\[(ev_\d+)\] p\.13: \"Capital and other", brief).group(1)
        return LLMResponse(
            structured={
                "narrative": f"Westpac's margin fell 3.0 bps to 1.89%. Liquid assets added "
                f"5.0 bps [{liquid}], and capital and other took 3.0 bps off [{capital}].",
                "drivers": [
                    {
                        "driver": "nim.liquids",
                        "why": f"Trading assets fell and liquid assets grew more slowly than "
                        f"lending [{liquid}].",
                    },
                    {
                        "driver": "nim.hedging",
                        "why": "A remediation provision, and items in the prior period that "
                        "did not repeat [ev_999].",
                    },
                ],
            },
            usage=Usage(5000, 1200),
        )
    if request.step == "verify":
        claims = re.findall(r"^(\d+)\. ", request.messages[0]["content"], re.MULTILINE)
        return LLMResponse(
            structured={
                "verdicts": [
                    {"claim": int(n), "verdict": "supported", "reason": "stated on p.13"}
                    for n in claims
                ]
            },
            usage=Usage(4000, 500),
        )
    raise AssertionError(f"no scripted answer for {request.step}")


@pytest.fixture(scope="module")
def written():
    from bankresearch.agent.orchestrator import Agent
    from bankresearch.memory.db import connect
    from bankresearch.memory.fact_store import FactStore
    from bankresearch.settings import get_settings

    settings = get_settings()
    if not settings.db_path.exists():
        pytest.skip("no ingested corpus: run `make ingest` first")
    client = ScriptedClient(on_exhausted=_responder)
    with connect(settings.db_path) as connection:
        agent = Agent(store=FactStore(connection), settings=settings, persist=False, client=client)
        run = agent.run("WBC", "1H26", "nim", as_of=date(2026, 5, 31))
    return run, client


def test_the_written_walk_is_used_and_no_chart_is_read(written):
    run, client = written
    result = run.result
    assert result.status == "OK"
    steps = [request.step for request in client.requests]
    assert "chart_read" not in steps and steps[0] == "text_walk"
    assert {d.driver_id: str(d.contribution) for d in result.drivers} == {
        "nim.lending_pricing": "-2",
        "nim.deposit_pricing": "-1",
        "nim.other": "-1",
        "nim.liquids": "5",
        "nim.hedging": "-3",
        "nim.markets_treasury": "-1",
    }
    walk_cards = [card for card in result.evidence if ".walk." in card.component]
    assert walk_cards and all(card.method == "text" and card.passage for card in walk_cards)


def test_a_stated_cause_citing_the_issuers_words_reaches_the_driver(written):
    run, _ = written
    by_id = {d.driver_id: d for d in run.result.drivers}
    liquids = by_id["nim.liquids"]
    assert liquids.cause.startswith("Trading assets fell")
    assert liquids.cause_evidence_ids
    cited = run.result.evidence_by_id()[liquids.cause_evidence_ids[0]]
    assert cited.passage and cited.passage.startswith("Liquid assets")


def test_a_cause_citing_evidence_that_does_not_exist_is_dropped(written):
    run, _ = written
    by_id = {d.driver_id: d for d in run.result.drivers}
    assert by_id["nim.hedging"].cause == ""


def test_written_walk_drivers_are_not_marked_down_as_a_chart(written):
    run, _ = written
    for driver in run.result.drivers:
        assert not any("chart reading" in reason for reason in driver.confidence.reasons)


# --- a capital ratio, walked over each half and chained ---------------------


def _chained_responder(request):
    """ANZ's two written CET1 walks, each given only when its own half is asked for."""
    from tests.integration.test_written_walks import ANZ_1H26, ANZ_2H25

    if request.step == "text_walk":
        asked = request.messages[0]["content"]
        if "from 2H25" in asked and "to 1H26" in asked:
            return LLMResponse(structured=ANZ_1H26, usage=Usage(4000, 900))
        if "from 1H25" in asked and "to 2H25" in asked:
            return LLMResponse(structured=ANZ_2H25, usage=Usage(4000, 900))
        return LLMResponse(
            structured={**ANZ_1H26, "found": False, "steps": [], "reads_as": "no such walk"},
            usage=Usage(4000, 100),
        )
    if request.step == "taxonomy_classify":
        return LLMResponse(structured={"mappings": []}, usage=Usage(300, 30))
    if request.step == "explain":
        return LLMResponse(
            structured={"narrative": "ANZ's CET1 ratio rose over the year.", "drivers": []},
            usage=Usage(5000, 600),
        )
    if request.step == "verify":
        return LLMResponse(structured={"verdicts": []}, usage=Usage(1000, 50))
    raise AssertionError(f"no scripted answer for {request.step}")


@pytest.fixture(scope="module")
def chained():
    from bankresearch.agent.orchestrator import Agent
    from bankresearch.memory.db import connect
    from bankresearch.memory.fact_store import FactStore
    from bankresearch.settings import get_settings

    settings = get_settings()
    if not settings.db_path.exists():
        pytest.skip("no ingested corpus: run `make ingest` first")
    client = ScriptedClient(on_exhausted=_chained_responder)
    with connect(settings.db_path) as connection:
        agent = Agent(store=FactStore(connection), settings=settings, persist=False, client=client)
        run = agent.run("ANZ", "1H26", "cet1_ratio", as_of=date(2026, 5, 31))
    return run, client, settings


def test_a_capital_ratio_is_explained_by_the_issuers_two_half_year_walks(chained):
    run, client, _ = chained
    result = run.result
    assert result.status == "OK"
    steps = [request.step for request in client.requests]
    assert "chart_read" not in steps
    assert 2 <= steps.count("text_walk") <= 3
    assert {driver.method for driver in result.drivers} == {"chained_walk"}
    by_id = {driver.driver_id: driver.contribution for driver in result.drivers}
    # Cash profit added 71bps in the second half of 2025 and 81bps in the first of 2026;
    # the dividends took 45 and 33.
    assert by_id["cet1.organic_capital"] == 152
    assert by_id["cet1.dividends"] == -78
    assert by_id["cet1.rwa_irrbb"] == -24
    assert sum(by_id.values()) == 61
    assert any("chained" in caveat for caveat in result.caveats)


def test_every_chained_step_cites_a_figure_its_own_page_prints(chained):
    from bankresearch.evals.scorers import citation_faithfulness

    run, _, settings = chained
    walk_cards = [card for card in run.result.evidence if ".walk." in card.component]
    assert {(card.source.doc_id, card.source.page) for card in walk_cards} == {
        ("anz-fy25-results-document", 43),
        ("anz-1h26-results-document", 40),
    }
    assert all(card.method == "text" and card.passage for card in walk_cards)
    checked, found, missing_page, _notes = citation_faithfulness(run.result, settings)
    assert checked and found == checked and missing_page == 0


# --- a stretched cause, narrowed in the one rewrite ---------------------------


def _narrowing_responder(request):
    """The first draft stretches the loan spread cause; told why, the rewrite narrows it."""
    import re

    if request.step == "text_walk":
        return LLMResponse(structured=WRITTEN, usage=Usage(4000, 900))
    if request.step == "taxonomy_classify":
        return LLMResponse(structured={"mappings": []}, usage=Usage(300, 30))
    brief = request.messages[0]["content"]
    loan = re.search(r"\[(ev_\d+)\] p\.13: \"Loan interest spread", brief)
    if request.step == "explain":
        cited = loan.group(1) if loan else "ev_999"
        why = (
            f"Tighter spreads in Australia, from competition, more than offset higher spreads "
            f"in New Zealand [{cited}]."
            if "These stated causes were left out" in brief
            else f"Competition across every market squeezed lending spreads [{cited}]."
        )
        return LLMResponse(
            structured={
                "narrative": "Westpac's margin fell 3.0 bps to 1.89%.",
                "drivers": [{"driver": "nim.lending_pricing", "why": why}],
            },
            usage=Usage(5000, 600),
        )
    if request.step == "verify":
        claims = re.findall(r"^(\d+)\. (.*)$", brief, re.MULTILINE)
        return LLMResponse(
            structured={
                "verdicts": [
                    {
                        "claim": int(n),
                        "verdict": "partial" if "every market" in text else "supported",
                        "reason": "the bank names Australia, not every market"
                        if "every market" in text
                        else "stated on p.13",
                    }
                    for n, text in claims
                ]
            },
            usage=Usage(3000, 300),
        )
    raise AssertionError(f"no scripted answer for {request.step}")


def test_a_stretched_cause_comes_back_narrower_and_is_kept():
    from bankresearch.agent.orchestrator import Agent
    from bankresearch.memory.db import connect
    from bankresearch.memory.fact_store import FactStore
    from bankresearch.settings import get_settings

    settings = get_settings()
    if not settings.db_path.exists():
        pytest.skip("no ingested corpus: run `make ingest` first")
    client = ScriptedClient(on_exhausted=_narrowing_responder)
    with connect(settings.db_path) as connection:
        agent = Agent(store=FactStore(connection), settings=settings, persist=False, client=client)
        run = agent.run("WBC", "1H26", "nim", as_of=date(2026, 5, 31))
    steps = [request.step for request in client.requests]
    assert steps.count("explain") == 2 and steps.count("verify") == 2
    repair = next(r for r in client.requests[::-1] if r.step == "explain")
    assert "the bank names Australia, not every market" in repair.messages[0]["content"]
    loan = {d.driver_id: d for d in run.result.drivers}["nim.lending_pricing"]
    assert loan.cause.startswith("Tighter spreads in Australia")
    # The narrative lost nothing, so the checked first draft stands.
    assert run.result.narrative.startswith("Westpac's margin fell 3.0 bps")
    assert not any("lending" in warning.lower() for warning in run.result.diagnostics.warnings)
