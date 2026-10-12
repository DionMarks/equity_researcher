"""Walks the issuers wrote out in prose, checked against their own pages."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from bankresearch.settings import get_settings

AS_OF = date(2026, 5, 31)


def _page(doc_id: str, page_no: int):
    from bankresearch.ingestion.artifacts import read_processed

    settings = get_settings()
    document = read_processed(doc_id, settings.processed_dir)
    if document is None:
        pytest.skip("no ingested corpus: run `make ingest` first")
    page = document.page(page_no)
    assert page is not None
    return page


def _spec(bank: str):
    from bankresearch.agent.steps.resolve import resolve_request

    return resolve_request(bank, "1H26", "nim", as_of=AS_OF)


def _check(bank: str, doc_id: str, page_no: int, answer: dict, opening: str, closing: str):
    from bankresearch.ingestion.text_walks import check_text_walk

    return check_text_walk(
        answer,
        {page_no: _page(doc_id, page_no)},
        spec=_spec(bank),
        fy_end_month=9,
        reported_opening=Decimal(opening),
        reported_closing=Decimal(closing),
        unit="pct",
    )


NAB = {
    "found": True,
    "reads_as": "Group NIM, March 2026 v March 2025",
    "page": 26,
    "opening": None,
    "closing": None,
    "endpoint_unit": "pct",
    "step_unit": "bps",
    "steps": [
        {
            "label": "Markets and Treasury",
            "value": 4,
            "quote": "Excluding an increase of 4 basis points in M&T and 3 basis points from a "
            "lower mix of lower yielding HQLA, the margin increased by 4 basis points due to:",
        },
        {
            "label": "Liquid assets",
            "value": 3,
            "quote": "Excluding an increase of 4 basis points in M&T and 3 basis points from a "
            "lower mix of lower yielding HQLA, the margin increased by 4 basis points due to:",
        },
        {
            "label": "Replicating portfolios",
            "value": 6,
            "quote": "an increase of 6 basis points primarily driven by higher earnings on "
            "deposit and capital replicating portfolios,",
        },
        {
            "label": "Funding costs",
            "value": 1,
            "quote": "an increase of 1 basis point driven by lower short-term wholesale funding "
            "costs, and",
        },
        {
            "label": "Deposits",
            "value": 1,
            "quote": "an increase of 1 basis point mainly driven by lower deposit costs, combined "
            "with deposit mix benefits",
        },
        {
            "label": "Lending margin",
            "value": -4,
            "quote": "These increases were partially offset by a decrease of 4 basis points in "
            "lending margin primarily driven by competitive pressures",
        },
    ],
}


def test_nabs_written_year_on_year_walk_is_accepted():
    steps, page, refused = _check("NAB", "nab-1h26-results-document", 26, NAB, "1.70", "1.81")
    assert not refused, refused
    assert page == 26
    assert sum(value for _, value, _ in steps) == Decimal(11)


def test_the_half_on_half_walk_beside_it_is_refused():
    """The same page explains +3 bps against the September half in nearly the same words.
    Its steps add up to that, not to the +11 bps year on year."""
    half = dict(NAB)
    half["steps"] = [
        {
            "label": "Replicating portfolio",
            "value": 3,
            "quote": "an increase of 3 basis points primarily driven by higher earnings on the "
            "deposit replicating portfolio, and",
        },
        {
            "label": "Lending margin",
            "value": -4,
            "quote": "These increases were offset by a decrease of 4 basis points in lending "
            "margin primarily driven by competitive pressures",
        },
    ]
    steps, _, refused = _check("NAB", "nab-1h26-results-document", 26, half, "1.70", "1.81")
    assert steps == [] and "add to" in refused


def test_a_quote_that_is_not_on_the_page_fails_the_whole_walk():
    made_up = dict(NAB)
    made_up["steps"] = [dict(step) for step in NAB["steps"]]
    made_up["steps"][2]["quote"] = "an increase of 6 basis points driven by a new pricing strategy"
    steps, _, refused = _check("NAB", "nab-1h26-results-document", 26, made_up, "1.70", "1.81")
    assert steps == [] and "not on the pages" in refused


def test_a_sign_the_wording_contradicts_fails_the_whole_walk():
    flipped = dict(NAB)
    flipped["steps"] = [dict(step) for step in NAB["steps"]]
    flipped["steps"][5]["value"] = 4
    flipped["steps"][2]["value"] = -2  # keeps the total, so only the sign check can catch it
    steps, _, refused = _check("NAB", "nab-1h26-results-document", 26, flipped, "1.70", "1.81")
    assert steps == [] and refused


def test_endpoints_from_another_basis_are_refused():
    other = dict(NAB, opening=1.95, closing=2.06)
    steps, _, refused = _check("NAB", "nab-1h26-results-document", 26, other, "1.70", "1.81")
    assert steps == [] and "not the published" in refused


ANZ = {
    "found": True,
    "reads_as": "Group NIM, March 2026 Half Year v March 2025 Half Year",
    "page": 18,
    "opening": 1.56,
    "closing": 1.53,
    "endpoint_unit": "pct",
    "step_unit": "bps",
    "steps": [
        {
            "label": "Assets pricing",
            "value": -4,
            "quote": "Assets pricing (-4 bps): driven by ongoing competition across most divisions and timing impact of RBA rate changes.",
        },
        {
            "label": "Deposits pricing",
            "value": -3,
            "quote": "Deposits pricing (-3 bps): driven by lower cash rates in New Zealand and international geographies and pricing competition.",
        },
        {
            "label": "Wholesale funding",
            "value": 0,
            "quote": "Wholesale funding (0 bps): largely flat with increased funding volume and lower average spread.",
        },
        {
            "label": "Capital and replicating portfolio",
            "value": 4,
            "quote": "Capital and replicating portfolio (+4 bps): driven by higher volumes and average hedge rates.",
        },
        {
            "label": "Assets and funding mix",
            "value": 2,
            "quote": "Assets and funding mix (+2 bps): favourable funding mix primarily from stronger growth in at-call deposits",
        },
        {
            "label": "Group Centre liquids",
            "value": 3,
            "quote": "Group Centre liquids (+3 bps): driven by a reduction in the average liquid asset balance in the Group Centre division.",
        },
        {
            "label": "Markets activities",
            "value": -5,
            "quote": "Markets activities (-5 bps): primarily due to higher average interest earning assets relative to the rest of the Group.",
        },
    ],
}


def test_anzs_bulleted_walk_is_accepted_with_its_bracketed_signs():
    steps, _page_no, refused = _check("ANZ", "anz-1h26-results-document", 18, ANZ, "1.56", "1.53")
    assert not refused, refused
    assert [str(value) for _, value, _ in steps] == ["-4", "-3", "0", "4", "2", "3", "-5"]


WBC = {
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


def test_westpacs_walk_is_accepted_and_a_footnote_marker_is_not_a_figure():
    steps, _, refused = _check("WBC", "wbc-1h26-results-document", 13, WBC, "1.92", "1.89")
    assert not refused, refused
    assert sum(value for _, value, _ in steps) == Decimal(-3)


def test_the_pages_offered_for_a_written_walk_are_the_year_on_year_ones():
    """Every major writes its margin walk out; capital walks are over the half, so no
    capital page is offered and no call is spent finding nothing."""
    from bankresearch.agent.steps.resolve import resolve_request
    from bankresearch.ingestion.artifacts import read_processed
    from bankresearch.ingestion.text_walks import rank_text_walk_pages
    from bankresearch.memory.db import connect
    from bankresearch.memory.fact_store import FactStore

    settings = get_settings()
    if not settings.db_path.exists():
        pytest.skip("no ingested corpus: run `make ingest` first")
    expected = {"NAB": 26, "ANZ": 18, "WBC": 13}
    ends = {"NAB": ("1.70", "1.81"), "ANZ": ("1.56", "1.53"), "WBC": ("1.92", "1.89")}
    with connect(settings.db_path) as connection:
        store = FactStore(connection)
        for bank, page_no in expected.items():
            docs = [
                (document, meta)
                for meta in store.documents_for(bank, ["1H26"], AS_OF)
                if (document := read_processed(meta.doc_id, settings.processed_dir)) is not None
            ]
            if not docs:
                pytest.skip("no ingested corpus: run `make ingest` first")
            spec = resolve_request(bank, "1H26", "nim", as_of=AS_OF)
            ranked = rank_text_walk_pages(
                docs,
                spec,
                fy_end_month=9,
                endpoints=(Decimal(ends[bank][0]), Decimal(ends[bank][1]), "pct"),
            )
            assert page_no in [page.page_no for _, _, _, page in ranked[:2]], bank
            capital = resolve_request(bank, "1H26", "cet1_ratio", as_of=AS_OF)
            assert not any(
                "CET1 ratio" in (page.text or "")[:200]
                for _, _, _, page in rank_text_walk_pages(
                    docs, capital, fy_end_month=9, endpoints=(Decimal("12"), Decimal("12"), "pct")
                )[:1]
                if bank in {"NAB", "ANZ"}
            )


# --- a balance walked over each half and chained -----------------------------

ANZ_2H25 = {
    "found": True,
    "reads_as": "ANZBGL Group CET1 ratio, September 2025 v March 2025",
    "page": 43,
    "opening": None,
    "closing": 12.03,
    "endpoint_unit": "pct",
    "step_unit": "bps",
    "steps": [
        {
            "label": "Cash profit",
            "value": 71,
            "quote": "Cash profit (Level 2) excluding 2H25 significant items increased the CET1 "
            "ratio by +71 bps.",
        },
        {
            "label": "Significant items",
            "value": -19,
            "quote": "2H25 significant items decreased the CET1 ratio by -19 bps",
        },
        {
            "label": "Dividend",
            "value": -45,
            "quote": "Payment of the 2025 interim dividend (net of DRP and BOP) reduced the CET1 "
            "ratio by -45 bps.",
        },
        {
            "label": "Underlying RWA and other items",
            "value": 14,
            "quote": "Underlying RWA (excluding IRRBB) and other items increased the CET1 ratio "
            "by +14 bps",
        },
        {
            "label": "Operational risk and IRRBB",
            "value": -18,
            "quote": "Operational risk and IRRBB reduced CET1 ratio by -18 bps",
        },
        {
            "label": "Capital floor",
            "value": 22,
            "quote": "A decrease in the capital floor increased the CET1 ratio by +22 bps",
        },
    ],
}

ANZ_1H26 = {
    "found": True,
    "reads_as": "ANZBGL Group CET1 ratio, March 2026 v September 2025",
    "page": 40,
    "opening": None,
    "closing": 12.39,
    "endpoint_unit": "pct",
    "step_unit": "bps",
    "steps": [
        {
            "label": "Cash profit",
            "value": 81,
            "quote": "Cash profit (Level 2) increased the CET1 ratio by +81 bps.",
        },
        {
            "label": "Reinvestment of NOHC surplus capital",
            "value": 22,
            "quote": "Reinvestment of NOHC surplus capital, including the remaining $0.8 "
            "billion of the share buy-back, increased the CET1 ratio by +22 bps.",
        },
        {
            "label": "Dividend",
            "value": -33,
            "quote": "Payment of the 2025 final dividend (net of DRP discount and BOP) reduced "
            "the CET1 ratio by -33 bps.",
        },
        {
            "label": "Underlying RWA growth",
            "value": -19,
            "quote": "Underlying RWA (excluding IRRBB) growth decreased the CET1 ratio by -19 bps",
        },
        {
            "label": "Capital deductions and others",
            "value": 7,
            "quote": "Capital deductions and others increased the CET1 ratio by +7 bps",
        },
        {
            "label": "IRRBB RWA growth",
            "value": -24,
            "quote": "IRRBB RWA growth decreased the CET1 ratio by -24 bps",
        },
        {
            "label": "Capital floor",
            "value": 2,
            "quote": "A decrease in the capital floor adjustment increased the CET1 ratio by +2 "
            "bps",
        },
    ],
}


def _legs():
    from bankresearch.agent.steps.resolve import resolve_request
    from bankresearch.ingestion.text_walks import half_year_legs

    return half_year_legs(resolve_request("ANZ", "1H26", "cet1_ratio", as_of=AS_OF), 9)


def test_a_half_year_question_on_a_balance_runs_through_two_legs():
    legs = _legs()
    assert [(leg.opening, leg.closing) for leg in legs] == [("1H25", "2H25"), ("2H25", "1H26")]
    assert [(leg.request.comparison_period, leg.request.period) for leg in legs] == [
        ("1H25", "2H25"),
        ("2H25", "1H26"),
    ]


def test_a_full_year_question_runs_through_the_half_between():
    from bankresearch.agent.steps.resolve import resolve_request
    from bankresearch.ingestion.text_walks import half_year_legs

    spec = resolve_request("CBA", "FY26", "cet1_ratio", as_of=date(2026, 8, 31))
    legs = half_year_legs(spec, 6)
    assert [(leg.opening, leg.closing) for leg in legs] == [("FY25", "1H26"), ("1H26", "FY26")]
    assert [(leg.request.comparison_period, leg.request.period) for leg in legs] == [
        ("2H25", "1H26"),
        ("1H26", "2H26"),
    ]


def test_each_half_of_anzs_capital_walk_is_accepted_against_the_rounded_balances():
    """ANZ prints its CET1 ratio to one decimal place in its tables and walks it to two:
    11.8% to 12.0% is +20bps as published, and the walk says +25bps from 11.78% to 12.03%.
    The rounding of the two published balances is allowed for, and nothing more."""
    from bankresearch.ingestion.text_walks import check_text_walk

    legs = _legs()
    for leg, (doc_id, page_no, answer, opening, closing) in zip(
        legs,
        (
            ("anz-fy25-results-document", 43, ANZ_2H25, "11.8", "12.0"),
            ("anz-1h26-results-document", 40, ANZ_1H26, "12.0", "12.4"),
        ),
        strict=True,
    ):
        steps, _where, refused = check_text_walk(
            answer,
            {page_no: _page(doc_id, page_no)},
            spec=leg.request,
            fy_end_month=9,
            reported_opening=Decimal(opening),
            reported_closing=Decimal(closing),
            unit="pct",
        )
        assert not refused, (leg.closing, refused)
        assert sum(value for _, value, _ in steps) == sum(s["value"] for s in answer["steps"])


def test_a_leg_whose_steps_miss_the_balances_by_more_than_their_rounding_is_refused():
    from bankresearch.ingestion.text_walks import check_text_walk

    leg = _legs()[1]
    steps, _where, refused = check_text_walk(
        ANZ_1H26,
        {40: _page("anz-1h26-results-document", 40)},
        spec=leg.request,
        fy_end_month=9,
        reported_opening=Decimal("11.8"),
        reported_closing=Decimal("12.4"),
        unit="pct",
    )
    assert not steps and "add to" in refused


def test_the_two_halves_chain_and_every_step_cites_its_own_page():
    from bankresearch.decomposition.taxonomy import map_label
    from bankresearch.decomposition.walks import Walk, WalkStep, choose_path_a
    from bankresearch.domain.models import SourceRef

    def leg_walk(opening, closing, start, end, answer, doc_id, published):
        source = SourceRef(
            doc_id=doc_id, doc_type="RESULTS_DOCUMENT", page=answer["page"], publish_date=published
        )
        return Walk(
            opening_period=opening,
            closing_period=closing,
            opening_value=Decimal(start),
            closing_value=Decimal(end),
            steps=[
                WalkStep(
                    label=s["label"],
                    driver_id=map_label("cet1_ratio", s["label"]),
                    value=Decimal(s["value"]),
                    quote=s["quote"],
                )
                for s in answer["steps"]
            ],
            unit="bps",
            source=source,
            read_from="text",
        )

    legs = [
        leg_walk(
            "1H25",
            "2H25",
            "1180",
            "1200",
            ANZ_2H25,
            "anz-fy25-results-document",
            date(2025, 11, 10),
        ),
        leg_walk(
            "2H25", "1H26", "1200", "1240", ANZ_1H26, "anz-1h26-results-document", date(2026, 5, 7)
        ),
    ]
    chained = choose_path_a(legs, period="1H26", pcp="1H25", tolerance=Decimal("10"))
    assert chained is not None and chained.method == "chained_walk"
    assert chained.read_from == "text"
    assert sum(step.value for step in chained.steps) == Decimal("61")
    pages = {(step.source.doc_id, step.source.page) for step in chained.steps}
    assert pages == {("anz-fy25-results-document", 43), ("anz-1h26-results-document", 40)}


# Westpac's 1H24 margin walk (2024 Interim Financial Results Announcement p.19), as its own
# text and chart labels print it, and the reading a live run returned. The chart carries a
# "Liquid Assets 0bps" bar the text never mentions, and the text gives the notable items step
# only as "detracted 5 basis points compared to negative 2 basis points".
WBC_1H24_PAGE = (
    "First Half 2024 \u2013 First Half 2023\n5bps\nLoans\n(3bps)\nNotable Items\n1.80\n1H24\n"
    "1.89\n1H23\n(14bps)\n(1bp)\nDeposits\n0bps\nLiquid Assets\n(2bps)\nT&M2\nWSF1\n8bps\n1.96\n"
    "Capital & Other\nNIM down 7bps\nThe NIM decreased by 7 basis points to 1.89%. NIM comprised:\n"
    "Core NIM of 1.80%, down 9 basis points with key drivers described below;\n"
    "Treasury and Markets contribution of 14 basis points, up 5 basis points due to a higher "
    "contribution from favourable positioning for interest rate volatility; and\n"
    "Notable Items from unrealised fair value losses for accounting purposes related to economic "
    "hedges of term funding detracted 5 basis points compared to negative 2 basis points in the "
    "prior corresponding period.\nThe 9 basis points decrease in Core NIM was driven by:\n"
    "Loan interest spread: 14 basis point decrease from tighter spreads on mortgage lending;\n"
    "Wholesale funding: 2 basis point decrease from higher wholesale funding costs;\n"
    "Deposit interest spread: 1 basis point decrease included the impacts of competition;\n"
    "Capital & Other: 8 basis point increase primarily a rise in earnings on capital balances."
)
WBC_1H24 = {
    "found": True,
    "reads_as": "Group NIM walk from 1H23 (1.96%) to 1H24 (1.89%)",
    "page": 19,
    "opening": 1.96,
    "closing": 1.89,
    "endpoint_unit": "pct",
    "step_unit": "bps",
    "steps": [
        {
            "label": "Loan interest spread",
            "value": -14,
            "quote": "Loan interest spread: 14 basis point decrease from tighter spreads on mortgage lending",
        },
        {
            "label": "Wholesale funding",
            "value": -2,
            "quote": "Wholesale funding: 2 basis point decrease from higher wholesale funding costs",
        },
        {
            "label": "Deposit interest spread",
            "value": -1,
            "quote": "Deposit interest spread: 1 basis point decrease included the impacts of competition",
        },
        {"label": "Liquid Assets", "value": 0, "quote": "Liquid Assets"},
        {
            "label": "Capital & Other",
            "value": 8,
            "quote": "Capital & Other: 8 basis point increase primarily a rise in earnings on capital balances",
        },
        {
            "label": "Treasury and Markets",
            "value": 5,
            "quote": "Treasury and Markets contribution of 14 basis points, up 5 basis points",
        },
        {
            "label": "Notable Items",
            "value": -3,
            "quote": "Notable Items from unrealised fair value losses for accounting purposes related to economic hedges of term funding detracted 5 basis points compared to negative 2 basis points in the prior corresponding period.",
        },
    ],
}


def _check_westpac_1h24(answer: dict):  # type: ignore[no-untyped-def]
    from bankresearch.agent.steps.resolve import resolve_request
    from bankresearch.ingestion.parse_pdf import ParsedPage
    from bankresearch.ingestion.text_walks import check_text_walk

    return check_text_walk(
        answer,
        {19: ParsedPage(page_no=19, text=WBC_1H24_PAGE, kind="mixed")},
        spec=resolve_request("WBC", "1H24", "nim", as_of=date(2024, 5, 31)),
        fy_end_month=9,
        reported_opening=Decimal("1.96"),
        reported_closing=Decimal("1.89"),
        unit="pct",
    )


def test_a_zero_step_quoted_only_by_its_chart_label_doesnt_cost_the_walk():
    steps, page, refused = _check_westpac_1h24(WBC_1H24)
    assert not refused and page == 19
    assert [label for label, _, _ in steps] == [
        "Loan interest spread",
        "Wholesale funding",
        "Deposit interest spread",
        "Capital & Other",
        "Treasury and Markets",
        "Notable Items",
    ]
    assert sum(value for _, value, _ in steps) == Decimal(-7)


def test_a_step_given_as_two_figures_compared_is_their_difference():
    steps, _, _ = _check_westpac_1h24(WBC_1H24)
    assert dict((label, value) for label, value, _ in steps)["Notable Items"] == Decimal(-3)
    # Without the words that say two figures are being compared, it's still refused.
    uncompared = dict(WBC_1H24)
    uncompared["steps"] = [dict(step) for step in WBC_1H24["steps"]]
    uncompared["steps"][-1]["quote"] = (
        "Notable Items from unrealised fair value losses for accounting purposes related to "
        "economic hedges of term funding detracted 5 basis points"
    )
    steps, _, refused = _check_westpac_1h24(uncompared)
    assert steps == [] and "not printed" in refused


def test_a_non_zero_step_whose_quote_isnt_on_the_page_still_fails_the_walk():
    invented = dict(WBC_1H24)
    invented["steps"] = [dict(step) for step in WBC_1H24["steps"]]
    invented["steps"][0]["quote"] = "Loan pricing: 14 basis point decrease from a new strategy"
    steps, _, refused = _check_westpac_1h24(invented)
    assert steps == [] and "not on the pages" in refused
