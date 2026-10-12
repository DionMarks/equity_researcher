"""Reading a bank nobody has read: its units, its layout and its own names for rows."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from bankresearch.agent.orchestrator import _found_profile, _register
from bankresearch.domain.models import DocMeta
from bankresearch.domain.registry import get_registry
from bankresearch.domain.units import normalise_aud, value_tolerance
from bankresearch.ingestion.facts import cell_under, extract_facts, line_item_index
from bankresearch.ingestion.parse_pdf import ParsedDocument, ParsedPage, ParsedTable
from bankresearch.ingestion.tables import detect_units
from bankresearch.llm.client import LLMResponse

_BANK = "ZZB"
_HEADER = ("", "Key Metric", "", "", "30 June 2026", "", "", "30 June 2025", "", "", "Movement")
_ROWS = (
    ("", "Amounts in thousands of AUD", "", "", "($\u2019000s)", "", "", "($\u2019000s)"),
    ("Net interest income", "", "", "", "24,042", "", "21,882", "", "", "9.9%", ""),
    ("Total assets", "", "", "", "1,284,523", "", "1,233,688", "", "", "4.1%", ""),
    ("Net interest margin (average)", "", "", "", "1.96%", "", "1.52%", "", "", "44 bps", ""),
    ("Number of branches", "", "", "", "7", "", "8", "", "", "", ""),
)


@pytest.fixture
def bank():  # type: ignore[no-untyped-def]
    """A bank found on the ASX, registered for the test and taken away after it."""
    profile = _found_profile(_BANK, "Zed Banking Corporation Limited", 6, (), ("Zed",))
    _register(profile)
    yield profile
    get_registry().banks.pop(_BANK, None)
    line_item_index.cache_clear()


def _document() -> tuple[ParsedDocument, DocMeta]:
    table = ParsedTable(
        table_id="t1",
        page_no=9,
        caption="Key metrics",
        header=_HEADER,
        rows=_ROWS,
        units="aud_k",
    )
    page = ParsedPage(page_no=9, text="Key metrics for the year", kind="mixed", tables=(table,))
    document = ParsedDocument(
        doc_id="zzb-fy26-results-document", sha256="x", page_count=9, pages=(page,)
    )
    meta = DocMeta(
        doc_id=document.doc_id,
        bank_id=_BANK,
        period="FY26",
        doc_type="RESULTS_DOCUMENT",
        publish_date=date(2026, 8, 27),
        url="https://announcements.asx.com.au/x.pdf",
    )
    return document, meta


# --- units and layout ------------------------------------------------------------------


@pytest.mark.parametrize(
    "text", ["($\u2019000s)", "$'000", "$000", "Amounts in thousands of AUD", "In thousands of AUD"]
)
def test_thousands_are_recognised_however_they_are_printed(text):
    assert detect_units(text) == "aud_k"


def test_a_figure_in_thousands_is_held_in_millions_to_its_printed_precision():
    assert normalise_aud(Decimal("24042"), "aud_k") == (Decimal("24.042"), "aud_m")
    assert value_tolerance("aud_k", reported=Decimal("24042")) == Decimal("0.0005")


def test_a_figure_one_column_off_its_heading_is_still_read():
    row = _ROWS[1]
    headed = [i for i, header in enumerate(_HEADER) if header]
    assert cell_under(row, 4, headed) == "24,042"
    assert cell_under(row, 7, headed) == "21,882"
    # Equally near two headings, a figure belongs to neither.
    assert cell_under(("x", "", "5", "", ""), 3, [1, 3]) == ""


def test_a_new_banks_thousands_are_read_as_millions_and_quoted_as_printed(bank):
    document, meta = _document()
    facts = {(f.component, f.period): f for f in extract_facts(document, meta, ("FY26", "FY25"))}
    nii = facts[("nii", "FY25")]
    assert (nii.value, nii.unit, nii.published_value) == (Decimal("21.882"), "aud_m", 21882)


def test_a_new_bank_is_held_to_ranges_for_a_much_smaller_bank(bank):
    small = line_item_index(_BANK).plausible["aiea"][0]
    configured = line_item_index("BEN").plausible["aiea"][0]
    assert small < Decimal(1500) < configured


# --- rows named in the bank's own words ------------------------------------------------


class _Answers:
    """A model that names rows: ``mappings`` is what it returns."""

    available = True

    def __init__(self, mappings):  # type: ignore[no-untyped-def]
        self.mappings, self.requests = mappings, []

    def complete(self, request):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        return LLMResponse(structured={"mappings": self.mappings})


def test_only_unmatched_rows_near_a_missing_figure_are_offered(bank):
    from bankresearch.agent.steps.map_rows import offer_rows

    document, meta = _document()
    rows = offer_rows(document, meta, 6, ("FY26", "FY25"), line_item_index(_BANK), ["nim"])
    labels = [row.label for row in rows]
    assert "Net interest margin (average)" in labels
    assert "Net interest income" not in labels  # already matched
    assert "Number of branches" not in labels  # no word in common with the margin
    margin = next(row for row in rows if row.label.startswith("Net interest margin"))
    assert margin.cells == (("FY26", "1.96%"), ("FY25", "1.52%"))


def test_a_model_may_only_name_offered_rows_for_the_figures_asked(bank):
    from bankresearch.agent.steps.map_rows import OfferedRow, map_rows

    rows = [
        OfferedRow(1, "Net interest margin (average)", 9, "", (("FY26", "1.96%"),)),
        OfferedRow(2, "Total assets", 9, "", (("FY26", "1,284,523"),)),
        OfferedRow(3, "Margin", 9, "", (("FY26", "1.96%"),)),
    ]
    answers = _Answers(
        [
            {"row": 1, "component": "nim"},
            {"row": 2, "component": "rwa"},  # not asked for
            {"row": 9, "component": "aiea"},  # not offered
            {"row": 3, "component": "aiea"},  # one ordinary word
        ]
    )
    mapping = map_rows(["nim", "aiea"], rows, answers, model="m", effort="low", bank_name="Zed")
    assert mapping.labels == {"Net interest margin (average)": "nim"}
    assert set(mapping.refused) == {"Total assets", "9", "Margin"}


def test_two_rows_for_one_figure_are_both_refused(bank):
    from bankresearch.agent.steps.map_rows import OfferedRow, map_rows

    rows = [
        OfferedRow(1, "Net interest margin (average)", 9, "", (("FY26", "1.96%"),)),
        OfferedRow(2, "Net interest margin (spot)", 9, "", (("FY26", "2.00%"),)),
    ]
    answers = _Answers([{"row": 1, "component": "nim"}, {"row": 2, "component": "nim"}])
    mapping = map_rows(["nim"], rows, answers, model="m", effort="low", bank_name="Zed")
    assert not mapping.labels and len(mapping.refused) == 2


def test_a_matched_name_is_read_like_any_configured_label(bank):
    document, meta = _document()
    before = {f.component for f in extract_facts(document, meta, ("FY26", "FY25"))}
    _register(replace(bank, row_labels=(("Net interest margin (average)", "nim"),)))
    after = {
        (f.component, f.period, f.value) for f in extract_facts(document, meta, ("FY26", "FY25"))
    }
    assert "nim" not in before
    assert {("nim", "FY26", Decimal("1.96")), ("nim", "FY25", Decimal("1.52"))} <= after


def _spec_and_agent(tmp_path, held, ingest, mappings, monkeypatch):  # type: ignore[no-untyped-def]
    from bankresearch.agent import orchestrator
    from bankresearch.agent.orchestrator import Agent
    from bankresearch.agent.steps.resolve import resolve_request
    from bankresearch.settings import get_settings

    document, meta = _document()
    meta = meta.model_copy(update={"sha256": "abc"})
    settings = get_settings().model_copy(update={"data_dir": tmp_path})
    settings.raw_dir.mkdir(parents=True)
    (settings.raw_dir / "abc.pdf").write_bytes(b"%PDF-")

    class Store:
        def __init__(self) -> None:
            self.saved: tuple[tuple[str, str], ...] = ()
            self.sought: set[str] = set()

        def get_facts(self, spec):  # type: ignore[no-untyped-def]
            return [SimpleNamespace(component=c, period=p) for c, p in held]

        def put_row_labels(self, bank_id, labels):  # type: ignore[no-untyped-def]
            self.saved = tuple(labels)

        def rows_sought(self, bank_id):  # type: ignore[no-untyped-def]
            return set(self.sought)

        def discovered_banks(self):  # type: ignore[no-untyped-def]
            return []

        def put_rows_sought(self, bank_id, sought):  # type: ignore[no-untyped-def]
            self.sought = self.sought | set(sought)

    store = Store()
    agent = Agent(store=store, settings=settings, persist=False)  # type: ignore[arg-type]
    monkeypatch.setattr(agent, "_model", lambda: _Answers(mappings))
    monkeypatch.setattr(orchestrator, "read_processed", lambda doc_id, root: document)
    monkeypatch.setattr(orchestrator, "ingest_document", lambda *a, **k: ingest())
    spec = resolve_request(_BANK, "FY26", "nim", as_of=date(2026, 10, 1))
    availability = SimpleNamespace(current_docs=[meta])
    return agent, store, spec, availability


def test_a_matched_name_is_kept_once_its_figures_pass(bank, tmp_path, monkeypatch):
    from bankresearch.llm.budget import Budget

    held = {("nii", "FY26"), ("nii", "FY25")}
    agent, store, spec, availability = _spec_and_agent(
        tmp_path,
        held,
        lambda: held.update({("nim", "FY26"), ("nim", "FY25")}),
        [{"row": 1, "component": "nim"}],
        monkeypatch,
    )
    caveats: list[str] = []
    diagnostics = SimpleNamespace(warnings=[])
    matched = agent._learn_rows(spec, availability, Budget(), caveats, diagnostics)
    assert matched == ["Net interest margin (average) as nim"]
    assert store.saved == (("Net interest margin (average)", "nim"),)
    assert get_registry().bank(_BANK).row_labels == store.saved
    assert "names some rows in its own words" in caveats[0]


def test_a_matched_name_that_costs_a_figure_is_dropped(bank, tmp_path, monkeypatch):
    from bankresearch.llm.budget import Budget

    held = {("nii", "FY26"), ("nii", "FY25")}
    agent, store, spec, availability = _spec_and_agent(
        tmp_path,
        held,
        # The wrong row put the margin identity out, and net interest income went with it.
        lambda: held.discard(("nii", "FY25")),
        [{"row": 1, "component": "nim"}],
        monkeypatch,
    )
    diagnostics = SimpleNamespace(warnings=[])
    assert agent._learn_rows(spec, availability, Budget(), [], diagnostics) == []
    assert store.saved == () and get_registry().bank(_BANK).row_labels == ()
    assert any("wasn't kept" in w for w in diagnostics.warnings)


def test_one_wrong_match_doesnt_cost_the_right_one(bank, tmp_path, monkeypatch):
    from bankresearch.llm.budget import Budget

    held = {("nii", "FY26"), ("nii", "FY25")}

    def ingest() -> None:
        labels = dict(get_registry().bank(_BANK).row_labels)
        held.clear()
        held.update({("nii", "FY26"), ("nii", "FY25")})
        if "Net interest margin (average)" in labels:
            held.update({("nim", "FY26"), ("nim", "FY25")})
        if "Total assets" in labels:
            # Total assets read as the average balance put the margin identity out.
            held.difference_update({("nii", "FY26"), ("nii", "FY25")})

    agent, store, spec, availability = _spec_and_agent(
        tmp_path,
        held,
        ingest,
        [{"row": 2, "component": "aiea"}, {"row": 1, "component": "nim"}],
        monkeypatch,
    )
    diagnostics = SimpleNamespace(warnings=[])
    matched = agent._learn_rows(spec, availability, Budget(), [], diagnostics)
    assert matched == ["Net interest margin (average) as nim"]
    assert ("nii", "FY25") in held and ("nim", "FY26") in held
    assert store.saved == (("Net interest margin (average)", "nim"),)
    assert any("as aiea" in w and "wasn't kept" in w for w in diagnostics.warnings)


def test_a_run_that_has_to_find_its_filing_gets_the_cold_guardrail():
    from bankresearch.llm.budget import Budget

    registry = get_registry()
    budget = Budget.from_config(registry.budgets, registry.models)
    assert budget.cost_guardrail_usd == Decimal("1.50")
    budget.go_cold(registry.budgets)
    assert (budget.cost_guardrail_usd, budget.max_calls) == (Decimal("5.00"), 30)
    roomy = Budget(max_calls=40, cost_guardrail_usd=Decimal("9"))
    roomy.go_cold(registry.budgets)
    assert (roomy.cost_guardrail_usd, roomy.max_calls) == (Decimal("9"), 40)


def test_a_figure_already_looked_for_in_a_filing_isnt_asked_for_again(bank, tmp_path, monkeypatch):
    from bankresearch.llm.budget import Budget

    held = {("nii", "FY26"), ("nii", "FY25"), ("nim", "FY26"), ("nim", "FY25")}
    agent, store, spec, availability = _spec_and_agent(
        tmp_path, held, lambda: None, [], monkeypatch
    )
    asked = agent._model()
    monkeypatch.setattr(agent, "_model", lambda: asked)
    diagnostics = SimpleNamespace(warnings=[])
    assert agent._learn_rows(spec, availability, Budget(), [], diagnostics) == []
    assert store.sought == {"zzb-fy26-results-document aiea"} and len(asked.requests) == 1
    agent._learn_rows(spec, availability, Budget(), [], diagnostics)
    assert len(asked.requests) == 1  # BNK's filing has no average balances to find


def test_a_found_banks_estimate_says_when_it_must_fetch_its_slide_pack(bank, tmp_path, monkeypatch):
    """BNK's filing gives its margin but no average balances, so a question about it reads
    the slide pack too. Its estimate said "saved", priced as an answer from saved data."""
    from bankresearch.agent import orchestrator

    held = {("nii", "FY26"), ("nii", "FY25"), ("nim", "FY26"), ("nim", "FY25")}
    agent, _, _, availability = _spec_and_agent(tmp_path, held, lambda: None, [], monkeypatch)
    monkeypatch.setattr(
        orchestrator,
        "check_availability",
        lambda spec, store: SimpleNamespace(
            reportable=True, current_docs=availability.current_docs
        ),
    )
    estimate = agent.estimate(_BANK, "FY26", "nim", date(2026, 10, 1))
    assert estimate.path == "fetch" and estimate.outside_saved_data
    assert any("results presentation" in step for step in estimate.steps)
    held.update({("aiea", "FY26"), ("aiea", "FY25")})
    assert agent.estimate(_BANK, "FY26", "nim", date(2026, 10, 1)).path == "saved"


# --- remembered with the bank ----------------------------------------------------------


def test_matched_names_are_remembered_with_the_bank(tmp_path):
    from bankresearch.memory.db import connect
    from bankresearch.memory.fact_store import FactStore

    store = FactStore(connect(tmp_path / "research.sqlite"))
    store.put_discovered_bank(
        _BANK, "Zed", 6, ["x.com.au"], ["Zed"], confirmed_by="doc", discovered_at=date(2026, 10, 1)
    )
    store.put_row_labels(_BANK, [("Net interest margin (average)", "nim")])
    assert store.discovered_banks()[0]["row_labels"] == (("Net interest margin (average)", "nim"),)


def test_a_database_made_before_the_names_were_kept_gains_the_column(tmp_path):
    from bankresearch.memory.db import connect

    path = tmp_path / "research.sqlite"
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE discovered_banks (bank_id TEXT PRIMARY KEY, name TEXT NOT NULL, "
        "fy_end_month INTEGER NOT NULL, ir_domains TEXT NOT NULL, aliases TEXT NOT NULL, "
        "confirmed_by TEXT NOT NULL, discovered_at TEXT NOT NULL)"
    )
    old.commit()
    old.close()
    columns = {row["name"] for row in connect(path).execute("PRAGMA table_info(discovered_banks)")}
    assert "row_labels" in columns


# --- the slide pack --------------------------------------------------------------------


def _deck_claim(kind: str = "unknown", end: str | None = None) -> dict[str, object]:
    return {
        "issuer": "BNK Banking Corporation Limited",
        "doc_type": "INVESTOR_PRESENTATION",
        "period_kind": kind,
        "period_end": end,
        "publish_date": "2026-08-27",
    }


def test_a_slide_pack_is_placed_by_the_period_on_its_cover(monkeypatch):
    from bankresearch.acquisition.discover import classification_agrees

    _register(_found_profile("BBC", "BNK Banking Corporation Limited", 6, (), ()))
    try:
        cover = "FY26 Financial Results\n27th August 2026\nInvestor Presentation"
        agree = lambda claim, text: classification_agrees(  # noqa: E731
            claim,
            text,
            bank_id="BBC",
            period="FY26",
            fy_end_month=6,
            doc_type="INVESTOR_PRESENTATION",
        )
        assert agree(_deck_claim(), cover) == (True, "")
        assert agree(_deck_claim("full_year", "2026-06-30"), cover) == (True, "")
        # A deck for another year, or a classifier that reads another period, is refused.
        assert not agree(_deck_claim(), cover.replace("FY26", "FY25"))[0]
        assert not agree(_deck_claim("half_year", "2025-12-31"), cover)[0]
        # A results document still has to say which year it ended.
        assert not classification_agrees(
            _deck_claim(), cover, bank_id="BBC", period="FY26", fy_end_month=6
        )[0]
    finally:
        get_registry().banks.pop("BBC", None)


def test_a_walk_titled_by_its_two_periods_is_a_walk_page():
    page = ParsedPage(page_no=19, text="1.52%\nLending\n1.96%\nNIM FY25 to FY26", kind="mixed")
    assert page.likely_walk_page
    assert not ParsedPage(page_no=5, text="FY26 Key Outcomes", kind="text").likely_walk_page


# --- the ASX's own copies ----------------------------------------------------------------


def test_a_word_of_the_asx_stamp_in_front_of_a_label_is_read_past():
    """Every PDF on the ASX's announcements site is stamped "For personal use only" down the
    margin, and its words land among the row labels."""
    from bankresearch.ingestion.facts import label_candidates
    from bankresearch.ingestion.sections import is_wrap_candidate

    assert "Corporate tax expense" in label_candidates("only Corporate tax expense (4,301) 4")
    assert "Net interest income" in label_candidates("personal use Net interest income")
    assert not is_wrap_candidate("use") and not is_wrap_candidate("Only")
    # A name that really starts with one of the words still matches as printed first.
    assert label_candidates("Use of funds")[0] == "Use of funds"


def _westpac_1h24(rows: tuple[tuple[str, ...], ...], heading_path: tuple[str, ...] = ()):  # type: ignore[no-untyped-def]
    table = ParsedTable(
        table_id="t",
        page_no=14,
        caption="Half Year Half Year Half Year % Mov't",
        header=("$m", "March 2024", "Sept 2023", "March 2023"),
        rows=rows,
        units="aud_m",
    )
    page = ParsedPage(
        page_no=14,
        text="Group performance",
        kind="mixed",
        tables=(table,),
        heading_path=heading_path,
    )
    document = ParsedDocument(doc_id="wbc-1h24", sha256="x", page_count=14, pages=(page,))
    meta = DocMeta(
        doc_id="wbc-1h24",
        bank_id="WBC",
        period="1H24",
        doc_type="RESULTS_DOCUMENT",
        publish_date=date(2024, 5, 6),
        url="https://announcements.asx.com.au/x.pdf",
    )
    return {
        (f.component, f.period): f.value for f in extract_facts(document, meta, ("1H24", "1H23"))
    }


def test_westpacs_non_controlling_interests_are_read_through_the_stamp():
    """Westpac's 1H24 report, as the ASX holds it (p.14). Without the line, the 1H23 bridge
    missed by A$4m and every income line of the half was quarantined."""
    facts = _westpac_1h24(
        (
            # The table's own costs, in brackets, are what say how it prints a deduction.
            ("Operating expenses", "(5,395)", "(5,704)", "(4,988)"),
            ("Profit after income tax expense", "3,342", "3,196", "4,005"),
            ("only Profit attributable to non-controlling", "", "", ""),
            ("interests (NCI)", "-", "(2)", "(4)"),
            ("Net profit attributable to owners of WBC", "3,342", "3,194", "4,001"),
        )
    )
    assert facts[("minority_interests", "1H23")] == Decimal(4)
    assert facts[("minority_interests", "1H24")] == Decimal(0)


def test_a_numbered_division_heading_is_still_a_division():
    """Westpac's 2024 interim report numbers its segment pages ("2.1. Consumer"), and the
    Consumer division's margin of 1.69% was stored as the group's."""
    rows = (("Net interest margin (Ex Notable Items)", "1.69%", "1.76%", "2.00%"),)
    assert ("nim", "1H24") in _westpac_1h24(rows)
    assert ("nim", "1H24") not in _westpac_1h24(rows, heading_path=("2.1. Consumer",))


def test_a_caption_cant_turn_an_amount_into_a_ratio_or_back():
    """Macquarie's FY26 results table is captioned with its movement column's "%", and its
    profit of A$4,847m was read as a percentage and refused."""
    index = line_item_index("WBC")
    assert index.unit_for("headline_earnings", "Net profit after tax", "pct", "4,847") == "aud_m"
    assert index.unit_for("nim", "Net interest margin", "aud_m", "1.86") == "pct"
    # Within a kind, the caption still decides: an amount in a table of billions.
    assert index.unit_for("aiea", "Average interest earning assets", "aud_bn", "1,177") == "aud_bn"


def _ratio_and_its_parts(bank_id: str, ratio_table: str = "p49"):  # type: ignore[no-untyped-def]
    from bankresearch.domain.models import Fact
    from bankresearch.ingestion.validate import validate_facts

    def fact(component: str, value: str, unit: str, table: str = "p7") -> Fact:
        return Fact(
            fact_id="",
            bank_id=bank_id,
            period="FY26",
            component=component,
            value=Decimal(value),
            unit=unit,  # type: ignore[arg-type]
            doc_id="d",
            doc_type="INVESTOR_PRESENTATION",
            page_no=7,
            table_id=table,
            publish_date=date(2026, 5, 8),
        )

    facts = [
        # A division's, on a page read as the group's.
        fact("cost_to_income", "51", "pct", table=ratio_table),
        fact("operating_expenses", "12748", "aud_m"),
        fact("operating_income", "19477", "aud_m"),
    ]
    outcome = validate_facts(facts, bank_id, ("FY26",))
    return {fact.component for fact, _ in outcome.quarantined}


def test_a_found_banks_ratio_read_apart_from_its_parts_is_not_trusted(bank):
    """Configured, the printed ratio stands and its parts go. Found, a ratio read from
    another table than the group's own totals may be a division's, so it goes as well; read
    from the same table, it's the bank's own figure and stands as it would anywhere."""
    assert _ratio_and_its_parts("WBC") == {"operating_expenses", "operating_income"}
    assert _ratio_and_its_parts(_BANK) == {
        "cost_to_income",
        "operating_expenses",
        "operating_income",
    }
    assert _ratio_and_its_parts(_BANK, ratio_table="p7") == {
        "operating_expenses",
        "operating_income",
    }


def test_a_row_offered_to_the_model_is_named_without_the_stamp(bank):
    """A name kept with the bank has to match the row in a copy that isn't stamped:
    Macquarie's deck offered "use Net credit impairment charges"."""
    from bankresearch.agent.steps.map_rows import offer_rows
    from bankresearch.ingestion.facts import without_stamp

    assert without_stamp("use Net credit impairment charges") == "Net credit impairment charges"
    assert without_stamp("Use") == "Use"
    document, meta = _document()
    stamped = replace(
        document.pages[0].tables[0],
        rows=(
            *document.pages[0].tables[0].rows,
            ("only Net interest margin (average)", "", "", "", "1.96%", "", "1.52%"),
        ),
    )
    page = replace(document.pages[0], tables=(stamped,))
    rows = offer_rows(
        replace(document, pages=(page,)), meta, 6, ("FY26", "FY25"), line_item_index(_BANK), ["nim"]
    )
    assert all(not row.label.lower().startswith(("only ", "use ")) for row in rows)


def test_a_figures_own_word_is_a_name_and_an_ordinary_word_isnt():
    """BNK labels its tax line just "Tax", and refusing it left its bridge without tax."""
    from bankresearch.agent.steps.map_rows import _specific

    assert _specific("Tax", "tax")
    assert _specific("NIM", "nim") and _specific("Credit Loss Provision", "credit_impairment")
    assert not _specific("Total", "tax") and not _specific("Interest", "nii")
    assert not _specific("Tax", "credit_impairment")


def test_a_line_that_needs_the_others_in_place_gets_its_turn_after_them(
    bank, tmp_path, monkeypatch
):
    """BNK's NPAT switches its bridge on, which can't close until its tax is read too: on
    its own it costs the bank the lines it had, so it has to be tried again after them."""
    held = {("nii", "FY26"), ("nii", "FY25")}

    def ingest() -> None:
        labels = dict(get_registry().bank(_BANK).row_labels)
        held.clear()
        held.update({("nii", "FY26"), ("nii", "FY25")})
        if "Tax" in labels:
            held.update({("tax", "FY26"), ("tax", "FY25")})
        if "NPAT" in labels:
            held.update({("headline_earnings", "FY26"), ("headline_earnings", "FY25")})
            if "Tax" not in labels:
                held.difference_update({("nii", "FY26"), ("nii", "FY25")})
        if "Total assets" in labels:
            held.difference_update({("nii", "FY26"), ("nii", "FY25")})

    agent, _, _, availability = _spec_and_agent(tmp_path, held, ingest, [], monkeypatch)
    from bankresearch.agent.steps.resolve import resolve_request

    spec = resolve_request(_BANK, "FY26", "cash earnings", as_of=date(2026, 10, 1))
    meta = availability.current_docs[0]
    diagnostics = SimpleNamespace(warnings=[])
    kept = agent._keep_what_passes(
        spec,
        meta,
        tmp_path / "raw" / "abc.pdf",
        {"NPAT": "headline_earnings", "Tax": "tax", "Total assets": "aiea"},
        lambda: set(held),
        diagnostics,
    )
    assert kept == {"Tax": "tax", "NPAT": "headline_earnings"}
    assert ("nii", "FY25") in held and ("headline_earnings", "FY26") in held
    assert any("('Total assets')" in w and "wasn't kept" in w for w in diagnostics.warnings)


def test_a_bridge_printed_to_the_thousand_is_held_to_the_thousand():
    """Sized for whole millions, the allowance let BNK's bridge miss by A$3m on a movement
    of A$5m. Whole millions are unchanged."""
    from bankresearch.decomposition.metrics import reconciliation_tolerance

    majors = reconciliation_tolerance("aud_m", 6, Decimal("10072"), Decimal("9836"), "aud_m")
    bnk = reconciliation_tolerance("aud_m", 6, Decimal("1.283"), Decimal("-3.706"), "aud_m")
    assert majors == Decimal("4.0")  # 3 for six lines, and the two published figures' 1
    assert bnk < Decimal("0.01")


def test_a_movement_is_shown_to_the_thousand_when_the_bank_prints_thousands():
    from bankresearch.output.render_md import amount_places

    assert amount_places(Decimal("1.283"), Decimal("-3.706")) == 3
    assert amount_places(Decimal("125.6"), Decimal("168.1")) == 1
    assert amount_places(Decimal("10072"), Decimal("9836")) == 0


def test_a_found_banks_lines_are_named_as_it_names_them():
    """BNK's A$3.5m was called the "notable item" it is in the taxonomy rather than the
    goodwill impairment it is."""
    from bankresearch.agent.steps.decompose import _in_the_banks_words
    from bankresearch.domain.models import Decomposition, DriverContribution

    def driver(driver_id: str, component: str) -> DriverContribution:
        return DriverContribution(
            driver_id=driver_id,
            label="taxonomy name",
            contribution=Decimal(-1),
            direction="-",
            method="computed",
            evidence_type="computed",
            from_components=(component,),
        )

    decomposition = Decomposition(
        metric_id="headline_earnings",
        path="B",
        change_unit="aud_m",
        headline_change=Decimal(-2),
        drivers=[driver("earn.notable_items", "notable_items"), driver("earn.nii", "nii")],
    )
    _in_the_banks_words(decomposition, {"notable_items": "Goodwill Impairment"})
    assert [d.label for d in decomposition.drivers] == ["Goodwill Impairment", "taxonomy name"]


def test_the_cost_to_income_expense_note_doesnt_cap_its_drivers():
    """The scorer caps a driver computed from an aggregate by the words "in aggregate" in the
    decomposition's notes. The expense note isn't that, and once said it by accident, which
    took every cost-to-income answer from High to Medium."""
    from bankresearch.decomposition.metrics import cost_to_income

    source = Path(cost_to_income.__file__).read_text(encoding="utf-8")
    assert "in aggregate" not in source
