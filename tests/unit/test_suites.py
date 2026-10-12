"""The perturbation, stability and typography checks.

These measure how the system degrades, which an eval over clean cases can't: a run that
answers the same way with its evidence removed wasn't using it.
"""

from __future__ import annotations

import pathlib
from datetime import date
from decimal import Decimal

import pytest

from bankresearch.evals.cases import EvalCase
from bankresearch.evals.perturb import (
    SUITES,
    UNEXPLAINED_CAP,
    PerturbationResult,
    ablations_for,
    conflicts_for,
    look_ahead_for,
    perturbations_for,
)
from bankresearch.evals.stability import StabilityReport, jaccard
from bankresearch.ingestion.outline import Span, headings_from_typography
from bankresearch.llm.client import LLMResponse, ScriptedClient


def _case() -> EvalCase:
    return EvalCase(
        case_id="anz_1h26_cet1",
        split="dev",
        bank="ANZ",
        period="1H26",
        metric="cet1_ratio",
        as_of=date(2026, 5, 31),
    )


def _fact(component: str = "cet1_ratio", value: str = "11.8", unit: str = "pct"):
    from bankresearch.domain.models import Basis, Fact

    return Fact(
        fact_id="f1",
        bank_id="ANZ",
        period="1H25",
        component=component,
        value=Decimal(value),
        published_value=Decimal(value),
        unit=unit,
        basis=Basis(),
        doc_id="anz-1h26-results-document",
        doc_type="RESULTS_DOCUMENT",
        page_no=11,
        publish_date=date(2026, 5, 1),
        validated=True,
    )


def _body(count: int = 30) -> list[Span]:
    return [Span(1, f"body text line {n}", 9.0, 300.0 + n, 800.0) for n in range(count)]


# --- ablation --------------------------------------------------------------


def test_an_ablation_expects_the_movement_to_survive_and_the_drivers_not_to():
    """The movement is the issuer's own published figure. The attribution needs every
    input, so a run that still attributes with one missing is attributing from somewhere
    other than the inputs."""
    [first, *_] = ablations_for(_case(), ("cet1_ratio", "cet1_capital", "rwa"))
    assert first.expect_answer is True
    assert first.expect_drivers is False
    assert first.drop_components == ("cet1_capital",)


def test_the_headline_figure_is_never_ablated():
    """Without it there's no movement, which tests the abstention path rather than
    attribution, and that path has its own cases.
    """
    dropped = {p.drop_components[0] for p in ablations_for(_case(), ("cet1_ratio", "rwa"))}
    assert dropped == {"rwa"}


def test_a_perturbation_removes_the_component_from_what_the_agent_sees():
    perturbation = ablations_for(_case(), ("cet1_ratio", "rwa"))[0]
    facts = [_fact("cet1_ratio"), _fact("rwa", "462100", "aud_m")]
    assert [f.component for f in perturbation.applies_to(facts)] == ["cet1_ratio"]


# --- look-ahead ------------------------------------------------------------


def test_the_look_ahead_perturbation_moves_the_as_of_behind_publication():
    """The one perturbation whose failure is a correctness bug rather than a confidence
    bug: an answer means a backtest could read a figure before it existed."""
    perturbation = look_ahead_for(_case(), date(2026, 5, 1))
    assert perturbation.as_of_override == date(2026, 4, 30)
    assert perturbation.expect_answer is False
    assert perturbation.expect_drivers is False


# --- conflict injection ----------------------------------------------------


def test_a_rounding_conflict_is_sized_in_the_figures_own_unit():
    """``value_tolerance`` answers in basis points for a ratio, and adding that to a
    percentage turns 11.8% into 14.3%: a 250bp change labelled as a rounding difference.
    """
    perturbation = conflicts_for(_case(), _fact(value="11.8"), rounding=True)
    injected = perturbation.inject[0].value
    assert Decimal("11.8") < injected < Decimal("11.85")
    assert perturbation.expect_conflict == "ROUNDING"


def test_an_unexplained_conflict_is_material_and_capped():
    perturbation = conflicts_for(_case(), _fact(value="11.8"), rounding=False)
    assert perturbation.inject[0].value > Decimal("16")
    assert perturbation.expect_conflict == "UNEXPLAINED"
    assert perturbation.expect_confidence_at_most == UNEXPLAINED_CAP


def test_a_more_specific_conflict_class_counts_as_a_pass():
    """Injecting a materially different prior into CBA's impairment produced BASIS, because
    the corpus really does hold two bases for it, and BASIS comes before UNEXPLAINED in the
    order, so it's a better answer than the one asked for.
    """
    result = PerturbationResult(
        perturbation=conflicts_for(_case(), _fact(value="11.8"), rounding=False),
        answered=True,
        drivers=2,
        classification="BASIS",
        confidence=Decimal("0.82"),
        baseline_confidence=Decimal("0.82"),
    )
    assert result.passed, "a cap that belongs to UNEXPLAINED must not be applied to BASIS"


def test_a_conflict_nobody_detected_is_a_failure():
    result = PerturbationResult(
        perturbation=conflicts_for(_case(), _fact(value="11.8"), rounding=False),
        answered=True,
        drivers=2,
        classification="",
    )
    assert not result.passed


def test_an_unexplained_conflict_that_did_not_respect_the_cap_is_a_failure():
    result = PerturbationResult(
        perturbation=conflicts_for(_case(), _fact(value="11.8"), rounding=False),
        answered=True,
        drivers=2,
        classification="UNEXPLAINED",
        confidence=Decimal("0.92"),
        baseline_confidence=Decimal("0.92"),
    )
    assert not result.passed


def test_whether_an_attribution_survives_a_conflict_is_not_asserted():
    """It is downstream of the identity and legitimately varies: a conflict resolved by
    precedence may leave the bridge intact or not."""
    assert conflicts_for(_case(), _fact(value="11.8"), rounding=True).expect_drivers is None


def test_every_suite_is_named_and_the_distractor_one_says_why_it_is_absent():
    from bankresearch.evals.perturb import distractor_note

    assert set(SUITES) == {"ablation", "look_ahead", "conflict", "distractor"}
    assert "needs a model" in distractor_note()["skipped"]


def test_a_case_with_no_prior_or_publish_date_contributes_only_ablations():
    assert all(p.suite == "ablation" for p in perturbations_for(_case(), ("cet1_ratio", "rwa")))


# --- stability -------------------------------------------------------------


def test_identical_runs_score_one():
    report = StabilityReport(
        case_id="c",
        statuses=["OK"] * 3,
        levels=["High"] * 3,
        driver_sets=[frozenset({"a", "b"})] * 3,
        changes=[Decimal("1.5")] * 3,
    )
    assert report.stable
    assert report.driver_jaccard == 1.0
    assert report.change_agreement == 1.0


def test_a_headline_figure_that_differs_between_runs_is_not_stable():
    """The strictest of the three and the one that matters most: a driver set can
    legitimately vary once a model is involved, and a headline figure cannot."""
    report = StabilityReport(
        case_id="c",
        statuses=["OK"] * 3,
        levels=["High"] * 3,
        driver_sets=[frozenset({"a"})] * 3,
        changes=[Decimal("1.5"), Decimal("1.5"), Decimal("1.6")],
    )
    assert not report.stable
    assert report.change_agreement == 0.0


def test_jaccard_is_pairwise_so_one_divergent_run_does_not_collapse_it():
    same = frozenset({"a", "b"})
    assert jaccard([same, same, frozenset({"a"})]) == pytest.approx((1.0 + 0.5 + 0.5) / 3)
    assert jaccard([frozenset(), frozenset()]) == 1.0


# --- typography outline ----------------------------------------------------


def test_headings_are_recovered_from_size_and_position():
    spans = [Span(1, "Financial information", 18.0, 40.0, 800.0), *_body()]
    assert [h.text for h in headings_from_typography(spans)] == ["Financial information"]


def test_a_figure_set_in_display_type_is_not_a_heading():
    """These packs set their key numbers in large type ("$3,780m", "83c", "12.4%"), which is
    exactly what a size-based rule would otherwise pick up.
    """
    spans = [
        Span(1, text, 24.0, 40.0, 800.0) for text in ("$3,780m", "12.4%", "(1,234)", "505,310")
    ]
    assert headings_from_typography([*spans, *_body()]) == []


def test_a_sentence_is_not_a_heading_however_it_is_set():
    spans = [Span(1, "The margin fell three basis points in the half.", 16.0, 40.0, 800.0)]
    assert headings_from_typography([*spans, *_body()]) == []


def test_a_heading_low_on_the_page_is_not_a_heading():
    spans = [Span(1, "Financial information", 18.0, 700.0, 800.0), *_body()]
    assert headings_from_typography(spans) == []


def test_too_little_text_returns_nothing_rather_than_a_guess():
    """Every judgement here is relative to the body font size, and without enough text to
    establish one they are all arbitrary."""
    assert headings_from_typography([Span(1, "Big Words", 30.0, 10.0, 800.0)]) == []


def test_the_body_font_is_weighted_by_characters_not_by_span_count():
    """A document with many short large-set labels and a few long paragraphs would
    otherwise take a heading size as its body, and recover nothing."""
    spans = [Span(1, "Label", 18.0, 40.0 + n, 800.0) for n in range(25)]
    spans += [Span(1, "x" * 400, 9.0, 400.0 + n, 800.0) for n in range(3)]
    assert [h.text for h in headings_from_typography(spans)] == ["Label"] * 25


def test_a_documents_own_outline_is_never_overridden():
    source = pathlib.Path("src/bankresearch/ingestion/parse_pdf.py").read_text(encoding="utf-8")
    assert "if not outline:" in source, "typography must only fill an empty outline"


# --- the taxonomy fallback -------------------------------------------------


def test_a_proposed_id_the_metric_does_not_declare_is_refused():
    """A plausible id that does not exist would roll up under a parent that was never
    defined, and no test would catch it because the row came from a document."""
    from bankresearch.agent.steps.taxonomy_llm import map_labels_with_model

    client = ScriptedClient(
        responses=[
            LLMResponse(
                structured={
                    "mappings": [{"label": "Zqx widget adjustment", "driver_id": "nim.invented"}]
                }
            )
        ]
    )
    outcome = map_labels_with_model("nim", ["Zqx widget adjustment"], client)
    assert not outcome.from_model
    assert "Zqx widget adjustment" in outcome.refused
    assert "Zqx widget adjustment" in outcome.unmapped


def test_null_leaves_the_row_unmapped_and_costs_nothing():
    from bankresearch.agent.steps.taxonomy_llm import map_labels_with_model

    client = ScriptedClient(
        responses=[
            LLMResponse(
                structured={
                    "mappings": [{"label": "Something outside the taxonomy", "driver_id": None}]
                }
            )
        ]
    )
    outcome = map_labels_with_model("nim", ["Something outside the taxonomy"], client)
    assert outcome.unmapped == ["Something outside the taxonomy"]
    assert not outcome.refused


def test_a_declared_id_is_accepted_and_recorded_as_a_model_mapping():
    from bankresearch.agent.steps.taxonomy_llm import map_labels_with_model
    from bankresearch.decomposition.taxonomy import canonical_ids

    target = canonical_ids("nim")[0]
    client = ScriptedClient(
        responses=[
            LLMResponse(
                structured={
                    "mappings": [{"label": "An unrecognised row label", "driver_id": target}]
                }
            )
        ]
    )
    outcome = map_labels_with_model("nim", ["An unrecognised row label"], client)
    assert outcome.from_model == {"An unrecognised row label": target}
    assert outcome.mapping["An unrecognised row label"] == target


def test_a_row_the_index_already_mapped_is_never_sent_to_a_model():
    from bankresearch.agent.steps.taxonomy_llm import map_labels_with_model
    from bankresearch.decomposition.taxonomy import map_labels

    mapped, _ = map_labels("nim", ["Deposit margin"])
    if not mapped:
        pytest.skip("no deterministic mapping available for the probe label")
    client = ScriptedClient(responses=[])
    outcome = map_labels_with_model("nim", list(mapped), client)
    assert client.requests == []
    assert outcome.mapping == mapped


def test_every_leftover_row_is_mapped_in_one_call_with_the_whole_walk_as_context():
    """One call, not one per row, and the model sees every bar so a bare label can be read
    against its neighbours."""
    from bankresearch.agent.steps.taxonomy_llm import map_labels_with_model

    client = ScriptedClient(
        responses=[
            LLMResponse(
                structured={
                    "mappings": [
                        {"label": "Zqx one", "driver_id": "nim.funding_costs"},
                        {"label": "Zqx two", "driver_id": None},
                    ]
                }
            )
        ]
    )
    walk = ["Loans", "Zqx one", "Zqx two", "Liquid assets"]
    outcome = map_labels_with_model("nim", walk, client, context=walk)
    assert len(client.requests) == 1
    assert "Loans | Zqx one | Zqx two | Liquid assets" in str(client.requests[0].messages)
    assert outcome.from_model == {"Zqx one": "nim.funding_costs"}
    assert outcome.unmapped == ["Zqx two"]
    assert outcome.mapping["Loans"] == "nim.lending_pricing"


def test_an_aggregate_node_or_the_residual_is_never_accepted_for_a_bar():
    from bankresearch.agent.steps.taxonomy_llm import map_labels_with_model

    client = ScriptedClient(
        responses=[
            LLMResponse(
                structured={
                    "mappings": [
                        {"label": "Zqx one", "driver_id": "nim.income"},
                        {"label": "Zqx two", "driver_id": "nim.residual"},
                    ]
                }
            )
        ]
    )
    outcome = map_labels_with_model("nim", ["Zqx one", "Zqx two"], client)
    assert not outcome.from_model
    assert set(outcome.refused) == {"Zqx one", "Zqx two"}


def test_without_a_model_unmapped_rows_say_what_they_become():
    from bankresearch.agent.steps.taxonomy_llm import map_labels_with_model

    class _Unavailable:
        available = False

        def complete(self, request: object) -> object:  # pragma: no cover
            raise AssertionError("must not be called")

    outcome = map_labels_with_model("nim", ["zzz unmapped"], _Unavailable())  # type: ignore[arg-type]
    assert outcome.unmapped == ["zzz unmapped"]
    assert "reported as 'other'" in outcome.notes[0]


# --- chained walks, end to end ---------------------------------------------


def _leg(opening: str, closing: str, open_value: str, close_value: str, label: str):
    from bankresearch.decomposition.walks import Walk, WalkStep
    from bankresearch.domain.models import SourceRef

    return Walk(
        opening_period=opening,
        closing_period=closing,
        opening_value=Decimal(open_value),
        closing_value=Decimal(close_value),
        steps=[
            WalkStep(
                label=label,
                driver_id="nim.deposit_pricing",
                value=Decimal(close_value) - Decimal(open_value),
            )
        ],
        unit="pct",
        source=SourceRef(
            doc_id="d",
            doc_type="RESULTS_DOCUMENT",
            page=1,
            publish_date=date(2026, 5, 1),
        ),
    )


def test_two_half_on_half_legs_chain_into_the_year_on_year_walk():
    """The majors publish half-on-half walks. A year-on-year comparison has to chain two,
    and the chain is what `choose_path_a` falls back to when no direct walk exists."""
    from bankresearch.decomposition.walks import choose_path_a

    legs = [
        _leg("1H25", "2H25", "1.56", "1.58", "first half"),
        _leg("2H25", "1H26", "1.58", "1.53", "second half"),
    ]
    chained = choose_path_a(legs, period="1H26", pcp="1H25", tolerance=Decimal("2.0"))
    assert chained is not None
    assert chained.method == "chained_walk"
    assert chained.opening_value == Decimal("1.56")
    assert chained.closing_value == Decimal("1.53")
    # Each leg's step is kept with its own page, and the two net to the movement that has
    # to reconcile; the decomposition sums them into one driver.
    assert [step.driver_id for step in chained.steps] == ["nim.deposit_pricing"] * 2
    assert sum(step.value for step in chained.steps) == Decimal("-0.03")


def test_a_chain_whose_midpoints_disagree_is_refused_not_patched():
    """Changes in a level telescope, so a leg that does not start where the previous one
    ended is not a leg of the same walk. Patching it would hide the mismatch inside a
    residual that looks like ordinary unexplained movement."""
    from bankresearch.decomposition.walks import choose_path_a

    legs = [
        _leg("1H25", "2H25", "1.56", "1.58", "first half"),
        _leg("2H25", "1H26", "1.70", "1.53", "second half"),
    ]
    assert choose_path_a(legs, period="1H26", pcp="1H25", tolerance=Decimal("2.0")) is None


def test_a_direct_walk_is_preferred_over_a_chain():
    from bankresearch.decomposition.walks import choose_path_a

    direct = _leg("1H25", "1H26", "1.56", "1.53", "full year")
    legs = [
        direct,
        _leg("1H25", "2H25", "1.56", "1.58", "first half"),
        _leg("2H25", "1H26", "1.58", "1.53", "second half"),
    ]
    chosen = choose_path_a(legs, period="1H26", pcp="1H25", tolerance=Decimal("2.0"))
    assert chosen is direct
    assert chosen.method == "management_walk"


# --- which pages the chart reader is pointed at ---------------------------


def test_walk_pages_are_ranked_with_the_known_walk_first():
    """``likely_walk_page`` matches any page that says "bps", which is every KPI table.

    It flagged 54 of NAB's 108 pages and 76 of ANZ's 131, and the first three by page order
    were taken, so NAB's margin was read off its cover, summary and contents pages, and
    ANZ's walk was only found because page 25 happens to be the second candidate in its
    pack.

    ANZ page 25 is the one page in this corpus known to be a group margin walk: it's what
    produced the six reconciling drivers.
    """
    from bankresearch.ingestion.artifacts import read_processed
    from bankresearch.ingestion.chart_walks import rank_walk_pages
    from bankresearch.settings import get_settings

    settings = get_settings()
    if not settings.db_path.exists():
        pytest.skip("no ingested corpus: run `make ingest` first")
    document = read_processed("anz-1h26-investor-presentation", settings.processed_dir)
    if document is None:
        pytest.skip("the ANZ investor pack is not in the processed corpus")

    unranked = [page.page_no for page in document.pages if page.likely_walk_page]
    ranked = [page.page_no for page in rank_walk_pages(document, "nim", "INVESTOR_PRESENTATION")]

    assert 25 in unranked, "the premise: page 25 is a candidate either way"
    assert ranked[0] == 25, f"the known walk must rank first, got {ranked[:3]}"
    assert len(ranked) < len(unranked) / 3, (
        f"the gate must cut the field hard: {len(ranked)} of {len(unranked)}"
    )


def test_a_page_that_does_not_name_the_metric_is_not_a_candidate():
    """Westpac charts its liquidity coverage ratio as a waterfall, with real labelled
    bars. Reading it as a margin walk spends a vision call to produce a reading the
    checksum then has to throw away, and says nothing about where the margin walk is."""
    from bankresearch.ingestion.artifacts import read_processed
    from bankresearch.ingestion.chart_walks import rank_walk_pages
    from bankresearch.settings import get_settings

    settings = get_settings()
    if not settings.db_path.exists():
        pytest.skip("no ingested corpus: run `make ingest` first")
    document = read_processed("wbc-1h26-investor-presentation", settings.processed_dir)
    if document is None:
        pytest.skip("the Westpac pack is not in the processed corpus")

    lcr_page = document.page(86)
    assert lcr_page is not None and "LIQUIDITY COVERAGE" in lcr_page.text.upper()
    assert lcr_page.likely_walk_page, "the premise: it used to be a candidate"

    ranked = [page.page_no for page in rank_walk_pages(document, "nim", "INVESTOR_PRESENTATION")]
    assert 86 not in ranked, "a liquidity waterfall is not a margin walk"


def test_a_wrapped_chart_label_is_still_recognised():
    """ANZ draws "Assets pricing" as "Assets" above "pricing".

    Neither line maps to a driver on its own, so requiring a single capitalised line
    scored the page that is the answer below one that merely mentions margins.
    """
    from bankresearch.ingestion.chart_walks import _label_candidates

    lines = ["Net interest margin (NIM)", "bps", "Assets", "pricing", "Deposits", "pricing"]
    candidates = _label_candidates(lines)
    assert "Assets pricing" in candidates
    assert "Deposits pricing" in candidates


def test_prose_about_drivers_is_not_a_chart_label():
    """CBA's divisional margin page is full of driver words inside sentences."""
    from bankresearch.ingestion.chart_walks import _is_label

    assert _is_label("Capital & replicating portfolio")
    assert not _is_label("Higher deposit margins and earnings on the replicating portfolio,")
    assert not _is_label("unfavourable deposit mix and lower lending margins.")
