"""The harness, and the line it draws around what it measures."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import ClassVar

import pytest
import yaml

from bankresearch.agent.steps.resolve import resolve_request
from bankresearch.domain.models import Basis, Diagnostics, EvidenceCard, SourceRef
from bankresearch.evals.cases import load_cases
from bankresearch.evals.report import write_report
from bankresearch.evals.runner import EvalRun, run_split
from bankresearch.evals.scorers import CaseScore, score_case
from bankresearch.output.schema import Confidence, HeadlineValue, ResearchResult
from bankresearch.settings import Settings

REPO = Path(__file__).parents[2]
PUB = date(2026, 5, 1)


@pytest.fixture
def dev_cases(tmp_path):
    return load_cases("dev", root=REPO / "evals" / "cases")


def a_result(status: str = "OK", reason: str | None = None, cards=()) -> ResearchResult:
    spec = resolve_request("ANZ", "1H26", "nim", as_of=date(2026, 5, 31))
    return ResearchResult(
        run_id=spec.run_id,
        created_at=datetime.now(UTC),
        as_of=spec.as_of,
        request=spec,
        status=status,  # type: ignore[arg-type]
        abstain_reason=reason,  # type: ignore[arg-type]
        current=HeadlineValue(
            value=Decimal("1.53"), unit="pct", label_at_bank="NIM", evidence_ids=["ev_001"]
        )
        if status != "ABSTAIN"
        else None,
        prior=HeadlineValue(
            value=Decimal("1.56"), unit="pct", label_at_bank="NIM", evidence_ids=["ev_002"]
        )
        if status != "ABSTAIN"
        else None,
        overall_confidence=Confidence(level="High", score=0.9) if status != "ABSTAIN" else None,
        evidence=list(cards),
        diagnostics=Diagnostics(latency_ms=42),
    )


def a_card(evidence_id: str, value: str, published: str | None, page: int) -> EvidenceCard:
    return EvidenceCard(
        evidence_id=evidence_id,
        component="nim",
        period="1H26",
        value=Decimal(value),
        published_value=Decimal(published) if published is not None else None,
        unit="pct",
        basis=Basis(),
        statement="read from a table",
        source=SourceRef(
            doc_id="missing-doc", doc_type="RESULTS_DOCUMENT", page=page, publish_date=PUB
        ),
        method="table",
    )


# --- the case set ---------------------------------------------------------


def test_the_dev_split_covers_the_plans_standard_block(dev_cases):
    """The standard block is 4 banks by 6 metrics, and every one is answerable from the
    ingested corpus. The abstention and robustness rows of the same matrix follow."""
    standard = [case for case in dev_cases if "standard" in case.categories]
    assert len(standard) >= 24
    for bank, period in (("ANZ", "1H26"), ("NAB", "1H26"), ("WBC", "1H26"), ("CBA", "FY26")):
        covered = {case.metric for case in standard if case.bank == bank and case.period == period}
        assert covered == {
            "nim",
            "cet1_ratio",
            "roe",
            "cost_to_income",
            "headline_earnings",
            "credit_impairment_charge",
        }, (bank, period, sorted(covered))


def test_every_gold_label_is_verified_and_says_who_checked_it(dev_cases):
    """Gold that nobody owns is gold nothing should be scored against.

    The accuracy figures in the report rest entirely on this file, so a case with figures
    must also carry the verification and the labeller behind them. ``labeller_kind`` is what
    lets the report say whether the measure is independent of the system.
    """
    with_gold = [case for case in dev_cases if case.has_gold]
    assert with_gold, "the dev split carries no gold at all"
    for case in with_gold:
        assert case.gold_verified, f"{case.case_id} carries unverified gold"
        assert case.labeller, f"{case.case_id} does not say who checked it"
        assert case.labeller_kind in ("human", "automated"), case.case_id


def test_gold_verified_without_a_labeller_is_refused():
    """What this check exists to stop: gold that looks checked when nobody checked it."""
    from bankresearch.evals.cases import _case_from

    raw = {"case_id": "x", "bank": "CBA", "period": "FY26", "metric": "nim"}
    with pytest.raises(ValueError, match="labeller"):
        _case_from({**raw, "gold_verified": True}, "dev")
    with pytest.raises(ValueError, match="labeller_kind"):
        _case_from({**raw, "gold_verified": True, "labeller": "Someone"}, "dev")
    # Unverified gold needs no labeller: it is scored against nothing.
    assert _case_from(raw, "dev").gold_verified is False


def test_every_case_pins_its_as_of_date(dev_cases):
    """Otherwise the default is derived from the release month and the result depends on
    when the harness was run."""
    assert all(case.as_of is not None for case in dev_cases)


def test_every_abstention_case_names_its_reason_code(dev_cases):
    for case in dev_cases:
        if case.expects_abstention:
            assert case.expected_abstain_reason, case.case_id


def test_the_case_set_covers_all_six_metrics(dev_cases):
    assert {case.metric for case in dev_cases} >= {
        "nim",
        "cet1_ratio",
        "roe",
        "cost_to_income",
        "headline_earnings",
        "credit_impairment_charge",
    }


def test_the_case_set_covers_the_reachable_abstention_codes(dev_cases):
    """Five of the six codes are reachable from this corpus, and the sixth isn't.

    ``INSUFFICIENT_EVIDENCE`` needs a document that covers the period but leaves out the
    metric. Every document here that covers a period has all six metrics for it, so
    asserting that code would mean writing a case the corpus can't produce, one that would pass for
    the wrong reason.
    """
    reasons = {case.expected_abstain_reason for case in dev_cases if case.expects_abstention}
    assert reasons == {
        "PERIOD_NOT_YET_REPORTED",
        "AMBIGUOUS_INPUT",
        "BANK_UNKNOWN",
        "METRIC_OUT_OF_SCOPE",
        "DOCS_UNAVAILABLE",
    }


def test_case_ids_are_unique(dev_cases):
    ids = [case.case_id for case in dev_cases]
    assert len(ids) == len(set(ids))


def test_the_test_split_is_frozen_and_says_so():
    """The split is only held out if the date it was frozen is recorded.

    Without that, nothing separates a number measured before the code changed from one
    measured after, and that's the whole value of running it once.
    """
    cases = load_cases("test", root=REPO / "evals" / "cases")
    assert cases, "the test split is empty"
    text = (REPO / "evals" / "cases" / "test" / "holdout.yaml").read_text(encoding="utf-8")
    assert re.search(r"FROZEN ON \d{4}-\d{2}-\d{2}", text), "no freeze date recorded"
    for case in cases:
        assert case.as_of is not None, case.case_id
        if case.has_gold:
            assert case.gold_verified and case.labeller, case.case_id


def test_the_test_split_shares_no_case_with_dev(dev_cases):
    """A case in both splits is not held out from anything."""
    test_cases = load_cases("test", root=REPO / "evals" / "cases")
    dev_requests = {(c.bank, c.period, c.metric) for c in dev_cases if c.has_gold}
    overlap = [
        c.case_id for c in test_cases if c.has_gold and (c.bank, c.period, c.metric) in dev_requests
    ]
    assert not overlap, f"scored in both splits: {overlap}"
    assert not {c.case_id for c in test_cases} & {c.case_id for c in dev_cases}


# --- the test split is protected -----------------------------------------


def test_running_the_test_split_without_final_is_refused():
    with pytest.raises(ValueError) as exc:
        run_split("test", cases=[], final=False)
    assert "--final" in str(exc.value)


# --- scoring --------------------------------------------------------------


def test_a_case_without_gold_scores_no_accuracy_measure(dev_cases, tmp_path):
    settings = Settings(
        config_dir=REPO / "config", data_dir=tmp_path, runs_dir=tmp_path, evals_dir=tmp_path
    )
    # An abstention case has nothing to label, so nothing gold-dependent may be scored.
    case = next(c for c in dev_cases if c.case_id == "unknown_bank")
    assert not case.has_gold
    score = score_case(case, a_result("OK"), settings)
    assert score.headline_exact is None
    assert score.driver_precision is None
    assert score.change_error is None


def test_status_is_scored_against_the_assertion(dev_cases, tmp_path):
    settings = Settings(
        config_dir=REPO / "config", data_dir=tmp_path, runs_dir=tmp_path, evals_dir=tmp_path
    )
    case = next(c for c in dev_cases if c.case_id == "anz_1h26_nim")
    assert score_case(case, a_result("PARTIAL"), settings).status_correct
    assert not score_case(case, a_result("ABSTAIN", "DOCS_UNAVAILABLE"), settings).status_correct


def test_an_abstention_with_the_wrong_reason_fails(dev_cases, tmp_path):
    settings = Settings(
        config_dir=REPO / "config", data_dir=tmp_path, runs_dir=tmp_path, evals_dir=tmp_path
    )
    case = next(c for c in dev_cases if c.case_id == "unknown_bank")
    wrong = score_case(case, a_result("ABSTAIN", "DOCS_UNAVAILABLE"), settings)
    assert wrong.abstain_reason_correct is False
    assert not wrong.passed


def test_the_citation_check_compares_against_what_the_page_prints(dev_cases, tmp_path):
    """A bracketed deduction is stored as a magnitude; checking the magnitude against the
    page would fail every one of them."""
    settings = Settings(
        config_dir=REPO / "config", data_dir=tmp_path, runs_dir=tmp_path, evals_dir=tmp_path
    )
    case = next(c for c in dev_cases if c.case_id == "anz_1h26_nim")
    card = a_card("ev_001", value="5534", published="-5534", page=10)
    result = a_result("OK", cards=[card])
    result.current.evidence_ids = ["ev_001"]
    score = score_case(case, result, settings)
    # The document is absent from this temp store, so the figure cannot be found; what
    # matters is that the published value was the thing looked for.
    assert score.cited_numbers == 1
    assert any("-5534" in note for note in score.notes)


def test_a_citation_with_no_page_fails_the_case(dev_cases, tmp_path):
    settings = Settings(
        config_dir=REPO / "config", data_dir=tmp_path, runs_dir=tmp_path, evals_dir=tmp_path
    )
    case = next(c for c in dev_cases if c.case_id == "anz_1h26_nim")
    card = a_card("ev_001", value="1.53", published="1.53", page=0)
    result = a_result("OK", cards=[card])
    result.current.evidence_ids = ["ev_001"]
    score = score_case(case, result, settings)
    assert score.citations_without_a_page == 1
    assert not score.passed


# --- the report -----------------------------------------------------------


def test_the_summary_says_which_measures_were_not_scored(dev_cases, tmp_path):
    run = EvalRun(split="dev", started=datetime.now(UTC))
    run.scores = [
        CaseScore(case_id=case.case_id, status="OK", expected_status=("OK",), status_correct=True)
        for case in dev_cases
    ]
    target = write_report(run, dev_cases, tmp_path)
    summary = (target / "summary.md").read_text(encoding="utf-8")
    assert "No accuracy measure is reported" in summary
    assert "labelled by hand" in summary
    # Counted off the case file, not the scores: a case with gold that this run did not
    # score is a different fact from a case that has no gold to score.
    no_gold = sum(1 for case in dev_cases if not case.has_gold)
    assert f"{no_gold} case(s) carry no gold labels" in summary


def test_the_summary_states_who_checked_the_gold_it_scored(dev_cases, tmp_path):
    """An accuracy figure without its labelling provenance is a claim, not a measure."""
    run = EvalRun(split="dev", started=datetime.now(UTC))
    run.scores = [
        CaseScore(
            case_id=case.case_id,
            status="OK",
            expected_status=("OK",),
            status_correct=True,
            headline_exact=True,
        )
        for case in dev_cases
        if case.has_gold
    ]
    summary = (write_report(run, dev_cases, tmp_path) / "summary.md").read_text(encoding="utf-8")
    assert "Accuracy, against gold" in summary
    assert "Gold provenance" in summary
    labellers = {case.labeller for case in dev_cases if case.gold_verified}
    assert all(labeller in summary for labeller in labellers)
    if any(case.labeller_kind == "automated" for case in dev_cases if case.gold_verified):
        assert "not a person" in summary
        assert "independent holdout" in summary


def test_the_report_writes_a_row_per_case(dev_cases, tmp_path):
    run = EvalRun(split="dev", started=datetime.now(UTC))
    run.scores = [
        CaseScore(case_id=case.case_id, status="OK", expected_status=("OK",), status_correct=True)
        for case in dev_cases
    ]
    target = write_report(run, dev_cases, tmp_path)
    rows = (target / "results.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == len(dev_cases) + 1


def test_a_run_records_its_provenance():
    run = EvalRun(split="dev", started=datetime.now(UTC), git_sha="abc1234", config_hash="deadbeef")
    assert run.git_sha and run.config_hash


def test_an_unmeasurable_measure_reports_as_not_measured(dev_cases, tmp_path):
    run = EvalRun(split="dev", started=datetime.now(UTC))
    run.scores = [
        CaseScore(case_id="x", status="ABSTAIN", expected_status=("ABSTAIN",), status_correct=True)
    ]
    assert run.reconciliation_rate is None
    assert run.citation_faithfulness is None
    summary = (write_report(run, dev_cases, tmp_path) / "summary.md").read_text(encoding="utf-8")
    assert "not measured" in summary


# --- the committed case file ---------------------------------------------


def test_the_case_file_states_where_its_gold_came_from():
    """A reader of the file must be able to see what the labels are worth."""
    text = (REPO / "evals" / "cases" / "dev" / "behaviour.yaml").read_text(encoding="utf-8")
    assert "labeller_kind" in text
    assert "independent holdout" in text
    assert yaml.safe_load(text)["cases"]


# --- a claim must cite the page that carries it ---------------------------


def test_a_claim_citing_a_card_with_another_figure_is_caught():
    """Westpac's 1H26 net profit is $3,414m in the interim report and $3,483m in the
    investor pack's appendix. Both cards match their own page, so the per-card check passes
    either way; what mustn't pass is the claim citing both.
    """
    from bankresearch.evals.scorers import claims_cite_their_own_figure

    result = a_result(
        "OK",
        cards=[
            a_card("ev_001", "1.53", None, 10),  # the current figure
            a_card("ev_002", "1.56", None, 10),  # the prior figure
            a_card("ev_003", "1.89", None, 119),  # another issuer's reading of neither
        ],
    )
    result.current.evidence_ids = ["ev_001", "ev_003"]
    result.prior.evidence_ids = ["ev_002"]

    wrong = claims_cite_their_own_figure(result)
    assert len(wrong) == 1
    assert "ev_003" in wrong[0]

    result.current.evidence_ids = ["ev_001"]
    assert claims_cite_their_own_figure(result) == []


def test_a_case_fails_when_a_claim_cites_another_figure(dev_cases, tmp_path):
    settings = Settings(
        config_dir=REPO / "config", data_dir=tmp_path, runs_dir=tmp_path, evals_dir=tmp_path
    )
    case = next(c for c in dev_cases if c.case_id == "unknown_bank")
    result = a_result(
        "OK", cards=[a_card("ev_001", "1.56", None, 10), a_card("ev_002", "9.99", None, 11)]
    )
    result.current.evidence_ids = ["ev_002"]  # claims 1.53, cites a page printing 9.99
    result.prior.evidence_ids = ["ev_001"]
    score = score_case(case, result, settings)
    assert score.claims_citing_another_figure == 1
    assert not score.passed


# --- materiality cuts both ways -------------------------------------------


def test_an_immaterial_driver_is_neither_credit_nor_blame(dev_cases, tmp_path):
    """ANZ's minority interests moved $1m on a $212m half and Westpac's didn't move.

    The gold declares a $2m materiality threshold, which drops both from the expected set.
    Scoring precision against the unfiltered prediction then penalised a run for reporting a
    line the gold itself had chosen to ignore.
    """
    from bankresearch.evals.cases import EvalCase, GoldDriver
    from bankresearch.output.schema import Driver

    case = EvalCase(
        case_id="x",
        split="dev",
        bank="ANZ",
        period="1H26",
        metric="headline_earnings",
        drivers=(
            GoldDriver(driver_id="earn.nii", contribution=Decimal("19")),
            GoldDriver(driver_id="earn.minority_interests", contribution=Decimal("1")),
        ),
        materiality=Decimal("2"),
        gold_verified=True,
        labeller="test",
        labeller_kind="human",
    )
    settings = Settings(
        config_dir=REPO / "config", data_dir=tmp_path, runs_dir=tmp_path, evals_dir=tmp_path
    )
    result = a_result("OK")
    result.drivers = [
        Driver(
            driver_id="earn.nii",
            label="NII",
            contribution=Decimal("19"),
            direction="+",
            method="computed",
            evidence_type="computed",
            confidence=Confidence(level="High", score=0.9),
        ),
        Driver(
            driver_id="earn.minority_interests",
            label="NCI",
            contribution=Decimal("1"),
            direction="+",
            method="computed",
            evidence_type="computed",
            confidence=Confidence(level="High", score=0.9),
        ),
    ]
    score = score_case(case, result, settings)
    assert score.driver_precision == 1.0
    assert score.driver_recall == 1.0


def _driver(driver_id, contribution, score=0.9, level="High"):
    from bankresearch.output.schema import Driver

    return Driver(
        driver_id=driver_id,
        label=driver_id,
        contribution=Decimal(contribution),
        direction="+" if Decimal(contribution) >= 0 else "-",
        method="computed",
        evidence_type="computed",
        confidence=Confidence(level=level, score=score),
    )


def _gold_case(*drivers):
    from bankresearch.evals.cases import EvalCase, GoldDriver

    return EvalCase(
        case_id="x",
        split="dev",
        bank="CBA",
        period="FY26",
        metric="credit_impairment_charge",
        drivers=tuple(
            GoldDriver(driver_id=d, contribution=Decimal(c), tolerance=Decimal("1"))
            for d, c in drivers
        ),
        materiality=Decimal("2"),
        gold_verified=True,
        labeller="test",
        labeller_kind="human",
    )


def test_a_driver_named_right_but_sized_wrong_is_caught(tmp_path):
    """Precision and recall only ask whether the right drivers were named. A run that
    names both and swaps their sizes scores perfectly on both, and is wrong."""
    settings = Settings(
        config_dir=REPO / "config", data_dir=tmp_path, runs_dir=tmp_path, evals_dir=tmp_path
    )
    result = a_result("OK")
    result.drivers = [_driver("cic.collective", "-88"), _driver("cic.individual", "150")]
    score = score_case(
        _gold_case(("cic.collective", "150"), ("cic.individual", "-88")), result, settings
    )
    assert score.driver_precision == 1.0 and score.driver_recall == 1.0
    assert (score.driver_values_right, score.driver_values_checked) == (0, 2)
    assert any("cic.collective sized -88" in note for note in score.notes)


def test_each_named_driver_is_one_calibration_claim(tmp_path):
    settings = Settings(
        config_dir=REPO / "config", data_dir=tmp_path, runs_dir=tmp_path, evals_dir=tmp_path
    )
    result = a_result("OK")
    result.drivers = [
        _driver("cic.collective", "150.4", score=0.92),
        _driver("cic.individual", "-88", score=0.75, level="Medium"),
        _driver("cic.overlays", "12", score=0.75, level="Medium"),
    ]
    score = score_case(
        _gold_case(("cic.collective", "150"), ("cic.individual", "-88")), result, settings
    )
    assert (score.driver_values_right, score.driver_values_checked) == (2, 2)
    # A driver the gold doesn't have is a wrong claim, not an unscored one.
    assert score.driver_claims == [
        ("cic.collective", 0.92, "High", True),
        ("cic.individual", 0.75, "Medium", True),
        ("cic.overlays", 0.75, "Medium", False),
    ]


# --- the abstention must not understate what is held --------------------


def test_a_missing_period_document_says_what_is_actually_held():
    """It said "Found: nothing" for ANZ 1H25 while 24 figures for the period were stored.

    They're the comparison column of the 1H26 pack. The refusal is right, because a
    comparative is the restated value, so answering from one would describe a period using
    numbers published after it closed. But the message blamed the period for being missing
    when it was there, and made the system look blind to its own data.
    """
    from datetime import date as _date

    from bankresearch.agent.steps.availability import check_availability
    from bankresearch.agent.steps.resolve import resolve_request
    from bankresearch.memory.db import connect
    from bankresearch.memory.fact_store import FactStore
    from bankresearch.settings import get_settings

    settings = get_settings()
    if not settings.db_path.exists():
        pytest.skip("no ingested corpus: run `make ingest` first")

    spec = resolve_request("ANZ", "1H25", "nim", as_of=_date(2026, 5, 31))
    with connect(settings.db_path) as connection:
        availability = check_availability(spec, FactStore(connection))

    assert availability.reportable is False
    assert availability.reason == "DOCS_UNAVAILABLE"
    detail = availability.detail
    assert "Found: nothing" not in detail
    assert "comparative" in detail, "it must say where the figures it holds came from"
    assert "restated" in detail, "and why a comparative is not a substitute"
    assert "1H25" in detail


# --- model modes ---------------------------------------------------------------------
#
# `--replay` used to set the label in the summary and nothing else: no recorded response was
# ever replayed, and a key left in the shell let cached answers into one run and not the
# next. The mode is now chosen, and each one is checked here by what the agent is given.


class _CapturingAgent:
    """Stands in for the agent and records how the runner built it."""

    built: ClassVar[dict[str, object]] = {}

    def __init__(self, store, settings, client=None, consent=None, persist=True):  # type: ignore[no-untyped-def]
        _CapturingAgent.built = {"settings": settings, "client": client, "consent": consent}

    def run(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("no case should run in these tests")


def _one_case(monkeypatch):  # type: ignore[no-untyped-def]
    """A single case run by the capturing agent, which is built per case."""
    from bankresearch.evals import runner
    from bankresearch.evals.cases import load_cases

    monkeypatch.setattr(_CapturingAgent, "run", lambda self, *a, **k: None)
    monkeypatch.setattr(runner, "score_case", lambda *a, **k: None)
    return [load_cases("dev")[0]]


def _settings_with_a_key(tmp_path) -> Settings:  # type: ignore[no-untyped-def]
    return Settings(
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        evals_dir=tmp_path / "evals",
        **{"ANTHROPIC_API_KEY": "sk-ant-test-placeholder-not-a-key"},
    )


def test_an_eval_with_the_model_off_never_sees_a_key_left_in_the_shell(tmp_path, monkeypatch):
    from bankresearch.evals import runner

    monkeypatch.setattr(runner, "Agent", _CapturingAgent)
    run = runner.run_split(
        "dev", settings=_settings_with_a_key(tmp_path), cases=_one_case(monkeypatch)
    )
    built = _CapturingAgent.built
    assert run.model_mode == "off"
    assert built["client"] is None and built["consent"] is None
    assert not built["settings"].has_api_key  # type: ignore[attr-defined]


def test_a_replay_answers_from_recordings_only_and_never_goes_live(tmp_path, monkeypatch):
    from bankresearch.evals import runner
    from bankresearch.llm.cache import CachedClient

    monkeypatch.setattr(runner, "Agent", _CapturingAgent)
    run = runner.run_split(
        "dev",
        settings=_settings_with_a_key(tmp_path),
        cases=_one_case(monkeypatch),
        model="replay",
    )
    client = _CapturingAgent.built["client"]
    assert run.model_mode == "replay"
    assert isinstance(client, CachedClient)
    assert client.replay_only and client.upstream is None
    assert not _CapturingAgent.built["settings"].has_api_key  # type: ignore[attr-defined]


def test_a_live_eval_needs_a_key_and_a_grant(tmp_path, monkeypatch):
    from bankresearch.evals import runner
    from bankresearch.llm.consent import BudgetedGrant

    monkeypatch.setattr(runner, "Agent", _CapturingAgent)
    keyless = Settings(data_dir=tmp_path / "data", runs_dir=tmp_path / "runs")
    grant = BudgetedGrant(calls=2, ceiling_usd=Decimal("0.10"))
    with pytest.raises(ValueError, match="needs ANTHROPIC_API_KEY"):
        runner.run_split("dev", settings=keyless, cases=[], model="live", consent=grant)
    with pytest.raises(ValueError, match="needs a grant"):
        runner.run_split("dev", settings=_settings_with_a_key(tmp_path), cases=[], model="live")

    # Agents are built per case now, so one case is needed to see how it was built.
    run = runner.run_split(
        "dev",
        settings=_settings_with_a_key(tmp_path),
        cases=_one_case(monkeypatch),
        model="live",
        consent=grant,
    )
    assert run.model_mode == "live"
    assert _CapturingAgent.built["consent"] is grant
    assert _CapturingAgent.built["settings"].has_api_key  # type: ignore[attr-defined]


def test_an_unknown_model_mode_is_refused(tmp_path):
    with pytest.raises(ValueError, match="model mode"):
        run_split("dev", settings=_settings_with_a_key(tmp_path), cases=[], model="sometimes")


def test_the_suites_never_see_a_key(tmp_path, monkeypatch):
    """They measure the deterministic path; a key in the shell mustn't change them."""
    from bankresearch.evals import runner

    monkeypatch.setattr(runner, "Agent", _CapturingAgent)
    settings = _settings_with_a_key(tmp_path)
    assert runner.run_stability([], settings=settings) == []
    assert runner.without_a_key(settings).has_api_key is False
    assert settings.has_api_key  # the caller's settings are left alone


def test_replay_and_grant_together_are_refused_by_the_command():
    from typer.testing import CliRunner

    from bankresearch.cli import app

    outcome = CliRunner().invoke(app, ["eval", "--replay", "--grant", "2:0.10"])
    assert outcome.exit_code == 1
    assert "contradict" in outcome.stdout


def test_a_cold_path_case_expects_an_answer_only_when_a_model_is_used(tmp_path):
    """Out-of-corpus cases are "OK if discovery works, else ABSTAIN
    DOCS_UNAVAILABLE". Without a model there is no discovery, so abstaining is right;
    with one, abstaining means the cold path failed."""
    from bankresearch.evals.cases import _case_from

    case = _case_from(
        {
            "case_id": "ben_cold",
            "bank": "BEN",
            "period": "1H26",
            "metric": "nim",
            "expected_status": ["ABSTAIN"],
            "expected_abstain_reason": "DOCS_UNAVAILABLE",
            "with_model": {"expected_status": ["OK", "PARTIAL"]},
        },
        "dev",
    )
    settings = Settings(data_dir=tmp_path / "data", runs_dir=tmp_path / "runs")
    abstained = a_result("ABSTAIN", "DOCS_UNAVAILABLE")
    assert score_case(case, abstained, settings).status_correct
    assert score_case(case, abstained, settings).abstain_reason_correct
    assert not score_case(case, abstained, settings, "live").status_correct
    assert score_case(case, a_result("OK"), settings, "live").status_correct
    assert not score_case(case, a_result("OK"), settings).status_correct
