"""Recording that a person checked a case's gold, without disturbing the file around it."""

from __future__ import annotations

from datetime import date

import pytest

from bankresearch.evals.cases import load_cases
from bankresearch.evals.gold_check import record_verification

CASES = """\
cases:
  # --- a section -----------------------------------------------------------
  - case_id: one
    bank: ANZ
    period: FY25
    metric: nim
    expected_status: [OK]
    notes: >-
      The reasoning behind the label, which must survive.
    gold_verified: true
    labeller: "automated"
    labeller_kind: automated
    gold:
      current: {value: "1.55", unit: pct, tolerance: "0.005",
                source: {doc_id: d, page: 12}}
      prior: {value: "1.57", unit: pct, tolerance: "0.005",
                source: {doc_id: d, page: 12}}

  - case_id: two
    bank: ANZ
    period: FY25
    metric: roe
    expected_status: [OK]
    gold_verified: true
    labeller: "automated"
    labeller_kind: automated
    gold:
      current: {value: "8.1", unit: pct, tolerance: "0.005",
                source: {doc_id: d, page: 12}}
"""


def test_a_verification_records_the_person_and_the_date_and_nothing_else(tmp_path):
    (tmp_path / "dev").mkdir()
    path = tmp_path / "dev" / "cases.yaml"
    path.write_text(CASES, encoding="utf-8")
    record_verification(tmp_path / "dev", "one", "Jo Analyst", date(2026, 10, 11))
    text = path.read_text(encoding="utf-8")
    assert "The reasoning behind the label, which must survive." in text
    assert "# --- a section" in text
    cases = {case.case_id: case for case in load_cases("dev", root=tmp_path)}
    assert (cases["one"].labeller, cases["one"].labeller_kind) == ("Jo Analyst", "human")
    assert cases["one"].independently_labelled
    assert (cases["two"].labeller, cases["two"].labeller_kind) == ("automated", "automated")
    assert "    verified_on: 2026-10-11" in text
    # Checking it again replaces the date rather than adding a second one.
    record_verification(tmp_path / "dev", "one", "Jo Analyst", date(2026, 10, 12))
    assert path.read_text(encoding="utf-8").count("verified_on:") == 1


def test_a_verification_needs_a_name_and_a_case(tmp_path):
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev" / "cases.yaml").write_text(CASES, encoding="utf-8")
    with pytest.raises(ValueError, match="name"):
        record_verification(tmp_path / "dev", "one", "  ", date(2026, 10, 11))
    with pytest.raises(ValueError, match="no case"):
        record_verification(tmp_path / "dev", "three", "Jo", date(2026, 10, 11))


def test_the_command_records_nothing_when_a_figure_is_rejected(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from bankresearch.cli import app
    from bankresearch.settings import get_settings

    settings = get_settings()
    if not settings.db_path.exists():
        pytest.skip("no ingested corpus: run `make ingest` first")
    before = (settings.evals_dir / "cases" / "dev" / "behaviour.yaml").read_text(encoding="utf-8")
    result = CliRunner().invoke(
        app, ["gold-check", "anz_fy25_nim", "--verifier", "Jo Analyst"], input="y\nn\n"
    )
    assert result.exit_code != 0
    assert "not confirmed" in result.output
    assert ">>1.55<<" in result.output
    assert (settings.evals_dir / "cases" / "dev" / "behaviour.yaml").read_text(
        encoding="utf-8"
    ) == before


# --- the quick way through: a review sheet, then one question ---------------------------


@pytest.fixture
def scratch_cases(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    """The two cases above as the whole answer key, so nothing real is written."""
    (tmp_path / "cases" / "dev").mkdir(parents=True)
    path = tmp_path / "cases" / "dev" / "cases.yaml"
    path.write_text(CASES, encoding="utf-8")
    monkeypatch.setenv("BANKRESEARCH_EVALS_DIR", str(tmp_path))
    return path


def _gold_check(*arguments: str, answers: str = ""):  # type: ignore[no-untyped-def]
    from typer.testing import CliRunner

    from bankresearch.cli import app

    return CliRunner().invoke(app, ["gold-check", *arguments], input=answers)


def test_the_sheet_lists_every_unchecked_case_and_records_nothing(scratch_cases, tmp_path):
    before = scratch_cases.read_text(encoding="utf-8")
    sheet = tmp_path / "review.html"
    result = _gold_check("--sheet", str(sheet))
    assert result.exit_code == 0, result.output
    html = sheet.read_text(encoding="utf-8")
    assert "value='one'" in html and "value='two'" in html
    assert "gold-check --split all --confirm" in html
    # The cited document isn't on disk here, and the sheet says so rather than failing.
    assert "isn't on disk" in html
    assert scratch_cases.read_text(encoding="utf-8") == before


def test_confirming_the_ticked_cases_asks_once_and_a_no_records_nothing(scratch_cases):
    before = scratch_cases.read_text(encoding="utf-8")
    refused = _gold_check("--confirm", "one,two", "--verifier", "Jo Analyst", answers="n\n")
    assert refused.exit_code != 0
    assert "Nothing recorded" in refused.output
    assert scratch_cases.read_text(encoding="utf-8") == before

    agreed = _gold_check("--confirm", "one,two", "--verifier", "Jo Analyst", answers="y\n")
    assert agreed.exit_code == 0, agreed.output
    cases = {case.case_id: case for case in load_cases("dev", root=scratch_cases.parents[1])}
    assert {case.labeller_kind for case in cases.values()} == {"human"}
    assert {case.labeller for case in cases.values()} == {"Jo Analyst"}


def test_the_sheets_placeholder_name_and_an_unknown_case_are_refused(scratch_cases):
    before = scratch_cases.read_text(encoding="utf-8")
    assert _gold_check("--confirm", "one", "--verifier", "YOUR NAME", answers="y\n").exit_code
    assert _gold_check("--confirm", "one,three", "--verifier", "Jo", answers="y\n").exit_code
    assert scratch_cases.read_text(encoding="utf-8") == before


def test_going_through_all_of_them_records_only_the_ones_said_yes_to(scratch_cases):
    result = _gold_check("--all", "--verifier", "Jo Analyst", answers="y\nn\n")
    assert result.exit_code == 0, result.output
    assert "Disputed, nothing recorded: two" in result.output
    cases = {case.case_id: case for case in load_cases("dev", root=scratch_cases.parents[1])}
    assert cases["one"].labeller_kind == "human"
    assert cases["two"].labeller_kind == "automated"
