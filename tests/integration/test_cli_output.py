"""What `bankresearch run` writes when its output goes to a file rather than a terminal."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from bankresearch.settings import get_settings


def _run(*arguments: str) -> str:
    """The command with a scripted model (the ``scripted_model`` fixture) and the estimate
    accepted, so nothing leaves the machine and nothing is asked."""
    from bankresearch.cli import app

    if not get_settings().db_path.exists():
        pytest.skip("no ingested corpus: run `make ingest` first")
    result = CliRunner().invoke(app, ["run", *arguments, "--yes"])
    assert result.exit_code == 0, result.output
    # The answer alone: notices about the run go to stderr, so a redirect gets just this.
    return result.stdout


def test_a_memo_sent_to_a_file_is_the_markdown_unwrapped(scripted_model):
    """Printed through the terminal's wrapping, the memo's long lines were broken, which
    breaks a Markdown table or list in the file it was redirected to."""
    memo = _run(
        "--bank", "CommBank", "--period", "FY25", "--metric", "cash_earnings", "--format", "md"
    )
    assert memo.startswith("# Commonwealth Bank of Australia")
    assert "| Cash NPAT | A$9,836m | A$10,252m |" in memo
    assert any(len(line) > 200 for line in memo.splitlines())
    # Every figure read from a table is quoted from the cell it came from.
    sources = [line for line in memo.splitlines() if line.startswith("- [ev_") and "table" in line]
    assert sources and all('"' in line for line in sources)


def test_json_output_is_json(scripted_model):
    data = json.loads(
        _run("--bank", "NAB", "--period", "1H26", "--metric", "nim", "--format", "json")
    )
    assert data["status"] == "OK"
