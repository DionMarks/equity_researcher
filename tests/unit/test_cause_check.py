"""Scoring the guards on driver causes against labelled claims."""

from __future__ import annotations

from pathlib import Path

import pytest

from bankresearch.evals.cause_check import load_claims, score_claims, todays_guard

CLAIMS = """\
claims:
  - {id: a, bank: X, metric: nim, driver: d1, claim: "Competition tightened spreads [ev_001].",
     outcome: shown, label: supported, passages: []}
  - {id: b, bank: X, metric: nim, driver: d2, claim: "The bank gives no further reason [ev_002].",
     outcome: shown, label: unsupported, passages: []}
  - {id: c, bank: X, metric: nim, driver: d3, claim: "A stretched cause [ev_003].",
     outcome: dropped, label: partial, passages: []}
  - {id: d, bank: X, metric: nim, driver: d4, claim: "A faithful cause [ev_004].",
     outcome: dropped, label: supported, passages: []}
"""


def test_the_runs_outcomes_are_scored_against_the_labels(tmp_path):
    path = tmp_path / "claims.yaml"
    path.write_text(CLAIMS, encoding="utf-8")
    score = score_claims(load_claims(path))
    assert (score.shown_supported, score.shown) == (1, 2)
    assert (score.flagged_removed, score.flagged) == (1, 2)
    assert (score.supported_removed, score.supported) == (1, 2)


def test_todays_guards_are_applied_on_top_of_what_the_run_did(tmp_path):
    path = tmp_path / "claims.yaml"
    path.write_text(CLAIMS, encoding="utf-8")
    claims = load_claims(path)
    assert todays_guard(claims[1]) == "claims the bank is silent"
    score = score_claims(claims, today=True)
    assert (score.shown_supported, score.shown) == (1, 1)
    assert (score.flagged_removed, score.flagged) == (2, 2)


def test_an_unknown_label_is_refused(tmp_path):
    path = tmp_path / "claims.yaml"
    path.write_text(CLAIMS.replace("label: partial", "label: maybe"), encoding="utf-8")
    with pytest.raises(ValueError, match="label"):
        load_claims(path)


def test_the_live_runs_dataset_loads_and_every_claim_is_labelled():
    path = Path(__file__).parents[2] / "evals" / "checker" / "claims.yaml"
    claims = load_claims(path)
    assert len(claims) == 78
    assert {claim.outcome for claim in claims} == {"shown", "dropped"}
    assert all(claim.note for claim in claims if claim.label != "supported")


def test_rejudging_sends_each_claim_with_its_own_passages_and_scores_the_verdicts():
    """Claims from different runs share a call, and ``ev_066`` is a different sentence in
    each run, so each claim's ids are made its own before they're sent."""
    from bankresearch.evals.cause_check import CauseClaim, Passage, rejudge
    from bankresearch.llm.client import LLMResponse, Usage

    claims = [
        CauseClaim(
            claim_id=f"c{n}",
            bank="X",
            metric="nim",
            driver=f"d{n}",
            claim=f"Deposit competition narrowed the margin [ev_066].{' ' * n}",
            outcome="shown",
            label="supported" if n % 2 else "partial",
            driver_label=f"Driver {n}",
            passages=(Passage("ev_066", "doc", n + 1, f"Sentence {n} of the bank's own."),),
            period="1H26",
            contribution=f"+A${n + 10}m",
            method="computed",
        )
        for n in range(5)
    ]

    class Judges:
        available = True

        def __init__(self) -> None:
            self.requests: list[str] = []

        def complete(self, request):  # type: ignore[no-untyped-def]
            content = request.messages[0]["content"]
            self.requests.append(content)
            claims_sent = [line for line in content.split("\n") if line[:1].isdigit()]
            return LLMResponse(
                structured={
                    "verdicts": [
                        {"claim": n, "verdict": "supported" if n % 2 else "partial", "reason": "r"}
                        for n in range(1, len(claims_sent) + 1)
                    ]
                },
                usage=Usage(input_tokens=10, output_tokens=5),
            )

    client = Judges()
    verdicts, unjudged = rejudge(claims, client, model="m", effort="medium", batch=2)
    assert unjudged == []
    assert len(client.requests) == 3
    assert "[c0_ev_066]" in client.requests[0] and "[c1_ev_066]" in client.requests[0]
    assert "Sentence 1 of the bank's own." in client.requests[0]
    # The driver's size goes with it, as in a run, named so two runs' drivers can't mix.
    assert "Driver 1 (X 1H26 nim): +A$11m, computed exactly" in client.requests[0]
    assert set(verdicts) == {"c0", "c1", "c2", "c3", "c4"}

    # The scripted checker passes the first claim of each batch and not the second, which
    # is the opposite of every label here: c0, c2 and c4 are shown though labelled partial,
    # and c1 and c3 are taken out though labelled supported.
    assert {key: verdict for key, (verdict, _) in verdicts.items()} == {
        "c0": "supported",
        "c1": "partial",
        "c2": "supported",
        "c3": "partial",
        "c4": "supported",
    }
    live = score_claims(claims, verdicts=verdicts)
    assert (live.shown, live.shown_supported, live.supported_removed) == (3, 0, 2)


def test_a_grant_that_runs_out_leaves_the_rest_unjudged_and_reports_the_rest():
    from bankresearch.evals.cause_check import CauseClaim, Passage, rejudge
    from bankresearch.llm.client import LLMResponse, Usage
    from bankresearch.llm.consent import ConsentDenied

    claims = [
        CauseClaim(
            claim_id=f"c{n}",
            bank="X",
            metric="nim",
            driver=f"d{n}",
            claim="A cause [ev_001].",
            outcome="shown",
            label="supported",
            passages=(Passage("ev_001", "doc", 1, "The bank's sentence."),),
        )
        for n in range(4)
    ]

    class RunsOut:
        available = True
        calls = 0

        def complete(self, request):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls > 1:
                raise ConsentDenied("the grant of 1 call(s) is used up")
            return LLMResponse(
                structured={
                    "verdicts": [
                        {"claim": 1, "verdict": "supported", "reason": "r"},
                        {"claim": 2, "verdict": "supported", "reason": "r"},
                    ]
                },
                usage=Usage(input_tokens=10, output_tokens=5),
            )

    verdicts, unjudged = rejudge(claims, RunsOut(), model="m", effort="medium", batch=2)
    assert set(verdicts) == {"c0", "c1"}
    assert unjudged == ["claims c2 to c3: the grant of 1 call(s) is used up"]
