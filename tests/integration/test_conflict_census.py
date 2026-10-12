"""The conflict census over the dev split: no unexplained conflict goes unreviewed."""

from __future__ import annotations

import pytest

from bankresearch.settings import get_settings


def test_no_unexplained_conflict_goes_unreviewed():
    from bankresearch.evals.cases import load_cases
    from bankresearch.evals.census import conflict_census

    settings = get_settings()
    if not settings.db_path.exists():
        pytest.skip("no ingested corpus: run `make ingest` first")
    census = conflict_census(load_cases("dev"), settings)
    assert census.seen, "the corpus raises basis conflicts, so an empty census means it didn't run"
    assert not census.unreviewed(), [
        (item.bank, item.component, item.readings) for item in census.unreviewed()
    ]
    # Every conflict is resolved by a stated rule.
    assert all(item.rule for item in census.seen)


def test_a_review_covers_only_the_conflict_it_names():
    from bankresearch.evals.census import Census, ConflictSeen

    item = ConflictSeen(
        bank="WBC",
        component="headline_earnings",
        classification="UNEXPLAINED",
        readings=(("wbc-a", 9, "3414"), ("wbc-b", 4, "3483")),
        rule="precedence",
    )
    census = Census(seen={item: {"x"}})
    assert census.unreviewed() == [item]
    census.reviewed = [{"bank": "WBC", "component": "nim", "documents": ["wbc-a"]}]
    assert census.unreviewed() == [item]
    census.reviewed = [{"bank": "WBC", "component": "headline_earnings", "documents": ["wbc-a"]}]
    assert census.unreviewed() == []
