"""What memory holds, and the rules that stop it holding more.

Each rule has a test of its own, and these are those tests.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal

import pytest

from bankresearch.agent.steps.resolve import resolve_request
from bankresearch.domain.models import Basis, DocMeta, Fact, HeadlineLabel, Restatement
from bankresearch.memory.asof import knowable_at
from bankresearch.memory.db import connect
from bankresearch.memory.fact_store import FactStore

AS_OF = date(2026, 5, 1)


@pytest.fixture
def db():
    connection = connect(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def store(db):
    return FactStore(db)


@pytest.fixture
def spec():
    return resolve_request("ANZ", "1H26", "cet1_ratio", as_of=AS_OF)


def doc(
    doc_id: str, bank_id: str = "ANZ", period: str = "1H26", published: date = AS_OF
) -> DocMeta:
    return DocMeta(
        doc_id=doc_id,
        bank_id=bank_id,
        period=period,
        doc_type="RESULTS_DOCUMENT",
        publish_date=published,
        url=f"https://example.invalid/{doc_id}",
        page_count=200,
    )


def fact(
    component: str,
    value: str,
    *,
    bank_id: str = "ANZ",
    period: str = "1H26",
    doc_id: str = "d1",
    published: date = AS_OF,
    validated: bool = True,
    restated: bool = False,
) -> Fact:
    return Fact(
        fact_id="",
        bank_id=bank_id,
        period=period,
        component=component,
        value=Decimal(value),
        unit="pct",
        basis=Basis(scope="level2"),
        doc_id=doc_id,
        doc_type="RESULTS_DOCUMENT",
        page_no=7,
        publish_date=published,
        validated=validated,
        is_restated=restated,
    )


# --- the schema itself ----------------------------------------------------


def test_no_table_has_a_narrative_or_conclusion_column(db):
    """There must be nowhere to persist a conclusion."""
    forbidden = re.compile(
        r"narrative|explanation|conclusion|summary|rationale|reasoning|confidence|verdict",
        re.IGNORECASE,
    )
    offenders: list[str] = []
    for table in db.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        for column in db.execute(f"PRAGMA table_info({table['name']})"):
            if forbidden.search(column["name"]):
                offenders.append(f"{table['name']}.{column['name']}")
    assert offenders == []


def test_values_are_stored_as_text_so_decimals_survive(db, store):
    store.put_document(doc("d1"))
    store.put_fact(fact("cet1.headline", "12.39"))
    row = db.execute("SELECT value, typeof(value) AS t FROM facts").fetchone()
    assert row["t"] == "text"
    assert row["value"] == "12.39"


def test_a_reported_value_round_trips_exactly(store, spec):
    store.put_document(doc("d1"))
    store.put_fact(fact("cet1.headline", "12.39"))
    assert store.get_facts(spec)[0].value == Decimal("12.39")


def test_trailing_precision_is_preserved(store, spec):
    store.put_document(doc("d1"))
    store.put_fact(fact("cet1.headline", "12.40"))
    assert str(store.get_facts(spec)[0].value) == "12.40"


# --- bank scoping --------------------------------------------------------


def test_another_banks_facts_are_unreachable(store, spec):
    """The distractor case: the corpus holds both, the run may see only one."""
    store.put_document(doc("anz", bank_id="ANZ"))
    store.put_document(doc("nab", bank_id="NAB"))
    store.put_fact(fact("cet1.headline", "12.39", bank_id="ANZ", doc_id="anz"))
    store.put_fact(fact("cet1.headline", "11.80", bank_id="NAB", doc_id="nab"))

    facts = store.get_facts(spec)
    assert [f.bank_id for f in facts] == ["ANZ"]
    assert facts[0].value == Decimal("12.39")


def test_documents_are_scoped_by_bank_too(store):
    store.put_document(doc("anz", bank_id="ANZ"))
    store.put_document(doc("nab", bank_id="NAB"))
    found = store.documents_for("ANZ", ["1H26"], AS_OF)
    assert [d.doc_id for d in found] == ["anz"]


# --- period scoping ------------------------------------------------------


def test_only_the_two_periods_under_comparison_are_readable(store, spec):
    store.put_document(doc("d1", period="1H26"))
    store.put_document(doc("d2", period="1H25", published=date(2025, 5, 1)))
    store.put_document(doc("d3", period="2H25", published=date(2025, 11, 1)))
    store.put_fact(fact("cet1.headline", "12.39", period="1H26", doc_id="d1"))
    store.put_fact(
        fact("cet1.headline", "12.21", period="1H25", doc_id="d2", published=date(2025, 5, 1))
    )
    store.put_fact(
        fact("cet1.headline", "12.30", period="2H25", doc_id="d3", published=date(2025, 11, 1))
    )

    periods = {f.period for f in store.get_facts(spec)}
    assert periods == {"1H26", "1H25"}


# --- the as-of filter ----------------------------------------------------


def test_a_later_vintage_is_invisible_under_the_default_as_of(store, spec):
    """Look-ahead leakage: a restatement published after as-of must not be read."""
    store.put_document(doc("d1"))
    store.put_document(doc("d_later", period="1H26", published=date(2026, 11, 10)))
    store.put_fact(fact("cet1.headline", "12.39", doc_id="d1"))
    store.put_fact(
        fact(
            "cet1.headline", "12.31", doc_id="d_later", published=date(2026, 11, 10), restated=True
        )
    )

    facts = store.get_facts(spec)
    assert [f.value for f in facts] == [Decimal("12.39")]


def test_opting_in_to_later_vintages_shows_both(store):
    store.put_document(doc("d1"))
    store.put_document(doc("d_later", period="1H26", published=date(2026, 11, 10)))
    store.put_fact(fact("cet1.headline", "12.39", doc_id="d1"))
    store.put_fact(
        fact(
            "cet1.headline", "12.31", doc_id="d_later", published=date(2026, 11, 10), restated=True
        )
    )
    later = resolve_request("ANZ", "1H26", "cet1_ratio", as_of=date(2026, 12, 31))
    values = {f.value for f in store.get_facts(later)}
    assert values == {Decimal("12.39"), Decimal("12.31")}


def test_the_latest_knowable_vintage_is_returned_first(store):
    store.put_document(doc("d1", published=date(2025, 5, 1)))
    store.put_document(doc("d2", published=date(2026, 5, 1)))
    store.put_fact(fact("cet1.headline", "12.30", doc_id="d1", published=date(2025, 5, 1)))
    store.put_fact(
        fact("cet1.headline", "12.39", doc_id="d2", published=date(2026, 5, 1), restated=True)
    )
    spec = resolve_request("ANZ", "1H26", "cet1_ratio", as_of=date(2026, 5, 1))
    assert store.get_facts(spec)[0].value == Decimal("12.39")


def test_documents_published_after_as_of_are_not_listed(store):
    store.put_document(doc("d1"))
    store.put_document(doc("d_later", published=date(2026, 11, 10)))
    assert [d.doc_id for d in store.documents_for("ANZ", ["1H26"], AS_OF)] == ["d1"]


def test_knowable_at_is_inclusive_of_the_as_of_date():
    assert knowable_at(AS_OF, AS_OF)
    assert not knowable_at(date(2026, 5, 2), AS_OF)


# --- quarantine ----------------------------------------------------------


def test_an_unvalidated_fact_cannot_be_written_to_the_fact_table(store):
    store.put_document(doc("d1"))
    with pytest.raises(ValueError) as exc:
        store.put_fact(fact("cet1.headline", "12.39", validated=False))
    assert "quarantine" in str(exc.value)


def test_a_quarantined_fact_is_unreachable_from_get_facts(store, spec):
    store.put_document(doc("d1"))
    store.quarantine(fact("cet1.headline", "99.99", validated=False), "CET1 identity off by 480bp")
    assert store.get_facts(spec) == []
    assert store.quarantined_count() == 1


def test_the_fact_store_exposes_no_method_to_read_quarantine():
    readers = [
        name
        for name in dir(FactStore)
        if "quarantine" in name and not name.startswith("_") and name != "quarantine"
    ]
    assert readers == ["quarantined_count"]


# --- restatements and labels ---------------------------------------------


def test_a_restatement_adds_a_row_and_never_overwrites(store, db):
    store.put_document(doc("d_old", period="1H25", published=date(2025, 5, 1)))
    store.put_document(doc("d_new", period="1H26"))
    original = store.put_fact(
        fact("earn.headline", "5016", period="1H25", doc_id="d_old", published=date(2025, 5, 1))
    )
    restated = store.put_fact(
        fact("earn.headline", "4990", period="1H25", doc_id="d_new", restated=True)
    )
    store.put_restatement(
        Restatement(
            original_fact_id=original,
            restated_fact_id=restated,
            footnoted=True,
            note="Restated for the sale of the insurance business",
        )
    )
    assert db.execute("SELECT COUNT(*) AS n FROM facts").fetchone()["n"] == 2
    found = store.restatements_for("ANZ", "1H25")
    assert len(found) == 1
    assert found[0].footnoted is True


def test_a_headline_label_is_recorded_per_bank_period_and_metric(store):
    store.put_document(doc("d1"))
    store.put_headline_label(
        HeadlineLabel(
            bank_id="ANZ",
            period="1H26",
            metric_id="headline_earnings",
            label="Cash profit",
            doc_id="d1",
            page_no=2,
        )
    )
    found = store.headline_label("ANZ", "1H26", "headline_earnings")
    assert found is not None
    assert found.label == "Cash profit"
    assert store.headline_label("NAB", "1H26", "headline_earnings") is None
