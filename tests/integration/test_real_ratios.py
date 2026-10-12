"""The extracted figures, checked against the ratios the issuers themselves published."""

from __future__ import annotations

from decimal import Decimal

import pytest

from bankresearch.memory.db import connect
from bankresearch.settings import get_settings

# (bank, metric, prior period, prior value, current period, current value, source page)
#
# Read from each issuer's own results document during this build. The page is the one the
# figure was read from, so a disagreement can be settled by opening it.
PUBLISHED: tuple[tuple[str, str, str, str, str, str, str], ...] = (
    # ANZ 1H26 results document, Financial information and the cash profit results.
    ("ANZ", "nim", "1H25", "1.56", "1H26", "1.53", "p.10"),
    ("ANZ", "cet1_ratio", "1H25", "11.8", "1H26", "12.4", "p.11"),
    ("ANZ", "roe", "1H25", "10.2", "1H26", "10.6", "p.10"),
    ("ANZ", "cost_to_income", "1H25", "52.2", "1H26", "49.4", "p.10"),
    ("ANZ", "headline_earnings", "1H25", "3568", "1H26", "3780", "p.10"),
    ("ANZ", "credit_impairment_charge", "1H25", "145", "1H26", "274", "p.10"),
    # NAB 1H26 results document. ROE and cost to income are the cash readings, which is
    # NAB's headline presentation; both moved on a $949m large notable item.
    ("NAB", "nim", "1H25", "1.70", "1H26", "1.81", "p.22"),
    ("NAB", "cet1_ratio", "1H25", "12.01", "1H26", "11.65", "p.22"),
    ("NAB", "roe", "1H25", "11.7", "1H26", "8.5", "p.12"),
    ("NAB", "cost_to_income", "1H25", "46.8", "1H26", "58.6", "p.22"),
    ("NAB", "headline_earnings", "1H25", "3583", "1H26", "2639", "p.101"),
    ("NAB", "credit_impairment_charge", "1H25", "348", "1H26", "706", "p.6"),
    # Westpac 1H26 results document. The margin and the expense ratio are published
    # excluding Notable Items, which is how Westpac leads; net profit is statutory.
    ("WBC", "nim", "1H25", "1.92", "1H26", "1.89", "p.11"),
    ("WBC", "cet1_ratio", "1H25", "12.24", "1H26", "12.42", "p.29"),
    ("WBC", "roe", "1H25", "9.42", "1H26", "9.58", "p.11"),
    ("WBC", "cost_to_income", "1H25", "51.83", "1H26", "51.67", "p.11"),
    ("WBC", "headline_earnings", "1H25", "3317", "1H26", "3414", "p.9"),
    # Charges, not write-backs: page 21 heads them "Total impairment (charges)/benefits".
    ("WBC", "credit_impairment_charge", "1H25", "250", "1H26", "443", "p.21"),
    # CBA FY26 results document. The group margin, four pages from Retail Banking's 2.50%.
    ("CBA", "nim", "FY25", "2.08", "FY26", "2.05", "p.18"),
    ("CBA", "cet1_ratio", "FY25", "12.3", "FY26", "12.0", "p.106"),
    ("CBA", "roe", "FY25", "13.5", "FY26", "14.0", "p.19"),
    ("CBA", "cost_to_income", "FY25", "45.7", "FY26", "45.5", "p.18"),
    # Page 16 prints the prior year twice: $10,253m as "Net profit after tax", and $10,252m
    # as "Net profit after tax from continuing operations". They differ by the $1m
    # discontinued result, and only the second is the basis CBA presents its headline on:
    # p.18 heads the block "Group Performance from continuing operations", and p.26 moves
    # the year by "$730 million", which is 10,982 - 10,252.
    ("CBA", "headline_earnings", "FY25", "10252", "FY26", "10982", "p.16"),
    ("CBA", "credit_impairment_charge", "FY25", "726", "FY26", "788", "p.16"),
    # The half-year pack repeats the full-year page's trap: $5,133m as "Net profit after
    # tax" and $5,132m from continuing operations, the basis CBA leads with. Here the
    # qualifier is quoted - 'from continuing operations - "cash basis"' - and bare on the
    # full-year page, so both spellings have to resolve to the same line item.
    ("CBA", "headline_earnings", "1H25", "5132", "1H26", "5445", "p.16"),
)


AS_OF = {"ANZ": "2026-05-31", "NAB": "2026-05-31", "WBC": "2026-05-31", "CBA": "2026-08-31"}


@pytest.fixture(scope="module")
def answers():
    """What the system actually answers for each figure, from a real run.

    Goes through ``Agent.run`` rather than reading the store, because the store legitimately
    holds more than one reading of a figure and choosing between them is what's being
    tested. Westpac states its net profit as $3,317m in its results document and $3,457m in
    an appendix excluding Notable Items; reading the store directly would find both and
    prove nothing about which one a reader gets.
    """
    from datetime import date

    from bankresearch.agent.orchestrator import Agent
    from bankresearch.memory.fact_store import FactStore

    settings = get_settings()
    if not settings.db_path.exists():
        pytest.skip("no ingested corpus: run `make ingest` first")

    found: dict[tuple[str, str, str], Decimal] = {}
    with connect(settings.db_path) as connection:
        store = FactStore(connection)
        if connection.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0:
            pytest.skip("the corpus holds no facts")
        for bank, metric, prior_period, _p, period, _c, _page in PUBLISHED:
            outcome = Agent(store=store, settings=settings, persist=False).run(
                bank, period, metric, date.fromisoformat(AS_OF[bank])
            )
            result = outcome.result
            if result.current is not None and result.current.value is not None:
                found[(bank, metric, period)] = result.current.value
            if result.prior is not None and result.prior.value is not None:
                found[(bank, metric, prior_period)] = result.prior.value
    return found


def _answer(answers, bank: str, metric: str, period: str) -> Decimal | None:
    return answers.get((bank, metric, period))


@pytest.mark.parametrize(
    ("bank", "metric", "prior_period", "prior", "period", "current", "page"),
    PUBLISHED,
    ids=[f"{bank}-{period}-{metric}" for bank, metric, _pp, _p, period, *_ in PUBLISHED],
)
def test_the_extracted_figure_is_the_one_the_issuer_published(
    answers,
    bank: str,
    metric: str,
    prior_period: str,
    prior: str,
    period: str,
    current: str,
    page: str,
) -> None:
    got_prior = _answer(answers, bank, metric, prior_period)
    got_current = _answer(answers, bank, metric, period)

    assert got_prior is not None, f"{bank} {metric} {prior_period} was not extracted ({page})"
    assert got_current is not None, f"{bank} {metric} {period} was not extracted ({page})"
    assert got_prior == Decimal(prior), f"{bank} {metric} {prior_period}: {page} says {prior}"
    assert got_current == Decimal(current), f"{bank} {metric} {period}: {page} says {current}"


def test_every_seed_pair_is_covered() -> None:
    """Four banks by six metrics. A gap here is a metric nobody checked against a page."""
    covered = {(bank, metric) for bank, metric, *_ in PUBLISHED}
    assert len(covered) == 24, sorted(covered)


def test_the_divisional_repeat_is_not_what_gets_answered(answers) -> None:
    """The most expensive mistake available on these documents. CBA's Retail Banking margin
    of 2.50% sits four pages from its group margin of 2.05%, under a heading row rather
    than a section title."""
    for period in ("FY25", "FY26"):
        value = _answer(answers, "CBA", "nim", period)
        assert value is not None and value < Decimal("2.2"), (
            f"CBA nim {period} is {value}; a divisional margin is 2.50%"
        )


def test_the_level_1_capital_structure_is_not_what_gets_answered() -> None:
    """Westpac's Level 1 capital over its Level 2 risk-weighted assets gives a CET1 ratio
    70bp below the published one, from two figures both read correctly."""
    settings = get_settings()
    if not settings.db_path.exists():
        pytest.skip("no ingested corpus: run `make ingest` first")
    with connect(settings.db_path) as connection:
        rows = connection.execute(
            "SELECT value FROM facts WHERE bank_id = 'WBC' AND component = 'cet1_capital'"
        ).fetchall()
    assert rows, "WBC CET1 capital was not extracted"
    assert all(Decimal(row[0]) > Decimal("54000") for row in rows), (
        f"got {[row[0] for row in rows]}; Level 1 is 53,722"
    )


# --- the whole matrix, compared against itself ---------------------------

#: The cap `overall_confidence` applies when nothing was attributed.
_UNEXPLAINED_CAP = Decimal("0.45")


@pytest.mark.parametrize(
    "bank,period,as_of",
    [
        ("ANZ", "1H26", "2026-05-31"),
        ("NAB", "1H26", "2026-05-31"),
        ("WBC", "1H26", "2026-05-31"),
        ("CBA", "FY26", "2026-08-31"),
    ],
)
def test_a_run_that_explains_nothing_never_reads_as_confident(bank, period, as_of):
    """Comparing all 24 bank-metric combinations is what found this.

    Westpac's margin and expense ratio came back High (0.92) with zero drivers, while ANZ's,
    NAB's and CBA's margins, in the same situation, came back Low (0.45). The difference was
    which code path had declined to explain the movement: a declined decomposition leaves
    the residual at its default of zero, which reads as "0% of the movement is unexplained",
    just like a perfectly reconciled walk.

    No single case could show that. The inconsistency across cases is the signal, so the
    rule is checked across all of them.
    """
    from datetime import date as _date

    from bankresearch.agent.orchestrator import Agent
    from bankresearch.memory.fact_store import FactStore

    settings = get_settings()
    if not settings.db_path.exists():
        pytest.skip("no ingested corpus: run `make ingest` first")

    metrics = (
        "nim",
        "headline_earnings",
        "roe",
        "cet1_ratio",
        "credit_impairment_charge",
        "cost_to_income",
    )
    confident_without_drivers = []
    with connect(settings.db_path) as connection:
        store = FactStore(connection)
        for metric in metrics:
            result = (
                Agent(store=store, settings=settings, persist=False)
                .run(bank, period, metric, as_of=_date.fromisoformat(as_of))
                .result
            )
            if result.status == "ABSTAIN":
                continue
            score = result.overall_confidence.score if result.overall_confidence else None
            if score is None or result.top_level_drivers:
                continue
            # Compared via ``str``, because ``float(0.45) > Decimal("0.45")`` is true: 0.45
            # has no exact binary form, so the float is a hair larger and every capped case
            # would read as a violation. It's the same trap as comparing a figure to a
            # tolerance.
            if Decimal(str(score)) > _UNEXPLAINED_CAP:
                confident_without_drivers.append(f"{bank} {metric}: {score}")

    assert not confident_without_drivers, (
        f"nothing was attributed, so nothing should read as explained: {confident_without_drivers}"
    )


def test_westpacs_expense_ratio_is_explained_on_its_own_basis() -> None:
    """Westpac publishes the ratio only excluding Notable Items, and the expenses and income
    both ways. Explained from the reported expenses ($5,937m) the movement doesn't
    reconcile; from the excluding-Notable-Items ones on page 10 ($5,830m against $5,698m,
    income $11,284m against $10,993m) it does, to within a basis point."""
    from datetime import date as _date

    from bankresearch.agent.orchestrator import Agent
    from bankresearch.memory.fact_store import FactStore

    settings = get_settings()
    if not settings.db_path.exists():
        pytest.skip("no ingested corpus: run `make ingest` first")
    with connect(settings.db_path) as connection:
        if connection.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0:
            pytest.skip("the corpus holds no facts")
        result = (
            Agent(store=FactStore(connection), settings=settings, persist=False)
            .run("WBC", "1H26", "cost_to_income", as_of=_date(2026, 5, 31))
            .result
        )

    assert result.status == "OK"
    assert result.current is not None and result.current.basis.ex_notable_items
    assert {d.driver_id for d in result.top_level_drivers} == {
        "cti.expense_growth",
        "cti.income_growth",
    }
    assert result.reconciliation is not None and result.reconciliation.within_tolerance
    assert any("only excluding notable items" in caveat for caveat in result.caveats)
