"""Management walks, checksums and chaining."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from bankresearch.decomposition.walks import (
    MidpointMismatch,
    Walk,
    WalkStep,
    chain_walks,
    choose_path_a,
    find_pcp_walk,
)
from bankresearch.domain.models import SourceRef

SRC = SourceRef(
    doc_id="anz-1h26-pack",
    doc_type="INVESTOR_PRESENTATION",
    page=12,
    publish_date=date(2026, 5, 1),
)


def step(driver_id: str, value: str) -> WalkStep:
    return WalkStep(label=driver_id, driver_id=driver_id, value=Decimal(value))


def walk(opening: str, closing: str, open_v: str, close_v: str, steps: list[WalkStep]) -> Walk:
    return Walk(
        opening_period=opening,
        closing_period=closing,
        opening_value=Decimal(open_v),
        closing_value=Decimal(close_v),
        steps=steps,
        unit="bps",
        source=SRC,
    )


# --- the checksum ----------------------------------------------------------


def test_a_walk_whose_steps_sum_to_the_close_reconciles():
    w = walk(
        "1H25",
        "1H26",
        "1200",
        "1239",
        [step("cet1.organic_capital", "60"), step("cet1.dividends", "-21")],
    )
    assert w.reconciles(tolerance=Decimal("1"))
    assert w.checksum_residual() == Decimal("0")


def test_a_walk_that_misses_by_more_than_tolerance_does_not_reconcile():
    w = walk(
        "1H25",
        "1H26",
        "1200",
        "1239",
        [step("cet1.organic_capital", "60"), step("cet1.dividends", "-10")],
    )
    assert not w.reconciles(tolerance=Decimal("1"))
    assert w.checksum_residual() == Decimal("-11")


def test_rounding_inside_tolerance_still_reconciles():
    w = walk(
        "1H25",
        "1H26",
        "1200",
        "1239",
        [step("cet1.organic_capital", "60"), step("cet1.dividends", "-20")],
    )
    assert w.reconciles(tolerance=Decimal("1.5"))


# --- period binding -------------------------------------------------------


def test_a_pcp_walk_is_found_by_the_periods_it_declares():
    pcp = walk("1H25", "1H26", "1200", "1239", [step("cet1.organic_capital", "39")])
    hoh = walk("2H25", "1H26", "1220", "1239", [step("cet1.organic_capital", "19")])
    assert find_pcp_walk([hoh, pcp], period="1H26", pcp="1H25") is pcp


def test_a_half_on_half_walk_is_not_mistaken_for_a_pcp_walk():
    """The error that reconciles perfectly and answers a different question."""
    hoh = walk("2H25", "1H26", "1220", "1239", [step("cet1.organic_capital", "19")])
    assert find_pcp_walk([hoh], period="1H26", pcp="1H25") is None


def test_a_previous_packs_walk_is_not_used():
    stale = walk("1H24", "1H25", "1180", "1200", [step("cet1.organic_capital", "20")])
    assert find_pcp_walk([stale], period="1H26", pcp="1H25") is None


# --- chaining -------------------------------------------------------------


def test_two_half_on_half_walks_chain_driver_by_driver():
    first = walk(
        "1H25",
        "2H25",
        "1200",
        "1220",
        [step("cet1.organic_capital", "35"), step("cet1.dividends", "-15")],
    )
    second = walk(
        "2H25",
        "1H26",
        "1220",
        "1239",
        [step("cet1.organic_capital", "30"), step("cet1.dividends", "-11")],
    )
    chained = chain_walks([first, second])
    assert chained.opening_period == "1H25"
    assert chained.closing_period == "1H26"
    assert chained.opening_value == Decimal("1200")
    assert chained.closing_value == Decimal("1239")
    assert chained.method == "chained_walk"
    by_driver: dict[str | None, Decimal] = {}
    for item in chained.steps:
        by_driver[item.driver_id] = by_driver.get(item.driver_id, Decimal(0)) + item.value
    assert by_driver["cet1.organic_capital"] == Decimal("65")
    assert by_driver["cet1.dividends"] == Decimal("-26")
    assert chained.reconciles(tolerance=Decimal("2"))


def test_a_chained_step_keeps_the_page_it_is_printed_on():
    """A summed step (+65bps of organic capital) is printed on neither leg's page, so it
    couldn't be checked against either. Each leg's steps are kept as printed."""
    first = walk("1H25", "2H25", "1200", "1220", [step("cet1.organic_capital", "35")])
    second = Walk(
        opening_period="2H25",
        closing_period="1H26",
        opening_value=Decimal("1220"),
        closing_value=Decimal("1250"),
        steps=[step("cet1.organic_capital", "30")],
        unit="bps",
        source=SourceRef(
            doc_id="anz-1h26-results-document",
            doc_type="RESULTS_DOCUMENT",
            page=40,
            publish_date=date(2026, 5, 7),
        ),
    )
    chained = chain_walks([first, second])
    assert [(item.value, item.source.doc_id) for item in chained.steps] == [
        (Decimal("35"), first.source.doc_id),
        (Decimal("30"), "anz-1h26-results-document"),
    ]


def test_a_chain_cites_every_document_that_fed_it():
    first = walk("1H25", "2H25", "1200", "1220", [step("cet1.organic_capital", "20")])
    second = Walk(
        opening_period="2H25",
        closing_period="1H26",
        opening_value=Decimal("1220"),
        closing_value=Decimal("1239"),
        steps=[step("cet1.organic_capital", "19")],
        unit="bps",
        source=SourceRef(
            doc_id="anz-fy25-pack",
            doc_type="INVESTOR_PRESENTATION",
            page=9,
            publish_date=date(2025, 11, 10),
        ),
    )
    chained = chain_walks([first, second])
    assert {s.doc_id for s in chained.sources} == {"anz-1h26-pack", "anz-fy25-pack"}


def test_a_mismatched_midpoint_is_refused_as_a_restatement_signal():
    """The first walk closes at 1220; the second opens at 1235. Something was restated."""
    first = walk("1H25", "2H25", "1200", "1220", [step("cet1.organic_capital", "20")])
    second = walk("2H25", "1H26", "1235", "1239", [step("cet1.organic_capital", "4")])
    with pytest.raises(MidpointMismatch) as exc:
        chain_walks([first, second], midpoint_tolerance=Decimal("1"))
    assert "restat" in str(exc.value).lower()


def test_non_contiguous_legs_are_refused():
    first = walk("1H25", "2H25", "1200", "1220", [step("cet1.organic_capital", "20")])
    third = walk("1H26", "2H26", "1239", "1250", [step("cet1.organic_capital", "11")])
    with pytest.raises(ValueError):
        chain_walks([first, third])


def test_a_single_leg_chain_is_refused():
    only = walk("1H25", "2H25", "1200", "1220", [step("cet1.organic_capital", "20")])
    with pytest.raises(ValueError):
        chain_walks([only])


# --- path selection -------------------------------------------------------


def test_a_pcp_walk_outranks_a_chain():
    pcp = walk("1H25", "1H26", "1200", "1239", [step("cet1.organic_capital", "39")])
    legs = [
        walk("1H25", "2H25", "1200", "1220", [step("cet1.organic_capital", "20")]),
        walk("2H25", "1H26", "1220", "1239", [step("cet1.organic_capital", "19")]),
    ]
    chosen = choose_path_a([pcp, *legs], period="1H26", pcp="1H25", tolerance=Decimal("1"))
    assert chosen is not None
    assert chosen.method == "management_walk"


def test_chaining_is_used_only_when_no_pcp_walk_exists():
    legs = [
        walk("1H25", "2H25", "1200", "1220", [step("cet1.organic_capital", "20")]),
        walk("2H25", "1H26", "1220", "1239", [step("cet1.organic_capital", "19")]),
    ]
    chosen = choose_path_a(legs, period="1H26", pcp="1H25", tolerance=Decimal("2"))
    assert chosen is not None
    assert chosen.method == "chained_walk"


def test_a_broken_chain_yields_no_path_a_rather_than_a_stretched_walk():
    """A withheld leg must not be covered by its partner."""
    legs = [walk("2H25", "1H26", "1220", "1239", [step("cet1.organic_capital", "19")])]
    assert choose_path_a(legs, period="1H26", pcp="1H25", tolerance=Decimal("1")) is None


def test_a_walk_that_fails_its_own_checksum_is_not_used():
    broken = walk("1H25", "1H26", "1200", "1239", [step("cet1.organic_capital", "5")])
    assert choose_path_a([broken], period="1H26", pcp="1H25", tolerance=Decimal("1")) is None


def test_unmapped_steps_are_kept_rather_than_forced():
    w = walk(
        "1H25",
        "1H26",
        "1200",
        "1239",
        [
            WalkStep(
                label="Something the taxonomy does not cover", driver_id=None, value=Decimal("39")
            ),
        ],
    )
    assert w.unmapped_steps() == 1
    assert w.reconciles(tolerance=Decimal("1"))


# --- a chart walk carries one unit, true of every number in it ------------


def test_a_chart_walk_is_put_into_one_unit():
    """`Walk` assumes a single unit: `checksum_residual` is declared change minus steps.

    ANZ states its margin as 1.53% and draws the walk explaining it in basis points, so
    the Walk held pct endpoints beside bps steps under a `unit` field true of neither.
    The residual came out -0.97 against a real -2, and the evidence card read
    "Capital & replicating portfolio | pct | 2" for two basis points.
    """
    from datetime import date
    from decimal import Decimal

    from bankresearch.domain.models import SourceRef
    from bankresearch.ingestion.chart_walks import _as_walk
    from bankresearch.ingestion.charts import ChartReading

    reading = ChartReading(
        steps=[
            ("Capital & replicating portfolio", Decimal("2")),
            ("Markets activities", Decimal("-3")),
        ],
        unit="bps",
    )
    walk = _as_walk(
        reading,
        metric_id="nim",
        source=SourceRef(
            doc_id="anz-1h26-investor-presentation",
            doc_type="INVESTOR_PRESENTATION",
            page=25,
            publish_date=date(2026, 5, 1),
        ),
        unit="pct",
        opening_period="1H25",
        closing_period="1H26",
        reported_opening=Decimal("1.56"),
        reported_closing=Decimal("1.53"),
    )

    assert walk.unit == "bps", "one canonical unit for a ratio walk"
    assert walk.opening_value == Decimal("156")
    assert walk.closing_value == Decimal("153")
    assert walk.declared_change == Decimal("-3")
    assert walk.sum_of_steps == Decimal("-1")
    # The number that was wrong: -0.03 - (-1) = 0.97 before the endpoints were converted.
    assert walk.checksum_residual() == Decimal("-2")


def test_an_amount_walk_is_put_into_millions():
    from datetime import date
    from decimal import Decimal

    from bankresearch.domain.models import SourceRef
    from bankresearch.ingestion.chart_walks import _as_walk
    from bankresearch.ingestion.charts import ChartReading

    walk = _as_walk(
        ChartReading(steps=[("NII", Decimal("0.019"))], unit="aud_bn"),
        metric_id="headline_earnings",
        source=SourceRef(
            doc_id="d", doc_type="INVESTOR_PRESENTATION", page=1, publish_date=date(2026, 5, 1)
        ),
        unit="aud_m",
        opening_period="1H25",
        closing_period="1H26",
        reported_opening=Decimal("3568"),
        reported_closing=Decimal("3780"),
    )
    assert walk.unit == "aud_m"
    assert walk.sum_of_steps == Decimal("19"), "0.019bn is 19m"
    assert walk.declared_change == Decimal("212")


def test_a_dollar_walk_against_a_ratio_metric_says_so():
    """Returning zeroes made the note read "miss by 0 against a tolerance of 0".

    That is my own fix reporting itself badly: a chart drawn in dollars cannot measure a
    movement in percent, which is a misreading rather than a near-miss, and the note
    should say which.
    """
    from decimal import Decimal

    from bankresearch.ingestion.charts import checksum

    passed, residual, _tolerance = checksum(
        [("x", Decimal("100"))], Decimal("1.56"), Decimal("1.53"), "pct", "aud_m"
    )
    assert passed is False
    assert residual.is_nan(), "a NaN residual is what the note keys off"
