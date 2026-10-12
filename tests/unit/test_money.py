"""Costs are shown and granted in Australian dollars, converted at the one rate in
config/models.yaml, and a memo's amounts are written as A$."""

from __future__ import annotations

from decimal import Decimal

from bankresearch.domain.money import aud, aud_amounts, to_usd, usd_per_aud
from bankresearch.domain.registry import get_registry


def test_the_rate_comes_from_the_config():
    assert get_registry().models["currency"]["usd_per_aud"] == "0.6979"
    assert usd_per_aud() == Decimal("0.6979")


def test_a_cost_billed_in_us_dollars_is_said_in_australian_dollars():
    assert aud("0.6979") == "A$1.00"
    assert aud(Decimal("3.58")) == "A$5.13"
    assert aud(0) == "A$0.00"


def test_a_ceiling_never_shows_less_than_it_can_cost():
    assert aud("0.0001", up=True) == "A$0.01"
    assert aud("0.0001") == "A$0.00"


def test_a_grant_is_taken_in_australian_dollars(capsys):
    from bankresearch.cli import _consent_gate

    gate = _consent_gate("2:1.00")
    assert gate.ceiling_usd == to_usd("1.00") == Decimal("0.6979")
    assert "A$1.00 in total" in capsys.readouterr().err


def test_a_memos_bare_dollars_are_written_as_australian():
    assert aud_amounts("includes $300 million of provisions") == (
        "includes A$300 million of provisions"
    )
    assert aud_amounts("a fall of ($27m)") == "a fall of (A$27m)"
    for kept in ("A$5m", "US$5m", "NZ$5m", "a $ sign alone"):
        assert aud_amounts(kept) == kept


def test_a_quotation_keeps_the_banks_words():
    text = 'it said "Net interest income increased by 9.9% to $24.0 million" and $2m more'
    assert aud_amounts(text) == (
        'it said "Net interest income increased by 9.9% to $24.0 million" and A$2m more'
    )
