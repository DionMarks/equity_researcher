"""What the memo says, and how it names what it says."""

from __future__ import annotations

from bankresearch.agent.steps.decompose import _spelled_out
from bankresearch.agent.steps.explain import _without_advice


def test_advice_to_the_reader_is_cut_back_to_the_fact_it_introduces():
    memo = (
        "Westpac's net interest margin fell 7.0 bps [ev_1]. Do not read the 7.0 bps fall as "
        "the underlying movement: the bank's Core NIM was 1.80%, down 9 basis points [ev_2]."
    )
    kept, changed = _without_advice(memo)
    assert changed == 1
    assert kept.endswith("The bank's Core NIM was 1.80%, down 9 basis points [ev_2].")
    assert "Do not" not in kept


def test_a_sentence_that_only_advises_goes():
    kept, changed = _without_advice("Costs rose [ev_3]. Investors should treat this with care.")
    assert (kept, changed) == ("Costs rose [ev_3].", 1)


def test_a_sentence_that_merely_starts_like_advice_stays():
    memo = "Never before had the margin moved so much in a half [ev_4]."
    assert _without_advice(memo) == (memo, 0)


def test_an_abbreviated_bar_is_spelled_out_and_a_named_one_is_left_alone():
    assert _spelled_out("WSF", "nim", "nim.funding_costs") == (
        "WSF (Wholesale and term funding costs)"
    )
    assert _spelled_out("T&M", "nim", "nim.markets_treasury").startswith("T&M (")
    assert _spelled_out("Capital & Other", "nim", "nim.hedging") == "Capital & Other"
    assert _spelled_out("Loans", "nim", "nim.lending_pricing") == "Loans"


def test_the_checker_is_told_the_movement_the_writer_was_told():
    """CBA's FY24 memo lost a correct "fell A$236m (2.3%)": the writer had been given the
    2.3% and the checker hadn't, and the page prints 2%."""
    from decimal import Decimal

    from bankresearch.agent.steps.explain import computed_lines
    from bankresearch.domain.models import Decomposition

    decomposition = Decomposition(
        metric_id="headline_earnings",
        path="B",
        change_unit="aud_m",
        headline_change=Decimal(-236),
        drivers=[],
    )
    movement = "fell A$236m (2.3%), from A$10,072m to A$9,836m"
    lines = computed_lines(decomposition, 0, movement=movement)
    assert lines == [f"The movement itself: {movement}, computed from the two published figures"]


def test_citations_side_by_side_show_each_page_once():
    from bankresearch.output.render_terminal import _adjacent_joined

    joined = _adjacent_joined("fell A$236m [ev_030][ev_008] [ev_045].")
    assert joined == "fell A$236m [ev_030, ev_008, ev_045]."


def test_saying_the_bank_doesnt_disclose_something_is_taken_out():
    """Bendigo's FY25 memo said its expense effect was aggregate "as expenses are not
    disclosed by category". Its annual report breaks them down; none was read, which isn't
    the same thing."""
    from bankresearch.agent.steps.explain import _without_claims_of_silence

    memo = (
        "Expenses added 444.0 bps [ev_1]. The expense effect is shown in aggregate as expenses "
        "are not disclosed by category."
    )
    assert _without_claims_of_silence(memo) == ("Expenses added 444.0 bps [ev_1].", 1)


def test_a_driver_two_chained_halves_name_differently_gets_one_name():
    """ANZ's 1H26 capital answer listed "Capital floor and Capital floor adjustment" as if
    they were two drivers."""
    from bankresearch.agent.steps.decompose import _shared_label

    assert _shared_label(["Capital floor", "Capital floor adjustment"]) == "Capital floor"
    assert (
        _shared_label(
            ["Cash profit (Level 2) excluding 2H25 significant items", "Cash profit (Level 2)"]
        )
        == "Cash profit (Level 2)"
    )
    assert (
        _shared_label(
            [
                "Underlying RWA (excluding IRRBB) and other items",
                "Underlying RWA (excluding IRRBB) growth",
            ]
        )
        == "Underlying RWA (excluding IRRBB)"
    )
    # Too little in common to name anything: the taxonomy's own name is used instead.
    assert (
        _shared_label(
            [
                "Payment of the 2025 interim dividend (net of DRP and BOP)",
                "Payment of the 2025 final dividend",
            ]
        )
        == ""
    )
    assert _shared_label(["IRRBB RWA growth"]) == "IRRBB RWA growth"


def test_a_label_shown_to_a_reader_drops_the_taxonomys_note_on_its_sign():
    """Judo's FY26 answer listed "Operating expenses (higher costs = negative)"."""
    from bankresearch.decomposition.taxonomy import display_label, driver_label

    assert "= negative" in driver_label("headline_earnings", "earn.operating_expenses")
    assert display_label("headline_earnings", "earn.operating_expenses") == "Operating expenses"
    assert display_label("headline_earnings", "earn.credit_impairment") == (
        "Credit impairment charge"
    )


def test_a_restated_comparative_is_named_in_words_and_only_if_the_answer_uses_it():
    """Westpac's 1H26 margin, explained wholly by its own walk, warned that its "nii"
    comparative was restated: a figure the answer never used, under an id nobody knows."""
    from bankresearch.agent.steps.reconcile import restatement_caveat
    from bankresearch.agent.steps.resolve import resolve_request

    spec = resolve_request("WBC", "1H26", "nim")
    assert restatement_caveat(spec, ["nii"], used={"nim"}, footnoted=True) is None
    said = restatement_caveat(spec, ["nii"], used={"nim", "nii", "aiea"}, footnoted=True)
    assert said is not None
    assert said.startswith("The 1H25 net interest income is the restated comparative.")
    assert "(nii)" not in said


def test_a_figure_is_named_in_words():
    from bankresearch.domain.registry import component_name

    assert component_name("nii") == "net interest income"
    assert component_name("rwa") == "risk-weighted assets"
    assert component_name("nim") == "net interest margin"
    assert component_name("cet1_ratio") == "CET1 ratio"


def test_an_earnings_heading_names_the_banks_own_measure():
    """Judo leads with profit before tax, and its answer was headed "Cash earnings"."""
    from bankresearch.agent.steps.resolve import resolve_request
    from bankresearch.output.schema import ResearchResult, request_names

    spec = resolve_request("JDO", "FY26", "headline_earnings")
    result = ResearchResult(
        run_id="example",
        created_at="2026-10-12T00:00:00Z",
        as_of=spec.as_of,
        request=spec,
        status="ABSTAIN",
        abstain_reason="DOCS_UNAVAILABLE",
    )
    bank, measure = request_names(result)
    assert bank.startswith("Judo")
    assert measure == (spec.headline_label or "Cash earnings")
    assert measure != "Cash earnings" or spec.headline_label is None


# --- every material driver is named, and the cross-check is shown ----------


def _bridge(narrative_ids: set[str] | None = None):  # type: ignore[no-untyped-def]
    from decimal import Decimal

    from bankresearch.domain.models import Decomposition, DriverContribution

    def driver(driver_id: str, label: str, value: str, *evidence: str) -> DriverContribution:
        amount = Decimal(value)
        return DriverContribution(
            driver_id=driver_id,
            label=label,
            contribution=amount,
            direction="+" if amount > 0 else "-" if amount < 0 else "0",
            evidence_ids=list(evidence),
        )

    return Decomposition(
        metric_id="headline_earnings",
        change_unit="aud_m",
        headline_change=Decimal("97"),
        drivers=[
            driver("earn.nii", "Net interest income", "420", "ev_022", "ev_073"),
            driver("earn.other_income", "Other income", "80", "ev_032", "ev_088"),
            driver("earn.operating_expenses", "Operating expenses", "-239", "ev_026", "ev_080"),
            driver("earn.credit_impairment", "Credit impairment charge", "-193", "ev_008"),
            driver("earn.tax", "Income tax", "29", "ev_038"),
            driver("earn.minority_interests", "Non-controlling interests", "0", "ev_020"),
        ],
    )


def test_a_material_driver_the_checked_memo_lost_is_stated_by_its_figure():
    """Westpac's memo explained a A$97m rise with -239, +80 and +29: the sentences for net
    interest income and the impairment charge were removed for causes on the other basis."""
    from bankresearch.agent.steps.explain import name_every_material_driver

    memo = (
        "Westpac's net profit rose A$97m [ev_015]. Operating expenses took A$239m off "
        "[ev_026, ev_080]. Other income added A$80m [ev_032] and lower tax added A$29m [ev_038]."
    )
    completed, named = name_every_material_driver(memo, _bridge())
    assert named == ["Net interest income", "Credit impairment charge"]
    assert completed.endswith(
        "Net interest income added A$420m [ev_022, ev_073]. "
        "Credit impairment charge took A$193m off [ev_008]."
    )


def test_a_driver_already_named_or_too_small_is_left_alone():
    from bankresearch.agent.steps.explain import name_every_material_driver

    memo = "Net interest income added A$420m [ev_022]. Expenses took A$239m off [ev_026]."
    completed, named = name_every_material_driver(memo, _bridge())
    # Other income (A$80m) and tax (A$29m) are under a quarter of the largest driver, and a
    # nil one is never added.
    assert named == ["Credit impairment charge"]
    assert "Other income" not in completed and "Non-controlling" not in completed


def test_an_empty_memo_stays_empty_and_the_word_limit_holds():
    from bankresearch.agent.steps.explain import (
        NARRATIVE_WORD_LIMIT,
        name_every_material_driver,
    )

    assert name_every_material_driver("", _bridge()) == ("", [])
    long = " ".join(["word"] * (NARRATIVE_WORD_LIMIT - 4)) + "."
    completed, named = name_every_material_driver(long, _bridge())
    assert named == [] and completed == long  # no sentence fits in the four words left


def test_the_identity_cross_check_states_what_it_left_unexplained():
    """The caveat used to say the identity's residual was "reported alongside", and nothing
    reported it."""
    from decimal import Decimal

    from bankresearch.agent.steps.decompose import _amount

    assert _amount(Decimal("0.04"), "bps") == "0.0bps"
    assert _amount(Decimal("-0.04"), "bps") == "0.0bps"
    assert _amount(Decimal("-12.46"), "aud_m") == "A$12.5m"
