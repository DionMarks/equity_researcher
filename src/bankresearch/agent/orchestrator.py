"""The fixed state machine that answers one question: twelve steps in order, with a model in some of
them and code for every figure."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

from ..acquisition.asx_announcements import AsxListing
from ..acquisition.bank_directory import BankMatch, find_bank
from ..acquisition.fetch import ASX_DOMAIN
from ..acquisition.identify import find_year_end
from ..domain.models import Diagnostics, DocType, EvidenceCard, RequestSpec
from ..domain.registry import BankProfile, get_registry, other_driver_id
from ..ingestion.artifacts import read_processed
from ..ingestion.chart_walks import rank_walk_candidates
from ..ingestion.facts import line_item_index
from ..ingestion.parse_pdf import ParsedDocument
from ..ingestion.pipeline import ingest_document
from ..ingestion.walk_tables import walks_in
from ..llm.budget import Budget
from ..llm.cache import CachedClient, ResponseCache
from ..llm.client import LLMClient, ModelUnreachable
from ..llm.consent import ConsentDenied, ConsentGate
from ..llm.tracing import Tracer
from ..memory.db import connect
from ..memory.fact_store import FactStore
from ..output.progress import RunProgress
from ..output.render_md import amount_places
from ..output.schema import ModelCall, ResearchResult, Trail
from ..settings import Settings, get_settings
from ..tools.doc_tools import Toolkit
from .estimate import Estimate
from .reporting import ReportingClient
from .state import Step
from .steps.availability import check_availability
from .steps.commentary import commentary_cards
from .steps.decompose import run_decomposition
from .steps.explain import (
    check_narrative,
    computed_lines,
    movement_line,
    name_every_material_driver,
    write_memo,
)
from .steps.extract import cards_from_facts, cards_from_walk
from .steps.locate import run_locate
from .steps.map_rows import map_rows, offer_rows
from .steps.reconcile import reconcile_inputs, restatement_caveat
from .steps.render import abstention_result, build_result, write_run
from .steps.resolve import AbstainSignal, caveats_for, expected_release_date, resolve_request
from .steps.score import score_run

__all__ = ["Agent", "RunOutcome"]


@dataclass
class RunOutcome:
    """A finished run and where its artefacts went."""

    result: ResearchResult
    artefacts: Path | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return 2 if self.result.status == "ABSTAIN" else 0


#: How a refused call reads in the trace and the memo. A refusal is a decision, not a
#: failure. Going over budget should weaken the answer rather than raise,
#: and the same goes for someone declining to spend. The deterministic path still produces
#: the figures, identities, decomposition, confidence and citations.
_REFUSED_CAVEAT = (
    "Consent for a paid model call was declined, so {step} was skipped. The figures, "
    "the accounting identities, the decomposition and the citations are unaffected: "
    "they are computed without a model."
)


#: The same note in ``diagnostics.warnings``, for anyone reading diagnostics rather than
#: prose. Without it, a declined run shows ``llm_calls: 0`` and nothing saying why, which
#: looks the same as a run that had no model to call.
_REFUSED_WARNING = (
    "No model call was made for {step}: consent for a paid call was declined. The "
    "deterministic path produced the figures, the identities and the citations."
)


def _declined(diagnostics: Diagnostics, step: str) -> None:
    """Records a paid call the gate turned down: as a warning for a reader, and by step, so
    a caller can tell a run that stopped short of its model calls from one that never needed
    them."""
    diagnostics.warnings.append(_REFUSED_WARNING.format(step=step))
    diagnostics.declined_steps.append(step)


def _refused(refusal: Exception) -> str:
    return f"consent declined: {refusal}"


def _record_spend(diagnostics: Diagnostics, budget: Budget, tracer: Tracer | None = None) -> None:
    """Copy what the run actually cost onto the result.

    The budget always counted correctly and refused calls when a cap was reached, but
    nothing copied its totals onto ``Diagnostics``, so every run reported ``llm_calls: 0``
    and ``cost_usd: 0`` however much it had spent. That made the cost in the eval report,
    the memo and the trace a constant rather than a measurement, and nobody can check a cost
    limit whose spending is invisible.
    """
    diagnostics.llm_calls = budget.calls
    diagnostics.input_tokens = budget.usage.input_tokens
    diagnostics.output_tokens = budget.usage.output_tokens
    diagnostics.cache_read_tokens = budget.usage.cache_read_tokens
    diagnostics.cost_usd = budget.cost
    diagnostics.budget_breached = budget.breached
    diagnostics.warnings.extend(budget.warnings)
    # Which model and which prompt version answered each step. Both were declared on
    # ``Diagnostics`` and never filled, so a live run recorded ``model_ids: {}`` and nobody
    # reading it later could tell which model said what.
    for event in tracer.events if tracer is not None else ():
        if event.get("model"):
            diagnostics.model_ids[str(event["step"])] = str(event["model"])
        if event.get("prompt") and event.get("prompt_sha256"):
            diagnostics.prompt_hashes[str(event["prompt"])] = str(event["prompt_sha256"])


@dataclass
class Agent:
    """One research question at a time, through a fixed sequence of steps."""

    store: FactStore
    settings: Settings = field(default_factory=get_settings)
    persist: bool = True
    client: LLMClient | None = None
    """The model, if there is one. ``None`` means the deterministic path, which is a real
    degraded mode rather than a failure: the figures, identities, decomposition, confidence
    and citations are all unaffected. Built lazily from the environment, so creating an
    ``Agent`` never needs a key.
    """
    consent: ConsentGate | None = None
    """Asked before each paid call. ``None`` leaves the client on its own default, which
    refuses, so an agent built without one can't spend money whatever else is set.
    """
    asx: AsxListing = field(default_factory=AsxListing)
    """The ASX's list of a company's announcements, read before any search for a filing."""
    progress: RunProgress | None = None
    """Told what the run is doing as it goes, for the line a terminal shows while a paid run
    works. It only watches: with or without it, the run makes the same calls."""
    _tracer: Tracer | None = field(default=None, init=False, repr=False)
    """The current run's model calls: model, effort, prompt hash, tokens and stop reason."""
    _found: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    """Banks identified from the ASX in this process and not yet confirmed by a filing,
    each with the name it was asked by."""

    def model_client(self) -> LLMClient | None:
        """The model client this agent would use, or ``None`` without a key."""
        return self._model()

    def _model(self) -> LLMClient | None:
        """The client, or ``None``. Resolved once per agent."""
        if self.client is None and not self.settings.has_api_key:
            return None
        if self.client is None:
            self.client = self._built_client()
        if self.progress is not None and not isinstance(self.client, ReportingClient):
            self.client = ReportingClient(self.client, self.progress)
        return self.client

    def _built_client(self) -> LLMClient:
        """The live client, read through the response cache."""
        from ..llm.anthropic_client import AnthropicClient

        # Read through the cache and write past it. A repeated request within a run, or
        # across a replayed eval, costs nothing and returns the same answer, which is what
        # makes an eval reproducible once a key exists.
        upstream = AnthropicClient(settings=self.settings)
        if self.consent is not None:
            # Replaced rather than defaulted here, so exactly one place decides what
            # refusing looks like: an agent given no gate keeps the client's own, which
            # is DenyAll.
            upstream = AnthropicClient(settings=self.settings, consent=self.consent)
        return CachedClient(
            cache=ResponseCache(connect(self.settings.db_path)),
            upstream=upstream,
            replay_only=False,
        )

    def _doing(self, activity: str) -> None:
        """Tells the progress line, if there is one, what the run is doing now."""
        if self.progress is not None:
            self.progress.doing(activity)

    def run(
        self,
        bank: str,
        period: str,
        metric: str,
        as_of: date | None = None,
    ) -> RunOutcome:
        """Answer one question, or abstain with a reason.

        Never raises, with one exception: ``ModelUnreachable``, when the model can't be used
        at all (no key, a key the API refuses, an account out of credit, a service that stays
        down). An answer without the model isn't the one that was asked for, so the caller
        says which it was instead of showing one.
        """
        trace: list[dict[str, Any]] = []
        started = time.perf_counter()
        self._tracer = Tracer(started=started)
        registry = get_registry()
        budget = Budget.from_config(registry.budgets, registry.models)

        def record(step: Step, **fields: Any) -> None:
            trace.append({"step": step.value, "at": time.perf_counter() - started, **fields})

        self._doing("Working out the question")
        # Step 1: Resolve, finding a bank that isn't configured on the way.
        spec, signal, found_notes = self._resolve(bank, period, metric, as_of, budget=budget)
        if spec is None:
            assert signal is not None
            return self._abstain_without_spec(bank, period, metric, as_of, signal, trace)
        record(
            Step.RESOLVE,
            period=spec.period,
            comparison=spec.comparison_period,
            as_of=spec.as_of.isoformat(),
            **({"found": found_notes} if found_notes else {}),
        )

        caveats = [*found_notes, *caveats_for(spec)]
        diagnostics = Diagnostics(config_hash=get_registry().config_hash, model_ids={}, warnings=[])

        # Step 2: Availability.
        availability = check_availability(spec, self.store)
        record(Step.AVAILABILITY, reportable=availability.reportable, detail=availability.detail)

        # Step 3: Discover. Only for a period that should have been reported by now and that
        # nothing in the store covers: an issuer outside the seed corpus, or a period not yet
        # ingested. "Not reported yet" is left alone, because there is nothing to find.
        if not availability.reportable and (
            availability.reason == "DOCS_UNAVAILABLE" or availability.release_pending
        ):
            client = self._model()
            if client is None or not client.available:
                record(Step.DISCOVER, skipped="no model configured")
                diagnostics.warnings.append(
                    "Cold-path discovery needs a model to search the issuer's site, and none "
                    "is configured."
                )
            else:
                budget.go_cold(get_registry().budgets)
                self._doing("Finding the filing on the ASX")
                acquired = self._discover(
                    spec,
                    client,
                    budget,
                    caveats,
                    diagnostics,
                    search=not availability.release_pending,
                )
                record(Step.DISCOVER, ingested=acquired.doc_ids, notes=acquired.notes)
                if acquired.doc_ids:
                    self._remember(spec, acquired.doc_ids[0])
                    availability = check_availability(spec, self.store)
                    record(
                        Step.AVAILABILITY,
                        reportable=availability.reportable,
                        detail=availability.detail,
                    )
                    caveats.append(
                        f"{get_registry().bank(spec.bank_id).name} {spec.period} wasn't in the "
                        "saved data. Its results document was found on the bank's own site or "
                        "the ASX, checked and read for this run: "
                        + "; ".join(n for n in acquired.notes if n.startswith("found"))
                    )
                elif availability.release_pending and acquired.not_on_the_asx:
                    # Due this month and not on the ASX's own list: not out yet, which is
                    # known for the price of reading the list, not a search.
                    profile = get_registry().bank(spec.bank_id)
                    availability = availability.model_copy(
                        update={
                            "reason": "PERIOD_NOT_YET_REPORTED",
                            "detail": (
                                f"{profile.name} usually releases its {spec.period} result "
                                f"by {expected_release_date(profile, spec.period).isoformat()}"
                                f", and the ASX's list of {profile.ticker}'s announcements has "
                                f"no {spec.period} result as at {spec.as_of.isoformat()}. It "
                                "hasn't been released yet."
                            ),
                        }
                    )
                else:
                    diagnostics.warnings.append(
                        "Cold-path discovery found no usable document: "
                        + ("; ".join(acquired.notes[-3:]) or "nothing was returned")
                    )
        if not availability.reportable:
            _record_spend(diagnostics, budget, self._tracer)
            diagnostics.latency_ms = int((time.perf_counter() - started) * 1000)
            return self._finish(
                abstention_result(
                    spec,
                    availability.reason or "DOCS_UNAVAILABLE",
                    availability.detail,
                    caveats,
                    diagnostics,
                ),
                trace,
            )

        # For a bank no one configured, the figures its filing gives under names of its own,
        # and its slide pack when the filing can't split the movement: a smaller bank often
        # draws its walk only there (BNK Banking Corporation's margin walk is on p.19 of its
        # FY26 deck, and nowhere in its Appendix 4E).
        if get_registry().bank(spec.bank_id).discovered:
            matched = self._learn_rows(spec, availability, budget, caveats, diagnostics)
            if matched:
                record(Step.DISCOVER, matched_rows=matched)
            # The measure called what the bank calls it: BNK reports no cash earnings, and
            # its own row is "NPAT".
            own = dict((c, label) for label, c in get_registry().bank(spec.bank_id).row_labels)
            if spec.headline_label is None and spec.measure in own:
                spec = spec.model_copy(update={"headline_label": own[spec.measure]})
            client = self._model()
            if client is not None and client.available and self._needs_the_deck(spec, availability):
                budget.go_cold(get_registry().budgets)
                self._doing("Finding the bank's results presentation")
                deck = self._discover(
                    spec, client, budget, caveats, diagnostics, doc_type="INVESTOR_PRESENTATION"
                )
                record(Step.DISCOVER, ingested=deck.doc_ids, notes=deck.notes)
                if deck.doc_ids:
                    availability = check_availability(spec, self.store)
                    # The slide pack names its rows its own way too: Macquarie's return on
                    # equity is only on its deck, as "Annualised return on equity (%)".
                    matched = self._learn_rows(spec, availability, budget, caveats, diagnostics)
                    if matched:
                        record(Step.DISCOVER, matched_rows=matched)
                    caveats.append(
                        f"{get_registry().bank(spec.bank_id).name}'s results document doesn't "
                        "give what's needed to split the movement, so its results presentation "
                        "was found, checked and read too."
                    )

        documents = self._load_documents(availability)
        client = self._model()
        # Step 4, Locate, used to run on every request with a model. It cost three quarters
        # of a run (seven turns of a growing conversation, A$0.96 of an A$1.22 Westpac
        # run) and nothing used the pages it chose: every figure comes from the validated store. It
        # now runs only where it can change the answer, which is when the ranking finds no
        # page that could hold the issuer's walk; see ``_page_search``.

        # Step 5: Extract. Every validated figure becomes a card, with no model involved.
        self._doing("Reading the saved figures")
        facts = self.store.get_facts(spec)
        cards: list[EvidenceCard] = cards_from_facts(facts, spec, documents)
        record(Step.EXTRACT, facts=len(facts), cards=len(cards), mode="deterministic")

        # Step 6: Validate. Figures were validated at ingestion and nothing unvalidated can
        # be read, so there's nothing left to fail here.
        record(Step.VALIDATE, note="identities checked at ingestion; store holds only passes")

        # Step 7: Reconcile.
        inputs = reconcile_inputs(cards, facts, spec)
        record(Step.RECONCILE, components=len(inputs.current), conflicts=len(inputs.conflicts))

        # Step 8: Decompose, including the issuer's own walk where it published one.
        self._doing("Splitting the movement")
        walks = []
        walk_cards: list[EvidenceCard] = []
        for document in documents.values():
            meta = next(
                (
                    doc
                    for doc in (*availability.current_docs, *availability.comparison_docs)
                    if doc.doc_id == document.doc_id
                ),
                None,
            )
            if meta is not None:
                walks.extend(walks_in(document, spec.metric_id, meta))

        # A walk read from a chart, where the issuer drew its waterfall instead of putting
        # it in a table, which is what the four majors do for margin and capital. Only tried
        # when no table walk was found and there's a model. The reading is checked against
        # the endpoints published in the text and thrown away whole if it doesn't bridge
        # them, so anything that reaches Path A has reconciled.
        reported = inputs.pair(spec.metric_id)
        charted = None
        located = "skipped: no model configured"
        # A metric that leads with its identity only needs a walk when the identity can't
        # run. Reading charts for one that can, ROE or earnings with every line in hand,
        # cost up to four vision calls for a reading the decomposition would not use.
        wants_walk = registry.metrics[spec.metric_id].primary_path in {
            "A",
            "A_else_B",
        } or not _identity_can_run(spec, inputs)
        # The issuer's walk written out in its own text, read before any chart: all four
        # majors print their year-on-year margin walk as prose, with a cause for each step,
        # and one text call is cheaper and faster than a vision call per page.
        written = None
        if (
            not walks
            and wants_walk
            and reported is not None
            and client is not None
            and client.available
        ):
            written = self._written_walks(
                spec, documents, availability, inputs, reported, client, budget
            )
            if written.refused:
                caveats.append(_REFUSED_CAVEAT.format(step="reading the written walk"))
                _declined(diagnostics, "reading the written walk")
            record(Step.EXTRACT, written_walks=len(written.walks), mode="text", notes=written.notes)
            if written.walks:
                walks.extend(written.walks)
                source = written.walks[0].source
                caveats.append(
                    f"The drivers are the issuer's own walk, written out on {source.doc_id} "
                    f"p.{source.page}: every step was found on the page and the steps "
                    "reconcile to the published movement."
                )
        # A balance is walked from the last balance date, not from a year earlier: every
        # major writes its CET1 walk over the half. Two halves, each read and checked against
        # the published balances at its own ends, chain into the year the question asks
        # about, where neither answers it alone.
        if (
            not walks
            and wants_walk
            and reported is not None
            and client is not None
            and client.available
            and registry.metrics[spec.metric_id].point_in_time
        ):
            legs = self._written_legs(spec, inputs, reported, client, budget)
            if legs.refused:
                caveats.append(_REFUSED_CAVEAT.format(step="reading the written walk"))
                _declined(diagnostics, "reading the written walk")
            record(Step.EXTRACT, written_legs=len(legs.walks), mode="text", notes=legs.notes)
            if len(legs.walks) == 2:
                walks.extend(legs.walks)
                earlier, later = sorted(legs.walks, key=lambda walk: walk.source.publish_date)
                caveats.append(
                    "The drivers are the issuer's own walks for the two halves, chained: "
                    f"{earlier.opening_period} to {earlier.closing_period} as written on "
                    f"{earlier.source.doc_id} p.{earlier.source.page}, and on to "
                    f"{later.closing_period} as written on {later.source.doc_id} "
                    f"p.{later.source.page}. Each was found on its page and reconciled to the "
                    "published balances at its own two ends."
                )
        if (
            not walks
            and wants_walk
            and reported is not None
            and client is not None
            and client.available
        ):
            readable = self._readable(documents, availability)
            unit = inputs.units.get(spec.metric_id, "pct")
            first: list[tuple[str, int]] | None = None
            if not readable:
                located = "skipped: no document's PDF is on disk to render"
            elif rank_walk_candidates(readable, spec.metric_id, (*reported, unit)):
                located = "skipped: the ranking found pages that could hold the walk"
            else:
                first, located = self._page_search(
                    spec, documents, availability, client, budget, caveats, diagnostics
                )
            # For a metric that only prefers a walk (capital), one page is worth a look when
            # the identity can explain it anyway, and four are not: capital walks are drawn
            # over the half, and live runs on ANZ and CBA spent four vision calls each to have
            # every one quarantined. A metric that leads with the walk (the margin) keeps all
            # four: CBA's FY26 margin walk is on the second page read, and limiting it to one
            # put that answer back on the identity's arithmetic.
            leads_with_walk = registry.metrics[spec.metric_id].primary_path == "A"
            pages_worth_reading = 4 if leads_with_walk or not _identity_can_run(spec, inputs) else 1
            try:
                charted = self._charted_walks(
                    readable, spec, inputs, reported, client, budget, first, pages_worth_reading
                )
            except ConsentDenied as refusal:
                charted = None
                record(Step.EXTRACT, skipped=_refused(refusal))
                caveats.append(_REFUSED_CAVEAT.format(step="chart reading"))
                _declined(diagnostics, "chart reading")
        elif client is not None and client.available:
            located = (
                "skipped: the issuer's walk is written out in its text"
                if written is not None and written.walks
                else "skipped: the issuer's walk is in a table"
                if walks
                else "skipped: the identity explains this metric and can run"
                if not wants_walk
                else "skipped: the headline figures aren't established"
            )
        record(Step.LOCATE, outcome=located)
        if charted is not None and charted.refused:
            record(Step.EXTRACT, skipped=f"consent declined: {charted.refused}")
            caveats.append(_REFUSED_CAVEAT.format(step="chart reading"))
            _declined(diagnostics, "chart reading")
        if charted is not None:
            walks.extend(charted.walks)
            record(
                Step.EXTRACT,
                charted_walks=len(charted.walks),
                quarantined=len(charted.quarantined),
                mode="vision",
                pages=charted.notes,
            )
            # The reader gets one sentence about the charts, not one per page tried; the
            # page-by-page account is in the trace.
            if charted.walks:
                source = charted.walks[0].source
                caveats.append(
                    f"The drivers are the issuer's own walk, read from the chart on "
                    f"{source.doc_id} p.{source.page} and reconciled to the published "
                    "movement before use."
                )
            elif charted.readings:
                caveats.append(
                    f"No chart of the issuer's walk could be read and reconciled "
                    f"({len(charted.readings)} page(s) tried), so there is no issuer "
                    "attribution. The run's trace says what each page was."
                )

        if walks and client is not None and client.available:
            walks = self._map_walk_rows(walks, spec, client, budget, caveats, diagnostics)

        walk_evidence: dict[str, list[str]] = {}
        decomposition, chosen_walk = run_decomposition(spec, inputs, walks, walk_evidence)
        if chosen_walk is not None:
            walk_cards = cards_from_walk(chosen_walk, spec, start_at=len(cards) + 1)
            # Every bar's card, including the ones that land in "other". Those used to get
            # no citation at all, so a driver read straight off the issuer's walk was
            # scored as uncited.
            for step, card in zip(chosen_walk.steps, walk_cards, strict=False):
                key = step.driver_id or other_driver_id(spec.metric_id)
                walk_evidence.setdefault(key, []).append(card.evidence_id)
            decomposition, chosen_walk = run_decomposition(spec, inputs, walks, walk_evidence)
            cards = [*cards, *walk_cards]
        # Invariant 2: a computed driver cites the cards its inputs came from.
        _attach_computed_citations(decomposition, inputs, spec)
        used = {spec.metric_id, spec.measure}
        used.update(c for driver in decomposition.drivers for c in driver.from_components)
        restated = restatement_caveat(
            spec, inputs.restated_components, used, inputs.restatement_footnoted
        )
        if restated is not None:
            caveats.append(restated)
        record(
            Step.DECOMPOSE,
            path=decomposition.path,
            drivers=len(decomposition.drivers),
            residual=str(decomposition.residual),
        )

        # The issuer's own explanation of this movement, found by code. Shown as it stands
        # when there's no model, and given to the writer as the only source of a cause
        # when there is one.
        commentary = commentary_cards(
            spec,
            decomposition,
            inputs,
            documents,
            self._metas(availability),
            start_at=len(cards) + 1,
        )
        cards = [*cards, *commentary]
        record(Step.EXTRACT, commentary=len(commentary), mode="commentary")

        # Step 9: Explain. The arithmetic is already done and isn't the model's to change;
        # it writes the narrative over the decomposition it's given.
        narrative = ""
        causes: dict[str, tuple[str, list[str]]] = {}
        if client is None or not client.available:
            record(Step.EXPLAIN, skipped="no model configured")
            diagnostics.warnings.append(
                "No model configured: drivers come from the issuer's disclosed walk and the "
                "accounting identity, and the narrative is omitted rather than written without "
                "evidence."
            )
        elif inputs.pair(spec.metric_id) is None:
            # With the headline not established the run abstains, and an abstention shows
            # no memo. Macquarie's first live run paid for one and its check all the same.
            record(Step.EXPLAIN, skipped="the headline figures aren't established")
        else:
            try:
                explain_model, explain_effort = registry.route("explain")
                draft = write_memo(
                    spec,
                    decomposition,
                    inputs,
                    cards,
                    client,
                    budget=budget,
                    model=explain_model,
                    effort=explain_effort,
                    tracer=self._tracer,
                )
                narrative, causes, explain_note = draft.narrative, draft.causes, draft.note
            except ConsentDenied as refusal:
                narrative, explain_note = "", _refused(refusal)
                caveats.append(_REFUSED_CAVEAT.format(step="the narrative"))
                _declined(diagnostics, "the narrative")
            if explain_note.startswith("consent declined"):
                record(Step.EXPLAIN, skipped=explain_note)
            else:
                record(Step.EXPLAIN, chars=len(narrative), causes=len(causes), note=explain_note)
            if explain_note:
                diagnostics.warnings.append(explain_note)

        # Step 10: Score.
        scored = score_run(spec, decomposition, inputs, facts=facts, cards=cards)
        record(Step.SCORE, overall=str(scored.overall.score), level=scored.overall.level)

        # Step 11: Verify. The entailment check needs a model. The code checks that every
        # cited evidence id exists, which doesn't.
        missing = [
            evidence_id
            for driver in decomposition.drivers
            for evidence_id in driver.evidence_ids
            if evidence_id not in {card.evidence_id for card in cards}
        ]
        if missing:
            diagnostics.warnings.append(
                f"{len(missing)} driver citation(s) referenced an evidence id that does "
                "not exist and were dropped."
            )
        entailment = "skipped, no model configured"
        if narrative and client is not None and client.available:
            # The narrative's own cited sentences, judged against the cards and the pages
            # they cite, in one call. The judge isn't shown any reasoning, because a verifier
            # that sees the agent's working tends to agree with it.
            pages = {
                (card.source.doc_id, card.source.page): page.text
                for card in cards
                if (parsed := documents.get(card.source.doc_id)) is not None
                and (page := parsed.page(card.source.page)) is not None
            }
            verify_model, verify_effort = registry.route("verify")
            explain_model, explain_effort = registry.route("explain")
            labels = {d.driver_id: d.label or d.driver_id for d in decomposition.drivers}
            computed = computed_lines(
                decomposition,
                amount_places(*(inputs.pair(spec.metric_id) or ())),
                movement=movement_line(spec, decomposition, inputs),
            )
            removed: list[Any] = []
            final_removed: list[Any] = []
            causes_removed: list[Any] = []
            try:
                checked = check_narrative(
                    narrative,
                    cards,
                    pages,
                    client,
                    budget=budget,
                    model=verify_model,
                    effort=verify_effort,
                    tracer=self._tracer,
                    causes=causes,
                    labels=labels,
                    computed=computed,
                )
                causes = checked.causes
                removed = [*checked.unsupported, *checked.partial]
                final_removed = list(removed)
                causes_removed = list(checked.causes_removed)
                if (removed or causes_removed) and not checked.skipped:
                    # The one repair allowed: the writer is told what was removed and
                    # why, writes the memo again, and the rewrite is checked once more.
                    # Anything still unsupported is removed with no further rewrite. Causes
                    # that passed are kept as they were; a removed cause may come back
                    # narrower, and is judged again like everything else in the rewrite.
                    retried = {v.driver_id for v in causes_removed if v.driver_id}
                    redraft = write_memo(
                        spec,
                        decomposition,
                        inputs,
                        cards,
                        client,
                        budget=budget,
                        model=explain_model,
                        effort=explain_effort,
                        tracer=self._tracer,
                        repair=[(v.claim, v.reason) for v in removed],
                        repair_causes=[
                            (v.driver_id, v.claim, v.reason) for v in causes_removed if v.driver_id
                        ],
                    )
                    narrowed = {d: c for d, c in redraft.causes.items() if d in retried}
                    # A narrative that lost nothing isn't redrafted: the rewrite was for the
                    # causes, and swapping a checked narrative for an unchecked one buys
                    # nothing.
                    text = redraft.narrative if removed else ""
                    if text or narrowed:
                        rechecked = check_narrative(
                            text,
                            cards,
                            pages,
                            client,
                            budget=budget,
                            model=verify_model,
                            effort=verify_effort,
                            tracer=self._tracer,
                            causes=narrowed,
                            labels=labels,
                            computed=computed,
                        )
                        if not rechecked.skipped:
                            causes = {**causes, **rechecked.causes}
                            causes_removed = [
                                *(v for v in causes_removed if v.driver_id not in rechecked.causes),
                                *rechecked.causes_removed,
                            ]
                            if text:
                                checked = rechecked
                                final_removed = [*rechecked.unsupported, *rechecked.partial]
                                removed = [*removed, *final_removed]
            except ConsentDenied as refusal:
                checked = None
                entailment = _refused(refusal)
                caveats.append(_REFUSED_CAVEAT.format(step="the check of the narrative"))
                _declined(diagnostics, "the check of the narrative")
            if checked is not None:
                narrative = checked.narrative
                kept = len(checked.verdicts) - len(checked.unsupported) - len(checked.partial)
                entailment = (
                    f"{len(checked.verdicts)} cited sentence(s) in the final narrative judged, "
                    f"{kept} supported; {len(removed)} removed across the draft and any rewrite"
                )
                if checked.skipped:
                    entailment = f"not checked: {checked.skipped}"
                    if checked.skipped != "no sentence cites any evidence":
                        caveats.append(
                            "The narrative's claims could not be checked against their "
                            f"evidence ({checked.skipped}), so they are shown as written."
                        )
                # A reader is told what the memo they're reading leaves out. A draft sentence
                # the rewrite then said correctly isn't news to them: CBA's FY24 memo listed
                # "Cash NPAT fell A$236m (2.3%)" as left out above a narrative that says it.
                # Those go to the trace.
                for verdict in removed:
                    how = (
                        "does not support"
                        if verdict.verdict == "unsupported"
                        else "only partly supports"
                    )
                    line = (
                        f"Left out of the narrative, because its evidence {how} it: "
                        f'"{verdict.claim}" ({verdict.reason})'
                    )
                    if verdict in final_removed:
                        caveats.append(line)
                    else:
                        diagnostics.warnings.append(f"{line}. The rewrite replaced it.")
                for verdict in causes_removed:
                    diagnostics.warnings.append(
                        f"A stated cause for {labels.get(verdict.driver_id or '', verdict.driver_id)} "
                        f'was left out, because its evidence does not support it: "{verdict.claim}" '
                        f"({verdict.reason})"
                    )
                if not narrative.strip():
                    diagnostics.warnings.append(
                        "No sentence of the narrative was supported by its evidence, so it was "
                        "removed in full."
                    )
                narrative, named = name_every_material_driver(
                    narrative,
                    decomposition,
                    amount_places(*(inputs.pair(spec.metric_id) or ())),
                )
                if named:
                    caveats.append(
                        "Stated by figure alone, with no cause, because the checked narrative "
                        "no longer named "
                        + (
                            named[0]
                            if len(named) == 1
                            else ", ".join(named[:-1]) + " or " + named[-1]
                        )
                        + "."
                    )
        record(
            Step.VERIFY,
            numeric_citations_checked=len(cards),
            dangling=len(missing),
            entailment=entailment,
        )

        # An abstention always carries a reason code and says what was missing. Without both
        # reported figures there's no movement to explain, and returning a result with no
        # status reason would be an abstention that doesn't say it abstained.
        if inputs.pair(spec.metric_id) is None:
            _record_spend(diagnostics, budget, self._tracer)
            diagnostics.latency_ms = int((time.perf_counter() - started) * 1000)
            available = sorted(set(inputs.current) | set(inputs.prior))
            record(Step.RENDER, status="ABSTAIN", reason="INSUFFICIENT_EVIDENCE")
            return self._finish(
                abstention_result(
                    spec,
                    "INSUFFICIENT_EVIDENCE",
                    f"{spec.metric_id} could not be established for both {spec.period} and "
                    f"{spec.comparison_period} from the available disclosures. "
                    + (
                        f"Components found: {', '.join(available)}."
                        if available
                        else "No components were extracted."
                    ),
                    caveats=[*caveats, *inputs.caveats, *diagnostics.warnings],
                    diagnostics=diagnostics,
                ),
                trace,
            )

        self._doing("Putting the answer together")
        _record_spend(diagnostics, budget, self._tracer)
        diagnostics.latency_ms = int((time.perf_counter() - started) * 1000)
        result = build_result(
            spec,
            inputs,
            decomposition,
            scored,
            cards,
            caveats,
            diagnostics,
            narrative,
            causes=causes,
            commentary=[card.evidence_id for card in commentary],
        )
        record(Step.RENDER, status=result.status)
        return self._finish(result, trace)

    # -- helpers ----------------------------------------------------------

    def _load_documents(self, availability) -> dict[str, ParsedDocument]:  # type: ignore[no-untyped-def]
        """Read the processed artefacts. A query never parses a PDF."""
        found: dict[str, ParsedDocument] = {}
        for meta in (*availability.current_docs, *availability.comparison_docs):
            document = read_processed(meta.doc_id, self.settings.processed_dir)
            if document is not None:
                found[meta.doc_id] = document
        return found

    def _discover(
        self,
        spec: RequestSpec,
        client: LLMClient,
        budget: Budget,
        caveats: list[str],
        diagnostics: Diagnostics,
        doc_type: DocType = "RESULTS_DOCUMENT",
        search: bool = True,
    ) -> Any:
        """The cold path: find, check, fetch and ingest the period's results document, or
        another of its documents. ``search=False`` reads the ASX's list and nothing else."""
        from .steps.acquire import Acquired, acquire_documents

        model, effort = get_registry().route("discover_classify")
        try:
            return acquire_documents(
                spec.bank_id,
                spec.period,
                spec.as_of,
                client,
                self.store,
                self.settings,
                budget=budget,
                tracer=self._tracer,
                model=model,
                effort=effort,
                doc_type=doc_type,
                asx=self.asx,
                doing=self._doing,
                search=search,
            )
        except ConsentDenied as refusal:
            caveats.append(_REFUSED_CAVEAT.format(step="finding the filing"))
            _declined(diagnostics, "finding the filing")
            return Acquired(notes=[_refused(refusal)])
        except Exception as error:  # a download or a parse that fails is an abstention
            if isinstance(error, ModelUnreachable):  # the run stops; see llm.client
                raise
            return Acquired(notes=[f"the cold path failed: {type(error).__name__}: {error}"])

    def _readable(
        self, documents: dict[str, ParsedDocument], availability: Any
    ) -> list[tuple[ParsedDocument, Any, Path]]:
        """The documents whose PDF is on disk, so a page can be rendered for a vision call."""
        metas = self._metas(availability)
        readable = []
        for doc_id, document in documents.items():
            meta = metas.get(doc_id)
            raw = self.settings.raw_dir / f"{document.sha256}.pdf"
            if meta is not None and raw.exists():
                readable.append((document, meta, raw))
        return readable

    def _page_search(
        self,
        spec: RequestSpec,
        documents: dict[str, ParsedDocument],
        availability: Any,
        client: LLMClient,
        budget: Budget,
        caveats: list[str],
        diagnostics: Diagnostics,
    ) -> tuple[list[tuple[str, int]], str]:
        """The locate loop, for when the ranking can't place the walk: the pages it pointed
        at, charts it read first, and what happened."""
        try:
            locate_model, locate_effort = get_registry().route("locate")
            located = run_locate(
                spec,
                Toolkit(spec=spec, documents=documents, metas=self._metas(availability)),
                client,
                budget=budget,
                model=locate_model,
                effort=locate_effort,
                tracer=self._tracer,
            )
        except ConsentDenied as refusal:
            caveats.append(_REFUSED_CAVEAT.format(step="page location"))
            _declined(diagnostics, "page location")
            return [], _refused(refusal)
        diagnostics.warnings.extend(located.notes)
        return located.pages(), (
            f"ran: {located.stopped} after {located.turns} turn(s), "
            f"{len(located.pages())} page(s) to read first"
        )

    def _written_walks(
        self,
        spec: RequestSpec,
        documents: dict[str, ParsedDocument],
        availability: Any,
        inputs: Any,
        reported: tuple[Decimal, Decimal],
        client: LLMClient,
        budget: Budget,
    ) -> Any:
        """The issuer's walk as it wrote it out, read and checked."""
        from ..ingestion.text_walks import TextWalkOutcome, read_text_walk

        registry = get_registry()
        bank = registry.banks.get(spec.bank_id)
        metas = self._metas(availability)
        held = [
            (document, metas[doc_id]) for doc_id, document in documents.items() if doc_id in metas
        ]
        if bank is None or not held:
            return TextWalkOutcome(notes=["no document of the period is held to read a walk from"])
        prior, current = reported
        model, effort = registry.route("text_walk")
        try:
            return read_text_walk(
                held,
                spec,
                fy_end_month=bank.fy_end_month,
                reported_opening=prior,
                reported_closing=current,
                unit=inputs.units.get(spec.metric_id, "pct"),
                client=client,
                budget=budget,
                model=model,
                effort=effort,
                tracer=self._tracer,
            )
        except ConsentDenied as refusal:
            return TextWalkOutcome(notes=[_refused(refusal)], refused=str(refusal))

    def _written_legs(
        self,
        spec: RequestSpec,
        inputs: Any,
        reported: tuple[Decimal, Decimal],
        client: LLMClient,
        budget: Budget,
    ) -> Any:
        """The issuer's walks for the two halves of a year-on-year movement in a balance.

        The later half is read first, from the question's own pack, and the earlier one only
        if that succeeds: one leg can't be chained, so a call after a failed leg would buy
        nothing. Each leg is looked for as written text first and then as a chart. The
        balance between the halves comes from the fact store, the same as the two ends, so
        the legs meet exactly or not at all.
        """
        from ..ingestion.text_walks import TextWalkOutcome, half_year_legs, read_text_walk

        registry = get_registry()
        bank = registry.banks.get(spec.bank_id)
        outcome = TextWalkOutcome()
        legs = half_year_legs(spec, bank.fy_end_month) if bank is not None else []
        if bank is None or not legs:
            outcome.notes.append("the movement doesn't run over two halves, so there's no chain")
            return outcome
        middle = self._balance_at(spec, legs[0].request.period, bank.fy_end_month)
        if middle is None:
            outcome.notes.append(
                f"no {spec.metric_id} is held for {legs[0].closing}, where the halves meet"
            )
            return outcome
        ends = {
            legs[0].closing: middle,
            spec.comparison_period: reported[0],
            spec.period: reported[1],
        }
        model, effort = registry.route("text_walk")
        chart_model, chart_effort = registry.route("chart_read")
        unit = inputs.units.get(spec.metric_id, "pct")
        for leg in reversed(legs):
            held = self._documents_covering(spec, leg.request.period)
            if not held:
                outcome.notes.append(f"no results document for {leg.request.period} is held")
                return outcome
            opening, closing = ends[leg.opening], ends[leg.closing]
            try:
                read = read_text_walk(
                    held,
                    leg.request,
                    fy_end_month=bank.fy_end_month,
                    reported_opening=opening,
                    reported_closing=closing,
                    unit=unit,
                    client=client,
                    budget=budget,
                    model=model,
                    effort=effort,
                    tracer=self._tracer,
                    labels=(leg.opening, leg.closing),
                )
                outcome.notes.extend(read.notes)
                outcome.pages.extend(read.pages)
                found = list(read.walks)
                # Westpac, NAB and CBA draw their capital walks; ANZ writes them out. The
                # chart is read only where the text gave nothing, and two pages at most.
                readable = [
                    (document, meta, raw)
                    for document, meta in held
                    if (raw := self.settings.raw_dir / f"{document.sha256}.pdf").exists()
                ]
                if not found and readable:
                    from ..ingestion.chart_walks import chart_walks_across

                    charted = chart_walks_across(
                        readable,
                        spec.metric_id,
                        reported_opening=opening,
                        reported_closing=closing,
                        unit=unit,
                        opening_period=leg.opening,
                        closing_period=leg.closing,
                        client=client,
                        budget=budget,
                        model=chart_model,
                        effort=chart_effort,
                        tracer=self._tracer,
                        max_pages=2,
                        period_ends=(leg.request.comparison_end, leg.request.period_end),
                    )
                    outcome.notes.extend(charted.notes)
                    if charted.refused:
                        outcome.refused = charted.refused
                    found = list(charted.walks)
            except ConsentDenied as refusal:
                outcome.notes.append(_refused(refusal))
                outcome.refused = str(refusal)
                return outcome
            if not found:
                return outcome
            outcome.walks.extend(found[:1])
        return outcome

    def _balance_at(self, spec: RequestSpec, half: str, fy_end_month: int) -> Decimal | None:
        """The metric's balance at the end of ``half``, chosen as the question's own are.

        A pack files its second half's closing balance under the full year, so 2H25 is looked
        for as FY25 first.
        """
        from ..ingestion.text_walks import leg_request

        labels = [f"FY{half[2:]}", half] if half.upper().startswith("2H") else [half]
        for label in labels:
            scoped = leg_request(spec, label, label, fy_end_month)
            facts = self.store.get_facts(scoped, components=[spec.metric_id])
            if not facts:
                continue
            resolved = reconcile_inputs(cards_from_facts(facts, scoped, {}), facts, scoped)
            value = resolved.current.get(spec.metric_id) or resolved.prior.get(spec.metric_id)
            if value is not None:
                return Decimal(value)
        return None

    def _documents_covering(self, spec: RequestSpec, half: str) -> list[tuple[ParsedDocument, Any]]:
        """Results documents for this half, published by the as-of date: its own, or for a
        second half the full year's."""
        labels = [half, f"FY{half[2:]}"] if half.upper().startswith("2H") else [half]
        held: list[tuple[ParsedDocument, Any]] = []
        for meta in self.store.documents_for(spec.bank_id, labels, spec.as_of):
            document = read_processed(meta.doc_id, self.settings.processed_dir)
            if document is not None:
                held.append((document, meta))
        return held

    def _charted_walks(
        self,
        readable: list[tuple[ParsedDocument, Any, Path]],
        spec: RequestSpec,
        inputs: Any,
        reported: tuple[Decimal, Decimal],
        client: LLMClient,
        budget: Budget,
        first: list[tuple[str, int]] | None = None,
        max_pages: int = 4,
    ) -> Any:
        """Walks read from charts across every document this run may read."""
        from ..ingestion.chart_walks import chart_walks_across

        prior, current = reported
        chart_model, chart_effort = get_registry().route("chart_read")
        return chart_walks_across(
            readable,
            spec.metric_id,
            reported_opening=prior,
            reported_closing=current,
            unit=inputs.units.get(spec.metric_id, "pct"),
            opening_period=spec.comparison_period,
            closing_period=spec.period,
            client=client,
            budget=budget,
            model=chart_model,
            effort=chart_effort,
            tracer=self._tracer,
            first=first,
            max_pages=max_pages,
        )

    def _map_walk_rows(
        self,
        walks: list[Any],
        spec: RequestSpec,
        client: LLMClient,
        budget: Budget,
        caveats: list[str],
        diagnostics: Diagnostics,
    ) -> list[Any]:
        """Walks with the rows the synonym list couldn't map put to the model, one call each.

        Each mapping it makes is stated, because a reader should know which drivers were
        named by a model rather than by the issuer's wording matching the taxonomy.
        """
        from dataclasses import replace

        from ..decomposition.taxonomy import display_label
        from .steps.taxonomy_llm import map_labels_with_model

        model, effort = get_registry().route("taxonomy_classify")
        out = []
        for walk in walks:
            leftover = [step.label for step in walk.steps if step.driver_id is None and step.label]
            if not leftover:
                out.append(walk)
                continue
            try:
                outcome = map_labels_with_model(
                    spec.metric_id,
                    leftover,
                    client,
                    budget=budget,
                    model=model,
                    effort=effort,
                    tracer=self._tracer,
                    context=[step.label for step in walk.steps],
                )
            except ConsentDenied:
                caveats.append(_REFUSED_CAVEAT.format(step="mapping the walk's unnamed bars"))
                _declined(diagnostics, "mapping the walk's unnamed bars")
                out.append(walk)
                continue
            for label, driver_id in outcome.from_model.items():
                caveats.append(
                    f'The walk\'s "{label}" bar was matched to '
                    f"{display_label(spec.metric_id, driver_id).lower()} by the model, not by "
                    "the issuer's wording."
                )
            out.append(
                replace(
                    walk,
                    steps=[
                        replace(step, driver_id=outcome.from_model[step.label])
                        if step.driver_id is None and step.label in outcome.from_model
                        else step
                        for step in walk.steps
                    ],
                )
            )
        return out

    def _metas(self, availability) -> dict[str, Any]:  # type: ignore[no-untyped-def]
        """The document metadata behind the loaded artefacts, keyed by id."""
        return {
            meta.doc_id: meta
            for meta in (*availability.current_docs, *availability.comparison_docs)
        }

    def estimate(self, bank: str, period: str, metric: str, as_of: date | None = None) -> Estimate:
        """What answering this will take, worked out without a model call or a download.

        Said to a person before the run, so they know whether the question is inside the
        saved data or needs external calls, and what it will cost, before agreeing to it.
        """
        asked = f"{bank} {period} {metric}"
        estimate = Estimate(asked=asked)
        self._recall_found_banks()
        try:
            spec = resolve_request(bank, period, metric, as_of=as_of)
        except AbstainSignal as signal:
            if signal.reason != "BANK_UNKNOWN":
                estimate.stop = signal.detail
                return estimate
            match = self._find(bank)
            if match.kind == "brand":
                estimate.notes.append(match.note)
                try:
                    spec = resolve_request(match.code or bank, period, metric, as_of=as_of)
                except AbstainSignal as owner_signal:
                    if owner_signal.reason != "BANK_UNKNOWN":
                        estimate.stop = owner_signal.detail
                        return estimate
                    match = BankMatch("listed_bank", bank, code=match.code, name=match.name)
            if match.kind == "listed_bank":
                try:
                    metric_name = get_registry().metric(metric).display_name
                except Exception:
                    metric_name = metric
                estimate.heading = f"{match.name} ({match.code}) · {metric_name} · {period}"
                estimate.path = "new_bank"
                estimate.walk_led = True
                estimate.notes.append(
                    f"{match.name} is listed on the ASX as a bank, but this tool hasn't read "
                    "it before."
                )
                estimate.steps = [
                    "Find its financial year from its announcements on the ASX.",
                    f"Search the ASX for its {period} results and check it's the right filing.",
                    "Download it, read its tables and check every figure against its "
                    "accounting identities. A model matches any row it names its own way.",
                    "If the filing can't split the movement, find its results presentation "
                    "for the bank's own walk.",
                    "Work out the drivers with exact arithmetic, read its own explanation, "
                    "write the answer and check each claim against its page.",
                ]
                return estimate
            if match.kind != "brand":
                estimate.stop = match.message
                return estimate
        registry = get_registry()
        profile = registry.bank(spec.bank_id)
        definition = registry.metric(spec.metric_id)
        if profile.discovered:
            # In no config file, but met before. Without this line the estimate reads like a
            # configured bank's, and the missing search of the ASX looks like a mistake.
            estimate.notes.append(
                f"{profile.name} isn't in this tool's config. It was found on the ASX in an "
                "earlier run and remembered with its year end and the names it gives its rows, "
                "so the bank itself doesn't need looking up again."
            )
        measure = definition.display_name
        if spec.metric_id == "headline_earnings" and spec.headline_label:
            measure = spec.headline_label  # the bank's own: Judo's profit before tax
        estimate.heading = (
            f"{profile.name} · {measure} · {spec.period} against {spec.comparison_period}"
        )
        estimate.walk_led = definition.primary_path in {"A", "A_else_B"}
        availability = check_availability(spec, self.store)
        if (
            availability.reportable
            and profile.discovered
            and self._needs_the_deck(spec, availability)
        ):
            # A found bank whose filing can't split the movement: its slide pack is fetched
            # during the run, which is a cold start's work and cost, not a saved answer's.
            estimate.path = "fetch"
            estimate.held = [doc.doc_id for doc in availability.current_docs]
            estimate.steps = [
                "Read the figures for both periods from the saved filings.",
                f"The saved filings can't split the movement: find {profile.name}'s results "
                "presentation on the ASX and read the bank's own walk from it.",
                "Work out the drivers with exact arithmetic, read the bank's own explanation, "
                "write the answer and check each claim against its page.",
            ]
        elif availability.reportable:
            estimate.path = "saved"
            estimate.held = [doc.doc_id for doc in availability.current_docs] or ["its filings"]
            estimate.steps = [
                "Read the figures for both periods from the saved filings.",
                "Work out the drivers with exact arithmetic"
                + (
                    ", reading the bank's own walk where it publishes one."
                    if estimate.walk_led
                    else "."
                ),
                "Read the bank's own explanation, write the answer and check each claim "
                "against its page.",
            ]
        elif availability.release_pending:
            estimate.path = "pending"
            estimate.notes.append(availability.detail)
            estimate.steps = [
                f"Check the ASX's list of {profile.name}'s announcements for its {spec.period} "
                "result. If it isn't there, the run stops and says it hasn't been released.",
                "If it is: download it, read its tables and check every figure against its "
                "accounting identities.",
                "Work out the drivers with exact arithmetic, read the bank's own explanation, "
                "write the answer and check each claim against its page.",
            ]
        elif availability.reason == "DOCS_UNAVAILABLE":
            estimate.path = "fetch"
            estimate.steps = [
                f"Search {profile.name}'s site and the ASX for its {spec.period} results and "
                "check it's the right filing.",
                "Download it, read its tables and check every figure against its accounting "
                "identities.",
                "Work out the drivers with exact arithmetic, read the bank's own explanation, "
                "write the answer and check each claim against its page.",
            ]
        else:
            estimate.stop = availability.detail
        return estimate

    def _find(self, bank: str) -> BankMatch:
        known = {
            bank_id: profile.name
            for bank_id, profile in get_registry().banks.items()
            if not profile.synthetic
        }
        return find_bank(
            bank, config_dir=self.settings.config_dir, data_dir=self.settings.data_dir, known=known
        )

    def _recall_found_banks(self) -> None:
        """Register the banks found on the ASX in earlier runs and confirmed by a filing."""
        registry = get_registry()
        for row in self.store.discovered_banks():
            bank_id = str(row["bank_id"])
            if bank_id in registry.banks and not registry.banks[bank_id].discovered:
                continue
            _register(
                _found_profile(
                    bank_id,
                    str(row["name"]),
                    int(row["fy_end_month"]),  # type: ignore[call-overload]
                    tuple(row["ir_domains"]),  # type: ignore[arg-type]
                    tuple(row["aliases"]),  # type: ignore[arg-type]
                    tuple(row["row_labels"]),  # type: ignore[arg-type]
                )
            )

    def _resolve(
        self, bank: str, period: str, metric: str, as_of: date | None, *, budget: Budget
    ) -> tuple[RequestSpec | None, AbstainSignal | None, list[str]]:
        """The request resolved, with a bank that isn't configured found on the way: by a
        brand, from an earlier run, or from its ASX listing. The notes say how."""
        self._recall_found_banks()
        try:
            return resolve_request(bank, period, metric, as_of=as_of), None, []
        except AbstainSignal as signal:
            if signal.reason != "BANK_UNKNOWN":
                return None, signal, []
        match = self._find(bank)
        notes: list[str] = []
        if match.kind == "brand":
            notes.append(match.note)
            try:
                return resolve_request(match.code or bank, period, metric, as_of=as_of), None, notes
            except AbstainSignal as signal:
                if signal.reason != "BANK_UNKNOWN":
                    return None, signal, notes
                match = BankMatch("listed_bank", bank, code=match.code, name=match.name)
        if match.kind != "listed_bank" or match.code is None or match.name is None:
            return None, AbstainSignal("BANK_UNKNOWN", match.message), notes
        client = self._model()
        if client is None or not client.available:
            return (
                None,
                AbstainSignal(
                    "BANK_UNKNOWN",
                    f"{match.name} ({match.code}) is listed on the ASX as a bank, but this tool "
                    "hasn't read it before. Finding and reading its filings needs the model: "
                    "run the question again with the API key, through scripts/with-key.",
                ),
                notes,
            )
        model, effort = get_registry().route("identify_bank")
        budget.go_cold(get_registry().budgets)
        try:
            year = find_year_end(
                match.name,
                match.code,
                client,
                model=model,
                effort=effort,
                budget=budget,
                tracer=self._tracer,
            )
        except ConsentDenied as refusal:
            return None, AbstainSignal("BANK_UNKNOWN", _refused(refusal)), notes
        if year.fy_end_month is None:
            return (
                None,
                AbstainSignal(
                    "BANK_UNKNOWN",
                    f"{match.name} ({match.code}) is listed on the ASX as a bank, but its "
                    "financial year couldn't be settled from its announcements: "
                    + "; ".join(year.notes),
                ),
                notes,
            )
        domains = tuple(d for d in (year.website, ASX_DOMAIN) if d)
        _register(_found_profile(match.code, match.name, year.fy_end_month, domains, (bank,)))
        self._found[match.code] = bank
        notes.append(
            f"{match.name} ({match.code}) isn't one of the banks configured here. It was found "
            f"on the ASX, with a financial year ending in {_MONTHS[year.fy_end_month - 1]}, "
            "and the filing read for this answer had to state a period ending there."
        )
        try:
            return resolve_request(match.code, period, metric, as_of=as_of), None, notes
        except AbstainSignal as signal:
            return None, signal, notes

    def _remember(self, spec: RequestSpec, doc_id: str) -> None:
        """Keep a bank found on the ASX, now that a filing has confirmed its year."""
        asked = self._found.pop(spec.bank_id, None)
        if asked is None:
            return
        profile = get_registry().bank(spec.bank_id)
        self.store.put_discovered_bank(
            profile.bank_id,
            profile.name,
            profile.fy_end_month,
            profile.ir_domains,
            {*profile.aliases, asked},
            confirmed_by=doc_id,
            discovered_at=date.today(),
            row_labels=profile.row_labels,
        )

    def _needs_the_deck(self, spec: RequestSpec, availability: Any) -> bool:
        """Whether the identity can't run on what's held and no slide pack is held yet."""
        from ..decomposition.identities import identity_for

        if any(doc.doc_type == "INVESTOR_PRESENTATION" for doc in availability.current_docs):
            return False
        identity = identity_for(spec.metric_id, pre_tax=spec.pre_tax)
        if identity is None:
            return True
        held = {(fact.component, fact.period) for fact in self.store.get_facts(spec)}
        return not all(
            (component, period) in held
            for component in (spec.measure, *identity.components)
            for period in (spec.period, spec.comparison_period)
        )

    def _learn_rows(
        self,
        spec: RequestSpec,
        availability: Any,
        budget: Budget,
        caveats: list[str],
        diagnostics: Diagnostics,
    ) -> list[str]:
        """The figures this question needs that a discovered bank's filings give under names
        of their own (DESIGN.md ADR-063).

        Each filing in turn, for whatever is still missing: a model is shown the rows nothing
        matched and says which holds each missing figure, and the filing is read again with
        those names, every figure checked as usual. A name is kept with the bank only if a
        figure it gave passed and the bank lost none it already had.
        """
        from ..decomposition.identities import identity_for

        profile = get_registry().bank(spec.bank_id)
        identity = identity_for(spec.metric_id, pre_tax=spec.pre_tax)
        needed = dict.fromkeys((spec.measure, *(identity.components if identity else ())))
        if identity is not None and identity.kind == "sum":
            # A bridge only closes with every line it has, and a smaller bank's one-off line
            # is easy to miss: BNK's FY26 loss includes a A$3.5m goodwill impairment.
            needed.update(
                dict.fromkeys(c for c in identity.optional_components if c in _BRIDGE_LINES)
            )
        periods = (spec.period, spec.comparison_period)

        def held() -> set[tuple[str, str]]:
            return {(fact.component, fact.period) for fact in self.store.get_facts(spec)}

        client = self._model()
        if client is None or not client.available:
            return []
        step = "matching the bank's own row names"
        sought = self.store.rows_sought(profile.bank_id)
        matched: dict[str, str] = {}
        for meta in availability.current_docs:
            before = held()
            missing = [c for c in needed if any((c, period) not in before for period in periods)]
            if not missing:
                break
            document = read_processed(meta.doc_id, self.settings.processed_dir)
            raw = self.settings.raw_dir / f"{meta.sha256}.pdf"
            # A figure already looked for in this filing isn't there to find: BNK publishes
            # no average interest-earning assets, and asking again cost a call every run.
            unsought = [c for c in missing if f"{meta.doc_id} {c}" not in sought]
            if document is None or not raw.exists() or not unsought:
                continue
            rows = offer_rows(
                document,
                meta,
                profile.fy_end_month,
                periods,
                line_item_index(spec.bank_id),
                unsought,
            )
            model, effort = get_registry().route("map_rows")
            try:
                mapping = map_rows(
                    unsought,
                    rows,
                    client,
                    model=model,
                    effort=effort,
                    bank_name=profile.name,
                    budget=budget,
                    tracer=self._tracer,
                )
            except ConsentDenied:
                caveats.append(_REFUSED_CAVEAT.format(step=step))
                _declined(diagnostics, step)
                break
            diagnostics.warnings.extend(mapping.notes)
            if mapping.settled:
                self.store.put_rows_sought(
                    profile.bank_id, (f"{meta.doc_id} {c}" for c in unsought)
                )
            if mapping.labels:
                kept = self._keep_what_passes(spec, meta, raw, mapping.labels, held, diagnostics)
                matched.update(kept)
                profile = get_registry().bank(spec.bank_id)
        if not matched:
            return []
        self.store.put_row_labels(profile.bank_id, profile.row_labels)
        caveats.append(
            f"{profile.name} names some rows in its own words. A model matched "
            + "; ".join(f'"{label}" as {component}' for label, component in matched.items())
            + ", and the figures read from them passed the same checks as every other."
        )
        return [f"{label} as {component}" for label, component in matched.items()]

    def _keep_what_passes(
        self,
        spec: RequestSpec,
        meta: Any,
        raw: Path,
        labels: dict[str, str],
        held: Any,
        diagnostics: Diagnostics,
    ) -> dict[str, str]:
        """The matched names worth keeping: every one at once when that reads cleanly, which
        is one more read of the filing; otherwise one at a time, round again while any new one
        passes, so a line that needs the others in place (BNK's NPAT closes its bridge only
        once its tax and provision are read) gets its turn after them."""
        profile = get_registry().bank(spec.bank_id)
        before = held()

        def read_with(names: dict[str, str]) -> set[tuple[str, str]]:
            _register(replace(profile, row_labels=_merged(profile.row_labels, names)))
            ingest_document(raw, meta, self.store, processed_root=self.settings.processed_dir)
            return set(held())

        def gained(
            after: set[tuple[str, str]], current: set[tuple[str, str]], component: str
        ) -> bool:
            return any(c == component for c, _ in after - current)

        after = read_with(labels)
        if not before - after and all(gained(after, before, c) for c in labels.values()):
            return dict(labels)
        kept: dict[str, str] = {}
        current = before
        pending = dict(labels)
        last: dict[str, str] = dict(labels)
        progress = True
        while pending and progress:
            progress = False
            for label, component in list(pending.items()):
                trial = {**kept, label: component}
                after = read_with(trial)
                last = trial
                if current - after or not gained(after, current, component):
                    continue
                kept, current = trial, after
                del pending[label]
                progress = True
        for label, component in pending.items():
            diagnostics.warnings.append(
                f"The row a model matched as {component} ({label!r}) gave no figure that "
                "passed its checks without costing one the filing already gave, so it wasn't kept."
            )
        if last != kept:
            read_with(kept)
        _register(replace(profile, row_labels=_merged(profile.row_labels, kept)))
        return kept

    def _abstain_without_spec(
        self,
        bank: str,
        period: str,
        metric: str,
        as_of: date | None,
        signal: AbstainSignal,
        trace: list[dict[str, Any]],
    ) -> RunOutcome:
        """Abstain before a RequestSpec could be built.

        Resolution failed, so there's no resolved period or metric to report. The request is
        echoed back as given, so the caller can see what was rejected.
        """
        import uuid

        spec = RequestSpec(
            run_id=uuid.uuid4().hex[:12],
            bank_id=bank.strip().upper()[:12] or "UNKNOWN",
            metric_id="nim",
            period=period.strip().upper()[:12] or "UNKNOWN",
            comparison_period="UNKNOWN",
            period_start=as_of or date.today(),
            period_end=as_of or date.today(),
            comparison_start=as_of or date.today(),
            comparison_end=as_of or date.today(),
            as_of=as_of or date.today(),
            asked_as=f"{bank.strip()} / {metric.strip()}",
        )
        trace.append({"step": Step.RESOLVE.value, "abstain": signal.reason})
        return self._finish(
            abstention_result(
                spec,
                signal.reason,
                signal.detail,
                caveats=[
                    f"Requested: bank={bank!r} period={period!r} metric={metric!r}. "
                    "Resolution failed, so no period or metric was established."
                ],
            ),
            trace,
        )

    def _finish(self, result: ResearchResult, trace: list[dict[str, Any]]) -> RunOutcome:
        # Each model call goes into the trace next to the step that made it: model, effort,
        # request hash, prompt hash, tokens and stop reason. Without them a trace said which
        # steps ran but not what any of them cost or which prompt produced the answer.
        if self._tracer is not None and self._tracer.events:
            calls = [{"event": "model_call", **event} for event in self._tracer.events]
            trace = sorted([*trace, *calls], key=lambda event: float(event.get("at", 0.0)))
        result = result.model_copy(update={"trail": _trail(self._tracer, trace)})
        artefacts = write_run(result, self.settings.runs_dir, trace) if self.persist else None
        return RunOutcome(result=result, artefacts=artefacts, trace=trace)


def _trail(tracer: Tracer | None, trace: list[dict[str, Any]]) -> Trail:
    """How the answer was reached, from what the run recorded: each model call in order, and
    each filing found and read on the way."""
    calls = []
    for event in tracer.events if tracer is not None else ():
        usage = event.get("usage")
        if not isinstance(usage, dict):
            continue  # a step that was skipped, not a call
        cost = event.get("cost_usd")
        calls.append(
            ModelCall(
                step=str(event.get("step", "")),
                model=str(event.get("model", "")),
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                web_searches=int(usage.get("web_searches", 0)),
                tool_calls=[str(name) for name in event.get("tool_calls") or []],
                cost_usd=Decimal(str(cost)) if cost is not None else None,
                cached=bool(event.get("cached", False)),
            )
        )
    fetched = [
        str(note)
        for record in trace
        if record.get("step") == Step.DISCOVER.value
        for note in record.get("notes") or []
        if str(note).startswith("found and ingested")
    ]
    return Trail(model_calls=calls, fetched=fetched)


_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def _found_profile(
    bank_id: str,
    name: str,
    fy_end_month: int,
    ir_domains: tuple[str, ...],
    aliases: tuple[str, ...],
    row_labels: tuple[tuple[str, str], ...] = (),
) -> BankProfile:
    """A bank found on the ASX: its code, name and year, and nothing assumed beyond them.

    Its release months aren't known, so a period is taken as reported two months after it
    ends, which is the ASX's own deadline for a listed company's results.
    """
    return BankProfile(
        bank_id=bank_id,
        name=name,
        ticker=bank_id,
        fy_end_month=fy_end_month,
        in_seed_corpus=False,
        ir_domains=ir_domains or (ASX_DOMAIN,),
        cet1_scope="level2",
        headline_earnings_label=None,
        results_release_months={},
        doc_name_hints={},
        verification={},
        aliases=aliases,
        discovered=True,
        row_labels=row_labels,
    )


#: The lines of a bridge a question doesn't strictly need but the bridge can't close
#: without, where a bank has them.
_BRIDGE_LINES: Final = frozenset({"minority_interests", "notable_items"})


def _merged(
    held: tuple[tuple[str, str], ...], learned: dict[str, str]
) -> tuple[tuple[str, str], ...]:
    return tuple(sorted({**dict(held), **learned}.items()))


def _register(profile: BankProfile) -> None:
    """Register a bank found while running. Its label index is built again on next use,
    with its own row labels and the wider ranges a bank no one configured is held to."""
    get_registry().register(profile)
    line_item_index.cache_clear()


def _identity_can_run(spec: RequestSpec, inputs: Any) -> bool:
    """Whether the metric's identity has every input, in both periods, on one basis."""
    from ..decomposition.identities import identity_for

    identity = identity_for(spec.metric_id, pre_tax=spec.pre_tax)
    if identity is None:
        return False
    usable = identity.usable_components(set(inputs.current) & set(inputs.prior))
    return usable is not None and bool(inputs.comparable(spec.metric_id, *usable))


def _attach_computed_citations(decomposition: Any, inputs: Any, spec: RequestSpec) -> None:
    """Give each computed driver the evidence ids of the figures behind it.

    A driver produced by arithmetic is only as traceable as its inputs, so the components it
    recorded are mapped back to the cards those figures came from, in both periods, since
    every contribution is a difference between them.
    """
    for contribution in decomposition.drivers:
        if contribution.evidence_ids or not contribution.from_components:
            continue
        found: list[str] = []
        for component in contribution.from_components:
            for period in (spec.period, spec.comparison_period):
                for evidence_id in inputs.citations_for(component, period):
                    if evidence_id not in found:
                        found.append(evidence_id)
        contribution.evidence_ids = found
