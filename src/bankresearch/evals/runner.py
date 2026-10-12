"""Runs a split through the agent and reports what it found."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from ..agent.orchestrator import Agent
from ..decomposition.identities import identity_for
from ..domain.models import Fact
from ..domain.registry import get_registry
from ..llm.cache import CachedClient, ResponseCache
from ..llm.client import ModelUnreachable
from ..llm.consent import ConsentGate
from ..memory.db import connect
from ..memory.fact_store import FactStore
from ..settings import Settings, get_settings
from .cases import EvalCase, load_cases
from .perturb import PerturbationResult, perturbations_for
from .scorers import CaseScore, score_case
from .stability import RUNS, StabilityReport, measure_stability
from .unseen import UnseenBankReport, WithoutBank, summarise_unseen

__all__ = [
    "MODEL_MODES",
    "EvalRun",
    "run_leave_one_bank_out",
    "run_perturbations",
    "run_split",
    "run_stability",
]


MODEL_MODES = ("off", "replay", "live")


def without_a_key(settings: Settings) -> Settings:
    """These settings with the API key taken out, so nothing in the run can reach a model.

    The agent builds its client from the key if it finds one. Without this, a key left in
    the shell would let recorded responses into one run of a suite and not the next, and a
    deterministic measurement would stop being one without anything saying so.
    """
    return settings.model_copy(update={"anthropic_api_key": None})


@dataclass
class EvalRun:
    """One pass over a split."""

    split: str
    started: datetime
    scores: list[CaseScore] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    git_sha: str = ""
    config_hash: str = ""
    model_mode: str = "off"

    # -- gold-free measures ----------------------------------------------

    @property
    def status_accuracy(self) -> float:
        return self._share(score.status_correct for score in self.scores)

    @property
    def abstention_accuracy(self) -> float | None:
        judged = [s for s in self.scores if s.abstain_reason_correct is not None]
        if not judged:
            return None
        return sum(1 for s in judged if s.abstain_reason_correct) / len(judged)

    @property
    def reconciliation_rate(self) -> float | None:
        judged = [s for s in self.scores if s.reconciles is not None]
        if not judged:
            return None
        return sum(1 for s in judged if s.reconciles) / len(judged)

    @property
    def citation_faithfulness(self) -> float | None:
        checked = sum(s.cited_numbers for s in self.scores)
        if not checked:
            return None
        return sum(s.cited_numbers_found for s in self.scores) / checked

    @property
    def citations_checked(self) -> int:
        return sum(s.cited_numbers for s in self.scores)

    @property
    def dangling_citations(self) -> int:
        return sum(s.citations_without_a_page for s in self.scores)

    @property
    def passed(self) -> int:
        return sum(1 for s in self.scores if s.passed)

    @property
    def latency_p50_ms(self) -> int:
        return self._percentile([s.latency_ms for s in self.scores], 0.50)

    @property
    def latency_p95_ms(self) -> int:
        return self._percentile([s.latency_ms for s in self.scores], 0.95)

    @property
    def total_cost_usd(self) -> Decimal:
        return sum((s.cost_usd for s in self.scores), Decimal(0))

    # -- gold-dependent --------------------------------------------------

    @property
    def gold_scored(self) -> int:
        return sum(1 for s in self.scores if s.headline_exact is not None)

    @property
    def headline_exact_rate(self) -> float | None:
        judged = [s for s in self.scores if s.headline_exact is not None]
        if not judged:
            return None
        return sum(1 for s in judged if s.headline_exact) / len(judged)

    @property
    def change_within_tolerance_rate(self) -> float | None:
        """How often the movement matched gold within the gold's own tolerance.

        Scored separately from the two levels, because a run can read both figures correctly
        and still state the movement wrongly (a sign convention, a percentage point quoted
        as a basis point), and the movement is what a memo leads with.
        """
        judged = [
            (s.change_error, s.change_tolerance)
            for s in self.scores
            if s.change_error is not None and s.change_tolerance is not None
        ]
        if not judged:
            return None
        return sum(error <= tolerance for error, tolerance in judged) / len(judged)

    @property
    def mean_driver_precision(self) -> float | None:
        values = [s.driver_precision for s in self.scores if s.driver_precision is not None]
        return sum(values) / len(values) if values else None

    @property
    def mean_driver_recall(self) -> float | None:
        values = [s.driver_recall for s in self.scores if s.driver_recall is not None]
        return sum(values) / len(values) if values else None

    @property
    def driver_values(self) -> tuple[int, int]:
        """Drivers sized right, out of those the run shared with the gold."""
        return (
            sum(s.driver_values_right for s in self.scores),
            sum(s.driver_values_checked for s in self.scores),
        )

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for score in self.scores:
            counts[score.status] = counts.get(score.status, 0) + 1
        return dict(sorted(counts.items()))

    @staticmethod
    def _share(flags) -> float:  # type: ignore[no-untyped-def]
        values = list(flags)
        return sum(1 for flag in values if flag) / len(values) if values else 0.0

    @staticmethod
    def _percentile(values: list[int], fraction: float) -> int:
        if not values:
            return 0
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
        return ordered[index]

    def mean_confidence(self) -> float | None:
        values = [s.overall_confidence for s in self.scores if s.overall_confidence is not None]
        return statistics.fmean(values) if values else None


def run_perturbations(
    cases: list[EvalCase],
    suite: str,
    settings: Settings | None = None,
) -> list[PerturbationResult]:
    """Run one perturbation suite and report whether the system degraded as it should.

    Each perturbation is applied to the fact set the agent is given, not to the store on
    disk. A suite that changed the corpus would leave it changed for the next run, and the
    first sign would be a clean eval that had quietly stopped being clean.
    """
    resolved = without_a_key(settings or get_settings())
    registry = get_registry()
    results: list[PerturbationResult] = []

    with connect(resolved.db_path) as connection:
        store = FactStore(connection)
        for case in cases:
            if case.expects_abstention or case.metric not in registry.metrics:
                continue  # an abstention has nothing to degrade
            baseline = Agent(store=store, settings=resolved, persist=False).run(
                case.bank, case.period, case.metric, case.as_of
            )
            # The identity the run used: Judo's earnings from 1H26 are struck before tax, so
            # taking tax away from them should change nothing.
            identity = identity_for(case.metric, pre_tax=baseline.result.request.pre_tax)
            base_confidence = _confidence_of(baseline)
            # The inputs this run's identity actually used: CBA's impairment charge is split
            # by division, so removing an individually assessed charge it never had would
            # test nothing.
            held = {card.component for card in baseline.result.evidence}
            components = (
                (identity.usable_components(held) or identity.components) if identity else ()
            )
            # Inputs another complete decomposition doesn't need, so removing one should
            # move the attribution there rather than end it.
            covered_elsewhere = frozenset(
                component
                for component in components
                for other in ((identity.components, *identity.alternatives) if identity else ())
                if other != components and set(other) <= held and component not in other
            )
            baseline_drivers = frozenset(driver.driver_id for driver in baseline.result.drivers)

            # The look-ahead date and the figure to contradict both come from the run that
            # just succeeded, so a perturbation is always against evidence that exists.
            published = max(
                (card.source.publish_date for card in baseline.result.evidence),
                default=None,
            )
            # The prior the answer actually used, not the first card that happens to be
            # for the prior period: with a variant held beside the headline, that card could
            # be the variant, and a perturbed variant tests nothing the answer depends on.
            used = set(baseline.result.prior.evidence_ids) if baseline.result.prior else set()
            prior = next(
                (
                    card
                    for card in baseline.result.evidence
                    if card.component == case.metric
                    and card.period != case.period
                    and (card.evidence_id in used or not used)
                ),
                None,
            )
            # A metric that already disagrees across documents isn't injected into, because
            # the suite would then be measuring the existing conflict. Westpac's net profit
            # is stated two ways in its own pack. A basis difference doesn't count: that's
            # the headline and its variant, and an injected figure is classified against
            # the headline's own readings regardless.
            already_conflicted = any(
                conflict.component == case.metric and conflict.classification != "BASIS"
                for conflict in baseline.result.data_conflicts
            )
            prior_fact = (
                _fact_from_card(prior, case.bank)
                if prior is not None and not already_conflicted
                else None
            )

            for perturbation in perturbations_for(
                case, components, published, prior_fact, covered_elsewhere
            ):
                if perturbation.suite != suite:
                    continue
                outcome = Agent(
                    # Duck-typed: the agent reads facts through `get_facts` and nothing
                    # else, which is what makes a wrapper safe here.
                    store=cast("FactStore", _ScopedStore(store, perturbation)),
                    settings=resolved,
                    persist=False,
                ).run(
                    case.bank,
                    case.period,
                    case.metric,
                    perturbation.as_of_override or case.as_of,
                )
                results.append(
                    PerturbationResult(
                        perturbation=perturbation,
                        answered=outcome.result.status != "ABSTAIN",
                        drivers=len(outcome.result.drivers),
                        confidence=_confidence_of(outcome),
                        baseline_confidence=base_confidence,
                        driver_ids=frozenset(d.driver_id for d in outcome.result.drivers),
                        baseline_driver_ids=baseline_drivers,
                        classification=_least_explained(
                            conflict.classification
                            for conflict in outcome.result.data_conflicts
                            if conflict.component == case.metric
                            and _involves(conflict, perturbation)
                        ),
                    )
                )
    return results


def _involves(conflict: Any, perturbation: Any) -> bool:
    """Whether a conflict includes a figure the perturbation injected, or the suite has
    nothing injected to look for."""
    injected = {fact.doc_id for fact in perturbation.inject}
    return not injected or any(
        candidate.source.doc_id in injected for candidate in conflict.candidates
    )


def _least_explained(classes: Any) -> str:
    """How the injected figure was classified, from the conflicts that involve it, or "".

    A metric can carry two: the basis difference between its headline and its variant,
    and a disagreement among the headline's own readings. An injected figure is on the
    headline's basis, so its class is the second; the first only describes readings on
    another basis. Among the rest, the class latest in the order is the one judged.
    """
    from ..reconcile.conflicts import CLASSIFICATION_ORDER

    order = list(CLASSIFICATION_ORDER)
    found = [name for name in classes if name in order]
    own = [name for name in found if name != "BASIS"] or found
    return max(own, key=order.index) if own else ""


def _fact_from_card(card: Any, bank_id: str) -> Fact | None:
    """The prior figure as a Fact, so a conflict can be injected against it.

    Rebuilt from the evidence card rather than re-queried, because the card is what the
    run actually used and a second query could resolve differently.
    """
    if card.value is None:
        return None
    return Fact(
        fact_id=f"{card.evidence_id}-baseline",
        bank_id=bank_id,
        period=card.period,
        component=card.component,
        value=card.value,
        published_value=card.published_value or card.value,
        unit=card.unit,
        basis=card.basis,
        doc_id=card.source.doc_id,
        doc_type=card.source.doc_type,
        page_no=card.source.page,
        publish_date=card.source.publish_date,
        validated=True,
    )


def _confidence_of(outcome: Any) -> Decimal | None:
    overall = outcome.result.overall_confidence
    score: Decimal | None = getattr(overall, "score", None) if overall is not None else None
    return score


@dataclass
class _ScopedStore:
    """The fact store with one perturbation applied to what ``get_facts`` returns.

    A wrapper rather than a change to the store, so the corpus on disk is untouched and the
    agent can't tell: it sees a store that happens to be missing a component or holding a
    second reading of one.
    """

    inner: FactStore
    perturbation: Any

    def get_facts(self, spec: Any, components: Any = None) -> list[Fact]:
        kept: list[Fact] = self.perturbation.applies_to(self.inner.get_facts(spec, components))
        return kept

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def run_leave_one_bank_out(
    cases: list[EvalCase],
    settings: Settings | None = None,
) -> list[UnseenBankReport]:
    """Each issuer in turn treated as one the system has never read."""
    resolved = without_a_key(settings or get_settings())
    banks = sorted({case.bank for case in cases if case.bank in get_registry().banks})
    reports: list[UnseenBankReport] = []

    with connect(resolved.db_path) as connection:
        store = FactStore(connection)
        for bank_id in banks:
            held_out = [case for case in cases if case.bank == bank_id]
            if not held_out:
                continue
            withheld = cast("FactStore", WithoutBank(store, bank_id))
            outcomes = [
                Agent(store=withheld, settings=resolved, persist=False).run(
                    case.bank, case.period, case.metric, case.as_of
                )
                for case in held_out
            ]
            reports.append(summarise_unseen(bank_id, outcomes, [case.case_id for case in held_out]))
    return reports


def run_stability(
    cases: list[EvalCase],
    settings: Settings | None = None,
    runs: int = RUNS,
) -> list[StabilityReport]:
    """Run each case several times and measure whether the answers agree."""
    resolved = without_a_key(settings or get_settings())
    outcomes: dict[str, list[Any]] = {}
    with connect(resolved.db_path) as connection:
        store = FactStore(connection)
        for _ in range(runs):
            for case in cases:
                outcome = Agent(store=store, settings=resolved, persist=False).run(
                    case.bank, case.period, case.metric, case.as_of
                )
                outcomes.setdefault(case.case_id, []).append(outcome)
    return measure_stability(outcomes)


def run_split(
    split: str,
    settings: Settings | None = None,
    cases: list[EvalCase] | None = None,
    final: bool = False,
    limit: int | None = None,
    model: str = "off",
    consent: ConsentGate | None = None,
    workers: int = 1,
) -> EvalRun:
    """Run a split and score it.

    Raises if the test split is run without ``--final``, if the model mode isn't one of
    :data:`MODEL_MODES`, or if a live run has no key or no grant: a live eval spends money,
    and the only way it may is within a limit someone chose for the whole split.
    """
    if split == "test" and not final:
        raise ValueError(
            "the test split runs only with --final: once, with no tuning afterwards, and "
            "every run logged to evals/test_runs.log"
        )
    if model not in MODEL_MODES:
        raise ValueError(f"model mode {model!r} is not one of {', '.join(MODEL_MODES)}")

    resolved = settings or get_settings()
    if model == "live" and not resolved.has_api_key:
        raise ValueError(
            "a live model eval needs ANTHROPIC_API_KEY; run it through scripts/with-key"
        )
    if model == "live" and consent is None:
        raise ValueError("a live model eval needs a grant for the split: --grant CALLS:DOLLARS")

    selected = cases if cases is not None else load_cases(split)
    if limit is not None:
        selected = selected[:limit]

    run = EvalRun(
        split=split,
        started=datetime.now(UTC),
        git_sha=_git_sha(),
        config_hash=_config_hash(),
        model_mode=model,
    )

    def make_agent() -> tuple[Agent, Any]:
        # One connection per agent, so cases can run on separate threads.
        connection = connect(resolved.db_path)
        store = FactStore(connection)
        if model == "live":
            # One grant for the whole split, so its limits bound the eval rather than each
            # case.
            return Agent(store=store, settings=resolved, consent=consent), connection
        if model == "replay":
            return (
                Agent(
                    store=store,
                    settings=without_a_key(resolved),
                    client=CachedClient(cache=ResponseCache(connection), replay_only=True),
                ),
                connection,
            )
        return Agent(store=store, settings=without_a_key(resolved)), connection

    # The cold path ingests what it finds into the shared corpus. A live eval that left
    # Bendigo's filing behind turned three no-model abstention cases into answers, so the
    # corpus is put back as it was found. A case whose period the corpus doesn't hold is the
    # one that will ingest, and run beside the others its document was read by them: Judo's
    # FY26 answer cited the FY25 report a concurrent cold start had just fetched. So those
    # cases run after the rest, one at a time, and each is put back before the next.
    with connect(resolved.db_path) as connection:
        store = FactStore(connection)
        before = store.document_ids()
        banks_before = {str(row["bank_id"]) for row in store.discovered_banks()}
        cold = {
            case.case_id
            for case in selected
            if not store.documents_for(case.bank, [case.period], case.as_of or date.max)
        }

    def put_back() -> None:
        with connect(resolved.db_path) as connection:
            store = FactStore(connection)
            for doc_id in sorted(store.document_ids() - before):
                store.remove_document(doc_id)
            # A bank found on the ASX during the split is forgotten with its filing.
            for row in store.discovered_banks():
                if str(row["bank_id"]) not in banks_before:
                    store.forget_discovered_bank(str(row["bank_id"]))

    def run_case(case: EvalCase) -> CaseScore | tuple[str, str]:
        agent, connection = make_agent()
        try:
            outcome = agent.run(case.bank, case.period, case.metric, as_of=case.as_of)
            return score_case(case, outcome.result, resolved, model)
        except Exception as exc:  # a harness that dies on one case reports nothing
            if isinstance(exc, ModelUnreachable):  # the run stops; see llm.client
                raise
            return (case.case_id, f"{type(exc).__name__}: {exc}")
        finally:
            connection.close()

    started = time.perf_counter()
    warm = [case for case in selected if case.case_id not in cold]
    by_case: dict[str, CaseScore | tuple[str, str]] = {}
    try:
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as pool:
                by_case.update(
                    zip((c.case_id for c in warm), pool.map(run_case, warm), strict=True)
                )
        else:
            by_case.update((case.case_id, run_case(case)) for case in warm)
        for case in selected:
            if case.case_id in cold:
                by_case[case.case_id] = run_case(case)
                put_back()
    finally:
        put_back()
    outcomes = [by_case[case.case_id] for case in selected if case.case_id in by_case]
    for outcome in outcomes:
        if isinstance(outcome, tuple):
            run.errors.append(outcome)
        else:
            run.scores.append(outcome)
    run.elapsed_seconds = time.perf_counter() - started

    if split == "test":
        _log_test_run(run, resolved)
    return run


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - a tarball has no git
        return "unknown"


def _config_hash() -> str:
    from ..domain.registry import get_registry

    return get_registry().config_hash


def _log_test_run(run: EvalRun, settings: Settings) -> None:
    """Append-only record of every test-split run, so a second one is visible."""
    path = Path(settings.evals_dir) / "test_runs.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{run.started.isoformat()} git={run.git_sha} config={run.config_hash} "
            f"cases={len(run.scores)} passed={run.passed} "
            f"status_accuracy={run.status_accuracy:.3f}\n"
        )
