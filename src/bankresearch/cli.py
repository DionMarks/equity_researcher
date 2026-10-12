"""The command line: ``bankresearch run``, and the commands that ingest, inspect and evaluate around
it."""

from __future__ import annotations

import json
import sys
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .agent.orchestrator import Agent
from .agent.state import ABSTAIN_REASONS
from .agent.steps.resolve import AbstainSignal, caveats_for, resolve_request
from .domain.models import DocMeta
from .domain.money import aud, to_usd
from .domain.registry import get_registry
from .ingestion.pipeline import ingest_document
from .llm.client import ModelUnreachable
from .llm.consent import BudgetedGrant, ConsentGate, TerminalConsent
from .memory.db import connect
from .memory.fact_store import FactStore
from .output.progress import TerminalProgress
from .output.render_md import render_markdown
from .output.render_terminal import render_terminal
from .output.schema import describe_basis
from .settings import Settings, get_settings

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Bank equity research agent: how a headline metric moved against the "
    "same period last financial year, attributed to drivers with cited evidence.",
)
# On Windows, output to a pipe uses the system code page, which has no box-drawing
# characters or arrows, so it's switched to UTF-8 rather than failing on the first one.
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        _reconfigure = getattr(_stream, "reconfigure", None)
        if _reconfigure is not None:
            _reconfigure(encoding="utf-8", errors="replace")

console = Console()
# Notices about the run itself, kept off stdout so `--format json` prints only the JSON.
notices = Console(stderr=True)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_ABSTAIN = 2


def _ingest_from_manifest(bank: str | None, period: str | None) -> NoReturn:
    """Fetch and ingest every manifest document, or a filtered subset."""
    from .acquisition.corpus import build_corpus

    settings = get_settings()
    registry = get_registry()
    store = FactStore(connect(settings.db_path))
    allowed = {bank_id: profile.ir_domains for bank_id, profile in registry.banks.items()}
    report = build_corpus(store, settings, allowed, only_bank=bank, only_period=period)
    for ingested in report.reports:
        console.print(f"  {ingested.summary()}")
    for doc_id, reason in report.failures:
        console.print(f"[red]failed[/red] {doc_id}: {reason}")
    console.print(report.summary())
    raise typer.Exit(EXIT_OK if report.ingested else EXIT_ERROR)


def _consent_gate(grant: str | None) -> ConsentGate:
    """The consent gate for this run: a --grant budget, or a prompt before each call.

    There's no flag that skips the prompt altogether. --grant is the closest thing, and it
    still makes you set a limit.
    """
    if grant is None:
        return TerminalConsent()
    calls, _, ceiling = grant.partition(":")
    if not calls.strip().isdigit() or not ceiling.strip():
        raise ValueError(
            f"--grant {grant!r} is not CALLS:DOLLARS, e.g. --grant 12:3.00. "
            "Both halves are required: a call count with no ceiling, or a ceiling with "
            "no count, leaves one of the two limits open."
        )
    try:
        limit = Decimal(ceiling.strip())
    except InvalidOperation as exc:
        raise ValueError(f"--grant {grant!r}: {ceiling.strip()!r} is not a dollar amount") from exc
    if limit <= 0 or int(calls) <= 0:
        raise ValueError(f"--grant {grant!r}: both the call count and the ceiling must be above 0")
    notices.print(
        f"[yellow]Granting up to {int(calls)} call(s) and A${limit} in total for this run.[/yellow] "
        "Each call is still counted against both, and the run stops at whichever is "
        "reached first."
    )
    return BudgetedGrant(calls=int(calls), ceiling_usd=to_usd(limit))


@app.command()
def run(
    bank: Annotated[
        str,
        typer.Option(
            "--bank",
            help="Ticker, name or brand: NAB, Westpac, Up, or any ASX-listed Australian bank.",
        ),
    ],
    period: Annotated[str, typer.Option("--period", help="1H26, 2H25, FY25.")],
    metric: Annotated[
        str,
        typer.Option(
            "--metric",
            help="nim, headline_earnings, roe, cet1_ratio, credit_impairment_charge, "
            "cost_to_income.",
        ),
    ],
    as_of: Annotated[
        str | None,
        typer.Option(
            "--as-of",
            help="YYYY-MM-DD, or 'today'. Defaults to the period's results release.",
        ),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            help="pretty (the terminal view), md (the memo as Markdown), json, or both.",
        ),
    ] = "pretty",
    grant: Annotated[
        str | None,
        typer.Option(
            "--grant",
            help="Approve up to N calls and A$X in total, e.g. --grant 12:3.00, instead of "
            "the estimate's own ceiling.",
        ),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Accept the estimate without being asked."),
    ] = False,
) -> None:
    """Explain how a metric moved against the prior corresponding period.

    Every answer is written and checked by a model, the same way every time. Before anything
    runs, an estimate says whether the question is inside the saved data or needs external
    calls, what it will cost and how long it usually takes, and asks to go ahead. If the
    model can't be used (no key, a key the API refuses, no credit left, a service that stays
    down) or the run uses up what was approved, it stops and says which.
    """
    try:
        resolved_as_of = _parse_as_of(as_of)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_ERROR) from exc

    settings = _with_the_key(get_settings())
    store = FactStore(connect(settings.db_path))
    estimator = Agent(store=store, settings=_without_key(settings), persist=False)
    estimate = estimator.estimate(bank, period, metric, as_of=resolved_as_of)
    _print_estimate(estimate)
    if estimate.stop is not None:
        raise typer.Exit(EXIT_ABSTAIN)
    if not settings.has_api_key:
        notices.print(f"[red]{_STOPPED['no_key']}[/red]")
        raise typer.Exit(EXIT_ERROR)

    try:
        consent = _approval(estimate, grant, yes)
    except ValueError as exc:
        notices.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_ERROR) from exc
    if consent is None:
        notices.print("Nothing was run, and nothing was spent.")
        raise typer.Exit(EXIT_OK)

    # Agreed: from here the run spends, so a line on stderr says what it's doing, how long
    # it has taken against how long a run like it usually takes, and what it has spent.
    usual = estimate.cost.seconds if estimate.cost is not None else None
    try:
        with TerminalProgress(notices, usual) as progress:
            agent = Agent(store=store, settings=settings, consent=consent, progress=progress)
            outcome = agent.run(bank, period, metric, as_of=resolved_as_of)
    except ModelUnreachable as stopped:
        notices.print(f"[red]{_STOPPED[stopped.kind]}[/red]")
        if stopped.detail:
            notices.print(f"[dim]The API said: {escape(stopped.detail)}.[/dim]")
        notices.print("No answer was given.")
        raise typer.Exit(EXIT_ERROR) from stopped
    declined = outcome.result.diagnostics.declined_steps
    if declined:
        spent = outcome.result.diagnostics
        notices.print(
            f"[red]{_STOPPED['budget']}[/red] It got as far as {', '.join(declined)}, after "
            f"{spent.llm_calls} model call{'' if spent.llm_calls == 1 else 's'} and "
            f"{aud(spent.cost_usd)}. The run record is "
            f"in {outcome.artefacts or 'runs/'}."
        )
        raise typer.Exit(EXIT_ERROR)
    _show(outcome, output_format)
    raise typer.Exit(outcome.exit_code)


#: What the command line says when a run can't use the model, by reason. Each says what to
#: do next, because every one of them is fixed outside this tool.
_STOPPED: dict[str, str] = {
    "no_key": "Every answer is written by a model, and no API key was found. Store it once "
    "with `make key-set`, or put ANTHROPIC_API_KEY=... in a .env file in the repo folder, "
    "and ask again.",
    "key_rejected": "The API didn't accept the key, so the run stopped. Check it is the "
    "right key and still active (`make key-set` replaces the one in the keychain), then "
    "ask again.",
    "out_of_credit": "The API account has run out of credit, so the run stopped. Add credit "
    "in the Anthropic console, then ask again.",
    "rate_limited": "The API kept limiting requests on this key after three tries, so the "
    "run stopped. Wait a minute and ask again.",
    "unreachable": "The API couldn't be reached after three tries, so the run stopped. Check "
    "the network connection, then ask again.",
    "budget": "The run used up the budget approved for it before it finished, so it stopped "
    "rather than answer without the model. Ask again with a larger --grant CALLS:DOLLARS.",
}


def _show(outcome: Any, output_format: str) -> None:
    result = outcome.result
    if output_format == "pretty":
        render_terminal(result, console, artefacts=outcome.artefacts)
    elif output_format == "json":
        # Bare JSON on stdout, for piping into jq or another tool.
        print(result.model_dump_json(indent=2))
    else:
        if output_format in {"md", "both"}:
            _print_memo(render_markdown(result))
        if outcome.artefacts is not None and console.is_terminal:
            notices.print(f"[dim]The memo is saved as {outcome.artefacts}/result.md[/dim]")
        if result.status == "ABSTAIN":
            notices.print(f"[bold]ABSTAIN[/bold] [cyan]{result.abstain_reason}[/cyan]")


def _print_estimate(estimate: Any) -> None:
    """The estimate, on stderr, so a redirected answer stays exactly the answer."""
    lines = estimate.lines()
    notices.print(lines[0], style="bold", highlight=False, markup=False)
    for line in lines[1:]:
        stopped = estimate.stop is not None and line == estimate.stop
        notices.print(line, style="red" if stopped else None, highlight=False, markup=False)


def _approval(estimate: Any, grant: str | None, yes: bool) -> ConsentGate | None:
    """The run's approval: the person's own --grant, the estimate's ceiling when they've
    said yes, or ``None`` when they haven't. Never assumed: a run with nobody to ask and no
    --yes or --grant refuses rather than spends."""
    if grant is not None:
        return _consent_gate(grant)
    cost = estimate.cost
    if not yes:
        if not sys.stdin.isatty():
            raise ValueError(
                "This run needs approval and there's nobody to ask: add --yes to accept the "
                "estimate, or --grant CALLS:DOLLARS to set your own limit."
            )
        if not typer.confirm("Go ahead?", default=False, err=True):
            return None
    return BudgetedGrant(calls=cost.ceiling_calls, ceiling_usd=cost.ceiling_usd)


def _without_key(settings: Settings) -> Settings:
    return settings.model_copy(update={"anthropic_api_key": None})


def _with_the_key(settings: Settings) -> Settings:
    """These settings with the API key, from the environment or else the macOS keychain
    entry `make key-set` writes. Never printed."""
    if settings.has_api_key:
        return settings
    import getpass
    import subprocess

    try:
        found = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-a",
                getpass.getuser(),
                "-s",
                "bankresearch-anthropic",
                "-w",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return settings
    key = found.stdout.strip()
    if found.returncode != 0 or not key:
        return settings
    from pydantic import SecretStr

    return settings.model_copy(update={"anthropic_api_key": SecretStr(key)})


def _print_memo(memo: str) -> None:
    """The memo, rendered in a terminal and exactly as written anywhere else.

    Printed as source in a terminal it was pipes and asterisks, and the terminal's own
    wrapping broke its long lines, which would also have broken them in a file the output
    was redirected to. A terminal now gets it rendered; a file or another program gets the
    Markdown unchanged and unwrapped, so it renders wherever it's opened.
    """
    if console.is_terminal:
        from rich.markdown import Markdown

        console.print(Markdown(memo))
    else:
        sys.stdout.write(memo if memo.endswith("\n") else memo + "\n")


@app.command()
def resolve(
    bank: Annotated[str, typer.Option("--bank")],
    period: Annotated[str, typer.Option("--period")],
    metric: Annotated[str, typer.Option("--metric")],
    as_of: Annotated[str | None, typer.Option("--as-of")] = None,
) -> None:
    """Show how a request resolves: fiscal dates, the pcp and the headline label.

    A bank that isn't configured is looked up the way a run looks it up: a brand is
    resolved as its owner, and any other name is placed among Australia's banks.
    """
    try:
        spec = resolve_request(bank, period, metric, as_of=_parse_as_of(as_of))
    except AbstainSignal as signal:
        if signal.reason != "BANK_UNKNOWN":
            _print_abstention(signal)
            raise typer.Exit(EXIT_ABSTAIN) from None
        settings = get_settings()
        estimator = Agent(
            store=FactStore(connect(settings.db_path)),
            settings=_without_key(settings),
            persist=False,
        )
        _print_estimate(estimator.estimate(bank, period, metric, as_of=_parse_as_of(as_of)))
        raise typer.Exit(EXIT_ABSTAIN) from None
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_ERROR) from exc
    _print_spec(spec)
    raise typer.Exit(EXIT_OK)


@app.command()
def coverage() -> None:
    """Issuers and metrics in scope, from config."""
    registry = get_registry()

    banks = Table(title="Issuers")
    for column in ("bank_id", "Name", "FY ends", "Seed corpus", "CET1 scope"):
        banks.add_column(column)
    for profile in registry.banks.values():
        if profile.discovered:
            continue
        banks.add_row(
            profile.bank_id,
            profile.name,
            f"month {profile.fy_end_month}",
            "yes" if profile.in_seed_corpus else "cold path",
            profile.cet1_scope,
        )
    settings = get_settings()
    found = (
        FactStore(connect(settings.db_path)).discovered_banks() if settings.db_path.exists() else []
    )
    for row in found:
        banks.add_row(
            str(row["bank_id"]),
            str(row["name"]),
            f"month {row['fy_end_month']}",
            "found on the ASX",
            "level2",
        )
    console.print(banks)
    console.print(
        "Any other ASX-listed Australian bank is found on the ASX when it's asked about, and "
        "a brand is answered as its owner's results."
    )

    metrics = Table(title="Metrics")
    for column in ("metric_id", "Display name", "Unit", "Change unit", "Primary path", "Drivers"):
        metrics.add_column(column)
    for definition in registry.metrics.values():
        metrics.add_row(
            definition.metric_id,
            definition.display_name,
            definition.unit,
            definition.change_unit,
            definition.primary_path,
            str(len(registry.driver_ids(definition.metric_id))),
        )
    console.print(metrics)
    console.print(f"Abstention reasons: {', '.join(ABSTAIN_REASONS)}")
    console.print(f"Config hash: {registry.config_hash}")


@app.command(name="config-check")
def config_check() -> None:
    """Fail if any 'verify during build' config field is still unverified.

    These values have to come from the issuer's own filings, recorded with a source URL, not
    from memory.
    """
    registry = get_registry()
    outstanding = registry.unverified
    if not outstanding:
        console.print("[green]All config fields verified against a source.[/green]")
        raise typer.Exit(EXIT_OK)

    table = Table(title="Config fields awaiting verification")
    table.add_column("bank_id")
    table.add_column("Fields")
    for bank_id, fields in outstanding.items():
        table.add_row(bank_id, ", ".join(fields))
    console.print(table)
    console.print(
        f"[yellow]{sum(len(f) for f in outstanding.values())} field(s) unverified.[/yellow] "
        "Each MUST be checked against the bank's own filing and recorded in config/ "
        "with a source URL before these values are relied on."
    )
    raise typer.Exit(EXIT_ERROR)


@app.command()
def ingest(
    path: Annotated[
        Path | None,
        typer.Option("--path", help="A local PDF to ingest, instead of the manifest."),
    ] = None,
    bank: Annotated[str | None, typer.Option("--bank")] = None,
    period: Annotated[str | None, typer.Option("--period")] = None,
    doc_type: Annotated[str, typer.Option("--doc-type")] = "RESULTS_DOCUMENT",
    publish_date: Annotated[
        str | None, typer.Option("--publish-date", help="YYYY-MM-DD. Defaults to today.")
    ] = None,
    doc_id: Annotated[str | None, typer.Option("--doc-id")] = None,
) -> None:
    """Ingest a filing: parse, extract, validate and store.

    With --path, ingests one local PDF. Without it, ingests everything in the manifest,
    which needs network access to the issuers' sites.
    """
    if path is None:
        _ingest_from_manifest(bank, period)

    if not bank or not period:
        console.print("[red]--bank and --period are required with --path.[/red]")
        raise typer.Exit(EXIT_ERROR)
    if not path.exists():
        console.print(f"[red]{path} does not exist.[/red]")
        raise typer.Exit(EXIT_ERROR)

    try:
        registry = get_registry()
        profile = registry.bank(bank)
        resolved_period = resolve_request(
            profile.bank_id, period, "nim", as_of=date(2099, 1, 1)
        ).period
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_ERROR) from exc

    settings = get_settings()
    meta = DocMeta(
        doc_id=doc_id or f"{profile.bank_id.lower()}-{resolved_period.lower()}-{doc_type.lower()}",
        bank_id=profile.bank_id,
        period=resolved_period,
        doc_type=doc_type,  # type: ignore[arg-type]
        publish_date=_parse_as_of(publish_date) or date.today(),
        url=path.resolve().as_uri(),
    )
    store = FactStore(connect(settings.db_path))
    report = ingest_document(path, meta, store, processed_root=settings.processed_dir)
    console.print(report.summary())
    for reason in report.quarantine_reasons:
        console.print(f"[yellow]quarantined[/yellow] {reason}")
    raise typer.Exit(EXIT_OK)


@app.command()
def discover(
    bank: Annotated[str, typer.Option("--bank", help="Ticker or name, e.g. BEN.")],
    period: Annotated[str, typer.Option("--period", help="1H26, 2H25, FY25.")],
    as_of: Annotated[
        str | None,
        typer.Option("--as-of", help="YYYY-MM-DD, or 'today' (the default)."),
    ] = None,
    grant: Annotated[
        str | None,
        typer.Option(
            "--grant",
            help="Pre-approve up to N calls and A$X in total, e.g. --grant 6:1.50. "
            "Without it, every paid call asks first.",
        ),
    ] = None,
) -> None:
    """Cold path: find, check and ingest a period's results document, from the ASX's list of
    the bank's announcements or else a search of its site.

    `run` does this by itself when it needs to; this only does the finding, so the document
    is in the store before any question is asked. It needs a key, and every call is priced
    and asked about like any other.
    """
    from .agent.steps.acquire import acquire_documents
    from .domain.calendar import parse_period
    from .llm.budget import Budget

    try:
        resolved_as_of = _parse_as_of(as_of) or date.today()
        consent = _consent_gate(grant)
        label = parse_period(period)
        profile = get_registry().bank(bank)
    except (ValueError, KeyError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_ERROR) from exc

    settings = _with_the_key(get_settings())
    agent = Agent(store=FactStore(connect(settings.db_path)), settings=settings, consent=consent)
    client = agent.model_client()
    if client is None or not client.available:
        console.print(
            "[red]Checking a filing needs a model, and no API key was found.[/red] Store it "
            "once with `make key-set` or in a .env file, or ingest from the manifest."
        )
        raise typer.Exit(EXIT_ERROR)
    registry = get_registry()
    model, effort = registry.route("discover_classify")
    budget = Budget.from_config(registry.budgets, registry.models)
    budget.go_cold(registry.budgets)
    acquired = acquire_documents(
        profile.bank_id,
        label,
        resolved_as_of,
        client,
        agent.store,
        settings,
        budget=budget,
        model=model,
        effort=effort,
        asx=agent.asx,
    )
    for note in acquired.notes:
        console.print(f"  {note}", markup=False)
    if acquired.doc_ids:
        console.print(f"[green]Ingested[/green] {', '.join(acquired.doc_ids)}")
        raise typer.Exit(EXIT_OK)
    console.print("[yellow]No usable document was found.[/yellow]")
    raise typer.Exit(EXIT_ABSTAIN)


@app.command(name="leave-one-out")
def leave_one_out(
    split: Annotated[str, typer.Option("--split", help="dev or test.")] = "dev",
) -> None:
    """Treat each issuer in turn as one the system has never seen.

    Removes one bank's stored facts and documents, keeps only its config entry, and reruns
    its cases. It can't answer them any more, so this checks that it abstains cleanly, with
    a reason code saying what's missing.
    """
    from .evals.cases import load_cases
    from .evals.runner import run_leave_one_bank_out

    settings = get_settings()
    cases = [case for case in load_cases(split) if not case.expects_abstention]
    reports = run_leave_one_bank_out(cases, settings=settings)
    if not reports:
        console.print(f"[red]No answerable cases in the {split} split.[/red]")
        raise typer.Exit(EXIT_ERROR)

    table = Table(title=f"Leave-one-bank-out ({split} split)", box=box.SIMPLE)
    table.add_column("held out")
    table.add_column("cases", justify="right")
    table.add_column("abstained", justify="right")
    table.add_column("reason codes")
    table.add_column("leaked")
    for report in reports:
        reasons = ", ".join(f"{code} x{n}" for code, n in sorted(report.reasons.items()))
        table.add_row(
            report.bank_id,
            str(report.cases),
            str(report.abstained),
            reasons or "n/a",
            "[green]none[/green]"
            if report.clean
            else f"[red]{', '.join(report.answered + report.crashed)}[/red]",
        )
    console.print(table)

    clean = sum(1 for report in reports if report.clean)
    console.print(f"{clean}/{len(reports)} issuers abstained cleanly when held out")
    if clean != len(reports):
        console.print(
            "[red]An answer for a bank whose figures were withheld came from another bank's.[/red]"
        )
    raise typer.Exit(EXIT_OK if clean == len(reports) else EXIT_ERROR)


def _checker_suite(settings: Any, grant: str | None = None, replay: bool = False) -> NoReturn:
    """Score the guards on driver causes over every cause the live runs drafted.

    With --grant the checker judges every claim again, live, with today's prompt; with
    --replay it does so from the responses a live re-judge recorded, for nothing.
    """
    from .evals.cause_check import load_claims, rejudge, score_claims

    claims = load_claims(settings.evals_dir / "checker" / "claims.yaml")
    verdicts: dict[str, tuple[str, str]] | None = None
    judged: list[Any] = []
    gate: Any = None
    if grant is not None or replay:
        from .llm.anthropic_client import AnthropicClient
        from .llm.cache import CachedClient, ResponseCache
        from .memory.db import connect

        if grant is not None and not settings.has_api_key:
            console.print("[red]--grant needs an API key: scripts/with-key bankresearch ...[/red]")
            raise typer.Exit(EXIT_ERROR)
        gate = _consent_gate(grant) if grant is not None else None
        client = CachedClient(
            cache=ResponseCache(connect(settings.db_path)),
            upstream=AnthropicClient(settings=settings, consent=gate) if gate else None,
            replay_only=gate is None,
        )
        model, effort = get_registry().route("verify")
        verdicts, unjudged = rejudge(claims, client, model=model, effort=effort)
        for reason in unjudged:
            console.print(f"[yellow]Not judged: {reason}[/yellow]")
        # Scored over the claims actually judged: a batch that wasn't is no verdict at all,
        # not a checker that removed everything in it.
        judged = [claim for claim in claims if claim.claim_id in verdicts]
        verdicts = verdicts or None

    table = Table(title=f"Driver causes: {len(claims)} drafted in the live runs", box=box.SIMPLE)
    table.add_column("measure")
    table.add_column("as the runs did it", justify="right")
    table.add_column("with today's code guards", justify="right")
    if verdicts is not None:
        table.add_column(f"and today's checker, re-taken ({len(judged)})", justify="right")

    def share(numerator: int, denominator: int) -> str:
        return (
            f"{numerator}/{denominator} ({numerator / denominator:.0%})" if denominator else "n/a"
        )

    run, today = score_claims(claims), score_claims(claims, today=True)
    live = score_claims(judged, verdicts=verdicts) if verdicts is not None else None
    for measure, *columns in (
        ("shown causes the bank's words support", "shown_supported", "shown"),
        ("partial or unsupported causes kept from the reader", "flagged_removed", "flagged"),
        ("supported causes taken out anyway", "supported_removed", "supported"),
    ):
        numerator, denominator = columns
        cells = [
            share(getattr(score, numerator), getattr(score, denominator))
            for score in (run, today, *([live] if live is not None else []))
        ]
        table.add_row(measure, *cells)
    console.print(table)
    if verdicts is not None:
        agreed = sum(1 for claim in judged if verdicts[claim.claim_id][0] == claim.label)
        console.print(
            f"The checker's verdict matched the claim's label on {agreed}/{len(judged)} claims."
        )
        for claim in judged:
            verdict, reason = verdicts.get(claim.claim_id, ("no verdict", ""))
            if verdict != claim.label:
                console.print(
                    f"  {claim.claim_id} {claim.bank} {claim.driver}: labelled {claim.label}, "
                    f"judged {verdict} ({reason})",
                    markup=False,
                )
        if gate is not None:
            console.print(f"{gate.used} call(s), {aud(gate.spent_usd)}")
    raise typer.Exit(EXIT_OK)


@app.command(name="gold-check")
def gold_check(
    case_id: Annotated[str | None, typer.Argument(help="The case to check.")] = None,
    split: Annotated[str, typer.Option("--split", help="dev, test or all.")] = "dev",
    verifier: Annotated[
        str | None, typer.Option("--verifier", help="Your name, recorded as the labeller.")
    ] = None,
    sheet: Annotated[
        Path | None,
        typer.Option(
            "--sheet",
            help="Write every case not yet checked by a person to this HTML review sheet.",
        ),
    ] = None,
    confirm: Annotated[
        str | None,
        typer.Option(
            "--confirm", help="Case ids, comma-separated as the sheet copies them, to record."
        ),
    ] = None,
    every: Annotated[
        bool, typer.Option("--all", help="Go through every unchecked case, one question each.")
    ] = False,
) -> None:
    """Check the gold figures against their pages, and record who checked them.

    With no case, lists every case with gold and who verified it. With one, shows each
    figure beside the words on the page it cites and asks you to confirm it. Only when you
    confirm every figure is the case recorded as checked by you.

    The quick way through all of them: --sheet writes one page with every figure boxed on
    the page it cites; tick the cases that are right and run the command it copies, which
    is --confirm with their ids, asked once. --all does the same in the terminal, one
    question per case.
    """
    from .evals.cases import load_cases
    from .evals.gold_check import checks_for, record_verification, review_sheet

    settings = get_settings()
    splits = ("dev", "test") if split == "all" else (split,)
    cases = [case for name in splits for case in load_cases(name) if case.has_gold]
    pending = [case for case in cases if case.labeller_kind != "human"]

    def record(case: Any, name: str) -> Path:
        return record_verification(
            settings.evals_dir / "cases" / case.split, case.case_id, name, date.today()
        )

    if sheet is not None:
        sheet.parent.mkdir(parents=True, exist_ok=True)
        sheet.write_text(review_sheet(pending, settings), encoding="utf-8")
        figures = sum(len(checks_for(case, settings)) for case in pending)
        console.print(
            f"Wrote {sheet}: {len(pending)} case(s) and {figures} figure(s) not yet checked "
            "by a person. Open it in a browser."
        )
        raise typer.Exit(EXIT_OK)

    if case_id is None and confirm is None and not every:
        table = Table(title=f"Gold in the {split} split", box=box.SIMPLE)
        table.add_column("case")
        table.add_column("labeller")
        table.add_column("kind")
        for case in cases:
            table.add_row(case.case_id, case.labeller, case.labeller_kind)
        console.print(table)
        human = sum(1 for case in cases if case.labeller_kind == "human")
        console.print(f"{human}/{len(cases)} checked by a person")
        raise typer.Exit(EXIT_OK)

    name = " ".join((verifier or "").split())
    if not name or name == "YOUR NAME":
        console.print("[red]--verifier is required: the record says who checked the figures.[/red]")
        raise typer.Exit(EXIT_ERROR)

    if confirm is not None:
        wanted = [item.strip() for item in confirm.split(",") if item.strip()]
        by_id = {case.case_id: case for case in cases}
        unknown = [item for item in wanted if item not in by_id]
        if unknown or not wanted:
            console.print(
                f"[red]No case with gold in the {split} split: {unknown or confirm}[/red]"
            )
            raise typer.Exit(EXIT_ERROR)
        for item in wanted:
            figures = len(checks_for(by_id[item], settings))
            console.print(f"  {item}  ({figures} figure(s))")
        if not typer.confirm(
            f"Record these {len(wanted)} case(s) as checked by {name}, every figure against "
            "the page it cites?",
            default=False,
        ):
            console.print("[yellow]Nothing recorded.[/yellow]")
            raise typer.Exit(EXIT_ERROR)
        for item in wanted:
            record(by_id[item], name)
        console.print(f"[green]{len(wanted)} case(s) recorded as checked by {name}.[/green]")
        raise typer.Exit(EXIT_OK)

    if every:
        recorded: list[str] = []
        disputed: list[str] = []
        for case in pending:
            console.print(f"\n[bold]{case.case_id}[/bold]  {case.bank} {case.period} {case.metric}")
            for check in checks_for(case, settings):
                console.print(
                    f"  {check.what}: {check.value} {check.unit}, {check.doc_id} p.{check.page}"
                )
                console.print(f"    {check.excerpt}", markup=False)
            answer = typer.prompt("Right? [y]es, [n]o, [s]kip, [q]uit", default="s").lower()
            if answer.startswith("q"):
                break
            if answer.startswith("y"):
                record(case, name)
                recorded.append(case.case_id)
            elif answer.startswith("n"):
                disputed.append(case.case_id)
        console.print(f"[green]{len(recorded)} recorded as checked by {name}.[/green]")
        if disputed:
            console.print(f"[yellow]Disputed, nothing recorded: {', '.join(disputed)}[/yellow]")
        raise typer.Exit(EXIT_OK)

    chosen = next((item for item in cases if item.case_id == case_id), None)
    if chosen is None:
        console.print(f"[red]No case {case_id!r} with gold in the {split} split.[/red]")
        raise typer.Exit(EXIT_ERROR)
    for check in checks_for(chosen, settings):
        console.print(
            f"\n[bold]{check.what}[/bold]: {check.value} {check.unit}, "
            f"{check.doc_id} p.{check.page}"
        )
        console.print(check.excerpt, markup=False)
        if not typer.confirm("Is this right, as the page gives it?", default=False):
            console.print(f"[red]{check.what} not confirmed; nothing recorded.[/red]")
            raise typer.Exit(EXIT_ERROR)
    path = record(chosen, name)
    console.print(f"[green]{chosen.case_id} recorded as checked by {name} in {path}.[/green]")
    raise typer.Exit(EXIT_OK)


@app.command()
def conflicts(
    split: Annotated[str, typer.Option("--split", help="dev or test.")] = "dev",
) -> None:
    """Every conflict the eval cases raise, by class, failing on any unexplained one that
    nobody has reviewed (config/conflicts_reviewed.yaml)."""
    from .evals.cases import load_cases
    from .evals.census import conflict_census

    census = conflict_census(load_cases(split), get_settings())
    table = Table(title=f"Conflicts raised by the {split} split", box=box.SIMPLE)
    table.add_column("bank")
    table.add_column("component")
    table.add_column("class")
    table.add_column("readings")
    table.add_column("cases", justify="right")
    for item, cases in sorted(census.seen.items(), key=lambda kv: (kv[0].bank, kv[0].component)):
        table.add_row(
            item.bank,
            item.component,
            item.classification,
            "; ".join(f"{doc} p.{page}: {value}" for doc, page, value in item.readings),
            str(len(cases)),
        )
    console.print(table)
    console.print(
        "By class: " + (", ".join(f"{k} {v}" for k, v in census.by_class().items()) or "none")
    )
    unreviewed = census.unreviewed()
    for item in unreviewed:
        console.print(
            f"[red]Unreviewed UNEXPLAINED conflict: {item.bank} {item.component} "
            f"({item.rule})[/red]"
        )
    raise typer.Exit(EXIT_ERROR if unreviewed else EXIT_OK)


@app.command()
def stability(
    split: Annotated[str, typer.Option("--split", help="dev or test.")] = "dev",
    runs: Annotated[int, typer.Option("--runs", help="How many times to run each case.")] = 3,
) -> None:
    """Run each case several times and report whether the answers agree.

    The pipeline should give the same result every time. Anything that depends on set or
    dict ordering, float rounding or the clock can drift now and then, and that's easy to
    miss unless you check.
    """
    from .evals.cases import load_cases
    from .evals.runner import run_stability

    settings = get_settings()
    cases = [case for case in load_cases(split) if not case.expects_abstention]
    if not cases:
        console.print(f"[red]No answerable cases in the {split} split.[/red]")
        raise typer.Exit(EXIT_ERROR)

    reports = run_stability(cases, settings=settings, runs=runs)
    table = Table(title=f"Stability over {runs} runs ({split} split)", box=box.SIMPLE)
    table.add_column("case")
    table.add_column("status", justify="right")
    table.add_column("change", justify="right")
    table.add_column("level", justify="right")
    table.add_column("drivers (Jaccard)", justify="right")
    for report in reports:
        table.add_row(
            report.case_id,
            f"{report.status_agreement:.0%}",
            f"{report.change_agreement:.0%}",
            f"{report.level_agreement:.0%}",
            f"{report.driver_jaccard:.3f}",
        )
    console.print(table)

    stable = sum(1 for report in reports if report.stable)
    console.print(f"{stable}/{len(reports)} identical across {runs} runs")
    if stable != len(reports):
        console.print(
            "[red]A deterministic run that differs from itself is a defect, not a tolerance.[/red]"
        )
    raise typer.Exit(EXIT_OK if stable == len(reports) else EXIT_ERROR)


@app.command(name="eval")
def eval_command(
    split: Annotated[str, typer.Option("--split", help="dev or test.")] = "dev",
    replay: Annotated[
        bool,
        typer.Option(
            "--replay",
            help="Answer the model steps from recorded responses only: free and repeatable.",
        ),
    ] = False,
    grant: Annotated[
        str | None,
        typer.Option(
            "--grant",
            help="Run the model steps live, within CALLS:DOLLARS (A$) for the whole split. "
            "Needs a key.",
        ),
    ] = None,
    suite: Annotated[
        str | None,
        typer.Option("--suite", help="ablation, look_ahead, conflict, distractor or checker."),
    ] = None,
    final: Annotated[bool, typer.Option("--final", help="Required for the test split.")] = False,
    smoke: Annotated[bool, typer.Option("--smoke", help="A short CI subset.")] = False,
    limit: Annotated[int | None, typer.Option("--limit", help="First N cases only.")] = None,
    only: Annotated[
        str | None,
        typer.Option(
            "--cases", help="Comma-separated case ids to run, instead of the whole split."
        ),
    ] = None,
    workers: Annotated[
        int, typer.Option("--workers", help="Cases to run at once. Live runs share one grant.")
    ] = 1,
) -> None:
    """Run the eval harness and write its results.

    Scores what doesn't need labelled figures (status, reconciliation, citation accuracy,
    abstentions, latency and cost) and lists which accuracy measures weren't scored and why.

    Without --replay or --grant no model is used, even with a key in the environment.
    --replay answers the model steps from recorded responses only. --grant runs them live
    within one limit for the whole split, and records each response so the split can be
    replayed afterwards for nothing.
    """
    from .evals.cases import load_cases
    from .evals.report import write_report
    from .evals.runner import run_split

    settings = get_settings()

    if suite == "checker":
        _checker_suite(settings, grant=grant, replay=replay)
    if suite is not None:
        from .evals.perturb import SUITES, distractor_note
        from .evals.runner import run_perturbations

        if suite not in SUITES:
            console.print(f"[red]Unknown suite {suite!r}. One of: {', '.join(SUITES)}[/red]")
            raise typer.Exit(EXIT_ERROR)
        if suite == "distractor":
            console.print(f"[yellow]{distractor_note()['skipped']}[/yellow]")
            raise typer.Exit(EXIT_OK)

        cases = load_cases(split)
        results = run_perturbations(cases, suite, settings=settings)
        if not results:
            console.print(f"[yellow]No case in {split} supports the {suite} suite.[/yellow]")
            raise typer.Exit(EXIT_OK)

        table = Table(title=f"{suite} suite ({split} split)", box=box.SIMPLE)
        table.add_column("case")
        table.add_column("perturbation")
        table.add_column("answered")
        table.add_column("drivers", justify="right")
        table.add_column("degraded as expected")
        for result in results:
            table.add_row(
                result.perturbation.case_id,
                result.perturbation.description,
                "yes" if result.answered else "no",
                str(result.drivers),
                "[green]yes[/green]" if result.passed else "[red]no[/red]",
            )
        console.print(table)
        passed = sum(1 for r in results if r.passed)
        console.print(f"{passed}/{len(results)} degraded as expected")
        raise typer.Exit(EXIT_OK if passed == len(results) else EXIT_ERROR)

    cases = load_cases(split)
    if not cases:
        console.print(f"[red]No cases found for the {split} split.[/red]")
        raise typer.Exit(EXIT_ERROR)
    if smoke:
        cases = cases[:6]
    if limit is not None:
        cases = cases[:limit]
    if only:
        wanted = [name.strip() for name in only.split(",") if name.strip()]
        unknown = sorted(set(wanted) - {case.case_id for case in cases})
        if unknown:
            console.print(f"[red]No such case(s) in {split}: {', '.join(unknown)}[/red]")
            raise typer.Exit(EXIT_ERROR)
        cases = [case for case in cases if case.case_id in wanted]

    if replay and grant is not None:
        console.print(
            "[red]--replay and --grant contradict each other: a replay never calls a model.[/red]"
        )
        raise typer.Exit(EXIT_ERROR)
    model = "live" if grant is not None else "replay" if replay else "off"
    try:
        consent = _consent_gate(grant) if grant is not None else None
        run = run_split(
            split,
            settings=settings,
            cases=cases,
            final=final,
            model=model,
            consent=consent,
            workers=max(1, workers),
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_ERROR) from exc
    except ModelUnreachable as stopped:
        # A live eval stops the same way a run does: every case after this would fail for
        # the same reason, and a report of them would read as a result.
        notices.print(f"[red]{_STOPPED[stopped.kind]}[/red] The eval stopped; no report written.")
        raise typer.Exit(EXIT_ERROR) from stopped

    target = write_report(run, cases, settings.evals_dir / "results")
    console.print(
        (target / "summary.md").read_text(encoding="utf-8"), markup=False, highlight=False
    )
    console.print(f"[dim]Results: {target}[/dim]")
    raise typer.Exit(EXIT_OK if run.passed == len(run.scores) else EXIT_ERROR)


@app.command()
def trace(run_id: Annotated[str, typer.Argument(help="Run id under runs/.")]) -> None:
    """Print one run's timeline from its trace.jsonl: each step, and each model call."""
    settings = get_settings()
    folder = settings.runs_dir / run_id
    path = folder / "trace.jsonl"
    if not path.is_file():
        console.print(f"[red]No trace for run {run_id!r} under {settings.runs_dir}.[/red]")
        raise typer.Exit(EXIT_ERROR)

    events: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line cut short by a crash is skipped, not fatal
        if isinstance(loaded, dict):
            events.append(loaded)

    table = Table(title=f"run {run_id}", box=box.SIMPLE, title_justify="left")
    table.add_column("at", justify="right", no_wrap=True)
    table.add_column("step", no_wrap=True)
    table.add_column("what happened", overflow="fold")
    for event in events:
        at = event.get("at")
        when = f"{float(at):.2f}s" if isinstance(at, int | float) else ""
        if event.get("event") == "model_call":
            table.add_row(when, f"  model: {event.get('step', '')}", _model_call_line(event))
        else:
            table.add_row(when, str(event.get("step", "")), _step_line(event))
    console.print(table)

    result = folder / "result.json"
    if result.is_file():
        loaded = json.loads(result.read_text(encoding="utf-8"))
        diagnostics = loaded.get("diagnostics") or {}
        console.print(
            f"{loaded.get('status')}"
            + (f" ({loaded['abstain_reason']})" if loaded.get("abstain_reason") else "")
            + f" · {diagnostics.get('llm_calls', 0)} model call(s)"
            + f" · {aud(diagnostics.get('cost_usd') or '0')}"
            + f" · {int(diagnostics.get('latency_ms') or 0) / 1000:.1f}s"
        )


def _model_call_line(event: dict[str, object]) -> str:
    """One model call, as the trace recorded it: never the request, only its identity."""
    usage = event.get("usage")
    tokens = usage if isinstance(usage, dict) else {}
    sha = str(event.get("prompt_sha256") or "")
    parts = [
        f"{event.get('model')} at {event.get('effort')} effort",
        f"prompt {event.get('prompt')} ({sha[:8]})" if sha else "",
        f"{int(tokens.get('input_tokens') or 0):,} tokens in, "
        f"{int(tokens.get('output_tokens') or 0):,} out",
        "answered from the cache" if event.get("cached") else "",
        aud(str(event["cost_usd"]), 4) if event.get("cost_usd") else "",
    ]
    return ", ".join(part for part in parts if part)


def _step_line(event: dict[str, object]) -> str:
    """A step's own fields, each cut to a readable length; the file has them in full."""
    shown: list[str] = []
    for key, value in event.items():
        if key in {"step", "at"}:
            continue
        text = json.dumps(value) if isinstance(value, list | dict) else str(value)
        shown.append(f"{key}: {text if len(text) <= 160 else text[:157] + '...'}")
    return "; ".join(shown)


@app.command()
def facts(
    bank: Annotated[str, typer.Option("--bank", help="Ticker, e.g. NAB.")],
    period: Annotated[str, typer.Option("--period", help="1H26, 2H25, FY26.")],
    as_of: Annotated[
        str | None,
        typer.Option("--as-of", help="YYYY-MM-DD. Defaults to today."),
    ] = None,
) -> None:
    """Show the validated figures a run for this bank and period could read.

    Read exactly as a run reads them: this bank only, the period and its comparison period
    only, and only from documents published by the as-of date.
    """
    from .domain.calendar import AmbiguousPeriodError, fiscal_period, parse_period
    from .domain.models import RequestSpec
    from .domain.registry import UnknownBankError

    registry = get_registry()
    try:
        profile = registry.bank(bank)
        current = fiscal_period(parse_period(period), profile.fy_end_month)
        when = _parse_as_of(as_of) or date.today()
    except (UnknownBankError, AmbiguousPeriodError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_ERROR) from exc
    prior = current.pcp()
    # The same scoped read the agent makes. The metric only sets the headline label, which
    # the read doesn't use, so any metric in scope gives the same figures.
    spec = RequestSpec(
        run_id="facts",
        bank_id=profile.bank_id,
        metric_id="nim",
        period=current.label,
        comparison_period=prior.label,
        period_start=current.start,
        period_end=current.end,
        comparison_start=prior.start,
        comparison_end=prior.end,
        as_of=when,
    )
    store = FactStore(connect(get_settings().db_path))
    held = store.get_facts(spec)
    if not held:
        console.print(
            f"[yellow]No validated figures for {profile.bank_id} {current.label} or "
            f"{prior.label} published by {when.isoformat()}.[/yellow] "
            "Run `bankresearch ingest`, or `bankresearch discover` for a period outside the "
            "corpus."
        )
        raise typer.Exit(EXIT_OK)

    table = Table(
        title=f"{profile.bank_id} · {current.label} and {prior.label} · as of {when.isoformat()}",
        box=box.SIMPLE,
        title_justify="left",
    )
    table.add_column("component", no_wrap=True)
    table.add_column("period", no_wrap=True)
    table.add_column("as printed", justify="right", no_wrap=True)
    table.add_column("unit", no_wrap=True)
    table.add_column("basis")
    table.add_column("source", no_wrap=True)
    # The title names the bank, so each source drops the bank from its document id.
    prefix = f"{profile.bank_id.lower()}-"
    for fact in sorted(held, key=lambda f: (f.component, f.period != current.label, f.doc_id)):
        printed = fact.published_value if fact.published_value is not None else fact.value
        document = fact.doc_id.removeprefix(prefix)
        table.add_row(
            fact.component,
            fact.period,
            str(printed),
            fact.unit,
            describe_basis(fact.basis),
            f"{document} p.{fact.page_no}",
        )
    console.print(table)
    documents = len({fact.doc_id for fact in held})
    console.print(
        f"{len(held)} validated figure(s) from {documents} document(s). "
        f"{store.quarantined_count()} figure(s) across the whole store failed an identity "
        "check at ingestion and are quarantined, where no run can read them."
    )


# --- helpers --------------------------------------------------------------


def _parse_as_of(raw: str | None) -> date | None:
    if raw is None:
        return None
    if raw.strip().lower() == "today":
        return date.today()
    try:
        return date.fromisoformat(raw.strip())
    except ValueError as exc:
        raise ValueError(f"--as-of must be YYYY-MM-DD or 'today', got {raw!r}") from exc


def _print_spec(spec) -> None:  # type: ignore[no-untyped-def]
    registry = get_registry()
    definition = registry.metrics[spec.metric_id]
    table = Table(
        title=f"{spec.bank_id} · {definition.display_name} · "
        f"{spec.period} vs {spec.comparison_period}"
    )
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Comparison basis", "prior corresponding period (same period, FY-1)")
    table.add_row(
        spec.period, f"{spec.period_start} to {spec.period_end} ({spec.period_days} days)"
    )
    table.add_row(
        spec.comparison_period,
        f"{spec.comparison_start} to {spec.comparison_end} ({spec.comparison_days} days)",
    )
    table.add_row("Headline label", spec.headline_label or "(detected at ingest)")
    table.add_row("As of", spec.as_of.isoformat())
    table.add_row("Run id", spec.run_id)
    console.print(table)
    for caveat in caveats_for(spec):
        console.print(f"[yellow]Caveat:[/yellow] {caveat}")


def _print_abstention(signal: AbstainSignal) -> None:
    console.print(f"[bold]ABSTAIN[/bold] [cyan]{signal.reason}[/cyan]")
    console.print(signal.detail)


if __name__ == "__main__":  # pragma: no cover
    app()
