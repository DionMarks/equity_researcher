"""Test-wide fixtures.

The only thing here makes sure the suite doesn't read the developer's shell environment.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

#: Environment variables that change what the system does, so they mustn't leak in from
#: whoever is running the tests.
#:
#: ``ANTHROPIC_API_KEY`` is the one that matters. With a key set, ``client.available`` is
#: true, the orchestrator takes its model branch, the consent gate refuses the call, and the
#: run degrades down a different path than with no key at all: same figures, different
#: warnings. So ``make check`` passed on a machine with no key and failed on one with a key
#: exported. That kind of failure looks like a flake, only reproduces by accident, and makes
#: CI and your laptop disagree about whether the build is green.
#:
#: A test that wants a key sets one with ``monkeypatch.setenv``. That's function-scoped, so
#: it applies after this and still wins; the key just has to be asked for rather than
#: inherited.
LEAKY_ENV = (
    "ANTHROPIC_API_KEY",
    "BANKRESEARCH_LLM_CONSENT",
)


@pytest.fixture(autouse=True)
def _no_banks_left_behind() -> Iterator[None]:
    """Take away every bank found while a test ran, so the next test starts without it.

    The registry is shared by the whole process, and a run recalls the banks a live run kept
    in the store. Left registered, one test's bank decides whether the next one's name is
    known.
    """
    from bankresearch.domain.registry import get_registry
    from bankresearch.ingestion.facts import line_item_index

    yield
    banks = get_registry().banks
    found = [bank_id for bank_id, profile in banks.items() if profile.discovered]
    for bank_id in found:
        banks.pop(bank_id, None)
    if found:
        line_item_index.cache_clear()


@pytest.fixture(scope="session", autouse=True)
def _no_asx_site() -> Iterator[None]:
    """Keep the ASX's own site out of the tests.

    A run that has to find a filing reads the ASX's list of the bank's announcements
    first. A test that wants that list passes its own pages to ``AsxListing``; anything
    else gets an error, which sends the run on to its search, as an unreachable site would.
    """
    from bankresearch.acquisition import asx_announcements

    def refuse(url: str) -> str:
        raise OSError(f"the tests don't reach {url}")

    patch = pytest.MonkeyPatch()
    patch.setattr(asx_announcements, "network_get", refuse)
    yield
    patch.undo()


@pytest.fixture(scope="session", autouse=True)
def _hermetic_environment() -> Iterator[None]:
    """Clear those variables for the whole session.

    Session-scoped on purpose. A function-scoped fixture is set up after every module-scoped
    one, so module fixtures that build a run (which is exactly where the model branch is
    taken) would still see the key from the shell. That's how this was missed the first
    time.
    """
    patch = pytest.MonkeyPatch()
    for name in LEAKY_ENV:
        patch.delenv(name, raising=False)
    yield
    patch.undo()


@pytest.fixture
def scripted_model(monkeypatch):  # type: ignore[no-untyped-def]
    """`bankresearch run` with a model that answers from a script.

    Every answer goes through a model now, so a test that runs the command needs one. This
    gives the run a key it never sends and a client that never leaves the machine: each
    call gets an empty reply unless a test puts something else in ``client.responses``.
    """
    from pydantic import SecretStr

    from bankresearch import cli
    from bankresearch.agent.orchestrator import Agent
    from bankresearch.llm.client import LLMResponse, ScriptedClient, Usage

    monkeypatch.setattr(
        cli,
        "_with_the_key",
        lambda settings: settings.model_copy(update={"anthropic_api_key": SecretStr("test")}),
    )
    client = ScriptedClient(on_exhausted=lambda request: LLMResponse(usage=Usage(100, 10)))
    monkeypatch.setattr(Agent, "_built_client", lambda self: client)
    return client


def pytest_terminal_summary(terminalreporter, exitstatus, config):  # type: ignore[no-untyped-def]
    """Print a clear notice when the real-filing tests didn't run.

    A fresh clone has no ingested corpus, so every test that checks a figure against its
    page is skipped, correctly, since there's nothing to check. But the skips were silent
    apart from a lower-case "s" in the progress line, so ``make check`` passed on a clone
    while testing a lot less than on a machine with the data, and the next engineer had no
    way to know.
    """
    skipped = terminalreporter.stats.get("skipped", [])
    corpus = [
        report
        for report in skipped
        if "no ingested corpus" in str(getattr(report, "longrepr", ""))
        or "corpus holds no facts" in str(getattr(report, "longrepr", ""))
    ]
    if not corpus:
        return
    terminalreporter.write_sep("=", "the real-filing tests did not run", yellow=True, bold=True)
    terminalreporter.write_line(
        f"{len(corpus)} test(s) skipped because no corpus is ingested. They check "
        "extracted figures against the pages of the banks' own filings, so this run "
        "did not test extraction against real documents."
    )
    terminalreporter.write_line("Run `make ingest` (network, about two minutes), then test again.")
