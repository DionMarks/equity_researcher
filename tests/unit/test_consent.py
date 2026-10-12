"""The consent gate (see ``llm/consent.py``)."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import pytest

from bankresearch.llm.anthropic_client import AnthropicClient, build_payload
from bankresearch.llm.client import LLMRequest, LLMUnavailable
from bankresearch.llm.consent import (
    CHARS_PER_TOKEN,
    OVERRIDE_ENV,
    AlwaysApprove,
    BudgetedGrant,
    CallEstimate,
    ConsentDenied,
    DenyAll,
    TerminalConsent,
    default_gate,
    estimate_call,
)

PRICING = {
    "claude-opus-5-5": {"input": "4.00", "output": "20.00", "cache_read": "0.20"},
    "claude-sonnet-5-5": {"input": "2.00", "output": "10.00", "cache_read": "0.10"},
}


@dataclass
class SpySDK:
    """Records whether the provider was reached. Never reached is the assertion."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "end_turn"

    @property
    def messages(self) -> SpySDK:
        return self

    def create(self, **payload: Any) -> dict[str, Any]:
        self.calls.append(payload)
        return {
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": self.stop_reason,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }


def a_request(**overrides: Any) -> LLMRequest:
    base: dict[str, Any] = {
        "step": "explain",
        "model": "claude-opus-5-5",
        "effort": "high",
        "system": "You explain movements from evidence.",
        "messages": ({"role": "user", "content": "why did the margin fall?"},),
    }
    base.update(overrides)
    return LLMRequest(**base)


def a_client(gate: Any, sdk: SpySDK | None = None) -> tuple[AnthropicClient, SpySDK]:
    spy = sdk or SpySDK()
    return AnthropicClient(consent=gate, pricing=PRICING, _sdk=spy), spy


# --- default deny ---------------------------------------------------------


def test_the_default_gate_refuses():
    """An absent decision must not read as yes."""
    assert isinstance(default_gate(), DenyAll)


@pytest.mark.parametrize("value", ["", "deny", "DENY", "yes", "true", "1", "always", "on"])
def test_only_the_word_ask_installs_an_asking_gate(monkeypatch, value):
    """Every plausible "just let it through" spelling refuses.

    ``yes``, ``true``, ``1`` and ``on`` are what someone reaches for when a prompt is in the
    way, and each would switch off a safeguard by accident.
    """
    monkeypatch.setenv(OVERRIDE_ENV, value)
    assert isinstance(default_gate(), DenyAll)


def test_the_asking_gate_is_installed_only_by_the_full_word(monkeypatch):
    monkeypatch.setenv(OVERRIDE_ENV, "ask")
    assert isinstance(default_gate(), TerminalConsent)


def test_a_client_built_with_no_gate_cannot_reach_the_provider(monkeypatch):
    monkeypatch.delenv(OVERRIDE_ENV, raising=False)
    client, spy = a_client(default_gate())
    with pytest.raises(ConsentDenied):
        client.complete(a_request())
    assert spy.calls == [], "the provider was reached without consent"


def test_an_agent_given_no_gate_keeps_the_refusing_one():
    """The Agent must not weaken the client's default by passing something laxer."""
    import inspect

    from bankresearch.agent.orchestrator import Agent

    assert "consent" in inspect.signature(Agent).parameters
    source = inspect.getsource(Agent)
    assert "AlwaysApprove" not in source, "the orchestrator must never install AlwaysApprove"


# --- refusal paths --------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    ["n\n", "no\n", "\n", "", "  \n", "maybe\n", "Y E S\n", "yeah\n", "q\n", "sure\n"],
)
def test_anything_but_yes_refuses(answer):
    gate = TerminalConsent(stream=io.StringIO(answer), out=io.StringIO(), require_tty=False)
    client, spy = a_client(gate)
    with pytest.raises(ConsentDenied):
        client.complete(a_request())
    assert spy.calls == []
    assert gate.refusals == 1


@pytest.mark.parametrize("answer", ["y\n", "yes\n", "Y\n", "YES\n", " y \n"])
def test_only_yes_approves(answer):
    gate = TerminalConsent(stream=io.StringIO(answer), out=io.StringIO(), require_tty=False)
    client, spy = a_client(gate)
    client.complete(a_request())
    assert len(spy.calls) == 1
    assert gate.approvals == 1


def test_a_stream_that_cannot_be_read_refuses():
    """A closed or broken stdin is not a yes."""

    class Broken:
        def isatty(self) -> bool:
            return True

        def readline(self) -> str:
            raise OSError("stdin is closed")

    gate = TerminalConsent(stream=Broken(), out=io.StringIO())
    client, spy = a_client(gate)
    with pytest.raises(ConsentDenied):
        client.complete(a_request())
    assert spy.calls == []


def test_a_non_interactive_shell_refuses_even_when_the_answer_says_yes():
    """Where consent can't be asked for, it hasn't been given.

    This is the case that matters in CI and in a cron job, where something might otherwise
    read a stray "y" from a pipe and spend money unattended.
    """
    gate = TerminalConsent(stream=io.StringIO("y\n"), out=io.StringIO(), require_tty=True)
    client, spy = a_client(gate)
    with pytest.raises(ConsentDenied):
        client.complete(a_request())
    assert spy.calls == []


# --- every charged call is asked about ------------------------------------


def test_the_fallback_retry_is_asked_about_separately():
    """A fallback-model retry is a second charge, so it gets a second question.

    Asking once and retrying without asking would be the obvious shortcut, and it would mean
    one approval could authorise two calls on two different models.
    """
    answers = io.StringIO("y\nn\n")
    gate = TerminalConsent(stream=answers, out=io.StringIO(), require_tty=False)
    spy = SpySDK(stop_reason="refusal")
    client, _ = a_client(gate, spy)
    with pytest.raises(ConsentDenied):
        client.complete(a_request())
    assert len(spy.calls) == 1, "the first call was approved; the fallback was not"
    assert gate.approvals == 1
    assert gate.refusals == 1


def test_the_fallback_estimate_says_why_it_is_being_asked_again():
    out = io.StringIO()
    gate = TerminalConsent(stream=io.StringIO("y\nn\n"), out=out, require_tty=False)
    client, _ = a_client(gate, SpySDK(stop_reason="refusal"))
    with pytest.raises(ConsentDenied):
        client.complete(a_request())
    text = out.getvalue()
    assert "first attempt" in text
    assert "fallback model, after" in text


def test_the_step_and_model_reach_the_prompt():
    out = io.StringIO()
    gate = TerminalConsent(stream=io.StringIO("n\n"), out=out, require_tty=False)
    client, _ = a_client(gate)
    with pytest.raises(ConsentDenied):
        client.complete(a_request(step="verify", model="claude-sonnet-5-5"))
    text = out.getvalue()
    assert "verify" in text
    assert "claude-sonnet-5-5" in text
    assert "PAID API CALL" in text


# --- the estimate ---------------------------------------------------------


def test_the_estimate_errs_high_on_token_count():
    """Four characters per token understates Claude 4.7's tokenizer by about a quarter.

    Rounding the other way would make every quoted figure a minimum presented as an
    estimate.
    """
    assert CHARS_PER_TOKEN == 3
    estimate = estimate_call(
        build_payload(a_request(messages=({"role": "user", "content": "x" * 300},))),
        step="explain",
        pricing=PRICING,
    )
    assert estimate.input_tokens >= 100


def test_the_quoted_figure_is_the_ceiling_not_a_guess():
    """Output tokens are unknowable before the call, so the figure assumes all of them."""
    estimate = estimate_call(
        build_payload(a_request(max_tokens=1000)), step="explain", pricing=PRICING
    )
    # 1,000 output tokens at $20/MTok.
    assert estimate.output_ceiling_usd == Decimal("0.0200")
    assert estimate.ceiling_usd == estimate.input_usd + estimate.output_ceiling_usd


def test_a_cache_marked_prefix_is_priced_at_the_write_multiple():
    """A cache *write* costs 1.25x base input, so quoting base understates it."""
    estimate = CallEstimate(
        step="explain",
        model="claude-opus-5-5",
        effort="high",
        input_tokens=0,
        system_tokens=1_000_000,
        max_output_tokens=0,
        attempt="first attempt",
        input_price=Decimal("4.00"),
        output_price=Decimal("20.00"),
    )
    assert estimate.input_usd == Decimal("5.0000")


def test_a_server_side_search_is_priced_into_the_ceiling():
    """A$14.33 per 1,000 searches is a charge the token prices do not cover."""
    estimate = estimate_call(
        build_payload(
            a_request(tools=({"type": "web_search_20250305", "name": "web_search", "max_uses": 5},))
        ),
        step="discover_classify",
        pricing=PRICING,
    )
    assert estimate.searches == 5
    assert estimate.search_usd == Decimal("0.0500")
    assert estimate.search_usd <= estimate.ceiling_usd


def test_an_unpriced_model_is_refused_rather_than_treated_as_free():
    """A model missing from config/models.yaml has an unknown cost, not a zero one."""
    estimate = estimate_call(
        build_payload(a_request(model="claude-something-new")), step="explain", pricing=PRICING
    )
    assert estimate.priced is False
    assert "unknown" in " ".join(estimate.lines())
    assert BudgetedGrant(calls=5, ceiling_usd=Decimal("1.00")).approve(estimate) is False


def test_every_routed_model_can_be_priced():
    """A routing entry pointing at an unpriced model would quote 'unknown' at the prompt."""
    from bankresearch.domain.registry import get_registry

    models = get_registry().models
    pricing = models["pricing"]
    for step, spec in models["routing"].items():
        assert spec["model"] in pricing, f"{step} routes to unpriced {spec['model']}"
    assert models["fallback_model"] in pricing


# --- a grant is bounded by numbers, not by patience -----------------------


def test_a_grant_stops_at_its_call_count():
    grant = BudgetedGrant(calls=2, ceiling_usd=Decimal("100"))
    client, spy = a_client(grant)
    client.complete(a_request())
    client.complete(a_request(step="verify"))
    with pytest.raises(ConsentDenied):
        client.complete(a_request(step="chart_read"))
    assert len(spy.calls) == 2
    assert any("used up" in reason for reason in grant.refusals)


def test_a_grant_stops_at_its_dollar_ceiling():
    grant = BudgetedGrant(calls=100, ceiling_usd=Decimal("0.05"))
    client, spy = a_client(grant)
    client.complete(a_request(max_tokens=1000))
    with pytest.raises(ConsentDenied):
        client.complete(a_request(max_tokens=4096))
    assert len(spy.calls) == 1
    assert any("would take the grant to" in reason for reason in grant.refusals)


def test_a_grant_decides_on_the_ceiling_and_then_settles_to_actual():
    """The ceiling is right for a single call and wrong as a running total.

    It assumes every output token, so ten calls at an A$0.13 ceiling commit A$1.29, when the
    ten calls actually came to A$0.10. An A$0.72 grant refused its tenth call with 86% of the
    money unspent. Deciding on the ceiling keeps the guarantee; settling afterwards keeps
    the grant worth what the person chose.
    """
    grant = BudgetedGrant(calls=10, ceiling_usd=Decimal("1.00"))
    client, _spy = a_client(grant)
    client.complete(a_request(max_tokens=4096))

    assert grant.used == 1
    assert grant.pending_ceiling_usd == 0, "settled, so nothing is still in flight"
    # The spy reports 10 input and 5 output tokens, which on Opus 5.5 is a fraction of a
    # cent, far below the ceiling a 4,096-token response was judged against.
    assert 0 < grant.spent_usd < Decimal("0.001")
    assert grant.committed_usd == grant.spent_usd


def test_a_settled_grant_permits_the_calls_the_money_allows():
    """What the over-commitment cost: a run refused while it had barely spent anything."""
    grant = BudgetedGrant(calls=30, ceiling_usd=Decimal("0.10"))
    client, spy = a_client(grant)
    for _ in range(20):
        client.complete(a_request(max_tokens=4096))
    assert len(spy.calls) == 20, f"refused early: {grant.refusals[:2]}"
    assert grant.spent_usd < Decimal("0.10")


def test_a_failed_call_leaves_its_ceiling_committed():
    """A call that raised may still have been billed, so the safe thing is to keep the
    ceiling committed rather than settle it to nothing.
    """
    grant = BudgetedGrant(calls=5, ceiling_usd=Decimal("1.00"))
    sdk = SpySDK()

    def boom(**_payload):
        raise RuntimeError("invalid_request_error: bad schema")

    sdk.create = boom  # type: ignore[method-assign]
    client = AnthropicClient(consent=grant, pricing=PRICING, _sdk=sdk)
    with pytest.raises(LLMUnavailable):
        client.complete(a_request())
    assert grant.pending_ceiling_usd > 0
    assert grant.spent_usd == 0


# --- the key never reaches the gate, the prompt or a trace ----------------


def test_the_prompt_cannot_print_the_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-must-not-be-printed")
    out = io.StringIO()
    gate = TerminalConsent(stream=io.StringIO("n\n"), out=out, require_tty=False)
    client, _ = a_client(gate)
    with pytest.raises(ConsentDenied) as exc:
        client.complete(a_request())
    assert "sk-ant-must-not-be-printed" not in out.getvalue()
    assert "sk-ant-must-not-be-printed" not in str(exc.value)


def test_the_estimate_holds_no_credential():
    """The gate is given a payload and a price list and nothing else."""
    estimate = estimate_call(build_payload(a_request()), step="explain", pricing=PRICING)
    for field_name in type(estimate).__dataclass_fields__:
        assert "key" not in field_name
        assert "token" not in field_name or field_name.endswith("_tokens")


# --- the deliberate bypass is conspicuous ---------------------------------


def test_the_test_only_gate_is_not_reachable_from_the_cli_or_config():
    """``AlwaysApprove`` is for tests against an injected SDK and nowhere else.

    If it ever shows up in the CLI, the orchestrator or a config file, the safeguard has a
    known hole, and this test is what reports it.

    Matched on references rather than text, by parsing each module: naming the class in a
    comment or docstring ("optional on the gate: AlwaysApprove has no budget to settle
    against") is explanation, not a hole, and a check that can't tell the two apart ends up
    being worked around instead of listened to.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    offenders: list[str] = []

    for path in (root / "src").rglob("*.py"):
        if path.name == "consent.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            named = (
                (isinstance(node, ast.Name) and node.id == "AlwaysApprove")
                or (isinstance(node, ast.Attribute) and node.attr == "AlwaysApprove")
                or (
                    isinstance(node, ast.ImportFrom)
                    and any(alias.name == "AlwaysApprove" for alias in node.names)
                )
            )
            if named:
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")

    for path in (root / "config").rglob("*.yaml"):
        if "AlwaysApprove" in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(root)))

    assert not offenders, f"AlwaysApprove reachable from {offenders}"


def test_approving_everything_still_requires_saying_so():
    client, spy = a_client(AlwaysApprove())
    client.complete(a_request())
    assert len(spy.calls) == 1


# --- the key cannot reach the repository ----------------------------------


def test_the_credential_scanner_catches_a_real_key_and_spares_a_placeholder(tmp_path):
    """Both halves matter.

    A scanner that misses a real key is useless. One that trips on this project's own test
    fixtures (``sk-ant-must-not-leak``, which exists so a test can check a key never reaches
    a log) gets switched off, and then it's useless too.
    """
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts" / "scan_secrets.py"

    def scan(body: str) -> int:
        probe = tmp_path / "probe.py"
        probe.write_text(body, encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(script), str(probe)], capture_output=True, text=True
        ).returncode

    # Realistic-looking key material, in each form worth refusing. Built from fragments at
    # run time rather than written out, because a complete key-shaped string in a tracked
    # file is exactly what the scanner refuses, and it would refuse this file. (It did, when
    # they were written out in full. That's the scanner working.)
    anthropic = "sk-" + "ant-api03-" + "Zx8KqLm2NpQ7rStUvWxYz0123456789AbCdEf"
    key_id = "api" + "key_" + "01AbCdEf23GhIjKl45MnOpQr"
    github = "gh" + "p_" + "AbCdEf0123456789AbCdEf0123456789AbCd"
    aws = "AK" + "IA" + "IOSFODNN7EXAMPLE"
    for secret in (anthropic, key_id, github, aws):
        assert scan(f'K = "{secret}"') == 1, f"not caught: {secret[:10]}..."

    # This project's deliberate stand-ins, which must keep working.
    assert scan('setenv("ANTHROPIC_API_KEY", "sk-ant-must-not-leak")') == 0
    assert scan('setenv("ANTHROPIC_API_KEY", "sk-ant-must-not-appear")') == 0
    assert scan('K = "sk-ant-replace-me"') == 0
    assert scan("K = os.environ['ANTHROPIC_API_KEY']") == 0


def test_the_whole_tree_is_free_of_credentials():
    """Runs in `make check`, so a key pasted into any tracked file fails the build."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    done = subprocess.run(
        [sys.executable, str(root / "scripts" / "scan_secrets.py")],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert done.returncode == 0, done.stderr


def test_the_key_is_read_from_the_environment_and_never_from_the_tree():
    """A key in a file is a key that gets committed eventually."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    settings = (root / "src" / "bankresearch" / "settings.py").read_text(encoding="utf-8")
    assert 'alias="ANTHROPIC_API_KEY"' in settings
    assert "SecretStr" in settings
    # `.env` is read by pydantic-settings for local convenience and is gitignored; no
    # other path may hold it.
    assert ".env" in (root / ".gitignore").read_text(encoding="utf-8")


# --- a refusal degrades the answer; it does not raise ---------------------


def test_a_refusal_degrades_the_run_rather_than_crashing_it(tmp_path):
    """A budget overrun should weaken the answer rather than raise, and
    someone declining to spend should get the same treatment.

    An analyst who says no should get the deterministic answer with a note saying what was
    skipped. A traceback would make the safeguard something to work around.
    """
    from datetime import date

    from bankresearch.agent.orchestrator import Agent
    from bankresearch.memory.db import connect
    from bankresearch.memory.fact_store import FactStore
    from bankresearch.settings import get_settings

    settings = get_settings()
    if not settings.db_path.exists():
        pytest.skip("no ingested corpus: run `make ingest` first")

    class RefusingClient:
        """Available, but refuses every call, which is what a declined approval looks like."""

        available = True

        def complete(self, request: Any) -> Any:
            raise ConsentDenied("declined for the purposes of this test")

    with connect(settings.db_path) as connection:
        agent = Agent(
            store=FactStore(connection),
            settings=settings,
            persist=False,
            client=RefusingClient(),
        )
        outcome = agent.run("ANZ", "1H26", "nim", as_of=date(2026, 5, 31))

    result = outcome.result
    assert result.status in {"OK", "PARTIAL"}, "the deterministic path still answers"
    assert result.current is not None and result.prior is not None
    assert result.change is not None
    assert any("declined" in caveat for caveat in result.caveats), (
        "the memo must say a call was declined, not quietly omit the step"
    )
    assert any("declined" in str(event.get("skipped", "")) for event in outcome.trace), (
        "the trace must record which step was skipped and why"
    )


# --- an image is priced by its area, not by its encoding ------------------


def test_an_image_is_priced_by_area_not_by_payload_length():
    """A rendered PDF page is about 270,000 characters of base64 but about 3,000 tokens.

    Counting the encoding as text quotes 30 times the real cost. That isn't a cautious
    estimate, it's a wrong one: it trips the per-run limit on a call costing a few cents,
    and a prompt whose number is obviously absurd is one people learn to click through. The
    provider bills an image at roughly width x height / 750.
    """
    import base64

    from bankresearch.llm.consent import IMAGE_PIXELS_PER_TOKEN, _tokens_in

    # A 1,000 x 1,500 PNG, built by hand so the test needs no corpus.
    width, height = 1000, 1500
    ihdr = (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
    )
    # Pad so the encoded form is large enough to prove length is not what is counted.
    png = ihdr + b"\x00" * 200_000
    block = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.standard_b64encode(png).decode(),
        },
    }

    expected = width * height // IMAGE_PIXELS_PER_TOKEN
    assert _tokens_in(block) == expected == 2000
    # The encoded string is ~267,000 characters; counted as text that is ~89,000 tokens.
    assert len(block["source"]["data"]) > 200_000
    assert _tokens_in(block) < 3_000


def test_an_image_whose_header_is_unreadable_is_not_treated_as_free():
    """An unknown format gets a conservative page-sized figure, not zero."""
    from bankresearch.llm.consent import _FALLBACK_IMAGE_TOKENS, _tokens_in

    for data in ("not-base64-at-all!!", "", base64.b64encode(b"\xff\xd8\xff\xe0jpeg").decode()):
        block = {"type": "image", "source": {"type": "base64", "data": data}}
        assert _tokens_in(block) == _FALLBACK_IMAGE_TOKENS


def test_an_image_in_a_message_reaches_the_estimate():
    """The block is nested two deep in a real payload, so the walk has to recurse."""
    import base64

    from bankresearch.llm.consent import estimate_call

    ihdr = (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + (900).to_bytes(4, "big")
        + (1200).to_bytes(4, "big")
    )
    encoded = base64.standard_b64encode(ihdr + b"\x00" * 50_000).decode()
    request = a_request(
        messages=(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What does this chart show?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": encoded,
                        },
                    },
                ],
            },
        )
    )
    estimate = estimate_call(build_payload(request), step="chart_read", pricing=PRICING)
    # 900 x 1200 / 750 = 1,440, plus a few tokens of prose. Not the ~22,000 the base64
    # string would have contributed.
    assert 1_400 < estimate.input_tokens < 1_600


# --- the suite must not read the developer's shell ------------------------


def test_the_environment_guard_is_session_scoped():
    """A function-scoped guard is set up after every module-scoped fixture.

    That's how this was missed: ``make check`` passed with no key in the shell and failed
    with one exported, because a module fixture built its run before the guard ran. The
    scope is the whole fix, so the scope is what this checks.
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "conftest.py").read_text(encoding="utf-8")
    assert '"ANTHROPIC_API_KEY"' in source
    assert '"BANKRESEARCH_LLM_CONSENT"' in source
    assert 'scope="session"' in source, "a function-scoped guard runs too late"
    assert "autouse=True" in source


def test_no_ambient_key_is_visible_to_a_test():
    """Whatever is in the shell, a test sees none of it unless it asks."""
    import os

    assert os.environ.get("ANTHROPIC_API_KEY") is None
    assert os.environ.get("BANKRESEARCH_LLM_CONSENT") is None


def test_a_test_can_still_ask_for_a_key(monkeypatch):
    """The guard removes the default, not the ability."""
    from bankresearch.settings import Settings

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-asked-for-explicitly")
    assert Settings().has_api_key is True


# --- a declined call is visible in diagnostics, not only in prose ---------


def test_a_declined_call_warns_as_well_as_caveats():
    """``llm_calls: 0`` with no warning looks the same as having no model at all.

    Someone reading ``diagnostics.warnings`` (where "No model configured" already goes)
    would otherwise see a run that skipped its judgement steps without saying so.
    """
    from datetime import date

    from bankresearch.agent.orchestrator import Agent
    from bankresearch.memory.db import connect
    from bankresearch.memory.fact_store import FactStore
    from bankresearch.settings import get_settings

    settings = get_settings()
    if not settings.db_path.exists():
        pytest.skip("no ingested corpus: run `make ingest` first")

    class RefusingClient:
        available = True

        def complete(self, request: Any) -> Any:
            raise ConsentDenied("declined for the purposes of this test")

    with connect(settings.db_path) as connection:
        outcome = Agent(
            store=FactStore(connection),
            settings=settings,
            persist=False,
            client=RefusingClient(),
        ).run("ANZ", "1H26", "nim", as_of=date(2026, 5, 31))

    result = outcome.result
    assert result.diagnostics.llm_calls == 0
    assert any("consent" in w.lower() for w in result.diagnostics.warnings), (
        "a declined call must appear in diagnostics.warnings, not only in caveats"
    )
    assert any("declined" in caveat for caveat in result.caveats)


# --- what a run cost must reach the result --------------------------------


def test_what_a_run_spent_is_reported_on_the_result():
    """The budget counted correctly but nothing copied its totals out.

    Every run reported ``llm_calls: 0`` and ``cost_usd: 0`` however much it spent, which
    made the cost in the memo, the trace and the eval report a constant rather than a
    measurement. Nobody can check a cost limit whose spending is invisible.
    """
    from decimal import Decimal

    from bankresearch.agent.orchestrator import _record_spend
    from bankresearch.domain.models import Diagnostics
    from bankresearch.llm.budget import Budget
    from bankresearch.llm.client import Usage

    budget = Budget(pricing={"claude-sonnet-5-5": {"input": "2.00", "output": "10.00"}})
    budget.record("locate", "claude-sonnet-5-5", Usage(input_tokens=1_000, output_tokens=500))
    budget.record("chart_read", "claude-sonnet-5-5", Usage(input_tokens=3_000, output_tokens=200))

    diagnostics = Diagnostics()
    _record_spend(diagnostics, budget)

    assert diagnostics.llm_calls == 2
    assert diagnostics.input_tokens == 4_000
    assert diagnostics.output_tokens == 700
    # 4,000 in at 2 per million tokens plus 700 out at 10, in the US dollars it's billed in.
    assert diagnostics.cost_usd == Decimal("0.015")
    assert diagnostics.budget_breached is False


def test_a_breached_budget_is_reported_as_breached():
    from bankresearch.agent.orchestrator import _record_spend
    from bankresearch.domain.models import Diagnostics
    from bankresearch.llm.budget import Budget

    budget = Budget(max_calls=1)
    assert budget.may_call("locate") is True
    budget.calls = 1
    assert budget.may_call("locate") is False

    diagnostics = Diagnostics()
    _record_spend(diagnostics, budget)
    assert diagnostics.budget_breached is True
    assert any("call cap" in w for w in diagnostics.warnings)


def test_the_chart_reader_counts_against_the_budget():
    """It was the one model call nothing recorded, which is why the totals read zero.

    Chart reading is also the most expensive step per call, so leaving it out meant the
    limit was blindest exactly where it mattered most.
    """
    import inspect

    from bankresearch.ingestion import charts

    source = inspect.getsource(charts.read_walk_chart)
    assert 'budget.record("chart_read"' in source
    assert 'budget.may_call("chart_read")' in source
    assert "budget" in inspect.signature(charts.read_walk_chart).parameters


# --- a cache hit is not a charge ------------------------------------------


def test_a_cached_response_costs_nothing_and_is_not_a_call():
    """A run that made no requests reported 13 model calls and A$0.095.

    A cached response carries the original call's token counts, which is what makes it
    reusable and also what made it get billed twice. Over-reporting isn't harmless: the
    limit would refuse real work to stay under a figure nobody was charged, and a replayed
    eval would hit its call cap on calls it never made.
    """
    from decimal import Decimal

    from bankresearch.llm.budget import Budget
    from bankresearch.llm.client import Usage

    pricing = {"claude-sonnet-5-5": {"input": "2.00", "output": "10.00"}}
    used = Usage(input_tokens=10_000, output_tokens=1_000)

    live = Budget(pricing=pricing)
    live.record("locate", "claude-sonnet-5-5", used)
    assert live.calls == 1
    assert live.cost == Decimal("0.03")

    replayed = Budget(pricing=pricing)
    replayed.record("locate", "claude-sonnet-5-5", used, cached=True)
    assert replayed.calls == 0, "no request was made"
    assert replayed.cost == Decimal(0)
    assert replayed.cache_hits == 1, "counted, but as what it is"
    assert replayed.usage.input_tokens == 0, "the provider processed no tokens"


def test_a_cache_hit_does_not_consume_the_call_cap():
    """Otherwise a replayed eval runs out of budget on calls it never made."""
    from bankresearch.llm.budget import Budget
    from bankresearch.llm.client import Usage

    budget = Budget(max_calls=2)
    for _ in range(10):
        assert budget.may_call("locate") is True
        budget.record("locate", "m", Usage(input_tokens=5), cached=True)
    assert budget.may_call("locate") is True
    assert budget.breached is False


def test_the_cache_marks_what_it_serves():
    """Only the cache can tell; everything above counts from the response it is handed."""
    from bankresearch.llm.cache import CachedClient, ResponseCache
    from bankresearch.llm.client import LLMRequest, LLMResponse, Usage
    from bankresearch.memory.db import connect

    request = LLMRequest(
        step="locate",
        model="claude-sonnet-5-5",
        effort="low",
        system="s",
        messages=({"role": "user", "content": "q"},),
    )
    answer = LLMResponse(text="a", usage=Usage(input_tokens=10, output_tokens=2))

    class Upstream:
        available = True
        calls = 0

        def complete(self, _request: LLMRequest) -> LLMResponse:
            type(self).calls += 1
            return answer

    with connect(":memory:") as connection:
        client = CachedClient(
            cache=ResponseCache(connection), upstream=Upstream(), replay_only=False
        )
        first = client.complete(request)
        second = client.complete(request)

    assert first.cached is False, "the first call really was made"
    assert second.cached is True, "the second came from the cache"
    assert second.text == first.text
    assert Upstream.calls == 1


def test_a_grant_shared_by_parallel_cases_releases_only_each_calls_own_ceiling():
    """Two calls in flight at once. Settling the first used to zero the whole in-flight
    total, so the second call's ceiling stopped counting and the grant could be overrun."""
    grant = BudgetedGrant(calls=10, ceiling_usd=Decimal("1.00"))
    first = estimate_call(
        build_payload(a_request(max_tokens=10000)), step="explain", pricing=PRICING
    )
    second = estimate_call(
        build_payload(a_request(max_tokens=10000)), step="verify", pricing=PRICING
    )
    assert grant.approve(first) and grant.approve(second)
    grant.settle(Decimal("0.01"), first)
    assert grant.pending_ceiling_usd == second.ceiling_usd
    grant.settle(Decimal("0.02"), second)
    assert grant.pending_ceiling_usd == 0 and grant.spent_usd == Decimal("0.03")


def test_a_grant_decides_one_call_at_a_time_across_threads():
    from concurrent.futures import ThreadPoolExecutor

    grant = BudgetedGrant(calls=5, ceiling_usd=Decimal("100"))
    estimate = estimate_call(build_payload(a_request()), step="explain", pricing=PRICING)
    with ThreadPoolExecutor(max_workers=8) as pool:
        approved = list(pool.map(lambda _: grant.approve(estimate), range(40)))
    assert sum(approved) == 5 and grant.used == 5
