"""The consent gate: no call that costs money is made unless a person approved it, call by call or
as a grant with a ceiling."""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
from binascii import Error as BinasciiError
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, Decimal
from typing import Any, Final, Protocol, runtime_checkable

from ..domain.money import aud
from .client import CACHE_WRITE_MULTIPLIER as _CACHE_WRITE_MULTIPLIER

__all__ = [
    "AlwaysApprove",
    "BudgetedGrant",
    "CallEstimate",
    "ConsentDenied",
    "ConsentGate",
    "DenyAll",
    "TerminalConsent",
    "default_gate",
    "estimate_call",
]

#: Characters per token, set low on purpose so the estimate errs high.
#:
#: Claude 4.7 and later use a tokenizer that produces roughly 30% more tokens for the same
#: text than the usual four-characters-per-token rule, so that rule would understate every
#: estimate by about a quarter. Three corrects for that, rounded so the estimate comes out
#: higher rather than lower.
CHARS_PER_TOKEN: Final = 3

#: Image tokens depend on the picture's size, not on how many bytes it took to encode. The
#: provider's documented approximation is width x height / 750.
#:
#: This matters more than it looks. A rendered PDF page is about 1.4MB of base64, which
#: counted as text reads as 470,000 tokens and prices a single chart call at A$1.79. Billed
#: by area, the same image is nearer 1,900 tokens and under a cent. Estimating from the
#: payload length would quote 150 times the real cost, trip the per-run limit on a cheap
#: call, and teach people to ignore the number, which a consent prompt can't afford.
IMAGE_PIXELS_PER_TOKEN: Final = 750

#: A cache-marked prefix costs 1.25x the base input price the first time it's written. The
#: system prompt is always cache-marked, so its first send is charged at this rate, and
#: quoting the base rate would understate the first call of every step.
CACHE_WRITE_MULTIPLIER: Final = _CACHE_WRITE_MULTIPLIER

#: Env var that installs a non-interactive gate. Spelled out in full, with no short form and
#: no default, so it can't be set by accident or out of habit.
OVERRIDE_ENV: Final = "BANKRESEARCH_LLM_CONSENT"

#: An A4 page at the project's render resolution, for an image whose header says nothing.
_FALLBACK_IMAGE_TOKENS: Final = 2_500


class ConsentDenied(RuntimeError):
    """Raised when consent was refused or couldn't be obtained.

    Not a subclass of ``LLMUnavailable``. An unavailable provider is something to work
    around, but a refusal is a decision to respect, so the orchestrator may catch this to
    abstain and nothing retries it.
    """


def _png_dimensions(data: bytes) -> tuple[int, int] | None:
    """Width and height from a PNG's IHDR chunk, or ``None`` if it isn't a PNG.

    Read from the header rather than by decoding the image: 24 bytes answers the question,
    and pulling in an image library just to price a call would be overkill.
    """
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    return (
        int.from_bytes(data[16:20], "big"),
        int.from_bytes(data[20:24], "big"),
    )


def _image_tokens(block: dict[str, Any]) -> int:
    """Tokens for one image block, from its dimensions.

    Falls back to a conservative figure for formats whose header isn't read here. An unknown
    image isn't free, and the fallback is roughly a full page at the project's render
    resolution.
    """
    source = block.get("source") or {}
    encoded = source.get("data")
    if not isinstance(encoded, str):
        return _FALLBACK_IMAGE_TOKENS
    try:
        header = base64.b64decode(encoded[:64], validate=False)
    except (ValueError, BinasciiError):
        return _FALLBACK_IMAGE_TOKENS
    size = _png_dimensions(header)
    if size is None:
        return _FALLBACK_IMAGE_TOKENS
    width, height = size
    return max(1, -(-width * height // IMAGE_PIXELS_PER_TOKEN))


def _tokens_in(value: Any) -> int:
    """A conservative token estimate for any JSON-serialisable fragment.

    Image blocks are priced by area and their encoded bytes are left out, because the two
    differ by more than a hundred times.
    """
    if value is None:
        return 0
    if isinstance(value, dict):
        if value.get("type") == "image":
            return _image_tokens(value)
        return sum(_tokens_in(part) for part in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tokens_in(part) for part in value)
    if not isinstance(value, str):
        value = json.dumps(value, default=str)
    return -(-len(value) // CHARS_PER_TOKEN)  # ceiling division


def _ceil_cents(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.0001"), rounding=ROUND_CEILING)


@dataclass(frozen=True, slots=True)
class CallEstimate:
    """What a call is about to cost, before it is made."""

    step: str
    model: str
    effort: str
    input_tokens: int
    """Everything but the cache-marked system prompt."""
    system_tokens: int
    """The cache-marked prefix, charged at :data:`CACHE_WRITE_MULTIPLIER` on first send."""
    max_output_tokens: int
    attempt: str
    """``first attempt``, or why this is a repeat: a retry or a fallback model."""
    input_price: Decimal
    output_price: Decimal
    priced: bool = True
    """False when the model is absent from ``config/models.yaml``, which must not read
    as free."""
    tools: tuple[str, ...] = ()
    searches: int = 0
    """Server-side searches this call may run, at A$14.33 per 1,000."""
    body_cache_marked: bool = False
    """Whether the conversation itself is cache-marked, as a tool loop's is. Then all of it
    is priced as a cache write, which is the most it can cost."""

    @property
    def input_usd(self) -> Decimal:
        body_rate = CACHE_WRITE_MULTIPLIER if self.body_cache_marked else Decimal(1)
        plain = Decimal(self.input_tokens) * self.input_price * body_rate
        cached = Decimal(self.system_tokens) * self.input_price * CACHE_WRITE_MULTIPLIER
        return _ceil_cents((plain + cached) / Decimal(1_000_000))

    @property
    def output_ceiling_usd(self) -> Decimal:
        return _ceil_cents(Decimal(self.max_output_tokens) * self.output_price / Decimal(1_000_000))

    @property
    def search_usd(self) -> Decimal:
        return _ceil_cents(Decimal(self.searches) * Decimal("10") / Decimal(1_000))

    @property
    def ceiling_usd(self) -> Decimal:
        """The most this call can cost: every output token allowed, nothing cached."""
        return self.input_usd + self.output_ceiling_usd + self.search_usd

    def lines(self) -> list[str]:
        """The estimate as a person needs to read it, not as it is stored."""
        total_input = self.input_tokens + self.system_tokens
        out = [
            f"step            {self.step}  ({self.attempt})",
            f"model           {self.model}  effort={self.effort}",
            f"input           ~{total_input:,} tokens"
            + (f", of which {self.system_tokens:,} cache-marked" if self.system_tokens else ""),
            f"output          up to {self.max_output_tokens:,} tokens",
        ]
        if self.tools:
            out.append(f"tools           {', '.join(self.tools)}")
        if self.searches:
            out.append(f"searches        up to {self.searches} at {aud(10)}/1,000")
        if not self.priced:
            out.append(
                f"COST            unknown - {self.model} is not in config/models.yaml, "
                "so this cannot be priced"
            )
            return out
        out += [
            f"input cost      {aud(self.input_usd, up=True)}",
            f"output cost     {aud(self.output_ceiling_usd, up=True)} at most",
        ]
        if self.searches:
            out.append(f"search cost     {aud(self.search_usd, up=True)} at most")
        out.append(f"MOST IT CAN COST  {aud(self.ceiling_usd, up=True)}")
        return out


def estimate_call(
    payload: dict[str, Any],
    *,
    step: str,
    pricing: dict[str, dict[str, str]],
    attempt: str = "first attempt",
) -> CallEstimate:
    """Price a payload without sending it.

    Estimated from the payload rather than the provider's token-counting endpoint, because
    counting tokens is itself a call: answering the gate's question would mean making a call
    first, and that has to stop somewhere. The estimate errs high instead.
    """
    model = str(payload.get("model", "")) or "(unset)"
    prices = pricing.get(model) or {}
    system_tokens = _tokens_in(payload.get("system"))
    body_tokens = _tokens_in(payload.get("messages")) + _tokens_in(payload.get("tools"))
    tools = tuple(
        str(tool.get("name") or tool.get("type") or "?")
        for tool in payload.get("tools") or ()
        if isinstance(tool, dict)
    )
    searches = sum(
        int(tool.get("max_uses") or 0)
        for tool in payload.get("tools") or ()
        if isinstance(tool, dict) and "web_search" in str(tool.get("type", ""))
    )
    return CallEstimate(
        step=step,
        model=model,
        effort=str((payload.get("output_config") or {}).get("effort", "")),
        input_tokens=body_tokens,
        body_cache_marked="cache_control" in json.dumps(payload.get("messages") or [], default=str),
        system_tokens=system_tokens,
        max_output_tokens=int(payload.get("max_tokens") or 0),
        attempt=attempt,
        input_price=Decimal(str(prices.get("input", "0"))),
        output_price=Decimal(str(prices.get("output", "0"))),
        priced=bool(prices),
        tools=tools,
        searches=searches,
    )


@runtime_checkable
class ConsentGate(Protocol):
    """Whether this specific call may be made."""

    def approve(self, estimate: CallEstimate) -> bool: ...


@dataclass(frozen=True, slots=True)
class DenyAll:
    """Refuse every call. The default, and the only safe default.

    A missing gate mustn't mean no restriction. If this is what's installed, either nobody
    has granted anything or the shell can't ask, and both mean no.
    """

    reason: str = "no consent gate is installed, so no paid call may be made"

    def approve(self, estimate: CallEstimate) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class AlwaysApprove:
    """Approve every call without asking. For tests against an injected SDK.

    Named to stand out at the call site: a test that installs this is saying outright that
    no money is at stake. It's never the default, and the CLI never installs it.
    """

    def approve(self, estimate: CallEstimate) -> bool:
        return True


@dataclass
class BudgetedGrant:
    """A grant of up to ``calls`` calls and ``ceiling_usd`` in total, approved once.

    For the case the per-call prompt handles badly: a 12-step run that would otherwise ask
    twelve times. The grant states both limits, counts against both, and refuses any call
    that would exceed either, so the worst case is bounded by a number the person chose.

    Each call is decided on its ceiling and then settled to what it actually cost. The
    ceiling is pessimistic on purpose (it assumes every output token is used), which is
    right for asking whether one call could blow the budget but badly wrong as a running
    total: ten calls at an A$0.13 ceiling commit A$1.29, when the ten calls actually came
    to A$0.10. An A$0.72 grant refused its tenth call with 86% of the money unspent. Deciding on
    the ceiling keeps the guarantee that no single call can go over the grant, and settling
    afterwards keeps the grant worth what the person chose.
    """

    calls: int
    ceiling_usd: Decimal
    spent_usd: Decimal = Decimal(0)
    """What the completed calls actually cost."""
    pending_ceiling_usd: Decimal = Decimal(0)
    """The ceiling of the call in flight, not yet settled."""
    used: int = 0
    refusals: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    """One grant can cover several cases running at once (``eval --workers``), so deciding
    and settling are each done whole."""

    @property
    def committed_usd(self) -> Decimal:
        """Actual spend plus the call in flight: what the next decision is made against."""
        return self.spent_usd + self.pending_ceiling_usd

    def approve(self, estimate: CallEstimate) -> bool:
        with self._lock:
            return self._approve(estimate)

    def _approve(self, estimate: CallEstimate) -> bool:
        if not estimate.priced:
            self.refusals.append(f"{estimate.step}: unpriced model {estimate.model}")
            return False
        if self.used >= self.calls:
            self.refusals.append(f"{estimate.step}: the grant of {self.calls} call(s) is used up")
            return False
        would_be = self.committed_usd + estimate.ceiling_usd
        if would_be > self.ceiling_usd:
            self.refusals.append(
                f"{estimate.step}: at {aud(self.spent_usd)} spent, its ceiling of "
                f"{aud(estimate.ceiling_usd, up=True)} would take the grant to "
                f"{aud(would_be, up=True)} of {aud(self.ceiling_usd)}"
            )
            return False
        self.used += 1
        self.pending_ceiling_usd += estimate.ceiling_usd
        return True

    def settle(self, actual: Decimal, estimate: CallEstimate | None = None) -> None:
        """Replace the in-flight ceiling with what the call cost.

        Called once per approved call, after it returns. A call that raises is never
        settled, so its ceiling stays committed. That's the safe direction, because a failed
        call may still have been billed. With several calls in flight, only this call's own
        ceiling is released, which is why the caller says which estimate it was.
        """
        with self._lock:
            self.spent_usd += actual
            if estimate is None:
                self.pending_ceiling_usd = Decimal(0)
            else:
                self.pending_ceiling_usd = max(
                    Decimal(0), self.pending_ceiling_usd - estimate.ceiling_usd
                )


_ASKING = threading.Lock()


@dataclass
class TerminalConsent:
    """Ask in the terminal, once per call, and default to no.

    Only ``y`` or ``yes`` approves. End of file, a blank line, anything else, or a
    non-interactive stdin all refuse. A gate that treats an unreadable answer as consent
    isn't a gate.
    """

    stream: Any = None
    out: Any = None
    require_tty: bool = True
    approvals: int = 0
    refusals: int = 0

    def _streams(self) -> tuple[Any, Any]:
        return (self.stream or sys.stdin, self.out or sys.stderr)

    def approve(self, estimate: CallEstimate) -> bool:
        # One prompt at a time, so two calls in flight can't interleave their questions.
        with _ASKING:
            return self._ask(estimate)

    def _ask(self, estimate: CallEstimate) -> bool:
        stream, out = self._streams()
        if self.require_tty and not (hasattr(stream, "isatty") and stream.isatty()):
            print(
                "\nRefusing a paid API call: stdin is not a terminal, so consent cannot "
                "be asked for. Run this interactively, or grant a budget explicitly.",
                file=out,
            )
            self.refusals += 1
            return False

        print("\n" + "=" * 68, file=out)
        print("  A PAID API CALL IS ABOUT TO BE MADE", file=out)
        print("=" * 68, file=out)
        for line in estimate.lines():
            print(f"  {line}", file=out)
        print("-" * 68, file=out)
        print("  Approve this one call?  [y/N] ", end="", file=out, flush=True)

        try:
            answer = stream.readline()
        except (OSError, ValueError):
            answer = ""
        decided = answer.strip().lower() in {"y", "yes"}
        print(f"  -> {'approved' if decided else 'refused'}\n", file=out)
        if decided:
            self.approvals += 1
        else:
            self.refusals += 1
        return decided


def default_gate() -> ConsentGate:
    """The gate to use when none was passed.

    Refuses unless the environment explicitly says otherwise, and the only values that do
    are spelled out. ``ask`` is the one that costs money, and it still asks.
    """
    setting = os.environ.get(OVERRIDE_ENV, "").strip().lower()
    if setting == "ask":
        return TerminalConsent()
    if setting in {"", "deny"}:
        return DenyAll()
    return DenyAll(
        reason=f"{OVERRIDE_ENV}={setting!r} is not a recognised value; 'ask' or 'deny' only"
    )
