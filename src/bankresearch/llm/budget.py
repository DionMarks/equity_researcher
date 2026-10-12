"""The budget guard: an overspend weakens the answer but never crashes the run."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from ..domain.money import aud
from .client import Usage, cost_usd

__all__ = ["Budget", "BudgetExceeded", "StepBudget"]


class BudgetExceeded(RuntimeError):
    """Raised inside the guard so a step stops, never propagated to the caller."""


@dataclass(frozen=True, slots=True)
class StepBudget:
    """Per-step caps, from ``config/budgets.yaml``."""

    calls: int
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None


@dataclass
class Budget:
    """One run's spend against its caps."""

    max_calls: int = 25
    max_input_tokens: int = 300_000
    max_output_tokens: int = 60_000
    cost_guardrail_usd: Decimal = Decimal("1.50")
    step_caps: dict[str, StepBudget] = field(default_factory=dict)
    pricing: dict[str, dict[str, str]] = field(default_factory=dict)

    calls: int = 0
    cache_hits: int = 0
    """Served from the response cache: no request, no tokens, no charge."""
    usage: Usage = field(default=Usage())
    cost: Decimal = Decimal(0)
    per_step_calls: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    breached: bool = False

    def may_call(self, step: str, projected_usd: Decimal = Decimal(0)) -> bool:
        """Whether another call is allowed, recording why not when it is not.

        ``projected_usd`` is what the call is expected to cost, for a step whose calls are
        large and predictable. The guard used to be checked only against what had been spent,
        so a run at A$1.86 of an A$2.15 guardrail could start a web-search round costing
        A$1.05 and finish at A$2.91, which is what Judo's cold path did.
        """
        if self.calls >= self.max_calls:
            return self._refuse(f"run call cap of {self.max_calls} reached")
        if self.usage.input_tokens >= self.max_input_tokens:
            return self._refuse(f"input token cap of {self.max_input_tokens} reached")
        if self.usage.output_tokens >= self.max_output_tokens:
            return self._refuse(f"output token cap of {self.max_output_tokens} reached")
        if self.cost >= self.cost_guardrail_usd:
            return self._refuse(f"cost guardrail of {aud(self.cost_guardrail_usd)} reached")
        if projected_usd and self.cost + projected_usd > self.cost_guardrail_usd:
            return self._refuse(
                f"cost guardrail of {aud(self.cost_guardrail_usd)} would be passed by a call "
                f"expected to cost about {aud(projected_usd)}"
            )
        cap = self.step_caps.get(step)
        if cap is not None and self.per_step_calls.get(step, 0) >= cap.calls:
            return self._refuse(f"{step} call cap of {cap.calls} reached")
        return True

    def go_cold(self, budgets: dict[str, Any]) -> None:
        """Take the cold path's caps, for a run that has to find what it reads.

        Only ever raises them: a run that's already cold, or one given more, keeps its own.
        """
        cold = budgets.get("paths", {}).get("cold", {}) or {}
        self.max_calls = max(self.max_calls, int(cold.get("max_llm_calls", self.max_calls)))
        self.cost_guardrail_usd = max(
            self.cost_guardrail_usd,
            Decimal(str(cold.get("cost_guardrail_usd_per_new_period", self.cost_guardrail_usd))),
        )

    def record(self, step: str, model: str, usage: Usage, *, cached: bool = False) -> None:
        """Count what a step used. A cache hit counts as a hit and nothing else.

        A cached response carries the original call's token counts, which is what makes it
        reusable and also what made it get billed twice: a run that made no requests at all
        reported 13 model calls and A$0.095. Over-reporting isn't harmless: the limit would
        refuse real work to stay under a figure nobody was charged, and a replayed eval
        would hit a call cap on calls it never made.
        """
        if cached:
            self.cache_hits += 1
            return
        self.calls += 1
        self.per_step_calls[step] = self.per_step_calls.get(step, 0) + 1
        self.usage = self.usage + usage
        self.cost += cost_usd(usage, self.pricing.get(model, {}))

    def _refuse(self, reason: str) -> bool:
        self.breached = True
        message = f"Budget guard: {reason}; exploration stopped and confidence lowered."
        if message not in self.warnings:
            self.warnings.append(message)
        return False

    @property
    def remaining_calls(self) -> int:
        return max(0, self.max_calls - self.calls)

    def summary(self) -> str:
        return (
            f"{self.calls} call(s), {self.usage.input_tokens} in / "
            f"{self.usage.output_tokens} out tokens, {aud(self.cost, 4)}"
            + (f", {self.cache_hits} from cache" if self.cache_hits else "")
            + (" [budget breached]" if self.breached else "")
        )

    @classmethod
    def from_config(
        cls, budgets: dict[str, Any], models: dict[str, Any], path: str = "warm"
    ) -> Budget:
        """Build a run's budget from ``config/budgets.yaml`` and ``config/models.yaml``."""
        warm = budgets.get("paths", {}).get(path, {})
        caps = budgets.get("step_caps", {}) or {}
        token_caps = budgets.get("token_caps", {}) or {}
        step_caps: dict[str, StepBudget] = {}
        for step, spec in caps.items():
            allowed = spec.get("tool_calls") or spec.get("calls") or 1
            tokens = token_caps.get(step, {})
            step_caps[step] = StepBudget(
                calls=int(allowed) + int(spec.get("repairs", 0) or spec.get("rechecks", 0) or 0),
                max_input_tokens=tokens.get("input"),
                max_output_tokens=tokens.get("output"),
            )
        return cls(
            max_calls=int(warm.get("max_llm_calls", 25)),
            max_input_tokens=int(warm.get("max_input_tokens", 300_000)),
            max_output_tokens=int(warm.get("max_output_tokens", 60_000)),
            cost_guardrail_usd=Decimal(str(warm.get("cost_guardrail_usd_p95", "1.50"))),
            step_caps=step_caps,
            pricing=models.get("pricing", {}) or {},
        )
