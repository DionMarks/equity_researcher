"""A model client that tells the run's progress line about each call."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from ..domain.registry import get_registry
from ..llm.client import LLMClient, LLMRequest, LLMResponse, cost_usd
from ..output.progress import RunProgress

__all__ = ["ReportingClient"]


@dataclass
class ReportingClient:
    """Passes every request through unchanged, and reports what it was for and cost."""

    inner: LLMClient
    progress: RunProgress

    @property
    def available(self) -> bool:
        return self.inner.available

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.progress.calling(request.step)
        response = self.inner.complete(request)
        self.progress.called(request.step, _cost(request.model, response))
        return response


def _cost(model: str, response: LLMResponse) -> Decimal:
    """What one call cost at the configured price, or nothing for a cache hit. A price that
    can't be looked up shows as nothing rather than stopping a run that has already paid."""
    if response.cached:
        return Decimal(0)
    try:
        prices = (get_registry().models.get("pricing") or {}).get(model)
    except Exception:
        return Decimal(0)
    return cost_usd(response.usage, prices) if prices else Decimal(0)
