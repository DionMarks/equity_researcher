"""The twelve steps, which ones may call a model, and how a run can end."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

__all__ = ["ABSTAIN_REASONS", "RUN_STATUSES", "STATES", "Step"]


class Step(StrEnum):
    RESOLVE = "resolve"
    AVAILABILITY = "availability"
    DISCOVER = "discover"
    LOCATE = "locate"
    EXTRACT = "extract"
    VALIDATE = "validate"
    RECONCILE = "reconcile"
    DECOMPOSE = "decompose"
    EXPLAIN = "explain"
    SCORE = "score"
    VERIFY = "verify"
    RENDER = "render"

    @property
    def uses_llm(self) -> bool:
        """Whether this step routinely calls a model.

        ``RESOLVE`` is excluded: it calls one only to parse free-text input, and
        never to decide a date, a period or a metric.
        """
        return self in _LLM_STEPS

    @property
    def is_terminal(self) -> bool:
        return self is Step.RENDER


_LLM_STEPS: Final[frozenset[Step]] = frozenset(
    {Step.DISCOVER, Step.LOCATE, Step.EXTRACT, Step.EXPLAIN, Step.VERIFY}
)

STATES: Final[tuple[Step, ...]] = tuple(Step)

RUN_STATUSES: Final[tuple[str, ...]] = ("OK", "PARTIAL", "ABSTAIN")

ABSTAIN_REASONS: Final[tuple[str, ...]] = (
    "AMBIGUOUS_INPUT",
    "METRIC_OUT_OF_SCOPE",
    "BANK_UNKNOWN",
    "PERIOD_NOT_YET_REPORTED",
    "DOCS_UNAVAILABLE",
    "METRIC_NOT_REPORTED",
    "INSUFFICIENT_EVIDENCE",
)
