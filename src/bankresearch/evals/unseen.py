"""Leave-one-bank-out: each issuer treated as one the system has never seen."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..domain.models import Fact

__all__ = ["UnseenBankReport", "WithoutBank", "summarise_unseen"]


@dataclass
class WithoutBank:
    """The fact store with one issuer's figures held back.

    Duck-typed over ``FactStore``: the agent reads facts through ``get_facts`` and nothing
    else, which is why a wrapper is enough, and why the agent can't tell it's being tested.
    """

    inner: Any
    bank_id: str

    def get_facts(self, spec: Any, components: Any = None) -> list[Fact]:
        if getattr(spec, "bank_id", "").upper() == self.bank_id.upper():
            return []
        found: list[Fact] = self.inner.get_facts(spec, components)
        return found

    def documents_for(self, bank_id: str, periods: Any, as_of: Any) -> list[Any]:
        """Also held back, since an unseen bank has no ingested documents either.

        Holding back the facts but leaving the documents would describe a bank whose filings
        were read and yielded nothing, which is a different failure with a different correct
        answer.
        """
        if bank_id.upper() == self.bank_id.upper():
            return []
        docs: list[Any] = self.inner.documents_for(bank_id, periods, as_of)
        return docs

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@dataclass
class UnseenBankReport:
    """One issuer, held out."""

    bank_id: str
    cases: int = 0
    abstained: int = 0
    reasons: dict[str, int] = field(default_factory=dict)
    answered: list[str] = field(default_factory=list)
    """Cases that still produced an answer. Each is a leak and is named."""
    crashed: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        """Whether every case abstained, with a reason, without crashing.

        An answer here is a leak: the only figures available were another bank's.
        """
        return not self.answered and not self.crashed and self.abstained == self.cases


def summarise_unseen(bank_id: str, outcomes: list[Any], case_ids: list[str]) -> UnseenBankReport:
    """Collapse one held-out bank's runs into a report."""
    report = UnseenBankReport(bank_id=bank_id, cases=len(outcomes))
    for case_id, outcome in zip(case_ids, outcomes, strict=False):
        result = outcome.result
        if result.status == "ABSTAIN":
            report.abstained += 1
            reason = result.abstain_reason or "UNSTATED"
            report.reasons[reason] = report.reasons.get(reason, 0) + 1
            if result.abstain_reason is None:
                # An abstention with no reason code is as bad as an answer, because it tells
                # the reader nothing about what was missing.
                report.crashed.append(case_id)
        else:
            report.answered.append(case_id)
    return report
