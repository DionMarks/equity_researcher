"""The issuer's commentary as evidence cards."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Any

from ...domain.models import (
    Decomposition,
    DocMeta,
    EvidenceCard,
    RequestSpec,
    ResolvedInputs,
    SourceRef,
)
from ...domain.registry import get_registry
from ...ingestion.parse_pdf import ParsedDocument
from ...retrieval.commentary import find_commentary, find_division_commentary
from .extract import short_quote

__all__ = ["commentary_cards", "commentary_topics"]

#: Components whose drivers are named after something else in prose. The ROE identity's
#: earnings lines are explained wherever the pack explains its profit.
_ALSO = {
    "roe": ("headline_earnings", "average_equity"),
    "cet1_ratio": ("cet1_capital", "rwa"),
    "nim": ("nii", "aiea"),
}


def commentary_topics(spec: RequestSpec, decomposition: Decomposition) -> list[str]:
    """The metric first, then whatever its drivers were computed from, in driver order."""
    topics: list[str] = [spec.metric_id]
    for component in _ALSO.get(spec.metric_id, ()):
        if component not in topics:
            topics.append(component)
    for driver in decomposition.drivers:
        for component in driver.from_components:
            if component not in topics and component != spec.metric_id:
                topics.append(component)
    return topics


def commentary_cards(
    spec: RequestSpec,
    decomposition: Decomposition,
    inputs: ResolvedInputs,
    documents: dict[str, ParsedDocument],
    metas: dict[str, DocMeta],
    *,
    start_at: int,
) -> list[EvidenceCard]:
    """One card per sentence of the issuer's own explanation of this movement."""
    registry = get_registry()
    bank = registry.banks.get(spec.bank_id)
    if bank is None:
        return []
    topics = commentary_topics(spec, decomposition)
    figures: dict[str, Sequence[Decimal]] = {}
    for topic in topics:
        values = [inputs.current.get(topic), inputs.prior.get(topic)]
        figures[topic] = [value for value in values if value is not None]
    held: list[tuple[ParsedDocument, Any]] = [
        (document, metas[doc_id]) for doc_id, document in documents.items() if doc_id in metas
    ]
    passages = find_commentary(spec, held, topics, fy_end_month=bank.fy_end_month, figures=figures)
    # A division's driver is explained on the division's own pages, which the group's
    # commentary leaves out.
    by_division = {
        key: f"impairment_div_{key}"
        for key in bank.impairment_divisions.values()
        if f"impairment_div_{key}" in topics
    }
    if by_division:
        for topic in by_division.values():
            values = [inputs.current.get(topic), inputs.prior.get(topic)]
            figures[topic] = [value for value in values if value is not None]
        passages += find_division_commentary(
            spec,
            held,
            bank.impairment_divisions,
            by_division,
            fy_end_month=bank.fy_end_month,
            figures=figures,
        )
    unit = inputs.units.get(spec.metric_id, registry.metrics[spec.metric_id].unit)
    cards: list[EvidenceCard] = []
    for offset, passage in enumerate(passages, start=start_at):
        cards.append(
            EvidenceCard(
                evidence_id=f"ev_{offset:03d}",
                component=passage.topic,
                period=spec.period,
                value=None,
                unit=unit,  # type: ignore[arg-type]
                statement=(
                    f"The issuer's commentary on {passage.topic.replace('_', ' ')}, "
                    f"{spec.period} against {spec.comparison_period}."
                ),
                quote=short_quote(passage.text),
                passage=passage.text,
                source=SourceRef(
                    doc_id=passage.doc_id,
                    doc_type=passage.doc_type,  # type: ignore[arg-type]
                    page=passage.page_no,
                    publish_date=passage.publish_date,
                ),
                method="text",
                checks_passed=["quote_on_page", "same_comparison"],
            )
        )
    return cards
