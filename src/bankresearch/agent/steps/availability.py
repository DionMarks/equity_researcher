"""Step 2: is this period reportable, and are its documents here?"""

from __future__ import annotations

from ...domain.models import Availability, RequestSpec
from ...domain.registry import get_registry
from ...memory.fact_store import FactStore
from .resolve import expected_release_date

__all__ = ["check_availability"]


def check_availability(spec: RequestSpec, store: FactStore) -> Availability:
    """What documents exist for this request as at its as-of date."""
    profile = get_registry().bank(spec.bank_id)
    release = expected_release_date(profile, spec.period)
    current = store.documents_for(spec.bank_id, [spec.period], spec.as_of)
    comparison = store.documents_for(spec.bank_id, [spec.comparison_period], spec.as_of)

    # A filing held and published by the as-of date settles it, whatever the usual date
    # says: NAB released its 1H26 results on 7 May, and a question dated 8 May was refused
    # because the usual release is "by 31 May".
    if not current and spec.as_of < release:
        window_opens = release.replace(day=1)
        if spec.as_of < window_opens:
            return Availability(
                reportable=False,
                reason="PERIOD_NOT_YET_REPORTED",
                detail=(
                    f"{spec.bank_id} {spec.period} ends {spec.period_end.isoformat()} and is "
                    f"usually released in {release:%B %Y}, after the as-of date "
                    f"{spec.as_of.isoformat()}. Nothing has been published to read."
                ),
            )
        # The month it's due: it may already be out. A run with a model checks the ASX's
        # own list; this reason stands if the list doesn't have it, or can't be checked.
        return Availability(
            comparison_docs=list(comparison),
            reportable=False,
            reason="PERIOD_NOT_YET_REPORTED",
            release_pending=True,
            detail=(
                f"{spec.bank_id} usually releases its {spec.period} result by "
                f"{release.isoformat()}, and nothing for {spec.period} is held as at "
                f"{spec.as_of.isoformat()}, so it may not be out yet."
            ),
        )

    if not current:
        found = sorted({doc.doc_id for doc in comparison})
        # Figures for the period may still be in the store, as the comparison column of a
        # later pack. Saying "Found: nothing" when 24 of them are there makes the system
        # look blind to its own data, and hides the real reason it's declining: a
        # comparative is the restated figure, so answering from one would report a period
        # using numbers published after it closed. That's the point-in-time guarantee, and
        # it's worth saying so.
        comparatives = len(store.get_facts(spec))
        return Availability(
            comparison_docs=list(comparison),
            reportable=False,
            reason="DOCS_UNAVAILABLE",
            detail=(
                f"No {spec.bank_id} document for {spec.period} was published on or before "
                f"{spec.as_of.isoformat()}. "
                + (
                    f"{comparatives} figure(s) for the period are held, but only as the "
                    "comparative column of a later pack: those are the restated values, "
                    "so answering from them would describe the period using numbers "
                    "published after it closed. "
                    if comparatives
                    else "No figures for the period are held either. "
                )
                + (
                    f"Documents held for {spec.comparison_period}: {', '.join(found)}. "
                    if found
                    else f"Nor is any document held for {spec.comparison_period}. "
                )
                + f"Ingest the results document for {spec.period}."
            ),
        )

    detail = (
        f"{len(current)} document(s) for {spec.period} and {len(comparison)} for "
        f"{spec.comparison_period}, all published on or before {spec.as_of.isoformat()}."
    )
    if not comparison:
        detail += (
            f" No {spec.comparison_period} document is held, so the prior value comes from "
            "the current document's comparative column and the original vintage cannot be "
            "checked: a restatement would not be visible."
        )
    return Availability(
        current_docs=list(current),
        comparison_docs=list(comparison),
        reportable=True,
        detail=detail,
    )
