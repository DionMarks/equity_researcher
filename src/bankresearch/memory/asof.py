"""Point-in-time integrity: what was knowable when."""

from __future__ import annotations

from datetime import date

__all__ = ["knowable_at"]


def knowable_at(publish_date: date, as_of: date) -> bool:
    """Whether a document was available on the as-of date."""
    return publish_date <= as_of
