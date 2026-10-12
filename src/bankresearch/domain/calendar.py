"""Fiscal period arithmetic, resolved against each issuer's own calendar."""

from __future__ import annotations

import calendar as _calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum

__all__ = [
    "AmbiguousPeriodError",
    "FiscalPeriod",
    "PeriodKind",
    "fiscal_period",
    "parse_period",
]


class AmbiguousPeriodError(ValueError):
    """Raised when a period string can't be read as a fiscal period.

    Carries the accepted forms so the abstention can tell the caller what to type instead.
    Maps to the ``AMBIGUOUS_INPUT`` reason code.
    """


class PeriodKind(StrEnum):
    H1 = "H1"
    H2 = "H2"
    FY = "FY"

    @property
    def is_half(self) -> bool:
        return self is not PeriodKind.FY


_ACCEPTED = "1H26, 2H25, FY25, H1 FY26, 1HFY26, HY26, FY2025"

# A four-digit year is only unambiguous when "FY" marks it as fiscal.
_FY_ONLY = re.compile(r"^FY\s?(?P<y>\d{4}|\d{2})$")
_HALF_LEADING = re.compile(r"^(?P<h>[12])H\s?(?:FY\s?)?(?P<y>\d{4}|\d{2})$")
_HALF_TRAILING = re.compile(r"^H(?P<h>[12])\s?(?:FY\s?)?(?P<y>\d{4}|\d{2})$")
_HY = re.compile(r"^HY\s?(?P<y>\d{4}|\d{2})$")
# Calendar-style: a half against a bare four-digit year, with no FY marker.
_CALENDAR_STYLE = re.compile(r"^(?:(?P<h1>[12])H|H(?P<h2>[12]))\s*(?P<y>\d{4})$")


def _two_digit(year: str) -> int:
    """Return the two-digit fiscal year suffix for a 2- or 4-digit input."""
    value = int(year)
    return value % 100


def parse_period(raw: str) -> str:
    """Normalise a period string to a canonical label, or refuse.

    Returns one of ``1H26``, ``2H25``, ``FY25``. Raises
    :class:`AmbiguousPeriodError` for calendar-style and unparseable input.
    """
    if raw is None:
        raise AmbiguousPeriodError(f"No period given. Accepted forms: {_ACCEPTED}.")

    text = re.sub(r"\s+", " ", raw.strip().upper())
    if not text:
        raise AmbiguousPeriodError(f"No period given. Accepted forms: {_ACCEPTED}.")

    # Refuse calendar-style input before anything else can match it, and offer
    # both fiscal readings rather than picking one.
    if "FY" not in text:
        calendar_style = _CALENDAR_STYLE.match(text)
        if calendar_style is not None:
            half = calendar_style.group("h1") or calendar_style.group("h2")
            year = int(calendar_style.group("y"))
            raise AmbiguousPeriodError(
                f"{raw!r} is a calendar half, and which fiscal half it means depends on "
                f"the issuer's year end. Use the fiscal label: {half}H{year % 100:02d} "
                f"or {half}H{(year + 1) % 100:02d}."
            )

    match = _FY_ONLY.match(text)
    if match is not None:
        return f"FY{_two_digit(match.group('y')):02d}"

    for pattern in (_HALF_LEADING, _HALF_TRAILING):
        match = pattern.match(text)
        if match is not None:
            return f"{match.group('h')}H{_two_digit(match.group('y')):02d}"

    match = _HY.match(text)
    if match is not None:
        # HY26 means 1H26.
        return f"1H{_two_digit(match.group('y')):02d}"

    raise AmbiguousPeriodError(f"Cannot read {raw!r} as a period. Accepted forms: {_ACCEPTED}.")


def _last_day(year: int, month: int) -> date:
    return date(year, month, _calendar.monthrange(year, month)[1])


def _add_months(anchor: date, months: int) -> date:
    """Month arithmetic that clamps to the month's length."""
    total = anchor.month - 1 + months
    year = anchor.year + total // 12
    month = total % 12 + 1
    day = min(anchor.day, _calendar.monthrange(year, month)[1])
    return date(year, month, day)


@dataclass(frozen=True, slots=True)
class FiscalPeriod:
    """A reporting period resolved against one issuer's financial calendar."""

    label: str
    kind: PeriodKind
    fy: int
    """The four-digit year in which the financial year ends. ``FY26`` -> 2026."""
    fy_end_month: int
    start: date
    end: date

    @property
    def days(self) -> int:
        """Days in the period, inclusive of both ends. Used for annualisation."""
        return (self.end - self.start).days + 1

    @property
    def year_days(self) -> int:
        """Days in the financial year this period belongs to: 366 when it takes in a 29
        February, as ANZ's, NAB's and Westpac's FY24 (October 2023 to September 2024) does.

        A margin or a return is annualised over the issuer's own year. Always using 365
        put ANZ's 1H24 margin at 1.551% against the 1.56% it published: its net interest
        income and balances failed their identity check and were quarantined, and the
        question about 1H25 had nothing to decompose.
        """
        return fiscal_period(f"FY{self.fy % 100:02d}", self.fy_end_month).days

    @property
    def balance_date(self) -> date:
        """The date a point-in-time metric is measured at.

        For ``2H`` and ``FY`` this is the same date, which is why a
        balance-sheet metric at 2H end equals the full-year value.
        """
        return self.end

    def pcp(self) -> FiscalPeriod:
        """The same period one financial year earlier. Never the adjacent half."""
        return fiscal_period(f"{self.label[:-2]}{(self.fy - 1) % 100:02d}", self.fy_end_month)

    def hoh(self) -> FiscalPeriod:
        """The immediately preceding half.

        Not a comparison basis. It's only used to chain two published half-on-half walks
        into the year-on-year span, and the output says so when it is.
        """
        if self.kind is PeriodKind.FY:
            raise ValueError("A full year has no preceding half.")
        if self.kind is PeriodKind.H2:
            return fiscal_period(f"1H{self.fy % 100:02d}", self.fy_end_month)
        return fiscal_period(f"2H{(self.fy - 1) % 100:02d}", self.fy_end_month)

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.label} ({self.start.isoformat()} to {self.end.isoformat()})"


def fiscal_period(label: str, fy_end_month: int) -> FiscalPeriod:
    """Resolve a period label against a financial year ending in ``fy_end_month``."""
    canonical = parse_period(label)
    if not 1 <= fy_end_month <= 12:
        raise ValueError(f"fy_end_month must be 1-12, got {fy_end_month}")

    fy = 2000 + int(canonical[-2:])
    fy_end = _last_day(fy, fy_end_month)
    # The financial year starts the day after the previous one ended.
    fy_start = _last_day(fy - 1, fy_end_month) + timedelta(days=1)

    if canonical.startswith("FY"):
        return FiscalPeriod(canonical, PeriodKind.FY, fy, fy_end_month, fy_start, fy_end)

    first_half_end = _add_months(fy_start, 6) - timedelta(days=1)
    if canonical.startswith("1H"):
        return FiscalPeriod(canonical, PeriodKind.H1, fy, fy_end_month, fy_start, first_half_end)
    return FiscalPeriod(
        canonical,
        PeriodKind.H2,
        fy,
        fy_end_month,
        first_half_end + timedelta(days=1),
        fy_end,
    )
