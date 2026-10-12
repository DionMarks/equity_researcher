"""Reading a figure's basis off the page it's on."""

from __future__ import annotations

import re

from ..domain.models import Basis

__all__ = ["detect_basis", "headline_earnings_basis"]

_CASH = re.compile(r"\bcash (?:profit|earnings|npat|results)\b|\bcash basis\b", re.IGNORECASE)
_STATUTORY = re.compile(r"\bstatutory\b|\bnet profit (?:results|after tax)\b", re.IGNORECASE)
_UNDERLYING = re.compile(r"\bunderlying (?:profit|results|earnings)\b", re.IGNORECASE)
_LEVEL_1 = re.compile(r"\blevel 1\b", re.IGNORECASE)
_LEVEL_2 = re.compile(r"\blevel 2\b", re.IGNORECASE)
_PRO_FORMA = re.compile(r"\bpro\s?forma\b", re.IGNORECASE)
_CONTINUING = re.compile(r"\bcontinuing operations\b", re.IGNORECASE)
_DISCONTINUED = re.compile(r"\bdiscontinued operations\b", re.IGNORECASE)

# Only the top of a page carries its section title; further down is the table itself.
_TITLE_LINES = 6


def detect_basis(page_text: str, caption: str = "") -> Basis:
    """The basis a page states for its figures.

    Read from the first few lines, where the section title is. Reading the whole page would
    pick up a footnote that mentions the other basis and switch every figure on the page to
    it.
    """
    head = "\n".join(line for line in (page_text or "").splitlines() if line.strip()).splitlines()[
        :_TITLE_LINES
    ]
    context = " ".join([*head, caption])

    earnings: str = "n/a"
    if _CASH.search(context):
        earnings = "cash"
    elif _UNDERLYING.search(context):
        earnings = "underlying"
    elif _STATUTORY.search(context):
        earnings = "statutory"

    scope: str = "group"
    if _LEVEL_1.search(context):
        scope = "level1"
    elif _LEVEL_2.search(context):
        scope = "level2"

    continuing: bool | None = None
    if _CONTINUING.search(context):
        continuing = True
    elif _DISCONTINUED.search(context):
        continuing = False

    # `ex_notable_items` isn't set from the page, on purpose. A results page often mentions
    # notable items in a footnote while its table carries the headline figure, and marking
    # every row on such a page as ex-notable made the reconciler treat the headline as the
    # variant and the other way round. That distinction comes from the row label instead,
    # where `exclusions` in config/line_items.yaml handles it.
    return Basis(
        earnings_basis=earnings,  # type: ignore[arg-type]
        scope=scope,  # type: ignore[arg-type]
        pro_forma=bool(_PRO_FORMA.search(context)),
        continuing_ops=continuing,
    )


def headline_earnings_basis(headline_label: str | None) -> str:
    """Which earnings basis the issuer leads with, from the label it uses.

    Westpac reports net profit as its single performance measure from 1H23, so its headline
    basis is statutory while the others' is cash. Working this out from the label rather
    than hard-coding it keeps it right for an issuer we haven't seen.
    """
    label = (headline_label or "").strip().lower()
    if label.startswith("net profit") or "statutory" in label:
        return "statutory"
    if label.startswith("cash") or "cash" in label:
        return "cash"
    # MyState leads with underlying net profit after tax, which leaves out its merger costs.
    if "underlying" in label:
        return "underlying"
    return "n/a"
