"""Which Australian bank a name means, before any bank-specific work starts."""

from __future__ import annotations

import csv
import difflib
import io
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Literal

import yaml

__all__ = [
    "ASX_LIST_URL",
    "BankMatch",
    "ListedCompany",
    "find_bank",
    "listed_companies",
]

#: The ASX's list of every listed company, with its code and GICS industry group.
ASX_LIST_URL: Final = "https://www.asx.com.au/asx/research/ASXListedCompanies.csv"
_LIST_FILE: Final = "asx_listed_companies.csv"
#: How long a fetched copy of the list is trusted before a miss fetches it again.
_REFRESH_AFTER: Final = timedelta(days=1)
_BANKS_GROUP: Final = "Banks"
_SUFFIXES: Final = frozenset(
    {"limited", "ltd", "holdings", "group", "corporation", "corp", "plc", "the", "company", "inc"}
)

MatchKind = Literal[
    "brand",
    "listed_bank",
    "listed_not_bank",
    "overseas",
    "unlisted",
    "ambiguous",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class ListedCompany:
    name: str
    code: str
    industry: str


@dataclass(frozen=True, slots=True)
class BankMatch:
    """What a name turned out to mean."""

    kind: MatchKind
    asked: str
    code: str | None = None
    """The ASX code of the bank, or of the group that reports a brand."""
    name: str | None = None
    note: str = ""
    """For a brand, the sentence a reader is given: whose results these are."""
    message: str = ""
    """For anything that can't be answered, what to tell the person who asked."""

    @property
    def answerable(self) -> bool:
        return self.kind in {"brand", "listed_bank"}


def _plain(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower().replace("&", " and ")))


def _core(text: str) -> str:
    return " ".join(word for word in _plain(text).split() if word not in _SUFFIXES)


@lru_cache(maxsize=4)
def _reference(config_dir: str) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(
        (Path(config_dir) / "bank_brands.yaml").read_text(encoding="utf-8")
    )
    return loaded


def _parse_list(text: str) -> list[ListedCompany]:
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("Company name,")), None)
    if start is None:
        return []
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    return [
        ListedCompany(
            name=row["Company name"].strip().rstrip("."),
            code=row["ASX code"].strip().upper(),
            industry=(row.get("GICS industry group") or "").strip(),
        )
        for row in reader
        if row.get("Company name") and row.get("ASX code")
    ]


def listed_companies(
    config_dir: Path, data_dir: Path | None = None, *, refresh: bool = False, fetch: Any = None
) -> list[ListedCompany]:
    """The ASX's list of listed companies: the freshest copy held, fetched again on request.

    A fetched copy is kept in the data directory. With no network, or if the ASX's site
    can't be reached, the copy held is used and nothing fails: the list changes slowly, and
    a stale copy can only miss a company listed since it was taken.
    """
    held = data_dir / _LIST_FILE if data_dir is not None else None
    if refresh and held is not None:
        fresh = held.exists() and (
            datetime.now(UTC) - datetime.fromtimestamp(held.stat().st_mtime, UTC) < _REFRESH_AFTER
        )
        if not fresh:
            try:
                text = (fetch or _fetch_list)()
                if _parse_list(text):
                    held.parent.mkdir(parents=True, exist_ok=True)
                    held.write_text(text, encoding="utf-8")
            except Exception:  # the copy held is the fallback, however the fetch failed
                pass
    for source in (held, config_dir / _LIST_FILE):
        if source is not None and source.exists():
            companies = _parse_list(source.read_text(encoding="utf-8"))
            if companies:
                return companies
    return []


def _fetch_list() -> str:
    import httpx

    response = httpx.get(ASX_LIST_URL, timeout=10.0, follow_redirects=False)
    response.raise_for_status()
    return response.text


def find_bank(
    asked: str,
    *,
    config_dir: Path,
    data_dir: Path | None = None,
    fetch: Any = None,
    known: dict[str, str] | None = None,
) -> BankMatch:
    """What ``asked`` means, among Australia's banks.

    ``known`` maps each configured bank's id to its name, for brands and suggestions. A
    configured bank itself is found by the registry before this is called.
    """
    reference = _reference(str(config_dir))
    known = known or {}
    wanted = _core(asked)
    if not wanted:
        return BankMatch("unknown", asked, message="No bank was named.")

    for code, brands in (reference.get("brands") or {}).items():
        for brand in brands:
            if _core(brand) == wanted:
                owner = known.get(code) or _owner_name(code, config_dir, data_dir)
                return BankMatch(
                    "brand",
                    asked,
                    code=code,
                    name=owner,
                    note=(
                        f"{brand} is a brand of {owner}, which reports it within its own "
                        f"results, so these are {owner}'s figures."
                    ),
                )

    unlisted = _unlisted(asked, reference)
    if unlisted is not None:
        # Before the ASX's codes: "ING" is ING the bank, not Inghams' code.
        return unlisted

    companies = listed_companies(config_dir, data_dir)
    match = _listed(asked, companies)
    if match is None and data_dir is not None:
        match = _listed(asked, listed_companies(config_dir, data_dir, refresh=True, fetch=fetch))
    if isinstance(match, list):
        names = "; ".join(f"{_display(c.name)} ({c.code})" for c in match[:5])
        return BankMatch(
            "ambiguous",
            asked,
            message=f"{asked!r} matches more than one company listed on the ASX: {names}. "
            "Ask again with the ASX code.",
        )
    if match is not None:
        return _classify(asked, match, reference)

    candidates = {
        **{name: code for code, name in known.items()},
        **{
            brand: code
            for code, brands in (reference.get("brands") or {}).items()
            for brand in brands
        },
        **{_display(c.name): c.code for c in companies if c.industry == _BANKS_GROUP},
        **{
            name: "not listed"
            for names in (reference.get("unlisted") or {}).values()
            for name in names
        },
    }
    close = difflib.get_close_matches(asked.title(), list(candidates), n=4, cutoff=0.6)
    suggestion = (
        " Did you mean " + ", ".join(f"{name} ({candidates[name]})" for name in close) + "?"
        if close
        else ""
    )
    return BankMatch(
        "unknown",
        asked,
        message=f"No Australian bank goes by {asked!r}: it isn't a bank this tool knows, a "
        "brand of one, or a company listed on the ASX." + suggestion + " `bankresearch "
        "coverage` lists the banks it answers for.",
    )


def _unlisted(asked: str, reference: dict[str, Any]) -> BankMatch | None:
    wanted = _core(asked)
    for kind, names in (reference.get("unlisted") or {}).items():
        for name in names:
            if _core(name) == wanted:
                return BankMatch(
                    "unlisted",
                    asked,
                    name=name,
                    message=f"{name} is an Australian bank, a {kind}, but it isn't listed on "
                    "the ASX, so it publishes no half-year or full-year results there for "
                    "this tool to read.",
                )
    return None


#: Short words kept in capitals when the ASX's upper-case names are shown.
_ACRONYMS: Final = frozenset(
    {"AMP", "ANZ", "BNK", "BOQ", "BSP", "CBA", "HSBC", "ING", "NAB", "TPT"}
)
_SMALL_WORDS: Final = frozenset({"of", "and", "the", "for"})


def _display(name: str) -> str:
    """An ASX name in the case people write it: "BNK Banking Corporation Limited"."""
    words: list[str] = []
    for word in name.split():
        if word.upper() in _ACRONYMS or not re.search(r"[AEIOU]", word.upper()):
            words.append(word.upper())
        elif word.lower() in _SMALL_WORDS and words:
            words.append(word.lower())
        else:
            words.append(word.capitalize())
    return " ".join(words)


def _listed(
    asked: str, companies: list[ListedCompany]
) -> ListedCompany | list[ListedCompany] | None:
    """The listed company ``asked`` names: by its code, its name, or the start of its name."""
    code = asked.strip().upper()
    by_code = [c for c in companies if c.code == code]
    if by_code:
        return by_code[0]
    wanted = _core(asked)
    exact = [c for c in companies if _core(c.name) == wanted]
    if exact:
        return exact[0] if len(exact) == 1 else exact
    if len(wanted) < 3:
        return None
    words = set(wanted.split())
    partial = [
        c
        for c in companies
        if _core(c.name).startswith(wanted + " ")
        or (len(words) > 1 and words <= set(_core(c.name).split()))
    ]
    if not partial:
        return None
    banks = [c for c in partial if c.industry == _BANKS_GROUP]
    chosen = banks or partial
    return chosen[0] if len(chosen) == 1 else chosen


def _classify(asked: str, company: ListedCompany, reference: dict[str, Any]) -> BankMatch:
    name = _display(company.name)
    owners = reference.get("bank_owners") or {}
    overseas = reference.get("overseas") or {}
    if company.code in overseas:
        return BankMatch(
            "overseas",
            asked,
            code=company.code,
            name=name,
            message=f"{name} ({company.code}) is listed on the ASX, but it's "
            f"{overseas[company.code]}. This tool covers Australian banks reporting in "
            "Australian dollars.",
        )
    if company.industry == _BANKS_GROUP or company.code in owners:
        return BankMatch("listed_bank", asked, code=company.code, name=name)
    return BankMatch(
        "listed_not_bank",
        asked,
        code=company.code,
        name=name,
        message=f"{name} ({company.code}) is listed on the ASX in the "
        f"{company.industry or 'unclassified'} industry group, not as a bank, so it reports "
        "none of the bank measures this tool explains.",
    )


def _owner_name(code: str, config_dir: Path, data_dir: Path | None) -> str:
    for company in listed_companies(config_dir, data_dir):
        if company.code == code:
            return _display(company.name)
    return code
