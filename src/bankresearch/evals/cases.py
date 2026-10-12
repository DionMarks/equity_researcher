"""Loading eval cases."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from ..settings import get_settings

__all__ = ["EvalCase", "GoldDriver", "GoldValue", "load_cases"]


@dataclass(frozen=True, slots=True)
class GoldValue:
    value: Decimal
    unit: str
    tolerance: Decimal
    doc_id: str | None = None
    page: int | None = None
    restated: bool = False


@dataclass(frozen=True, slots=True)
class GoldDriver:
    driver_id: str
    contribution: Decimal | None = None
    tolerance: Decimal | None = None
    doc_id: str | None = None
    page: int | None = None


@dataclass(frozen=True, slots=True)
class EvalCase:
    case_id: str
    split: str
    bank: str
    period: str
    metric: str
    as_of: date | None = None
    categories: tuple[str, ...] = ()
    expected_status: tuple[str, ...] = ("OK",)
    expected_abstain_reason: str | None = None
    current: GoldValue | None = None
    prior: GoldValue | None = None
    change: GoldValue | None = None
    drivers: tuple[GoldDriver, ...] = ()
    walk_drivers: tuple[GoldDriver, ...] = ()
    """The bank's own walk, mapped to driver ids, where it publishes one.
    A run whose drivers came from a walk is scored against these, and one whose drivers came
    from the identity against ``drivers``: both are right, and they name different things."""
    must_mention: tuple[str, ...] = ()
    must_not_claim: tuple[str, ...] = ()
    materiality: Decimal | None = None
    gold_verified: bool = False
    labeller: str = ""
    labeller_kind: str = ""
    notes: str = ""
    expected_status_with_model: tuple[str, ...] = ()
    """What the case should reach when a model is used, where that differs. A bank outside
    the corpus abstains without a model and takes the cold path with one: OK if discovery
    works, else ABSTAIN with DOCS_UNAVAILABLE."""
    expected_abstain_reason_with_model: str | None = None

    def expectation(self, model_mode: str = "off") -> tuple[tuple[str, ...], str | None]:
        """The statuses this case may reach, and the abstention reason if it abstains, for a
        run in this model mode."""
        if model_mode != "off" and self.expected_status_with_model:
            return self.expected_status_with_model, self.expected_abstain_reason_with_model
        return self.expected_status, self.expected_abstain_reason

    @property
    def independently_labelled(self) -> bool:
        """Whether the gold was checked by someone other than the system under test."""
        return self.gold_verified and self.labeller_kind == "human"

    @property
    def has_gold(self) -> bool:
        return self.current is not None or bool(self.drivers)

    @property
    def expects_abstention(self) -> bool:
        return "ABSTAIN" in self.expected_status


def _gold_value(raw: dict[str, Any] | None) -> GoldValue | None:
    if not raw:
        return None
    source = raw.get("source", {}) or {}
    return GoldValue(
        value=Decimal(str(raw["value"])),
        unit=raw.get("unit", "pct"),
        tolerance=Decimal(str(raw.get("tolerance", "0.5"))),
        doc_id=source.get("doc_id"),
        page=source.get("page"),
        restated=bool(raw.get("restated", False)),
    )


_LABELLER_KINDS = ("human", "automated")


def _checked_labelling(raw: dict[str, Any], case_id: str) -> tuple[bool, str, str]:
    """The labelling provenance. Refuses a verified case that doesn't say who labelled it.

    ``gold_verified: true`` with no labeller is exactly what this check is for: it reads as
    checked and scores as checked, but nobody can say who checked it.
    """
    verified = bool(raw.get("gold_verified", False))
    labeller = str(raw.get("labeller", "") or "").strip()
    kind = str(raw.get("labeller_kind", "") or "").strip().lower()
    if not verified:
        return False, labeller, kind
    if not labeller:
        raise ValueError(
            f"{case_id}: gold_verified is true but `labeller` is empty. Record who "
            "checked the figures against their pages."
        )
    if kind not in _LABELLER_KINDS:
        raise ValueError(
            f"{case_id}: gold_verified is true but `labeller_kind` is {kind!r}; "
            f"it must be one of {', '.join(_LABELLER_KINDS)}."
        )
    return True, labeller, kind


def load_cases(split: str | None = None, root: Path | None = None) -> list[EvalCase]:
    """Every case in a split, sorted by id so a run is reproducible."""
    base = root or (get_settings().evals_dir / "cases")
    directories = [base / split] if split else [base / name for name in ("dev", "test")]

    cases: list[EvalCase] = []
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for item in raw.get("cases", [raw]):
                cases.append(_case_from(item, path.parent.name))
    return sorted(cases, key=lambda case: case.case_id)


def _gold_drivers(raw: Any) -> tuple[GoldDriver, ...]:
    return tuple(
        GoldDriver(
            driver_id=item["driver_id"],
            contribution=(Decimal(str(item["contribution"])) if "contribution" in item else None),
            tolerance=(Decimal(str(item["tolerance"])) if "tolerance" in item else None),
            doc_id=(item.get("source", {}) or {}).get("doc_id"),
            page=(item.get("source", {}) or {}).get("page"),
        )
        for item in raw or ()
    )


def _statuses(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    return tuple(raw) if isinstance(raw, list) else (str(raw),)


def _case_from(raw: dict[str, Any], split: str) -> EvalCase:
    gold = raw.get("gold", {}) or {}
    status = raw.get("expected_status", "OK")
    statuses = tuple(status) if isinstance(status, list) else (status,)
    as_of = raw.get("as_of")
    if as_of is not None and not isinstance(as_of, date):
        as_of = date.fromisoformat(str(as_of))
    verified, labeller, kind = _checked_labelling(raw, str(raw.get("case_id", "?")))

    return EvalCase(
        case_id=raw["case_id"],
        split=raw.get("split", split),
        bank=raw["bank"],
        period=str(raw["period"]),
        metric=raw["metric"],
        as_of=as_of,
        categories=tuple(raw.get("categories", ()) or ()),
        expected_status=statuses,
        expected_abstain_reason=raw.get("expected_abstain_reason"),
        current=_gold_value(gold.get("current")),
        prior=_gold_value(gold.get("prior")),
        change=_gold_value(gold.get("change")),
        drivers=_gold_drivers(gold.get("drivers")),
        walk_drivers=_gold_drivers(gold.get("walk_drivers")),
        must_mention=tuple(gold.get("must_mention", ()) or ()),
        must_not_claim=tuple(gold.get("must_not_claim", ()) or ()),
        materiality=(Decimal(str(gold["materiality"])) if "materiality" in gold else None),
        gold_verified=verified,
        labeller=labeller,
        labeller_kind=kind,
        notes=raw.get("notes", ""),
        expected_status_with_model=_statuses(
            (raw.get("with_model", {}) or {}).get("expected_status")
        ),
        expected_abstain_reason_with_model=(raw.get("with_model", {}) or {}).get(
            "expected_abstain_reason"
        ),
    )
