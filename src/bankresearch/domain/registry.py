"""Typed access to the versioned config in ``config/``."""

from __future__ import annotations

import functools
import hashlib
import os
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, cast

import yaml

from ..settings import get_settings
from .calendar import fiscal_period
from .models import DocType, MetricId

__all__ = [
    "BankProfile",
    "LabelRule",
    "MetricDefinition",
    "Registry",
    "UnknownBankError",
    "UnknownMetricError",
    "get_registry",
]


class UnknownBankError(KeyError):
    """Mapped to the ``BANK_UNKNOWN`` reason code."""


class UnknownMetricError(KeyError):
    """Mapped to the ``METRIC_OUT_OF_SCOPE`` reason code."""


@dataclass(frozen=True, slots=True)
class LabelRule:
    """One bank's label for a metric, valid over a span of periods."""

    label: str
    valid_from: str | None = None
    valid_to: str | None = None
    extra_driver: str | None = None
    caveat: str | None = None
    pre_tax: bool = False
    """The measure is struck before tax, as Judo's profit before tax has been since 1H26."""

    def covers(self, period: str, fy_end_month: int) -> bool:
        """Whether this label was the issuer's headline in ``period``.

        Compared by period end date, so FY22 comes before 1H23 for a September year end.
        That ordering decides whether a Westpac request resolves to cash earnings or net
        profit.
        """
        end = fiscal_period(period, fy_end_month).end
        starts_too_late = (
            self.valid_from is not None and end < fiscal_period(self.valid_from, fy_end_month).end
        )
        ends_too_early = (
            self.valid_to is not None and end > fiscal_period(self.valid_to, fy_end_month).end
        )
        return not (starts_too_late or ends_too_early)


@dataclass(frozen=True, slots=True)
class BankProfile:
    bank_id: str
    name: str
    ticker: str
    fy_end_month: int
    in_seed_corpus: bool
    ir_domains: tuple[str, ...]
    cet1_scope: str
    headline_earnings_label: str | None
    results_release_months: dict[str, int]
    doc_name_hints: dict[str, Any]
    """Document type to the issuer's name for it, or to one name per kind of period."""
    verification: dict[str, str]
    not_reported: dict[str, str] = field(default_factory=dict)
    """Metric id to the reason the issuer does not report it."""
    synthetic: bool = False
    """A fictional issuer used only for the demonstration and the tests (ADR-012)."""
    impairment_divisions: dict[str, str] = field(default_factory=dict)
    """Division name as the issuer prints it, to the canonical segment it's split by
    (ADR-038). Empty for a bank whose charge is split as individually and collectively
    assessed."""
    aliases: tuple[str, ...] = ()
    """Other names a person asks for the bank by: "CommBank", "ANZ Bank", "Judo Bank"."""
    discovered: bool = False
    """Not configured: found from its ASX listing, with its year end confirmed by a filing."""
    row_labels: tuple[tuple[str, str], ...] = ()
    """For a discovered bank, row labels in its own words and the component each holds,
    matched by a model and kept only once the figures they gave passed their checks."""

    def period(self, label: str) -> Any:
        return fiscal_period(label, self.fy_end_month)

    @property
    def unverified_fields(self) -> tuple[str, ...]:
        return tuple(k for k, v in self.verification.items() if v == "unverified")


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    metric_id: MetricId
    display_name: str
    unit: str
    change_unit: str
    higher_is_favourable: bool
    headline_basis: str
    primary_path: str
    bank_labels: dict[str, tuple[LabelRule, ...]]
    alternative_label_patterns: tuple[str, ...]
    related_metric_fallback: dict[str, Any]
    point_in_time: bool
    context_measures: tuple[str, ...]
    show_pct: bool
    show_pp: bool

    def label_for(self, bank: BankProfile, period: str) -> LabelRule | None:
        """The label this bank led with for this metric in this period."""
        for rule in self.bank_labels.get(bank.bank_id, ()):
            if rule.covers(period, bank.fy_end_month):
                return rule
        return None


def _as_label_rules(raw: Any) -> tuple[LabelRule, ...]:
    rules: list[LabelRule] = []
    for item in raw or ():
        if isinstance(item, str):
            rules.append(LabelRule(label=item))
        else:
            rules.append(
                LabelRule(
                    label=item["label"],
                    valid_from=item.get("valid_from"),
                    valid_to=item.get("valid_to"),
                    extra_driver=item.get("extra_driver"),
                    caveat=item.get("caveat"),
                    pre_tax=bool(item.get("pre_tax", False)),
                )
            )
    return tuple(rules)


@dataclass(frozen=True)
class Registry:
    """Everything loaded from ``config/``, validated on construction."""

    banks: dict[str, BankProfile]
    metrics: dict[str, MetricDefinition]
    drivers: dict[str, dict[str, dict[str, Any]]]
    any_metric_drivers: dict[str, dict[str, Any]]
    precedence: dict[str, Any]
    budgets: dict[str, Any]
    models: dict[str, Any]
    regime_changes: tuple[dict[str, Any], ...]
    playbooks: dict[str, dict[str, Any]]
    tolerances: dict[str, dict[str, str]]
    percentage_change: dict[str, Any]
    config_hash: str

    # -- lookups ----------------------------------------------------------

    def bank(self, bank_id_or_name: str) -> BankProfile:
        key = bank_id_or_name.strip().upper()
        if key in self.banks:
            return self.banks[key]
        for profile in self.banks.values():
            if key in {profile.ticker.upper(), profile.name.upper()}:
                return profile
            if key in {alias.upper() for alias in profile.aliases}:
                return profile
            # Accept a distinctive leading word, e.g. "Westpac" or "Bendigo".
            if profile.name.upper().startswith(key) and len(key) >= 3:
                return profile
        raise UnknownBankError(
            f"{bank_id_or_name!r} is not a configured bank. Known: {', '.join(sorted(self.banks))}."
        )

    def register(self, profile: BankProfile) -> None:
        """Add a bank found while running, for the rest of this process. The configured
        banks are never replaced."""
        if profile.bank_id not in self.banks or self.banks[profile.bank_id].discovered:
            self.banks[profile.bank_id] = profile

    def metric(self, metric_id: str) -> MetricDefinition:
        key = metric_id.strip().lower()
        if key in self.metrics:
            return self.metrics[key]
        # Written the way the ids are, "cash_earnings" or "net-interest-margin" was refused
        # as out of scope, though "cash earnings" was understood.
        spaced = " ".join(key.replace("_", " ").replace("-", " ").split())
        for canonical, definition in self.metrics.items():
            if spaced in {canonical.replace("_", " "), definition.display_name.lower()}:
                return definition
        for canonical, aliases in _METRIC_ALIASES.items():
            if key in aliases or spaced in aliases:
                return self.metrics[canonical]
        raise UnknownMetricError(
            f"{metric_id!r} is not a metric in scope. In scope: {', '.join(sorted(self.metrics))}."
        )

    def driver_ids(self, metric_id: str) -> tuple[str, ...]:
        """Canonical driver ids for a metric, including its own residual/other."""
        group = self.drivers.get(metric_id, {})
        extra = tuple(
            key.format(metric=_residual_prefix(metric_id)) for key in self.any_metric_drivers
        )
        return tuple(group) + extra

    def driver_label(self, metric_id: str, driver_id: str) -> str:
        group = self.drivers.get(metric_id, {})
        if driver_id in group:
            return cast(str, group[driver_id].get("label", driver_id))
        for key, spec in self.any_metric_drivers.items():
            if driver_id == key.format(metric=_residual_prefix(metric_id)):
                return cast(str, spec.get("label", driver_id))
        # A child taken from another metric's taxonomy: return on equity's earnings effect
        # splits into the earnings bridge's lines, and a reader was shown "earn.nii".
        for spec in group.values():
            source = self.drivers.get(str(spec.get("children_from", "")), {})
            if driver_id in source:
                return cast(str, source[driver_id].get("label", driver_id))
        return driver_id

    def driver_parent(self, metric_id: str, driver_id: str) -> str | None:
        group = self.drivers.get(metric_id, {})
        spec = group.get(driver_id)
        return cast("str | None", spec.get("parent")) if spec else None

    def metric_names(self, metric_id: str, bank_id: str | None = None) -> tuple[str, ...]:
        """Every name this metric goes by: display name, aliases and the issuer's labels.

        Used to tie a published walk to the metric it's actually about. A walk table can be
        spotted by its shape, but the shape doesn't say whether the waterfall is for margin
        or capital, and a correct walk applied to the wrong metric explains nothing.
        """
        definition = self.metrics[metric_id]
        names: list[str] = [definition.display_name, *_METRIC_ALIASES.get(metric_id, ())]
        names.extend(definition.alternative_label_patterns)
        if bank_id is not None:
            names.extend(rule.label for rule in definition.bank_labels.get(bank_id, ()))
        else:
            names.extend(rule.label for rules in definition.bank_labels.values() for rule in rules)
        return tuple(sorted({name.strip() for name in names if name and name.strip()}))

    def playbook(self, metric_id: str) -> dict[str, Any]:
        return self.playbooks[metric_id]

    def doc_precedence(self, doc_type: DocType) -> int:
        return int(self.precedence["doc_type_precedence"][doc_type])

    def regime_changes_for(
        self, bank_id: str, metric_id: str, start: date, end: date
    ) -> tuple[dict[str, Any], ...]:
        """Regime changes whose effective date falls inside the comparison span.

        A definition change between the two periods is what makes them not like-for-like, so
        the window runs from the start of the comparison period to the end of the current
        one.
        """
        hits: list[dict[str, Any]] = []
        for change in self.regime_changes:
            effective = change.get("effective_date")
            if effective is None:
                continue
            if bank_id not in change.get("banks", ()):
                continue
            if metric_id not in change.get("metrics", ()):
                continue
            if start <= effective <= end:
                hits.append(change)
        return tuple(hits)

    def route(self, step: str) -> tuple[str, str]:
        """The model and effort for a step, from ``config/models.yaml``.

        Raises if the step isn't listed instead of falling back to a default. Steps used to
        carry a hardcoded default model in their own signatures and nothing passed
        ``model=``, so the routing config was read and checked but never actually used. A
        missing entry is a config error and should fail loudly.
        """
        routing = self.models.get("routing") or {}
        entry = routing.get(step)
        if not entry:
            raise KeyError(
                f"no routing for step {step!r} in config/models.yaml; "
                f"routed steps: {', '.join(sorted(routing))}"
            )
        return str(entry["model"]), str(entry["effort"])

    def output_limit(self, step: str) -> int:
        """The most a step's call may produce, thinking included, from ``config/budgets.yaml``.

        Raises for a step with no cap, like :meth:`route` does for a step with no model. A
        default here is what let every call go out at 4,096 tokens for as long as it did.
        """
        caps = self.budgets.get("token_caps") or {}
        entry = caps.get(step) or {}
        if not entry.get("output"):
            raise KeyError(
                f"no output cap for step {step!r} under token_caps in config/budgets.yaml; "
                f"capped steps: {', '.join(sorted(caps))}"
            )
        return int(entry["output"])

    def tolerance(self, kind: str, key: str) -> Decimal:
        return Decimal(self.tolerances[kind][key])

    @property
    def unverified(self) -> dict[str, tuple[str, ...]]:
        """Fields still marked ``unverified``, for ``bankresearch config-check``.

        Model prices live under a ``pricing`` key rather than under a bank, and are listed
        alongside the banks so one command covers every value that has to come from a
        source.
        """
        outstanding = {
            bank_id: profile.unverified_fields
            for bank_id, profile in sorted(self.banks.items())
            if profile.unverified_fields
        }
        prices = tuple(
            model
            for model, entry in sorted(self.models.get("pricing", {}).items())
            if entry.get("verification") == "unverified"
        )
        if prices:
            outstanding["pricing"] = prices
        return outstanding


#: What a person might type, mapped to the metric they mean.
#:
#: This is for requests, not documents: ``config/line_items.yaml`` maps row labels inside
#: filings, and the two lists cover different things. "Efficiency ratio" is what bankers
#: usually call cost-to-income, and before it was added here that common term got
#: METRIC_OUT_OF_SCOPE.
_METRIC_ALIASES: dict[str, frozenset[str]] = {
    "nim": frozenset({"net interest margin", "margin", "nim", "interest margin", "net margin"}),
    "headline_earnings": frozenset(
        {
            "cash earnings",
            "cash profit",
            "cash npat",
            "net profit",
            "earnings",
            "profit",
            "cash net profit after tax",
            "npat",
            "headline earnings",
        }
    ),
    "roe": frozenset(
        {"return on equity", "roe", "return on ordinary equity", "return on average equity"}
    ),
    "cet1_ratio": frozenset(
        {
            "cet1",
            "cet1 ratio",
            "common equity tier 1",
            "common equity tier 1 ratio",
            "capital ratio",
            "cet1 capital ratio",
        }
    ),
    "credit_impairment_charge": frozenset(
        {
            "credit impairment",
            "impairment",
            "credit impairment charge",
            "loan impairment expense",
            "impairment charge",
            "loan losses",
            "provisions",
            "bad debts",
            "credit loss",
        }
    ),
    "cost_to_income": frozenset(
        {
            "cost to income",
            "cost-to-income",
            "cti",
            "expense to income",
            "efficiency ratio",
            "cost income ratio",
            "cost to income ratio",
            "operating efficiency",
        }
    ),
}

_RESIDUAL_PREFIX: dict[str, str] = {
    "nim": "nim",
    "headline_earnings": "earn",
    "roe": "roe",
    "cet1_ratio": "cet1",
    "credit_impairment_charge": "cic",
    "cost_to_income": "cti",
}


def _residual_prefix(metric_id: str) -> str:
    return _RESIDUAL_PREFIX.get(metric_id, metric_id)


def residual_driver_id(metric_id: str) -> str:
    return f"{_residual_prefix(metric_id)}.residual"


def other_driver_id(metric_id: str) -> str:
    return f"{_residual_prefix(metric_id)}.other"


def _load(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _build(config_dir: Path) -> Registry:
    banks_raw = _load(config_dir / "banks.yaml")
    metrics_raw = _load(config_dir / "metrics.yaml")
    drivers_raw = _load(config_dir / "drivers.yaml")
    precedence = _load(config_dir / "precedence.yaml")
    budgets = _load(config_dir / "budgets.yaml")
    models = _load(config_dir / "models.yaml")
    regimes = _load(config_dir / "regime_changes.yaml")

    banks = {
        entry["bank_id"]: BankProfile(
            bank_id=entry["bank_id"],
            name=entry["name"],
            ticker=entry["ticker"],
            fy_end_month=int(entry["fy_end_month"]),
            in_seed_corpus=bool(entry.get("in_seed_corpus", False)),
            ir_domains=tuple(entry.get("ir_domains", ())),
            cet1_scope=entry.get("cet1_scope", ""),
            headline_earnings_label=entry.get("headline_earnings_label"),
            results_release_months=dict(entry.get("results_release_months", {})),
            doc_name_hints=dict(entry.get("doc_name_hints", {})),
            verification=dict(entry.get("verification", {})),
            not_reported=dict(entry.get("not_reported", {}) or {}),
            synthetic=bool(entry.get("synthetic", False)),
            impairment_divisions=_impairment_divisions(entry),
            aliases=tuple(str(alias) for alias in entry.get("aliases", ()) or ()),
        )
        for entry in banks_raw["banks"]
    }

    metrics: dict[str, MetricDefinition] = {}
    for metric_id, spec in metrics_raw["metrics"].items():
        metrics[metric_id] = MetricDefinition(
            metric_id=cast(MetricId, metric_id),
            display_name=spec["display_name"],
            unit=spec["unit"],
            change_unit=spec["change_unit"],
            higher_is_favourable=bool(spec.get("higher_is_favourable", True)),
            headline_basis=spec.get("headline_basis", ""),
            primary_path=spec.get("primary_path", "B"),
            bank_labels={
                bank_id: _as_label_rules(rules)
                for bank_id, rules in (spec.get("bank_labels") or {}).items()
            },
            alternative_label_patterns=tuple(spec.get("alternative_label_patterns", ())),
            related_metric_fallback=dict(spec.get("related_metric_fallback", {})),
            point_in_time=bool(spec.get("point_in_time", False)),
            context_measures=tuple(spec.get("context_measures", ())),
            show_pct=bool(spec.get("show_pct", False)),
            show_pp=bool(spec.get("show_pp", False)),
        )

    playbooks: dict[str, dict[str, Any]] = {}
    for path in sorted((config_dir / "playbooks").glob("*.yaml")):
        spec = _load(path)
        playbooks[spec["metric"]] = spec

    # Fail loudly at start-up rather than mid-run.
    missing = set(metrics) - set(playbooks)
    if missing:
        raise ValueError(f"metrics without a playbook: {sorted(missing)}")
    missing_drivers = set(metrics) - set(drivers_raw["drivers"])
    if missing_drivers:
        raise ValueError(f"metrics without a driver taxonomy: {sorted(missing_drivers)}")
    # Only an earnings measure can be struck before tax; on a ratio the flag would swap the
    # figure for profit before tax.
    misplaced = sorted(
        f"{metric_id}/{bank_id}"
        for metric_id, definition in metrics.items()
        if metric_id != "headline_earnings"
        for bank_id, rules in definition.bank_labels.items()
        if any(rule.pre_tax for rule in rules)
    )
    if misplaced:
        raise ValueError(f"pre_tax is only meaningful for headline_earnings: {misplaced}")

    digest = hashlib.sha256()
    for path in sorted(config_dir.rglob("*.yaml")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())

    return Registry(
        banks=banks,
        metrics=metrics,
        drivers=drivers_raw["drivers"],
        any_metric_drivers=drivers_raw.get("any_metric", {}),
        precedence=precedence,
        budgets=budgets,
        models=models,
        regime_changes=tuple(regimes.get("regime_changes", ())),
        playbooks=playbooks,
        tolerances=metrics_raw["tolerances"],
        percentage_change=metrics_raw["percentage_change"],
        config_hash=digest.hexdigest()[:16],
    )


#: The canonical segments, in the order the impairment taxonomy lists them.
IMPAIRMENT_SEGMENTS: tuple[str, ...] = (
    "retail",
    "business",
    "institutional",
    "new_zealand",
    "other",
)


def _impairment_divisions(entry: dict[str, Any]) -> dict[str, str]:
    """A bank's division-to-segment map, checked whole at start-up.

    Each segment must be named exactly once. A split with a segment missing can't be used
    (the missing division would land in the residual and still look like an answer), and
    two divisions on one segment would mean storing a sum no page prints, which no citation
    could support.
    """
    raw = entry.get("impairment_divisions") or {}
    mapping = {str(name): str(segment) for name, segment in raw.items()}
    if mapping and sorted(mapping.values()) != sorted(IMPAIRMENT_SEGMENTS):
        raise ValueError(
            f"{entry.get('bank_id')}: impairment_divisions must name each of "
            f"{', '.join(IMPAIRMENT_SEGMENTS)} exactly once, got {sorted(mapping.values())}"
        )
    return mapping


@lru_cache(maxsize=4)
def _cached(config_dir: str) -> Registry:
    return _build(Path(config_dir))


#: The configured directory, by what could change it: the environment variable, the working
#: directory a ``.env`` file is read from, and that file's last change. A run asks for the
#: registry about seventy times, and building the settings object each time to find the
#: same directory was a fifth of a run with no model.
_CONFIG_DIRS: dict[tuple[str | None, str, float | None], Path] = {}


def _default_config_dir() -> Path:
    try:
        env_file = Path(".env")
        key = (
            os.environ.get("BANKRESEARCH_CONFIG_DIR"),
            os.getcwd(),
            env_file.stat().st_mtime if env_file.exists() else None,
        )
    except OSError:
        # The working directory couldn't be looked at. A live run on BNK Banking
        # Corporation crashed here, after a paid call had succeeded, when macOS briefly
        # refused every path under the Desktop folder. The directory found last is still
        # the right one.
        if _CONFIG_DIRS:
            return next(reversed(_CONFIG_DIRS.values()))
        raise
    found = _CONFIG_DIRS.get(key)
    if found is None:
        found = _CONFIG_DIRS[key] = get_settings().config_dir
    return found


#: Plainer than the first synonym line_items.yaml lists, where that is a full row label.
_PLAIN_NAMES: Final[dict[str, str]] = {
    "aiea": "average interest-earning assets",
    "average_equity": "average equity",
    "cet1_capital": "CET1 capital",
    "minority_interests": "non-controlling interests",
    "rwa": "risk-weighted assets",
    "rwa_credit": "credit risk-weighted assets",
    "rwa_market": "market risk-weighted assets",
    "rwa_operational": "operational risk-weighted assets",
    "tax": "income tax",
}


@functools.lru_cache(maxsize=4)
def _line_item_names(config_dir: str) -> dict[str, str]:
    raw = _load(Path(config_dir) / "line_items.yaml") or {}
    names: dict[str, str] = {}
    for component, spec in (raw.get("components") or {}).items():
        synonyms = (spec or {}).get("synonyms") or []
        if synonyms:
            names[str(component)] = str(synonyms[0])
    return names


def component_name(component: str, config_dir: Path | None = None) -> str:
    """A figure's name in words, for text a reader sees: "net interest income", not "nii".
    A metric is called by its display name; a line item by its plain name or the first
    label config/line_items.yaml gives it."""
    registry = get_registry(config_dir)
    if component in registry.metrics:
        name = registry.metrics[component].display_name
        return name[0].lower() + name[1:] if name[1:2].islower() else name
    if component in _PLAIN_NAMES:
        return _PLAIN_NAMES[component]
    directory = str(config_dir or _default_config_dir())
    return _line_item_names(directory).get(component, component.replace("_", " "))


def get_registry(config_dir: Path | None = None) -> Registry:
    directory = config_dir or _default_config_dir()
    return _cached(str(directory))
