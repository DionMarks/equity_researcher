"""The rules the design rests on, checked structurally."""

from __future__ import annotations

import ast
import re
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from bankresearch.agent.state import ABSTAIN_REASONS, RUN_STATUSES, STATES
from bankresearch.confidence import rubric
from bankresearch.confidence.rubric import CLAMP_HIGH, CLAMP_LOW, LEVEL_TARGETS
from bankresearch.decomposition.identities import IDENTITIES
from bankresearch.decomposition.shapley import MAX_FACTORS
from bankresearch.domain.registry import get_registry
from bankresearch.reconcile.conflicts import _DEFAULT_DOC_PRECEDENCE, CLASSIFICATION_ORDER

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "bankresearch"

# ADR-005: aggregate nodes added so that an undisclosed split is reported in
# aggregate rather than invented, and so the NII split can nest.
ADR_005_ADDITIONS = {
    "earn.nii",
    "cet1.capital",
    "cet1.rwa",
    "cic.collective",
    "nim.income",
    "nim.balance_growth",
}

#: The impairment charge by division, for an issuer that discloses it no other way.
ADR_038_ADDITIONS = {
    "cic.div_retail",
    "cic.div_business",
    "cic.div_institutional",
    "cic.div_new_zealand",
    "cic.div_other",
}


# --- the driver taxonomy -------------------------------------------------


def test_the_other_and_residual_drivers_exist_for_every_metric():
    registry = get_registry()
    for metric_id in registry.metrics:
        ids = registry.driver_ids(metric_id)
        assert any(i.endswith(".other") for i in ids), metric_id
        assert any(i.endswith(".residual") for i in ids), metric_id


def test_every_aggregate_driver_node_is_built_and_documented():
    """The nodes ADR-005 and ADR-038 add stand in for a split a bank doesn't disclose. Each
    has to exist in the taxonomy and be named in the decision that added it."""
    design = (REPO / "DESIGN.md").read_text(encoding="utf-8")
    built = {driver for group in get_registry().drivers.values() for driver in group}
    assert "ADR-005" in design and "ADR-038" in design
    for driver_id in ADR_005_ADDITIONS | ADR_038_ADDITIONS:
        assert driver_id in built, driver_id
    for driver_id in ADR_005_ADDITIONS:
        assert driver_id in design, driver_id


# --- invariants, checked structurally -------------------------------------


def _imports_of(package: str) -> set[str]:
    """Every module imported anywhere under ``src/bankresearch/<package>``.

    Relative imports are resolved to absolute names. Otherwise ``from ..evals import x``
    inside ``agent/`` would be invisible to the invariant-8 check, which would be a hole in
    the very test that enforces it.
    """
    found: set[str] = set()
    for path in (SRC / package).rglob("*.py"):
        own_package = ["bankresearch", *path.relative_to(SRC).parent.parts]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    if node.module:
                        found.add(node.module)
                    continue
                # level 1 is this package, level 2 its parent, and so on.
                base = own_package[: len(own_package) - (node.level - 1)]
                resolved = ".".join([*base, node.module]) if node.module else ".".join(base)
                found.add(resolved)
                for alias in node.names:
                    found.add(f"{resolved}.{alias.name}")
    return found


@pytest.mark.parametrize("package", ["domain", "decomposition", "reconcile", "confidence"])
def test_invariant_1_the_deterministic_cores_cannot_reach_a_model(package):
    """The model never does arithmetic, enforced by what the code can import."""
    imported = _imports_of(package)
    assert not any(name.startswith("anthropic") for name in imported), imported
    assert not any("bankresearch.llm" in name or name == "llm" for name in imported)


# The agent has to WRITE runs/<run_id>/; invariant 8 forbids it
# reading them back. Only these two modules may name the runs directory at all.
_MAY_WRITE_RUNS = {"agent/orchestrator.py", "agent/steps/render.py"}

_READS_RUNS = re.compile(
    r"runs_dir[^\n]*\.(?:read_text|read_bytes|open|iterdir|glob|rglob|load)"
    r"|(?:read_text|read_bytes|iterdir|glob|rglob)\([^)]*runs",
)


def test_invariant_8_agent_code_cannot_read_gold_labels_or_past_runs():
    """Agent code must not import the harness, reach evals/, or read back runs/."""
    for package in ("agent", "tools", "decomposition", "reconcile", "confidence", "output"):
        directory = SRC / package
        if not directory.exists():
            continue
        assert "bankresearch.evals" not in _imports_of(package), package
        for path in directory.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            relative = path.relative_to(SRC).as_posix()
            for forbidden in ('"evals', "'evals", "evals_dir"):
                assert forbidden not in source, f"{relative} references {forbidden}"
            if "runs_dir" in source or '"runs' in source or "'runs" in source:
                assert relative in _MAY_WRITE_RUNS, f"{relative} names the runs directory"
            assert not _READS_RUNS.search(source), f"{relative} reads back from runs/"


def test_the_modules_that_write_runs_only_write_them():
    """Writing a run's own artefacts is required; reading any run back is not."""
    for relative in _MAY_WRITE_RUNS:
        source = (SRC / relative).read_text(encoding="utf-8")
        assert not _READS_RUNS.search(source), relative


def test_invariant_3_the_comparison_period_is_always_the_pcp():
    """No code path offers half-on-half as a comparison basis."""
    for path in SRC.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        # `hoh()` exists on FiscalPeriod for chaining walks only, and nothing
        # may use it to set a comparison period.
        assert "comparison_period=.*hoh" not in source
    from bankresearch.domain.calendar import fiscal_period

    for label in ("1H26", "2H25", "FY25"):
        period = fiscal_period(label, 9)
        assert period.pcp().label[-2:] == f"{period.fy - 1 - 2000:02d}"


def test_the_decomposition_package_declares_the_invariant_in_its_docstring():
    import bankresearch.decomposition as package

    assert package.__doc__ is not None
    assert "never does arithmetic" in package.__doc__


# --- declared constants ---------------------------------------------------


def test_the_abstain_reasons_are_the_seven_codes():
    assert ABSTAIN_REASONS == (
        "AMBIGUOUS_INPUT",
        "METRIC_OUT_OF_SCOPE",
        "BANK_UNKNOWN",
        "PERIOD_NOT_YET_REPORTED",
        "DOCS_UNAVAILABLE",
        "METRIC_NOT_REPORTED",
        "INSUFFICIENT_EVIDENCE",
    )


def test_the_run_statuses():
    assert RUN_STATUSES == ("OK", "PARTIAL", "ABSTAIN")


def test_there_are_twelve_steps():
    assert len(STATES) == 12


def test_the_conflict_classes_and_their_order():
    assert CLASSIFICATION_ORDER == (
        "UNIT",
        "ROUNDING",
        "BASIS",
        "SCOPE",
        "PRO_FORMA",
        "RESTATEMENT",
        "TIMING",
        "UNEXPLAINED",
    )


def test_confidence_is_earned_from_evidence_up_to_the_ceiling():
    """No claim starts from a fixed score for its kind (ADR-072). A printed figure starts at
    the floor and earns the rest, and everything that can stand behind it reaches exactly the
    ceiling."""
    assert not hasattr(rubric, "BASE_SCORES")
    earned = rubric._PRIMARY + rubric._CORROBORATED + rubric._CONFIRMED
    assert rubric._PRINTED + earned == CLAMP_HIGH


def test_the_confidence_clamp_and_targets():
    assert (CLAMP_LOW, CLAMP_HIGH) == (Decimal("0.05"), Decimal("0.98"))
    assert LEVEL_TARGETS == {"High": Decimal("0.90"), "Medium": Decimal("0.70"), "Low": None}


def test_shapley_is_capped_at_five_factors():
    assert MAX_FACTORS == 5


def test_identities_exist_for_all_six_metrics():
    # Plus the earnings bridge struck before tax, which answers the earnings metric for a
    # bank that leads with profit before tax (DESIGN.md ADR-062).
    assert set(IDENTITIES) == set(get_registry().metrics) | {"profit_before_tax"}


def test_the_primary_path_for_each_metric():
    registry = get_registry()
    assert registry.metrics["nim"].primary_path == "A"
    assert registry.metrics["cet1_ratio"].primary_path == "A_else_B"
    for metric_id in ("headline_earnings", "roe", "credit_impairment_charge", "cost_to_income"):
        assert registry.metrics[metric_id].primary_path == "B"


def test_cet1_is_the_only_point_in_time_metric():
    registry = get_registry()
    point_in_time = {m for m, d in registry.metrics.items() if d.point_in_time}
    assert point_in_time == {"cet1_ratio"}


# --- config holds the values the design sets ------------------------------


def test_the_document_precedence():
    registry = get_registry()
    assert registry.precedence["doc_type_precedence"] == {
        "RESULTS_DOCUMENT": 1,
        "RESTATEMENT_NOTICE": 1,
        "RESULTS_RELEASE": 2,
        "INVESTOR_PRESENTATION": 2,
        "PILLAR3": 3,
        "ANNUAL_REPORT": 3,
    }


def test_the_fallback_precedence_table_agrees_with_config():
    assert _DEFAULT_DOC_PRECEDENCE == get_registry().precedence["doc_type_precedence"]


def test_the_token_caps():
    caps = get_registry().budgets["token_caps"]
    assert caps["locate"] == {"input": 10000, "output": 4000}
    assert caps["explain"] == {"input": 20000, "output": 16000}
    assert caps["verify"] == {"input": 15000, "output": 8000}


def test_the_step_caps():
    caps = get_registry().budgets["step_caps"]
    assert caps["locate"]["tool_calls"] == 12
    assert caps["explain"] == {"calls": 1, "repairs": 1}
    assert caps["verify"] == {"calls": 1, "rechecks": 1}


def test_the_warm_path_budget():
    warm = get_registry().budgets["paths"]["warm"]
    assert warm["max_llm_calls"] == 25
    assert warm["max_input_tokens"] == 300000
    assert warm["max_output_tokens"] == 60000
    assert Decimal(warm["cost_guardrail_usd_p95"]) == Decimal("1.50")
    assert warm["latency_p50_seconds"] == 120
    assert warm["latency_p95_seconds"] == 240


def test_the_cold_path_budget():
    cold = get_registry().budgets["paths"]["cold"]
    assert cold["latency_seconds"] == 600
    assert Decimal(cold["cost_guardrail_usd_per_new_period"]) == Decimal("5.00")
    assert cold["on_breach"] == "abstain_docs_unavailable"


def test_every_model_step_is_routed_and_priceable():
    """The steps and efforts are fixed; the model is the operator's choice.

    This test used to pin a mix of model ids (Sonnet for location and extraction, Opus for
    judgement). That turned a model change into a test failure, which is wrong: the point of
    ``config/models.yaml`` is that models can be swapped without touching code. What has to
    hold is that every model step is routed, that its effort is one the API accepts, and
    that its model has a price, because an unpriced model can't be quoted to the person
    paying for the call.
    """
    models = get_registry().models
    routing = models["routing"]
    pricing = models["pricing"]

    for step in ("locate", "text_walk", "chart_read", "explain", "verify"):
        assert step in routing, f"{step} is a model step and nothing routes it"
        assert routing[step]["model"] in pricing, f"{step} routes to an unpriced model"

    # The efforts set for the steps whose depth is fixed.
    assert routing["explain"]["effort"] == "high"
    assert routing["verify"]["effort"] == "medium"
    assert routing["locate"]["effort"] == "medium"
    assert routing["text_walk"]["effort"] == "high"

    # The fallback must be a different model from the one it rescues, or the retry a
    # refusal or an exceeded context window earns is the same call again.
    assert models["fallback_model"] in pricing
    assert models["fallback_model"] != routing["explain"]["model"]


def test_the_api_constraints():
    constraints = get_registry().models["api_constraints"]
    assert constraints["send_sampling_parameters"] is False
    assert constraints["assistant_prefill"] is False
    assert constraints["force_tool_choice"] is False
    assert constraints["thinking_always_on"] is True
    assert constraints["structured_outputs"] is True
    assert constraints["prompt_caching"] is True


def test_the_tolerances():
    registry = get_registry()
    assert registry.tolerance("ratio", "value_match_bps") == Decimal("0.5")
    assert registry.tolerance("ratio", "walk_floor_bps") == Decimal("1")
    assert registry.tolerance("ratio", "walk_per_step_bps") == Decimal("0.5")
    assert registry.tolerance("aud", "value_match_aud_m") == Decimal("0.5")
    assert registry.tolerance("aud", "walk_floor_aud_m") == Decimal("2")
    assert registry.tolerance("aud", "walk_per_step_aud_m") == Decimal("0.5")


def test_the_bank_calendars():
    registry = get_registry()
    real = {
        bank_id: profile.fy_end_month
        for bank_id, profile in registry.banks.items()
        if not profile.synthetic
    }
    # The first six, and two added since from their own filings (config/banks.yaml says
    # where each calendar was verified).
    assert real == {
        "CBA": 6,
        "WBC": 9,
        "NAB": 9,
        "ANZ": 9,
        "BEN": 6,
        "BOQ": 8,
        "JDO": 6,
        "MYS": 6,
    }


def test_the_seed_corpus_is_the_four_majors():
    registry = get_registry()
    seed = {b for b, p in registry.banks.items() if p.in_seed_corpus}
    assert seed == {"CBA", "WBC", "NAB", "ANZ"}


def test_the_cold_path_banks_have_no_seed_documents():
    registry = get_registry()
    cold = {
        bank_id
        for bank_id, profile in registry.banks.items()
        if not profile.in_seed_corpus and not profile.synthetic
    }
    assert cold == {"BEN", "BOQ", "JDO", "MYS"}


def test_the_only_synthetic_issuer_is_the_fixture_one_and_it_is_documented():
    """Invented figures must never be attributable to a real bank (ADR-012)."""
    registry = get_registry()
    synthetic = {b for b, p in registry.banks.items() if p.synthetic}
    assert synthetic == {"XMPL"}
    assert not registry.banks["XMPL"].in_seed_corpus
    assert "ADR-012" in (REPO / "DESIGN.md").read_text(encoding="utf-8")


# --- repo layout ----------------------------------------------------------


@pytest.mark.parametrize(
    "relative",
    [
        "config/banks.yaml",
        "config/metrics.yaml",
        "config/drivers.yaml",
        "config/precedence.yaml",
        "config/regime_changes.yaml",
        "config/budgets.yaml",
        "config/models.yaml",
        "config/playbooks",
        "data/manifest.yaml",
        "src/bankresearch/domain",
        "src/bankresearch/acquisition",
        "src/bankresearch/ingestion",
        "src/bankresearch/retrieval",
        "src/bankresearch/tools",
        "src/bankresearch/agent",
        "src/bankresearch/decomposition",
        "src/bankresearch/reconcile",
        "src/bankresearch/confidence",
        "src/bankresearch/memory",
        "src/bankresearch/llm",
        "src/bankresearch/output",
        "src/bankresearch/evals",
        "evals/cases/dev",
        "evals/cases/test",
        "tests/unit",
        "tests/integration",
        "tests/fixtures",
        "README.md",
        "DESIGN.md",
        "Makefile",
        "Dockerfile",
        ".env.example",
        ".importlinter",
        ".github/workflows/ci.yml",
    ],
)
def test_the_repo_layout_exists(relative):
    assert (REPO / relative).exists(), relative


def test_gold_labels_live_outside_src():
    assert (REPO / "evals" / "cases").exists()
    assert not (SRC / "cases").exists()
    assert not list(SRC.rglob("*.gold.yaml"))


# --- the deliverable documents hang together -------------------------------

_DELIVERABLES = ("README.md", "DESIGN.md")
_ANCHOR_LINK = re.compile(r"\]\((?:DESIGN\.md)?#(adr-[a-z0-9-]+)\)")


def _design_anchors() -> set[str]:
    """GitHub's slug for each ADR heading in DESIGN.md."""
    anchors: set[str] = set()
    for line in (REPO / "DESIGN.md").read_text(encoding="utf-8").splitlines():
        if not line.startswith("### ADR-"):
            continue
        slug = line[4:].strip().lower()
        slug = re.sub(r"[^a-z0-9 -]", "", slug).replace(" ", "-")
        anchors.add(slug)
    return anchors


@pytest.mark.parametrize("name", _DELIVERABLES)
def test_every_adr_link_in_the_documents_resolves(name):
    """A dead anchor sends a reviewer looking for a decision that appears absent."""
    anchors = _design_anchors()
    broken = [
        slug
        for slug in _ANCHOR_LINK.findall((REPO / name).read_text(encoding="utf-8"))
        if slug not in anchors
    ]
    assert broken == [], f"{name} links to missing ADR anchors: {broken}"


def test_the_adrs_are_numbered_without_gaps_or_duplicates():
    numbers = [
        int(line[8:11])
        for line in (REPO / "DESIGN.md").read_text(encoding="utf-8").splitlines()
        if line.startswith("### ADR-")
    ]
    assert numbers == sorted(numbers), "ADRs are out of order"
    assert len(numbers) == len(set(numbers)), "duplicate ADR numbers"
    assert numbers == list(range(1, len(numbers) + 1)), "gap in ADR numbering"


@pytest.mark.parametrize("name", _DELIVERABLES)
def test_every_file_the_documents_link_to_exists(name):
    """A link to a file that's been moved or deleted fails here, not in a reviewer's browser."""
    text = (REPO / name).read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text)
    gone = sorted({t for t in targets if not t.startswith("http") and not (REPO / t).exists()})
    assert gone == [], f"{name} links to files that don't exist: {gone}"


@pytest.mark.parametrize("name", _DELIVERABLES)
def test_no_deliverable_claims_an_eval_result(name):
    """A result claimed in prose can't be traced to a run.

    Measured figures belong in the tables copied from the eval report; a sentence like "ECE
    of 0.04" in a deliverable is a number nobody can check.
    """
    text = (REPO / name).read_text(encoding="utf-8").lower()
    for forbidden in ("ece of 0.0", "high-confidence error rate was", "driver f1 of"):
        assert forbidden not in text, f"{name} appears to claim an eval result"


# --- acquisition -----------------------------------------------------------


def test_the_fetcher_enforces_an_allowlist_and_rechecks_redirects():
    """Downloads are limited to the issuer's own domains plus the ASX,
    enforced in code. A redirect off the allowlist is how a download leaves the issuer's
    site without anyone noticing, so the final URL is checked too.
    """
    from bankresearch.acquisition.fetch import ASX_DOMAIN, MAX_BYTES, host_allowed

    assert host_allowed("https://www.anz.com/x.pdf", ("anz.com",))
    assert host_allowed("https://capital.nab.com.au/x.pdf", ("nab.com.au",))
    assert host_allowed(f"https://{ASX_DOMAIN}/x.pdf", ("anz.com",))
    assert not host_allowed("https://evil.example/x.pdf", ("anz.com",))
    # Suffix confusion must not pass.
    assert not host_allowed("https://notanz.com/x.pdf", ("anz.com",))
    assert MAX_BYTES == 50 * 1024 * 1024

    # Plaintext HTTP is a downgrade: these documents are the whole evidence base, and a
    # figure altered in transit would be cited, reconciled and reported with a page number.
    assert not host_allowed("http://www.anz.com/x.pdf", ("anz.com",))
    assert not host_allowed("file:///etc/passwd", ("anz.com",))
    # Credentials before the host must not spoof it.
    assert not host_allowed("https://anz.com@evil.example/x.pdf", ("anz.com",))


def test_the_manifest_validates_every_entry_before_anything_is_fetched():
    from bankresearch.acquisition.manifest import load_manifest

    entries = load_manifest(REPO / "data" / "manifest.yaml")
    assert entries
    registry = get_registry()
    for entry in entries:
        assert entry.bank_id in registry.banks
        assert entry.doc_type in {
            "RESULTS_DOCUMENT",
            "RESTATEMENT_NOTICE",
            "RESULTS_RELEASE",
            "INVESTOR_PRESENTATION",
            "PILLAR3",
            "ANNUAL_REPORT",
        }
        assert entry.url.startswith("https://")
        assert entry.verified_by in {"human", "auto"}


def test_every_manifest_url_is_on_its_issuers_allowlist():
    """A manifest entry pointing outside the issuer's own domains would be downloaded from
    somewhere the fetcher doesn't allow.
    """
    from bankresearch.acquisition.fetch import host_allowed
    from bankresearch.acquisition.manifest import load_manifest

    registry = get_registry()
    for entry in load_manifest(REPO / "data" / "manifest.yaml"):
        allowed = registry.banks[entry.bank_id].ir_domains
        assert host_allowed(entry.url, allowed), f"{entry.doc_id}: {entry.url}"


# --- the tool surface ------------------------------------------------------


def test_the_tool_surface():
    from bankresearch.tools.registry import LLM_CALLABLE, tool_definitions

    assert set(LLM_CALLABLE) == {
        "get_outline",
        "search_docs",
        "read_pages",
        "get_table",
        "read_chart",
        "submit_reading_list",
    }
    for definition in tool_definitions():
        assert definition["strict"] is True
        assert definition["input_schema"]["additionalProperties"] is False


def test_no_tool_can_compute_or_fetch():
    from bankresearch.tools.registry import LLM_CALLABLE, ORCHESTRATOR_ONLY

    for forbidden in ("decompose", "score_confidence", "verify", "fetch_url"):
        assert forbidden in ORCHESTRATOR_ONLY
        assert forbidden not in LLM_CALLABLE


def test_documents_are_returned_as_data():
    from bankresearch.tools.doc_tools import DOCUMENT_WRAPPER

    assert DOCUMENT_WRAPPER.startswith("<document_content>")


# --- the LLM layer ---------------------------------------------------------


def test_no_sampling_parameter_appears_anywhere_in_the_llm_layer():
    """Non-default temperature, top_p or top_k are rejected by the model, and
    repeatability comes from the cache instead."""
    for path in (SRC / "llm").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for forbidden in ("temperature=", "top_p=", "top_k="):
            assert forbidden not in source, f"{path.name} sends {forbidden}"


def test_the_cache_key_is_exact_not_fuzzy():
    source = (SRC / "llm" / "cache.py").read_text(encoding="utf-8")
    assert "request_hash" in source or "hashed()" in source
    assert "raises on a miss" in source or "CacheMiss" in source


def test_the_budget_guard_degrades_rather_than_raising():
    from bankresearch.llm.budget import Budget

    budget = Budget(max_calls=0)
    assert budget.may_call("locate") is False
    assert budget.breached is True


# --- the harness -----------------------------------------------------------


def test_the_harness_may_import_the_agent_but_not_the_reverse():
    assert "bankresearch.agent.orchestrator" in _imports_of("evals")
    assert "bankresearch.evals" not in _imports_of("agent")


def test_every_scorer_layer_exists():
    from bankresearch.evals.scorers import CaseScore

    fields = set(CaseScore.__annotations__)
    for layer in (
        "status_correct",  # status
        "reconciles",  # attribution
        "cited_numbers_found",  # citations
        "abstain_reason_correct",  # abstention
        "latency_ms",  # efficiency
        "headline_exact",  # extraction, gold-dependent
        "driver_precision",  # attribution, gold-dependent
    ):
        assert layer in fields, layer


def test_the_test_split_cannot_be_run_casually():
    import inspect

    from bankresearch.evals.runner import run_split

    assert "--final" in inspect.getsource(run_split)


def test_every_config_key_is_load_bearing():
    """A key declared in config and read by nothing is worse than no key at all.

    Changing it where you'd expect to changes nothing, and the value actually used is in a
    module you weren't looking at. Each key below is either read at runtime or mirrored in
    code with a test, and this checks that the mirror agrees, so the config and the code
    can't quietly drift apart.
    """
    import pathlib
    import re

    import yaml

    from bankresearch.domain.registry import get_registry

    src = pathlib.Path("src/bankresearch")
    blob = "\n".join(p.read_text(encoding="utf-8") for p in src.rglob("*.py"))
    tests_blob = "\n".join(
        p.read_text(encoding="utf-8") for p in pathlib.Path("tests").rglob("*.py")
    )

    unread: list[str] = []
    for config in sorted(pathlib.Path("config").glob("*.yaml")):
        data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            continue
        for key in data:
            if key == "schema_version":
                continue
            pattern = rf"\b{re.escape(key)}\b"
            if not re.search(pattern, blob) and not re.search(pattern, tests_blob):
                unread.append(f"{config.name}:{key}")
    assert not unread, f"declared in config and read nowhere: {unread}"

    # The mirrors, asserted rather than trusted.
    precedence = get_registry().precedence
    assert list(CLASSIFICATION_ORDER) == precedence["classification_order"]
    assert _DEFAULT_DOC_PRECEDENCE == precedence["doc_type_precedence"]

    effects = precedence["confidence_effects"]
    conflicts_src = (src / "reconcile" / "conflicts.py").read_text(encoding="utf-8")
    assert effects["UNEXPLAINED"]["cap"] in conflicts_src, (
        "the UNEXPLAINED cap in config does not appear in the code that applies it"
    )
    assert effects["UNEXPLAINED"]["cap_if_sign_flipped"] in conflicts_src
    assert effects["RESTATEMENT"]["penalty_if_unfootnoted"] in conflicts_src

    # The detailed precedence spec must rank document types the same way as the flat table
    # the code uses. Where it adds a scope nuance the code doesn't implement, that's
    # recorded in ADR-026 rather than left as an unstated disagreement.
    detailed = precedence["unexplained_precedence"]
    for tier, entry in enumerate(detailed):
        ranks = {_DEFAULT_DOC_PRECEDENCE[name] for name in entry["doc_types"]}
        assert len(ranks) <= 2, (
            f"tier {tier} of unexplained_precedence spans {ranks} in the flat table"
        )
    # `prefer: more_precise` is the tiebreak between equally ranked sources.
    assert any(entry.get("prefer") == "more_precise" for entry in detailed)
    assert "_own_tolerance(c)" in conflicts_src, "the more-precise tiebreak is not applied"

    # Third-party data is not evidence, and the allowlist is where that is enforced:
    # a fetch is confined to the issuer's own domains plus the ASX.
    assert precedence["third_party_data_is_evidence"] is False
    from bankresearch.acquisition.fetch import host_allowed

    assert not host_allowed("https://some-data-aggregator.example/anz.pdf", ("anz.com",))


def test_the_tool_result_cap_comes_from_config():
    """It was hardcoded here and declared there with nothing linking them, so raising it
    where you'd look for it changed nothing.
    """
    from bankresearch.domain.registry import get_registry
    from bankresearch.tools.doc_tools import _RESULT_TOKEN_CAP

    assert _RESULT_TOKEN_CAP == int(get_registry().budgets["tool_result_token_cap"])


def test_config_check_covers_model_prices_as_well_as_banks():
    """A price from memory is the same defect as a config field from memory.

    Pricing entries live outside the bank list, so a check that only walked the banks
    wouldn't see them. Neither may come from memory, so one
    command has to report both.
    """
    registry = get_registry()
    assert registry.unverified == {}, (
        f"unverified config: {registry.unverified}. Read each off its source and "
        "record the URL or page beside it."
    )

    pricing = registry.models["pricing"]
    model = next(iter(pricing))
    original = pricing[model]["verification"]
    try:
        pricing[model]["verification"] = "unverified"
        assert registry.unverified == {"pricing": (model,)}
    finally:
        pricing[model]["verification"] = original


def test_every_verified_config_field_records_where_it_was_read():
    """`verified` without a source is an assertion, not a verification."""
    raw = yaml.safe_load(Path("config/banks.yaml").read_text(encoding="utf-8"))
    missing: list[str] = []
    for entry in raw["banks"]:
        verification = entry.get("verification", {})
        for field, status in verification.items():
            if status != "verified" or field.endswith("_source"):
                continue
            if not verification.get(f"{field}_source"):
                missing.append(f"{entry['bank_id']}.{field}")
    assert not missing, f"verified with no source: {missing}"
