"""An eval leaves the corpus as it found it."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from bankresearch.domain.models import DocMeta
from bankresearch.evals.cases import EvalCase
from bankresearch.memory.db import connect
from bankresearch.memory.fact_store import FactStore
from bankresearch.settings import get_settings


def _case(case_id: str) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        split="dev",
        bank="BEN",
        period="1H26",
        metric="nim",
        as_of=date(2026, 5, 31),
        expected_status=("ABSTAIN",),
    )


def test_documents_the_cold_path_ingested_are_removed_afterwards(tmp_path, monkeypatch):
    from bankresearch.evals import runner

    settings = get_settings().model_copy(update={"data_dir": tmp_path})
    assert settings.db_path.is_relative_to(tmp_path)
    with connect(settings.db_path) as connection:
        FactStore(connection).put_document(
            DocMeta(
                doc_id="kept",
                bank_id="ANZ",
                period="1H26",
                doc_type="RESULTS_DOCUMENT",
                publish_date=date(2026, 5, 1),
                url="https://www.anz.com/x.pdf",
            )
        )

    class Ingesting:
        def __init__(self, store, **_):
            self.store = store

        def run(self, bank, period, metric, as_of=None):
            self.store.put_document(
                DocMeta(
                    doc_id=f"found-{period}",
                    bank_id=bank,
                    period=period,
                    doc_type="RESULTS_DOCUMENT",
                    publish_date=date(2026, 2, 16),
                    url="https://www.bendigoadelaide.com.au/x.pdf",
                )
            )
            return SimpleNamespace(result=None)

    monkeypatch.setattr(runner, "Agent", Ingesting)
    monkeypatch.setattr(runner, "score_case", lambda *a, **k: SimpleNamespace(case_id="x"))
    run = runner.run_split("dev", settings=settings, cases=[_case("a"), _case("b")], workers=2)
    assert not run.errors
    with connect(settings.db_path) as connection:
        assert FactStore(connection).document_ids() == {"kept"}


def test_a_case_that_will_ingest_runs_after_the_rest_and_alone(tmp_path, monkeypatch):
    """Judo's FY26 answer, run beside Judo's FY25 cold start, cited the FY25 report the
    cold start had just ingested. A case whose period the corpus doesn't hold runs after
    the others, and what it ingests is gone before anything else runs."""
    from bankresearch.evals import runner

    settings = get_settings().model_copy(update={"data_dir": tmp_path})
    with connect(settings.db_path) as connection:
        FactStore(connection).put_document(
            DocMeta(
                doc_id="anz-1h26",
                bank_id="ANZ",
                period="1H26",
                doc_type="RESULTS_DOCUMENT",
                publish_date=date(2026, 5, 1),
                url="https://www.anz.com/x.pdf",
            )
        )
    seen: list[tuple[str, frozenset[str]]] = []

    class Recording:
        def __init__(self, store, **_):
            self.store = store

        def run(self, bank, period, metric, as_of=None):
            seen.append((f"{bank} {period}", frozenset(self.store.document_ids())))
            if bank == "BEN":
                self.store.put_document(
                    DocMeta(
                        doc_id=f"ben-{period.lower()}",
                        bank_id=bank,
                        period=period,
                        doc_type="RESULTS_DOCUMENT",
                        publish_date=date(2026, 2, 16),
                        url="https://www.bendigoadelaide.com.au/x.pdf",
                    )
                )
            return SimpleNamespace(result=None)

    def anz(case_id: str) -> EvalCase:
        return EvalCase(
            case_id=case_id,
            split="dev",
            bank="ANZ",
            period="1H26",
            metric="nim",
            as_of=date(2026, 5, 31),
        )

    monkeypatch.setattr(runner, "Agent", Recording)
    monkeypatch.setattr(
        runner, "score_case", lambda case, *a, **k: SimpleNamespace(case_id=case.case_id)
    )
    cases = [_case("a_cold"), anz("b_warm"), anz("c_warm")]
    run = runner.run_split("dev", settings=settings, cases=cases, workers=3)
    assert not run.errors
    assert [score.case_id for score in run.scores] == ["a_cold", "b_warm", "c_warm"]
    assert seen[-1][0] == "BEN 1H26"
    assert all(documents == {"anz-1h26"} for _, documents in seen)
    with connect(settings.db_path) as connection:
        assert FactStore(connection).document_ids() == {"anz-1h26"}
