"""Documents prepared in parallel are prepared exactly as they would be one at a time.

Parsing, extraction and validation run in separate processes, one document each, and the
store is written afterwards in manifest order.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from bankresearch.acquisition.corpus import _prepare_all
from bankresearch.domain.models import DocMeta

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_results_document.pdf"


def _jobs() -> list[tuple[str, Path, DocMeta]]:
    return [
        (
            doc_id,
            FIXTURE,
            DocMeta(
                doc_id=doc_id,
                bank_id="XMPL",
                period="1H26",
                doc_type="RESULTS_DOCUMENT",
                publish_date=date(2026, 5, 1),
                url="https://example.invalid/x.pdf",
            ),
        )
        for doc_id in ("xmpl-a", "xmpl-b")
    ]


def _figures(prepared) -> list[tuple[str, str, str]]:  # type: ignore[no-untyped-def]
    return sorted((f.period, f.component, str(f.value)) for f in prepared.validated)


def test_parallel_and_serial_preparation_agree(tmp_path):
    serial = _prepare_all(_jobs(), tmp_path / "serial", workers=1)
    parallel = _prepare_all(_jobs(), tmp_path / "parallel", workers=2)
    assert set(serial) == set(parallel) == {"xmpl-a", "xmpl-b"}
    for doc_id in serial:
        assert not isinstance(serial[doc_id], str), serial[doc_id]
        assert _figures(serial[doc_id]) == _figures(parallel[doc_id])
        assert _figures(serial[doc_id]), "the fixture yields figures"


def test_a_document_that_fails_to_parse_is_reported_not_fatal(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")
    jobs = [
        *_jobs()[:1],
        ("xmpl-broken", broken, _jobs()[0][2].model_copy(update={"doc_id": "xmpl-broken"})),
    ]
    outcome = _prepare_all(jobs, tmp_path / "out", workers=2)
    assert isinstance(outcome["xmpl-broken"], str)
    assert not isinstance(outcome["xmpl-a"], str)
