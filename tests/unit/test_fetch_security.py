"""The fetcher's security properties, exercised rather than asserted about."""

from __future__ import annotations

import httpx
import pytest

from bankresearch.acquisition.fetch import (
    MAX_BYTES,
    DomainNotAllowed,
    Fetcher,
    TooLarge,
    TooManyRedirects,
)

ALLOWED = ("anz.com",)
PDF = b"%PDF-1.7\n" + b"0" * 64


def _fetcher(tmp_path, handler) -> Fetcher:
    return Fetcher(raw_root=tmp_path, transport=httpx.MockTransport(handler))


def test_a_document_on_the_allowlist_is_fetched_and_content_addressed(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=PDF, headers={"content-type": "application/pdf"})

    result = _fetcher(tmp_path, handler).fetch("https://www.anz.com/a.pdf", ALLOWED)
    assert result.bytes == len(PDF)
    assert result.path.read_bytes() == PDF
    assert result.path.name == f"{result.sha256}.pdf"


def test_a_redirect_off_the_allowlist_is_refused_before_it_is_requested(tmp_path):
    """The hop that matters is the one not yet made. If the client followed redirects
    itself, only the end of the chain would be checked, and by then this process would
    already have sent a request to every host in it, including any internal address.
    """
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if request.url.host == "www.anz.com":
            return httpx.Response(
                302, headers={"location": "http://169.254.169.254/latest/meta-data"}
            )
        return httpx.Response(200, content=b"secrets")

    with pytest.raises(DomainNotAllowed):
        _fetcher(tmp_path, handler).fetch("https://www.anz.com/a.pdf", ALLOWED)
    assert requested == ["https://www.anz.com/a.pdf"], "the disallowed hop was requested"


def test_a_relative_redirect_is_resolved_against_the_current_url(tmp_path):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path == "/a.pdf":
            return httpx.Response(302, headers={"location": "/reports/b.pdf"})
        return httpx.Response(200, content=PDF)

    result = _fetcher(tmp_path, handler).fetch("https://www.anz.com/a.pdf", ALLOWED)
    assert seen == ["https://www.anz.com/a.pdf", "https://www.anz.com/reports/b.pdf"]
    assert result.final_url == "https://www.anz.com/reports/b.pdf"


def test_a_redirect_loop_terminates(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://www.anz.com/again.pdf"})

    with pytest.raises(TooManyRedirects):
        _fetcher(tmp_path, handler).fetch("https://www.anz.com/a.pdf", ALLOWED)


def test_a_declared_length_over_the_cap_is_refused(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=PDF, headers={"content-length": str(MAX_BYTES + 1)})

    with pytest.raises(TooLarge):
        _fetcher(tmp_path, handler).fetch("https://www.anz.com/a.pdf", ALLOWED)


def test_a_server_that_lies_about_its_length_cannot_exhaust_memory(tmp_path):
    """The cap is enforced while streaming, not only against the declared length."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=httpx.ByteStream(b"0" * (MAX_BYTES + 1024)),
            headers={"content-length": "10"},
        )

    with pytest.raises(TooLarge):
        _fetcher(tmp_path, handler).fetch("https://www.anz.com/a.pdf", ALLOWED)


def test_a_host_off_the_allowlist_is_never_requested(tmp_path):
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(200, content=PDF)

    with pytest.raises(DomainNotAllowed):
        _fetcher(tmp_path, handler).fetch("https://evil.example/a.pdf", ALLOWED)
    assert requested == []


def test_the_api_key_is_held_so_it_cannot_reach_a_log_or_a_traceback(monkeypatch):
    """Pydantic prints every field in a model's repr, so a settings object that ends up in a
    log line or an unhandled traceback would carry the key with it, and a traceback is
    exactly where nobody looks.
    """
    from bankresearch.settings import Settings

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-must-not-appear")
    settings = Settings()

    for rendered in (
        repr(settings),
        str(settings),
        str(settings.model_dump()),
        settings.model_dump_json(),
    ):
        assert "must-not-appear" not in rendered

    assert settings.has_api_key
    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-ant-must-not-appear"


def test_a_blank_key_is_not_a_key(monkeypatch):
    from bankresearch.settings import Settings

    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    assert not Settings().has_api_key
