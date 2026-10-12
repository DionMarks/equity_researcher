"""Downloads a filing from the issuer's own site."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

__all__ = [
    "ASX_DOMAIN",
    "MAX_BYTES",
    "DomainNotAllowed",
    "FetchResult",
    "Fetcher",
    "NotAPdf",
    "TooLarge",
    "TooManyRedirects",
    "host_allowed",
]

ASX_DOMAIN = "announcements.asx.com.au"
MAX_BYTES = 50 * 1024 * 1024
_MIN_INTERVAL_SECONDS = 1.0
_MAX_REDIRECTS = 5

# A plain identifying agent. Several issuers reject requests without one.
_USER_AGENT = (
    "bankresearch/0.1 (equity research agent; public disclosure retrieval; "
    "contact: repository owner)"
)


class DomainNotAllowed(ValueError):
    """The URL's host is not on this issuer's allowlist."""


class TooLarge(ValueError):
    """The document exceeds the size cap."""


class NotAPdf(ValueError):
    """The response is not a PDF."""


class TooManyRedirects(ValueError):
    """The chain of redirects did not terminate."""


def host_allowed(url: str, allowed: tuple[str, ...]) -> bool:
    """Whether a URL is HTTPS and its host is, or is a subdomain of, an allowed domain.

    The scheme matters too. A document downloaded over plain HTTP can be altered in transit,
    and these documents are all the evidence there is: a figure changed on the wire would be
    cited, reconciled and reported with a page number, and nothing else in the system could
    catch it. Every URL in ``data/manifest.yaml`` is already HTTPS; this makes sure a
    proposed one is too.
    """
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https":
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        return False
    for domain in (*allowed, ASX_DOMAIN):
        domain = domain.lower().strip().rstrip(".")
        if not domain:
            continue
        if host == domain or host.endswith("." + domain):
            return True
    return False


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    final_url: str
    sha256: str
    path: Path
    bytes: int
    from_cache: bool


@dataclass
class Fetcher:
    """Cache-first, allowlisted, rate-limited retrieval into a content-addressed store."""

    raw_root: Path
    timeout: float = 60.0
    transport: httpx.BaseTransport | None = None
    """A hook for tests. The allowlist and the redirect guard are security properties, so CI
    tests them against a mock transport rather than taking the code's word for it.
    """
    _last_request: dict[str, float] = field(default_factory=dict)

    def fetch(self, url: str, allowed_domains: tuple[str, ...]) -> FetchResult:
        """Download a PDF, or return the cached copy if its bytes are already stored."""
        if not host_allowed(url, allowed_domains):
            raise DomainNotAllowed(
                f"{urlparse(url).hostname} is not on the allowlist {(*allowed_domains, ASX_DOMAIN)}"
            )

        content, final_url = self._get(url, allowed_domains)
        if content[:5] != b"%PDF-":
            raise NotAPdf(f"{final_url} did not return a PDF (starts {content[:8]!r})")

        digest = hashlib.sha256(content).hexdigest()
        target = self.raw_root / f"{digest}.pdf"
        from_cache = target.exists()
        if not from_cache:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

        return FetchResult(
            url=url,
            final_url=final_url,
            sha256=digest,
            path=target,
            bytes=len(content),
            from_cache=from_cache,
        )

    # -- internals --------------------------------------------------------

    def _wait_for(self, host: str) -> None:
        last = self._last_request.get(host)
        if last is not None:
            delay = _MIN_INTERVAL_SECONDS - (time.monotonic() - last)
            if delay > 0:
                time.sleep(delay)
        self._last_request[host] = time.monotonic()

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    )
    def _get(self, url: str, allowed_domains: tuple[str, ...]) -> tuple[bytes, str]:
        self._wait_for((urlparse(url).hostname or "").lower())
        headers = {"User-Agent": _USER_AGENT, "Accept": "application/pdf,*/*"}
        # Redirects are followed by hand. If the client followed them, only the end of the
        # chain would be checked, and by then every hop has already been requested, so a
        # redirect through an internal address would mean this process had made a request to
        # it.
        with httpx.Client(
            timeout=self.timeout, follow_redirects=False, transport=self.transport
        ) as client:
            return self._stream(client, url, headers, allowed_domains)

    def _stream(
        self,
        client: httpx.Client,
        url: str,
        headers: dict[str, str],
        allowed_domains: tuple[str, ...],
    ) -> tuple[bytes, str]:
        for _ in range(_MAX_REDIRECTS + 1):
            with client.stream("GET", url, headers=headers) as response:
                if response.is_redirect:
                    target = response.headers.get("location", "")
                    if not target:
                        response.raise_for_status()
                        raise NotAPdf(f"{url} redirected without a location")
                    # Resolved against the current URL, because Location may be relative,
                    # and checked *before* the next request is made.
                    nxt = str(response.url.join(target))
                    if not host_allowed(nxt, allowed_domains):
                        raise DomainNotAllowed(
                            f"{url} redirects to {nxt}, which is not on the allowlist"
                        )
                    url = nxt
                    self._wait_for((urlparse(url).hostname or "").lower())
                    continue
                return self._read(response, url, allowed_domains)
        raise TooManyRedirects(f"{url} exceeded {_MAX_REDIRECTS} redirects")

    def _read(
        self, response: httpx.Response, url: str, allowed_domains: tuple[str, ...]
    ) -> tuple[bytes, str]:
        """The body of a non-redirect response, within the size cap."""
        response.raise_for_status()
        final_url = str(response.url)
        # Checked once more on the URL actually served, so no path reaches the body
        # without its host having been on the allowlist.
        if not host_allowed(final_url, allowed_domains):
            raise DomainNotAllowed(f"{url} resolved to {final_url}, which is not on the allowlist")
        declared = response.headers.get("content-length")
        if declared is not None and int(declared) > MAX_BYTES:
            raise TooLarge(f"{final_url} declares {declared} bytes, over the cap")

        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_bytes():
            total += len(chunk)
            # Checked while streaming too, so a server that lies about
            # Content-Length cannot exhaust memory.
            if total > MAX_BYTES:
                raise TooLarge(f"{final_url} exceeded the {MAX_BYTES} byte cap while streaming")
            chunks.append(chunk)
        return b"".join(chunks), final_url
