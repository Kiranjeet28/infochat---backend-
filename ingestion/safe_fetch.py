"""Guarded outgoing HTTP GET for notice sources (https, public hosts, capped size and time)."""

from __future__ import annotations

import ipaddress
import os
import socket
import time
from urllib.parse import urljoin, urlparse

import requests

REDIRECT_STATUSES = (301, 302, 303, 307, 308)
# Notices come only from the notices API host (NOTICES_API_URL). Other hosts, such as the college site, must be
# allowed explicitly with SAFE_FETCH_ALLOWED_HOSTS (comma separated, exact or subdomain match).
DEFAULT_ALLOWED_HOSTS = ""
PDF_TYPES = ("application/pdf",)
HTML_TYPES = ("text/html",)
JSON_TYPES = ("application/json",)
MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_TEXT_BYTES = 5 * 1024 * 1024

_getaddrinfo = socket.getaddrinfo  # replaced in offline tests


class FetchError(requests.RequestException):
    """Base class; messages never contain query strings, headers or keys."""


class BlockedUrlError(FetchError):
    pass


class HostNotAllowedError(BlockedUrlError):
    """The host is not the notices API host and is not in SAFE_FETCH_ALLOWED_HOSTS (no connection was made)."""


class FetchTooLargeError(FetchError):
    pass


class UnsupportedContentTypeError(FetchError):
    pass


class TooManyRedirectsError(FetchError):
    pass


class FetchDeadlineError(FetchError):
    pass


def allowed_hosts(extra: tuple[str, ...] = ()) -> list[str]:
    """The host of NOTICES_API_URL, plus SAFE_FETCH_ALLOWED_HOSTS (default none), plus explicit extras."""
    hosts = [h.strip().lower().rstrip(".") for h in os.getenv("SAFE_FETCH_ALLOWED_HOSTS", DEFAULT_ALLOWED_HOSTS).split(",")]
    api_host = urlparse(os.getenv("NOTICES_API_URL", "")).hostname
    hosts += [api_host.lower()] if api_host else []
    hosts += [h.lower() for h in extra if h]
    return [h for h in hosts if h]


def _host_allowed(host: str, allowed: list[str]) -> bool:
    host = host.lower().rstrip(".")
    return any(host == d or host.endswith("." + d) for d in allowed)


def _public_address(value: str) -> bool:
    ip = ipaddress.ip_address(value.split("%")[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def validate_url(url: str, allowed: list[str]) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise BlockedUrlError("only https URLs are fetched")
    if not parsed.hostname or parsed.username or parsed.password:
        raise BlockedUrlError("URL has no host or contains credentials")
    if parsed.port not in (None, 443):
        raise BlockedUrlError("non-standard port refused")
    if not _host_allowed(parsed.hostname, allowed):
        raise HostNotAllowedError(f"host not allowed: {parsed.hostname}")
    try:
        addresses = _getaddrinfo(parsed.hostname, 443)
    except OSError:
        raise BlockedUrlError(f"host does not resolve: {parsed.hostname}") from None
    if not addresses or not all(_public_address(a[4][0]) for a in addresses):
        raise BlockedUrlError(f"non-public address refused: {parsed.hostname}")


def safe_get(url: str, *, allowed_content_types: tuple[str, ...], max_bytes: int,
             timeout: tuple[float, float] = (10, 60), max_redirects: int = 5,
             deadline_seconds: float | None = None, session=None, params=None,
             extra_allowed_hosts: tuple[str, ...] = ()) -> requests.Response:
    """GET with https/host/IP checks on every hop, a byte cap, a content-type allowlist and a deadline.

    Residual risk: DNS is resolved for the check and again by requests, so a rebinding
    host could still differ between the two lookups (the host allowlist limits this).
    """
    started = time.monotonic()
    allowed = allowed_hosts(extra_allowed_hosts)

    def remaining() -> float | None:
        if deadline_seconds is None:
            return None
        left = deadline_seconds - (time.monotonic() - started)
        if left <= 0:
            raise FetchDeadlineError("overall fetch deadline exceeded")
        return left

    getter = session.get if session is not None else requests.get
    current, query = url, params
    for _ in range(max_redirects + 1):
        validate_url(current, allowed)
        left = remaining()
        connect, read = timeout
        if left is not None:
            connect, read = min(connect, left), min(read, left)
        try:
            response = getter(current, params=query, timeout=(connect, read), allow_redirects=False, stream=True)
        except requests.RequestException as error:
            raise FetchError(f"request failed ({type(error).__name__})") from None
        try:
            if response.status_code in REDIRECT_STATUSES:
                location = response.headers.get("location")
                if not location:
                    raise BlockedUrlError("redirect without a location")
                current, query = urljoin(current, location), None
                continue
            try:
                response.raise_for_status()
            except requests.HTTPError:
                raise FetchError(f"http status {response.status_code}") from None
            content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type not in allowed_content_types:
                raise UnsupportedContentTypeError("content type not allowed")
            try:
                declared = int(response.headers.get("content-length") or 0)
            except ValueError:
                declared = 0
            if declared > max_bytes:
                raise FetchTooLargeError("response too large")
            chunks, total = [], 0
            for chunk in response.iter_content(65536):  # decoded bytes, so gzip bombs are counted decompressed
                remaining()
                total += len(chunk)
                if total > max_bytes:
                    raise FetchTooLargeError("response too large")
                chunks.append(chunk)
            response._content = b"".join(chunks)
            return response
        finally:
            response.close()
    raise TooManyRedirectsError("too many redirects")
