"""Single-user access control for the deployed app.

The threat model is narrow and worth stating: this is one person's research
tool. What must not happen is an open URL that (a) hands out odds data the
provider's terms keep private, and (b) lets a stranger trigger ``refresh`` and
burn a month of API credits in a few seconds.

So: one shared token, supplied either as a bearer header (for scripts and cron)
or exchanged once for a signed cookie (for the browser). The cookie carries only
an expiry and an HMAC of it -- the token itself is never stored client-side.

This is deliberately not a user system. If more than one person needs access,
put a real identity provider in front of it rather than growing this file.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass

COOKIE_NAME = "parlay_session"
#: Paths reachable without a token: the login page and what it needs to render.
PUBLIC_PATHS = frozenset(
    {"/login", "/api/login", "/styles.css", "/favicon.ico", "/api/health"}
)
LOGIN_WINDOW_SECONDS = 300
LOGIN_MAX_ATTEMPTS = 8


def constant_time_equal(candidate: str, expected: str) -> bool:
    """Compare two secrets without leaking their similarity through timing."""
    return hmac.compare_digest(candidate.encode(), expected.encode())


def issue_session(token: str, *, ttl_seconds: int, now: float | None = None) -> str:
    """Mint a cookie value: ``<expiry>.<hmac(expiry)>``."""
    now = time.time() if now is None else now
    expiry = int(now + ttl_seconds)
    return f"{expiry}.{_sign(token, expiry)}"


def verify_session(cookie: str | None, token: str, *, now: float | None = None) -> bool:
    """Is this cookie a live session signed by ``token``?"""
    if not cookie or not token:
        return False
    expiry_part, _, signature = cookie.partition(".")
    if not signature:
        return False
    try:
        expiry = int(expiry_part)
    except ValueError:
        return False
    if expiry <= (time.time() if now is None else now):
        return False
    return hmac.compare_digest(signature, _sign(token, expiry))


def _sign(token: str, expiry: int) -> str:
    return hmac.new(token.encode(), str(expiry).encode(), hashlib.sha256).hexdigest()


def bearer_token(header: str | None) -> str | None:
    """Pull the token out of an ``Authorization: Bearer <token>`` header."""
    if not header:
        return None
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer" or not value:
        return None
    return value.strip()


def is_public(path: str) -> bool:
    return path in PUBLIC_PATHS


def generate_token(length: int = 32) -> str:
    """A fresh token to paste into the environment."""
    return secrets.token_urlsafe(length)


@dataclass
class LoginThrottle:
    """Crude per-client attempt limiter, enough to blunt online guessing."""

    window_seconds: int = LOGIN_WINDOW_SECONDS
    max_attempts: int = LOGIN_MAX_ATTEMPTS

    def __post_init__(self) -> None:
        self._attempts: dict[str, deque[float]] = defaultdict(deque)

    def _prune(self, client: str, now: float) -> deque[float]:
        attempts = self._attempts[client]
        while attempts and now - attempts[0] > self.window_seconds:
            attempts.popleft()
        return attempts

    def blocked(self, client: str, *, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return len(self._prune(client, now)) >= self.max_attempts

    def record_failure(self, client: str, *, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self._prune(client, now).append(now)

    def reset(self, client: str) -> None:
        self._attempts.pop(client, None)


def loopback_only(host: str) -> bool:
    """Is this bind address reachable only from the machine itself?"""
    return host in {"127.0.0.1", "::1", "localhost"}
