"""Shared fixtures. Nothing here may touch the network."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion import db  # noqa: E402


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """Fresh initialised SQLite file per test."""
    path = tmp_path / "test.db"
    db.init_db(path)
    return str(path)


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep retry/backoff paths instant in unit tests."""
    import asyncio

    async def _instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _instant)


@pytest.fixture
def odds_event() -> dict:
    """One Odds API event carrying FanDuel and a competitor book."""
    return {
        "id": "evt-1",
        "sport_key": "americanfootball_nfl",
        "commence_time": "2026-09-20T17:00:00Z",
        "home_team": "Kansas City Chiefs",
        "away_team": "Buffalo Bills",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "Kansas City Chiefs", "price": -170},
                        {"name": "Buffalo Bills", "price": 145},
                    ]}
                ],
            },
            {
                "key": "fanduel",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "Kansas City Chiefs", "price": -160},
                        {"name": "Buffalo Bills", "price": 136},
                    ]},
                    {"key": "spreads", "outcomes": [
                        {"name": "Kansas City Chiefs", "price": -110, "point": -3.0},
                        {"name": "Buffalo Bills", "price": -110, "point": 3.0},
                    ]},
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "price": -105, "point": 47.5},
                        {"name": "Under", "price": -115, "point": 47.5},
                    ]},
                ],
            },
        ],
    }


@pytest.fixture
def props_payload() -> dict:
    """One event-odds payload with FanDuel player props."""
    return {
        "id": "evt-1",
        "commence_time": "2026-09-20T17:00:00Z",
        "home_team": "Kansas City Chiefs",
        "away_team": "Buffalo Bills",
        "bookmakers": [
            {
                "key": "fanduel",
                "markets": [
                    {"key": "player_pass_yds", "outcomes": [
                        {"name": "Over", "description": "Patrick Mahomes",
                         "price": -114, "point": 274.5},
                        {"name": "Under", "description": "Patrick Mahomes",
                         "price": -106, "point": 274.5},
                    ]},
                    {"key": "player_receptions", "outcomes": [
                        {"name": "Over", "description": "Travis Kelce",
                         "price": 105, "point": 5.5},
                        {"name": "Under", "description": "Travis Kelce",
                         "price": -125, "point": 5.5},
                    ]},
                    # no description -> not a player prop, must be dropped
                    {"key": "player_anytime_td", "outcomes": [
                        {"name": "Yes", "price": 120},
                    ]},
                ],
            },
            {
                "key": "betmgm",
                "markets": [
                    {"key": "player_pass_yds", "outcomes": [
                        {"name": "Over", "description": "Patrick Mahomes",
                         "price": -120, "point": 272.5},
                    ]},
                ],
            },
        ],
    }


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly on any HTTP call a test did not mock.

    Without this a test can pass for the wrong reason on a machine with no
    outbound access -- the call fails, the code degrades to its fallback, and
    the assertion holds -- then fail on CI, where the same call succeeds and
    returns real data. That is exactly how an empty-feed-url bug survived
    locally and broke the build.

    The guard sits at name resolution, below every HTTP library: respx answers
    mocked requests without ever resolving a host, so only a genuinely
    unmocked call reaches here.
    """
    import socket

    # A configured HTTPS_PROXY would make every lookup resolve to the proxy's
    # own address, hiding the real host from the guard. Tests never need a
    # proxy, so drop them and keep local behaviour identical to CI.
    for variable in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
    ):
        monkeypatch.delenv(variable, raising=False)

    real_getaddrinfo = socket.getaddrinfo
    allowed = {"localhost", "127.0.0.1", "::1", "testserver"}

    def guarded(host, *args, **kwargs):
        if host in allowed:
            return real_getaddrinfo(host, *args, **kwargs)
        raise RuntimeError(
            f"unmocked network call to {host!r} -- mock it with respx, "
            "or pass an explicit client/feed url"
        )

    monkeypatch.setattr(socket, "getaddrinfo", guarded)
