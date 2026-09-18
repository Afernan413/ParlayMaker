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
