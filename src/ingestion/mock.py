"""Offline fixture loading for ``--mock`` runs.

Fixtures live in ``data/mock/<sport>/`` in the same shapes the live providers
return, and are pushed through the same parsers, so a mock run exercises the
real ingestion code without spending an Odds API credit. Regenerate them with
``python scripts/generate_mock_data.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import MOCK_DIR
from src.ingestion import db, injuries, odds_api, weather
from src.ingestion.odds_api import IngestSummary


class MockDataMissing(RuntimeError):
    """A fixture file the mock run needs is not present."""


def fixture_path(sport: str, name: str) -> Path:
    return MOCK_DIR / sport.lower() / f"{name}.json"


def load_fixture(sport: str, name: str, default: Any = None) -> Any:
    """Load one fixture; ``default`` is returned when it is absent."""
    path = fixture_path(sport, name)
    if not path.exists():
        if default is not None:
            return default
        raise MockDataMissing(
            f"missing fixture {path}; run python scripts/generate_mock_data.py"
        )
    with path.open() as handle:
        return json.load(handle)


def ingest_mock_slate(sport: str, *, db_path: str | None = None) -> IngestSummary:
    """Load a cached slate (games, lines, props, weather, injuries) into SQLite."""
    sport = sport.lower()
    db.init_db(db_path)
    summary = IngestSummary(sport=sport)

    events = load_fixture(sport, "odds")
    summary.games = db.upsert_games(
        [odds_api.parse_game(event, sport) for event in events], db_path=db_path
    )
    summary.lines = db.insert_lines(
        [row for event in events for row in odds_api.parse_lines(event)], db_path=db_path
    )

    for payload in load_fixture(sport, "props", default={}).values():
        summary.props += db.insert_props(odds_api.parse_props(payload), db_path=db_path)
        summary.events_polled += 1

    games = db.fetch_all("SELECT * FROM games WHERE sport = ?", (sport,), db_path=db_path)
    store_mock_weather(sport, games, db_path=db_path)
    mock_injury_records(sport, db_path=db_path)

    summary.quota_remaining = None
    return summary


def store_mock_weather(
    sport: str, games: list[dict[str, Any]], *, db_path: str | None = None
) -> list[weather.WeatherSnapshot]:
    """Build snapshots from cached forecasts, domes included."""
    forecasts = load_fixture(sport, "weather", default={})
    snapshots: list[weather.WeatherSnapshot] = []
    for game in games:
        venue = weather.stadium_for(game.get("home_team", ""))
        if venue is None:
            continue
        if venue["dome"]:
            snapshots.append(
                weather.dome_snapshot(
                    game["game_id"], venue["stadium"], game.get("commence_time")
                )
            )
            continue
        payload = forecasts.get(game["game_id"])
        if not payload:
            continue
        kickoff = weather._parse_iso(game["commence_time"])
        slot = weather.select_forecast_slot(payload, kickoff)
        if slot:
            snapshots.append(
                weather.snapshot_from_slot(game["game_id"], venue["stadium"], slot)
            )
    weather.store_snapshots(snapshots, db_path=db_path)
    return snapshots


def mock_injury_records(
    sport: str, *, db_path: str | None = None
) -> list[injuries.InjuryRecord]:
    """Parse and persist the cached injury feed."""
    payload = load_fixture(sport, "injuries", default={"injuries": []})
    records = injuries.parse_espn_injuries(payload, sport, source="mock")
    injuries.store_records(records, db_path=db_path)
    return records


def mock_stat_frames(sport: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(weekly, pbp)`` for NFL or ``(players, teams)`` for NBA."""
    sport = sport.lower()
    if sport == "nfl":
        return (
            pd.DataFrame(load_fixture(sport, "weekly")),
            pd.DataFrame(load_fixture(sport, "pbp")),
        )
    if sport == "nba":
        return (
            pd.DataFrame(load_fixture(sport, "players")),
            pd.DataFrame(load_fixture(sport, "teams")),
        )
    raise MockDataMissing(f"no mock frames for sport {sport!r}")


def available_sports() -> list[str]:
    """Sports with a fixture directory on disk."""
    if not MOCK_DIR.exists():
        return []
    return sorted(
        path.name for path in MOCK_DIR.iterdir()
        if path.is_dir() and (path / "odds.json").exists()
    )
