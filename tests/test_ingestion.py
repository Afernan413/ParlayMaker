"""Ingestion tests. Every outbound HTTP call is mocked with respx."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest
import respx

from src.ingestion import db, injuries, odds_api, weather
from src.ingestion.odds_api import OddsAPIClient, QuotaExhaustedError

BASE = "https://api.test-odds.local"


# ----------------------------------------------------------------- database
def test_init_db_is_idempotent_and_uses_wal(db_path):
    db.init_db(db_path)  # second call must not raise
    with db.connect(db_path) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert mode.lower() == "wal"
    assert {
        "games",
        "fanduel_lines",
        "fanduel_props",
        "weather_snapshots",
        "injury_reports",
        "api_quota_log",
    } <= tables


def test_upsert_games_updates_existing_row(db_path):
    game = {
        "game_id": "g1",
        "sport": "nfl",
        "commence_time": "2026-09-20T17:00:00Z",
        "home_team": "KC",
        "away_team": "BUF",
    }
    db.upsert_games([game], db_path=db_path)
    db.upsert_games([{**game, "commence_time": "2026-09-20T20:05:00Z", "week": 3}], db_path=db_path)
    rows = db.fetch_all("SELECT * FROM games", db_path=db_path)
    assert len(rows) == 1
    assert rows[0]["commence_time"] == "2026-09-20T20:05:00Z"
    assert rows[0]["week"] == 3


def test_latest_props_returns_newest_capture(db_path):
    db.upsert_games(
        [{"game_id": "g1", "sport": "nfl", "commence_time": "t",
          "home_team": "KC", "away_team": "BUF"}],
        db_path=db_path,
    )
    base = {
        "game_id": "g1",
        "market": "player_pass_yds",
        "player_name": "Patrick Mahomes",
        "selection": "Over",
        "line": 274.5,
    }
    db.insert_props([{**base, "american_odds": -110, "captured_at": "2026-09-20T10:00:00+00:00"}], db_path=db_path)
    db.insert_props([{**base, "american_odds": -125, "captured_at": "2026-09-20T15:00:00+00:00"}], db_path=db_path)
    latest = db.latest_props("g1", db_path)
    assert [row["american_odds"] for row in latest] == [-125]


# ------------------------------------------------------------- odds parsers
def test_parse_lines_keeps_only_fanduel(odds_event):
    rows = odds_api.parse_lines(odds_event)
    assert len(rows) == 6  # 2 h2h + 2 spreads + 2 totals, FanDuel only
    h2h = [r for r in rows if r["market"] == "h2h"]
    assert {r["american_odds"] for r in h2h} == {-160, 136}  # not DraftKings' -170/145
    total = next(r for r in rows if r["market"] == "totals" and r["selection"] == "Over")
    assert total["line"] == 47.5


def test_parse_props_uses_description_and_drops_unnamed(props_payload):
    rows = odds_api.parse_props(props_payload)
    assert {r["player_name"] for r in rows} == {"Patrick Mahomes", "Travis Kelce"}
    assert all(r["game_id"] == "evt-1" for r in rows)
    # the anytime-TD outcome without a player, and the BetMGM block, are gone
    assert not [r for r in rows if r["market"] == "player_anytime_td"]
    assert -120 not in {r["american_odds"] for r in rows}


def test_parse_game_normalises_event(odds_event):
    row = odds_api.parse_game(odds_event, "NFL")
    assert row == {
        "game_id": "evt-1",
        "sport": "nfl",
        "commence_time": "2026-09-20T17:00:00Z",
        "home_team": "Kansas City Chiefs",
        "away_team": "Buffalo Bills",
    }


# -------------------------------------------------------------- odds client
@respx.mock
async def test_fetch_game_odds_records_quota(db_path, odds_event):
    route = respx.get(f"{BASE}/v4/sports/americanfootball_nfl/odds").mock(
        return_value=httpx.Response(
            200,
            json=[odds_event],
            headers={"x-requests-remaining": "412", "x-requests-used": "88",
                     "x-requests-last": "1"},
        )
    )
    async with OddsAPIClient("key", base_url=BASE, db_path=db_path) as client:
        events = await client.fetch_game_odds("nfl")

    assert route.called
    assert [e["id"] for e in events] == ["evt-1"]
    assert respx.calls[0].request.url.params["bookmakers"] == "fanduel"
    logged = db.latest_quota("odds_api", db_path=db_path)
    assert logged["requests_remaining"] == 412
    assert logged["requests_used"] == 88


@respx.mock
async def test_retries_on_rate_limit_then_succeeds(db_path, odds_event):
    route = respx.get(f"{BASE}/v4/sports/americanfootball_nfl/odds").mock(
        side_effect=[
            httpx.Response(429, text="slow down"),
            httpx.Response(200, json=[odds_event],
                           headers={"x-requests-remaining": "300"}),
        ]
    )
    async with OddsAPIClient("key", base_url=BASE, db_path=db_path) as client:
        events = await client.fetch_game_odds("nfl")
    assert route.call_count == 2
    assert events[0]["id"] == "evt-1"


@respx.mock
async def test_fatal_status_is_not_retried(db_path):
    route = respx.get(f"{BASE}/v4/sports/americanfootball_nfl/odds").mock(
        return_value=httpx.Response(401, text="bad key")
    )
    async with OddsAPIClient("key", base_url=BASE, db_path=db_path) as client:
        with pytest.raises(odds_api.OddsAPIError):
            await client.fetch_game_odds("nfl")
    assert route.call_count == 1


@respx.mock
async def test_quota_floor_blocks_further_requests(db_path):
    respx.get(f"{BASE}/v4/sports/americanfootball_nfl/odds").mock(
        return_value=httpx.Response(200, json=[], headers={"x-requests-remaining": "49"})
    )
    async with OddsAPIClient("key", base_url=BASE, db_path=db_path,
                             min_quota_remaining=50) as client:
        await client.fetch_game_odds("nfl")
        assert client.quota.remaining == 49
        with pytest.raises(QuotaExhaustedError):
            await client.fetch_game_odds("nfl")


@respx.mock
async def test_ingest_slate_halts_props_when_quota_drops(db_path, odds_event, props_payload):
    second_event = {**odds_event, "id": "evt-2"}
    respx.get(f"{BASE}/v4/sports/americanfootball_nfl/odds").mock(
        return_value=httpx.Response(
            200, json=[odds_event, second_event],
            headers={"x-requests-remaining": "60"},
        )
    )
    respx.get(f"{BASE}/v4/sports/americanfootball_nfl/events/evt-1/odds").mock(
        return_value=httpx.Response(
            200, json=props_payload, headers={"x-requests-remaining": "10"}
        )
    )
    evt2 = respx.get(f"{BASE}/v4/sports/americanfootball_nfl/events/evt-2/odds")

    async with OddsAPIClient("key", base_url=BASE, db_path=db_path,
                             min_quota_remaining=50) as client:
        summary = await client.ingest_slate("nfl")

    assert summary.games == 2
    assert summary.lines == 12
    assert summary.events_polled == 1
    assert summary.props == 4
    assert summary.skipped_events == ["evt-2"]
    assert not evt2.called  # never spent a request we could not afford


async def test_missing_api_key_raises_before_any_request(db_path):
    async with OddsAPIClient("", base_url=BASE, db_path=db_path) as client:
        with pytest.raises(odds_api.OddsAPIError, match="ODDS_API_KEY"):
            await client.fetch_game_odds("nfl")


# ------------------------------------------------------------------ weather
def test_dome_game_skips_network(db_path):
    snapshot = weather.dome_snapshot("g1", "Ford Field", "2026-09-20T17:00:00Z")
    assert snapshot.is_dome == 1
    assert snapshot.high_wind == 0 and snapshot.freezing == 0


@respx.mock
async def test_outdoor_forecast_flags_wind_and_cold(db_path):
    payload = {
        "list": [
            {"dt_txt": "2026-12-20 12:00:00", "main": {"temp": 40.0},
             "wind": {"speed": 8.0, "deg": 200}, "pop": 0.1,
             "weather": [{"description": "clear sky"}]},
            {"dt_txt": "2026-12-20 18:00:00", "main": {"temp": 21.0},
             "wind": {"speed": 23.0, "deg": 310}, "pop": 0.4,
             "weather": [{"description": "light snow"}]},
        ]
    }
    respx.get("https://weather.test/data/2.5/forecast").mock(
        return_value=httpx.Response(200, json=payload)
    )
    game = {
        "game_id": "g9",
        "sport": "nfl",
        "home_team": "Buffalo Bills",
        "away_team": "Miami Dolphins",
        "commence_time": "2026-12-20T18:00:00Z",
    }
    db.upsert_games([game], db_path=db_path)
    async with weather.WeatherClient(
        "wkey", base_url="https://weather.test", db_path=db_path
    ) as client:
        snapshots = await client.ingest_games([game])

    assert len(snapshots) == 1
    snap = snapshots[0]
    assert snap.forecast_for == "2026-12-20 18:00:00"  # nearest slot to kickoff
    assert snap.high_wind == 1 and snap.freezing == 1
    assert snap.is_dome == 0
    stored = db.fetch_all("SELECT * FROM weather_snapshots", db_path=db_path)
    assert stored[0]["wind_speed_mph"] == 23.0


@respx.mock
async def test_dome_game_makes_no_http_call(db_path):
    route = respx.get("https://weather.test/data/2.5/forecast")
    game = {"game_id": "g2", "sport": "nfl", "home_team": "Detroit Lions",
            "away_team": "Chicago Bears", "commence_time": "2026-12-20T18:00:00Z"}
    db.upsert_games([game], db_path=db_path)
    async with weather.WeatherClient(
        "wkey", base_url="https://weather.test", db_path=db_path
    ) as client:
        snapshots = await client.ingest_games([game])
    assert not route.called
    assert snapshots[0].is_dome == 1


async def test_unknown_stadium_is_skipped(db_path):
    async with weather.WeatherClient("wkey", db_path=db_path) as client:
        assert await client.snapshot_for_game(
            {"game_id": "x", "home_team": "Toronto Huskies",
             "commence_time": "2026-12-20T18:00:00Z"}
        ) is None


def test_forecast_window_is_three_hours_before_kickoff():
    kickoff = datetime(2026, 12, 20, 18, 0, tzinfo=timezone.utc)
    assert weather.forecast_window(kickoff).hour == 15


# ----------------------------------------------------------------- injuries
@respx.mock
async def test_espn_injury_feed_is_normalised(db_path):
    payload = {
        "injuries": [
            {
                "displayName": "Buffalo Bills",
                "injuries": [
                    {"status": "Out", "date": "2026-09-19",
                     "athlete": {"displayName": "Jim Player",
                                 "position": {"abbreviation": "WR"}},
                     "shortComment": "hamstring"},
                    {"status": "Game-Time Decision",
                     "athlete": {"displayName": "Ann Back",
                                 "position": {"abbreviation": "RB"}}},
                ],
            }
        ]
    }
    respx.get("https://feeds.test/nfl/injuries").mock(
        return_value=httpx.Response(200, json=payload)
    )
    async with injuries.InjuryClient(
        db_path=db_path, feed_urls={"nfl": "https://feeds.test/nfl/injuries"}
    ) as client:
        records = await client.ingest("nfl")

    index = injuries.status_index(records)
    assert index == {"Jim Player": "OUT", "Ann Back": "QUESTIONABLE"}
    assert injuries.status_multiplier(index["Jim Player"]) == 0.0
    assert db.fetch_all("SELECT * FROM injury_reports", db_path=db_path)[0]["position"] == "WR"


async def test_injury_fetch_failure_degrades_to_empty(db_path):
    async with injuries.InjuryClient(db_path=db_path, feed_urls={}) as client:
        assert await client.ingest("nfl") == []


def test_parse_inactive_list_defaults_to_out():
    records = injuries.parse_inactive_list(
        [{"player": "Sam Sub", "team": "KC"}], "nfl"
    )
    assert records[0].status == "OUT"
    assert records[0].multiplier == 0.0


def test_inactives_window_matches_league_lead_times():
    kickoff = datetime(2026, 9, 20, 17, 0, tzinfo=timezone.utc)
    assert injuries.inactives_window(kickoff, "nfl").hour == 15
    assert injuries.inactives_window(kickoff, "nfl").minute == 30
    assert injuries.inactives_window(kickoff, "nba").minute == 30
    assert injuries.inactives_window(kickoff, "nba").hour == 16
