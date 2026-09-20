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
GEO = "https://weather.test/locations/v1/cities/geoposition/search"
HOURLY = "https://weather.test/forecasts/v1/hourly/12hour/349727"
DAILY = "https://weather.test/forecasts/v1/daily/5day/349727"

BILLS = {
    "game_id": "g9",
    "sport": "nfl",
    "home_team": "Buffalo Bills",
    "away_team": "Miami Dolphins",
    "commence_time": "2026-12-20T18:00:00Z",
}


def hour(stamp: str, temp: float, wind: float, deg: int, pop: int, phrase: str) -> dict:
    """One AccuWeather hourly reading, in its own shape."""
    return {
        "DateTime": stamp,
        "IconPhrase": phrase,
        "Temperature": {"Value": temp, "Unit": "F"},
        "Wind": {
            "Speed": {"Value": wind, "Unit": "mi/h"},
            "Direction": {"Degrees": deg, "English": "NW"},
        },
        "PrecipitationProbability": pop,
    }


def mock_geo(key: str = "349727", name: str = "Orchard Park"):
    return respx.get(GEO).mock(
        return_value=httpx.Response(200, json={"Key": key, "LocalizedName": name})
    )


def client(db_path):
    return weather.WeatherClient("wkey", base_url="https://weather.test", db_path=db_path)


def test_dome_game_skips_network(db_path):
    snapshot = weather.dome_snapshot("g1", "Ford Field", "2026-09-20T17:00:00Z")
    assert snapshot.is_dome == 1
    assert snapshot.high_wind == 0 and snapshot.freezing == 0


@respx.mock
async def test_outdoor_forecast_flags_wind_and_cold(db_path, monkeypatch):
    """Kickoff inside the hourly horizon gets the hour nearest to it."""
    _freeze_now(monkeypatch, "2026-12-20T12:00:00Z")
    mock_geo()
    respx.get(HOURLY).mock(return_value=httpx.Response(200, json=[
        hour("2026-12-20T12:00:00+00:00", 40.0, 8.0, 200, 10, "Clear"),
        hour("2026-12-20T18:00:00+00:00", 21.0, 23.0, 310, 40, "Light snow"),
    ]))
    db.upsert_games([BILLS], db_path=db_path)
    async with client(db_path) as weather_client:
        snapshots = await weather_client.ingest_games([BILLS])

    assert len(snapshots) == 1
    snap = snapshots[0]
    assert snap.forecast_for == "2026-12-20T18:00:00+00:00"   # nearest to kickoff
    assert snap.high_wind == 1 and snap.freezing == 1
    assert snap.is_dome == 0
    assert snap.conditions == "Light snow"
    # A 0-100 chance is stored as the 0-1 fraction the rest of the engine uses.
    assert snap.precip_chance == pytest.approx(0.4)
    stored = db.fetch_all("SELECT * FROM weather_snapshots", db_path=db_path)
    assert stored[0]["wind_speed_mph"] == 23.0
    assert stored[0]["wind_deg"] == 310


def _freeze_now(monkeypatch, iso: str):
    """Pin "now" so the hourly/daily choice is deterministic."""
    fixed = datetime.fromisoformat(iso.replace("Z", "+00:00"))

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed if tz else fixed.replace(tzinfo=None)

    monkeypatch.setattr(weather, "datetime", Clock)


@respx.mock
async def test_a_kickoff_beyond_the_hourly_horizon_uses_the_daily_forecast(
    db_path, monkeypatch
):
    """AccuWeather's hourly forecast reaches twelve hours. A Saturday build for
    a Sunday kickoff is outside it, and no forecast at all would be worse."""
    _freeze_now(monkeypatch, "2026-12-18T12:00:00Z")   # two days out
    mock_geo()
    hourly = respx.get(HOURLY)
    respx.get(DAILY).mock(return_value=httpx.Response(200, json={"DailyForecasts": [
        {"Date": "2026-12-19T07:00:00-05:00",
         "Temperature": {"Maximum": {"Value": 50.0, "Unit": "F"}},
         "Day": {"IconPhrase": "Rain", "PrecipitationProbability": 80,
                 "Wind": {"Speed": {"Value": 12.0}, "Direction": {"Degrees": 180}}}},
        {"Date": "2026-12-20T07:00:00-05:00",
         "Temperature": {"Maximum": {"Value": 24.0, "Unit": "F"}},
         "Day": {"IconPhrase": "Snow", "PrecipitationProbability": 70,
                 "Wind": {"Speed": {"Value": 25.0}, "Direction": {"Degrees": 300}}}},
    ]}))
    db.upsert_games([BILLS], db_path=db_path)
    async with client(db_path) as weather_client:
        snapshots = await weather_client.ingest_games([BILLS])

    assert not hourly.called, "should not spend a call on a window that cannot reach"
    snap = snapshots[0]
    assert snap.temperature_f == 24.0        # the day of the game, not the day before
    assert snap.wind_speed_mph == 25.0
    assert snap.high_wind == 1 and snap.freezing == 1
    # Says which kind of forecast it is, so nothing mistakes it for an hourly read.
    assert "daily outlook" in snap.conditions


@respx.mock
async def test_a_location_key_is_looked_up_once_and_cached(db_path, monkeypatch):
    """The free tier is 50 calls a day; a venue's key must not cost one per run."""
    _freeze_now(monkeypatch, "2026-12-20T12:00:00Z")
    geo = mock_geo()
    respx.get(HOURLY).mock(return_value=httpx.Response(200, json=[
        hour("2026-12-20T18:00:00+00:00", 40.0, 8.0, 200, 10, "Clear"),
    ]))
    db.upsert_games([BILLS], db_path=db_path)
    for _ in range(3):
        async with client(db_path) as weather_client:
            await weather_client.ingest_games([BILLS])
    assert geo.call_count == 1
    assert db.cached_location_key("42.7738,-78.787", "accuweather", db_path=db_path) == "349727"


@respx.mock
async def test_a_venue_accuweather_cannot_place_is_skipped(db_path):
    respx.get(GEO).mock(return_value=httpx.Response(200, json={}))
    db.upsert_games([BILLS], db_path=db_path)
    async with client(db_path) as weather_client:
        assert await weather_client.ingest_games([BILLS]) == []


@respx.mock
async def test_the_daily_allowance_running_out_stops_the_slate(db_path, monkeypatch):
    """A 503 that says the allowance is spent is a clean stop, not an outage:
    every further call would answer the same, so the rest is left unforecast."""
    _freeze_now(monkeypatch, "2026-12-20T12:00:00Z")
    geo = respx.get(GEO).mock(
        return_value=httpx.Response(
            503, text="The allowed number of requests has been exceeded."
        )
    )
    games = [dict(BILLS, game_id=f"g{i}") for i in range(4)]
    db.upsert_games(games, db_path=db_path)
    async with client(db_path) as weather_client:
        assert await weather_client.ingest_games(games) == []
    assert geo.call_count == 1, "should not retry once the allowance is known to be gone"


@respx.mock
async def test_a_real_provider_outage_is_not_mistaken_for_the_quota(db_path, monkeypatch):
    _freeze_now(monkeypatch, "2026-12-20T12:00:00Z")
    geo = respx.get(GEO).mock(return_value=httpx.Response(503, text="Service Unavailable"))
    games = [dict(BILLS, game_id=f"g{i}") for i in range(3)]
    db.upsert_games(games, db_path=db_path)
    async with client(db_path) as weather_client:
        assert await weather_client.ingest_games(games) == []
    # Each game is still tried: an outage can clear, a spent allowance cannot.
    assert geo.call_count == 3


@respx.mock
async def test_a_runs_budget_stops_a_slate_rather_than_stalling_it(db_path, monkeypatch):
    """The location-key cache lives in SQLite and a hosted build starts with an
    empty database, so an unbounded slate pays two calls a venue -- 58 for a
    29-game Sunday, which is past the 50-a-day allowance on its own. The budget
    is also the only hard stop on a slow provider stalling a build."""
    _freeze_now(monkeypatch, "2026-12-20T12:00:00Z")
    geo = mock_geo()
    hourly = respx.get(url__startswith="https://weather.test/forecasts/").mock(
        return_value=httpx.Response(200, json=[
            hour("2026-12-20T18:00:00+00:00", 40.0, 8.0, 200, 10, "Clear"),
        ])
    )
    games = [dict(BILLS, game_id=f"g{i}") for i in range(20)]
    db.upsert_games(games, db_path=db_path)
    async with weather.WeatherClient(
        "wkey", base_url="https://weather.test", db_path=db_path, call_budget=5
    ) as weather_client:
        snapshots = await weather_client.ingest_games(games)
        assert weather_client.calls_made == 5

    # The venue is the same, so its key is cached after the first lookup and the
    # rest of the budget goes on forecasts.
    assert geo.call_count == 1
    assert hourly.call_count == 4
    assert len(snapshots) == 4, "whatever was fetched before the budget ran out is kept"


@respx.mock
async def test_dome_game_makes_no_http_call(db_path):
    geo = respx.get(GEO)
    hourly = respx.get(HOURLY)
    game = {"game_id": "g2", "sport": "nfl", "home_team": "Detroit Lions",
            "away_team": "Chicago Bears", "commence_time": "2026-12-20T18:00:00Z"}
    db.upsert_games([game], db_path=db_path)
    async with client(db_path) as weather_client:
        snapshots = await weather_client.ingest_games([game])
    assert not geo.called and not hourly.called
    assert snapshots[0].is_dome == 1


async def test_unknown_stadium_is_skipped(db_path):
    async with weather.WeatherClient("wkey", db_path=db_path) as weather_client:
        assert await weather_client.snapshot_for_game(
            {"game_id": "x", "home_team": "Toronto Huskies",
             "commence_time": "2026-12-20T18:00:00Z"}
        ) is None


async def test_no_key_is_a_clear_error(db_path):
    async with weather.WeatherClient("", db_path=db_path) as weather_client:
        with pytest.raises(RuntimeError, match="OPENWEATHER_API_KEY"):
            await weather_client.location_key(42.0, -78.0)


def test_the_hourly_reading_nearest_kickoff_wins():
    kickoff = datetime(2026, 12, 20, 18, tzinfo=timezone.utc)
    slots = [
        hour("2026-12-20T15:00:00+00:00", 40.0, 5.0, 10, 0, "Clear"),
        hour("2026-12-20T17:00:00+00:00", 38.0, 6.0, 20, 0, "Cloudy"),
        hour("2026-12-20T21:00:00+00:00", 30.0, 7.0, 30, 0, "Snow"),
    ]
    assert weather.select_forecast_slot(slots, kickoff)["IconPhrase"] == "Cloudy"


def test_an_empty_forecast_is_none_rather_than_an_exception():
    kickoff = datetime(2026, 12, 20, 18, tzinfo=timezone.utc)
    assert weather.select_forecast_slot([], kickoff) is None
    assert weather.select_daily_slot({}, kickoff) is None


def test_a_missing_measurement_does_not_flag_anything():
    """AccuWeather omits blocks rather than sending nulls; neither is a reading."""
    snap = weather.snapshot_from_slot("g1", "Highmark Stadium", {"DateTime": "x"})
    assert snap.temperature_f is None and snap.wind_speed_mph is None
    assert snap.high_wind == 0 and snap.freezing == 0


def test_a_night_kickoff_gets_its_own_day_not_tomorrows():
    """8:20pm Eastern is 01:20 the next day in UTC. Matching UTC dates would ask
    for tomorrow's forecast for tonight's game."""
    kickoff = datetime(2026, 12, 21, 1, 20, tzinfo=timezone.utc)
    payload = {"DailyForecasts": [
        {"Date": "2026-12-20T07:00:00-05:00", "Day": {"IconPhrase": "game day"}},
        {"Date": "2026-12-21T07:00:00-05:00", "Day": {"IconPhrase": "day after"}},
    ]}
    assert weather.select_daily_slot(payload, kickoff)["Day"]["IconPhrase"] == "game day"


def test_an_afternoon_kickoff_still_gets_its_own_day():
    kickoff = datetime(2026, 12, 20, 18, tzinfo=timezone.utc)
    payload = {"DailyForecasts": [
        {"Date": "2026-12-19T07:00:00-05:00", "Day": {"IconPhrase": "day before"}},
        {"Date": "2026-12-20T07:00:00-05:00", "Day": {"IconPhrase": "game day"}},
    ]}
    assert weather.select_daily_slot(payload, kickoff)["Day"]["IconPhrase"] == "game day"


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


# ------------------------------------------------------------ credit cost
def test_credit_estimate_matches_markets_times_regions(db_path):
    client = OddsAPIClient("key", base_url=BASE, db_path=db_path)
    # 3 game markets, 6 NFL prop markets, 1 region
    assert client.estimated_credits("nfl", events=0) == 3
    assert client.estimated_credits("nfl", events=1) == 3 + 6
    assert client.estimated_credits("nfl", events=13) == 3 + 13 * 6
    assert client.estimated_credits("nba", events=10) == 3 + 10 * 4
    assert client.estimated_credits("nfl", events=13, include_props=False) == 3
    assert client.estimated_credits(
        "nfl", events=13, prop_markets=["player_pass_yds"]
    ) == 3 + 13


@respx.mock
async def test_max_events_caps_the_expensive_prop_calls(db_path, odds_event, props_payload):
    events = [{**odds_event, "id": f"evt-{index}"} for index in range(1, 4)]
    respx.get(f"{BASE}/v4/sports/americanfootball_nfl/odds").mock(
        return_value=httpx.Response(
            200, json=events, headers={"x-requests-remaining": "400"}
        )
    )
    routes = {
        event["id"]: respx.get(
            f"{BASE}/v4/sports/americanfootball_nfl/events/{event['id']}/odds"
        ).mock(
            return_value=httpx.Response(
                200, json={**props_payload, "id": event["id"]},
                headers={"x-requests-remaining": "394"},
            )
        )
        for event in events
    }

    async with OddsAPIClient("key", base_url=BASE, db_path=db_path) as client:
        summary = await client.ingest_slate("nfl", max_events=1)

    assert summary.games == 3
    assert summary.events_polled == 1
    assert routes["evt-1"].called
    assert not routes["evt-2"].called and not routes["evt-3"].called


# ------------------------------------------------- the no-network guard
async def test_unmocked_http_calls_are_blocked_in_tests():
    """The guard that keeps a test from passing only because the sandbox has
    no internet. It caught a real bug: InjuryClient(feed_urls={}) fell back to
    the live ESPN feed, which failed locally and returned real data on CI."""
    async with httpx.AsyncClient() as client:
        with pytest.raises(RuntimeError, match="unmocked network call"):
            await client.get("https://site.api.espn.com/should-never-be-reached")


async def test_empty_feed_map_does_not_fall_back_to_the_live_feed(db_path):
    """Regression: `feed_urls or {...}` treated {} as 'not supplied'."""
    client = injuries.InjuryClient(db_path=db_path, feed_urls={})
    assert client.feed_urls == {}
    assert await client.ingest("nfl") == []


async def test_default_feed_map_is_used_when_none_is_given(db_path):
    client = injuries.InjuryClient(db_path=db_path)
    assert set(client.feed_urls) == {"nfl", "nba", "ncaaf"}
    for url in client.feed_urls.values():
        assert url.startswith("https://")


@respx.mock
async def test_client_works_against_a_database_that_does_not_exist_yet(tmp_path, odds_event):
    """Regression: the client primed its quota from the database on connect,
    before any table existed. Local runs always had a database left over from
    an earlier mock run; a fresh machine crashed on the first live call."""
    fresh = tmp_path / "nested" / "brand-new.db"
    assert not fresh.exists()

    respx.get(f"{BASE}/v4/sports/americanfootball_nfl/odds").mock(
        return_value=httpx.Response(
            200, json=[odds_event], headers={"x-requests-remaining": "480"}
        )
    )
    async with OddsAPIClient("key", base_url=BASE, db_path=str(fresh)) as client:
        events = await client.fetch_game_odds("nfl")

    assert events[0]["id"] == "evt-1"
    assert fresh.exists()
    assert db.latest_quota("odds_api", db_path=str(fresh))["requests_remaining"] == 480
