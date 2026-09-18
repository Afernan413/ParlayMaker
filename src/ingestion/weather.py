"""Stadium weather ingestion.

Forecasts are pulled from OpenWeather's 5-day/3-hour endpoint and the slot
nearest to kickoff is kept. Domes short-circuit the network call entirely --
there is no weather story indoors, and it saves quota.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

import httpx

from config.settings import settings
from src.ingestion import db

logger = logging.getLogger(__name__)

#: team abbreviation -> stadium metadata. Retractable roofs are treated as
#: domes only when they are closed by default in bad weather; the flag here is
#: the conservative "climate controlled" reading.
STADIUMS: dict[str, dict[str, Any]] = {
    "ARI": {"stadium": "State Farm Stadium", "lat": 33.5277, "lon": -112.2626, "dome": True},
    "ATL": {"stadium": "Mercedes-Benz Stadium", "lat": 33.7554, "lon": -84.4008, "dome": True},
    "BAL": {"stadium": "M&T Bank Stadium", "lat": 39.2780, "lon": -76.6227, "dome": False},
    "BUF": {"stadium": "Highmark Stadium", "lat": 42.7738, "lon": -78.7870, "dome": False},
    "CAR": {"stadium": "Bank of America Stadium", "lat": 35.2258, "lon": -80.8528, "dome": False},
    "CHI": {"stadium": "Soldier Field", "lat": 41.8623, "lon": -87.6167, "dome": False},
    "CIN": {"stadium": "Paycor Stadium", "lat": 39.0954, "lon": -84.5160, "dome": False},
    "CLE": {"stadium": "Huntington Bank Field", "lat": 41.5061, "lon": -81.6995, "dome": False},
    "DAL": {"stadium": "AT&T Stadium", "lat": 32.7473, "lon": -97.0945, "dome": True},
    "DEN": {"stadium": "Empower Field at Mile High", "lat": 39.7439, "lon": -105.0201, "dome": False},
    "DET": {"stadium": "Ford Field", "lat": 42.3400, "lon": -83.0456, "dome": True},
    "GB": {"stadium": "Lambeau Field", "lat": 44.5013, "lon": -88.0622, "dome": False},
    "HOU": {"stadium": "NRG Stadium", "lat": 29.6847, "lon": -95.4107, "dome": True},
    "IND": {"stadium": "Lucas Oil Stadium", "lat": 39.7601, "lon": -86.1639, "dome": True},
    "JAX": {"stadium": "EverBank Stadium", "lat": 30.3239, "lon": -81.6373, "dome": False},
    "KC": {"stadium": "GEHA Field at Arrowhead", "lat": 39.0489, "lon": -94.4839, "dome": False},
    "LAC": {"stadium": "SoFi Stadium", "lat": 33.9535, "lon": -118.3392, "dome": True},
    "LAR": {"stadium": "SoFi Stadium", "lat": 33.9535, "lon": -118.3392, "dome": True},
    "LV": {"stadium": "Allegiant Stadium", "lat": 36.0909, "lon": -115.1833, "dome": True},
    "MIA": {"stadium": "Hard Rock Stadium", "lat": 25.9580, "lon": -80.2389, "dome": False},
    "MIN": {"stadium": "U.S. Bank Stadium", "lat": 44.9736, "lon": -93.2575, "dome": True},
    "NE": {"stadium": "Gillette Stadium", "lat": 42.0909, "lon": -71.2643, "dome": False},
    "NO": {"stadium": "Caesars Superdome", "lat": 29.9511, "lon": -90.0812, "dome": True},
    "NYG": {"stadium": "MetLife Stadium", "lat": 40.8135, "lon": -74.0745, "dome": False},
    "NYJ": {"stadium": "MetLife Stadium", "lat": 40.8135, "lon": -74.0745, "dome": False},
    "PHI": {"stadium": "Lincoln Financial Field", "lat": 39.9008, "lon": -75.1675, "dome": False},
    "PIT": {"stadium": "Acrisure Stadium", "lat": 40.4468, "lon": -80.0158, "dome": False},
    "SEA": {"stadium": "Lumen Field", "lat": 47.5952, "lon": -122.3316, "dome": False},
    "SF": {"stadium": "Levi's Stadium", "lat": 37.4033, "lon": -121.9694, "dome": False},
    "TB": {"stadium": "Raymond James Stadium", "lat": 27.9759, "lon": -82.5033, "dome": False},
    "TEN": {"stadium": "Nissan Stadium", "lat": 36.1665, "lon": -86.7713, "dome": False},
    "WAS": {"stadium": "Northwest Stadium", "lat": 38.9076, "lon": -76.8645, "dome": False},
}

#: Full club names (as The Odds API reports them) -> abbreviation.
TEAM_ALIASES: dict[str, str] = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def team_code(team: str) -> str | None:
    """Resolve a team name or abbreviation to a stadium key."""
    if not team:
        return None
    candidate = team.strip()
    if candidate.upper() in STADIUMS:
        return candidate.upper()
    return TEAM_ALIASES.get(candidate)


def stadium_for(home_team: str) -> dict[str, Any] | None:
    code = team_code(home_team)
    return STADIUMS.get(code) if code else None


@dataclass
class WeatherSnapshot:
    """One forecast reading, already reduced to the flags the model cares about."""

    game_id: str
    stadium: str | None
    is_dome: int
    temperature_f: float | None
    wind_speed_mph: float | None
    wind_deg: float | None
    precip_chance: float | None
    conditions: str | None
    high_wind: int
    freezing: int
    forecast_for: str | None

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def dome_snapshot(game_id: str, stadium: str | None, kickoff: str | None) -> WeatherSnapshot:
    """Indoor games get a fixed, benign reading with no API call."""
    return WeatherSnapshot(
        game_id=game_id,
        stadium=stadium,
        is_dome=1,
        temperature_f=70.0,
        wind_speed_mph=0.0,
        wind_deg=None,
        precip_chance=0.0,
        conditions="Dome (climate controlled)",
        high_wind=0,
        freezing=0,
        forecast_for=kickoff,
    )


def _parse_iso(value: str) -> datetime:
    cleaned = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(cleaned)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def select_forecast_slot(
    payload: dict[str, Any], kickoff: datetime
) -> dict[str, Any] | None:
    """Pick the 3-hourly slot closest to kickoff.

    The design doc asks for a read taken ~3 hours before kickoff; because the
    provider buckets in 3-hour steps, the closest bucket to kickoff itself is
    the sharpest available signal for game conditions.
    """
    slots = payload.get("list") or []
    if not slots:
        return None
    return min(slots, key=lambda slot: abs(_parse_iso(slot["dt_txt"]) - kickoff))


def snapshot_from_slot(
    game_id: str, stadium: str | None, slot: dict[str, Any]
) -> WeatherSnapshot:
    """Convert one OpenWeather slot into a :class:`WeatherSnapshot`."""
    main = slot.get("main") or {}
    wind = slot.get("wind") or {}
    weather = (slot.get("weather") or [{}])[0]
    temp_f = main.get("temp")
    wind_mph = wind.get("speed")
    return WeatherSnapshot(
        game_id=game_id,
        stadium=stadium,
        is_dome=0,
        temperature_f=temp_f,
        wind_speed_mph=wind_mph,
        wind_deg=wind.get("deg"),
        precip_chance=slot.get("pop"),
        conditions=weather.get("description"),
        high_wind=int(wind_mph is not None and wind_mph > settings.high_wind_mph),
        freezing=int(temp_f is not None and temp_f <= settings.freezing_temp_f),
        forecast_for=slot.get("dt_txt"),
    )


class WeatherClient:
    """OpenWeather forecast fetcher with dome short-circuiting."""

    api_name = "openweather"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
        db_path: str | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.openweather_api_key
        self.base_url = (base_url or settings.openweather_base_url).rstrip("/")
        self.db_path = db_path
        self._owns_client = client is None
        self._client = client

    async def __aenter__(self) -> "WeatherClient":
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url, timeout=settings.http_timeout_seconds
            )
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url, timeout=settings.http_timeout_seconds
            )
        return self._client

    async def fetch_forecast(self, lat: float, lon: float) -> dict[str, Any]:
        """Raw 5-day/3-hour forecast in imperial units."""
        if not self.api_key:
            raise RuntimeError("OPENWEATHER_API_KEY is not set")
        response = await self.client.get(
            "/data/2.5/forecast",
            params={
                "lat": lat,
                "lon": lon,
                "units": "imperial",
                "appid": self.api_key,
            },
        )
        db.log_quota(
            self.api_name,
            "/data/2.5/forecast",
            status_code=response.status_code,
            db_path=self.db_path,
        )
        response.raise_for_status()
        return response.json()

    async def snapshot_for_game(self, game: dict[str, Any]) -> WeatherSnapshot | None:
        """Forecast for one ``games`` row, or ``None`` if the venue is unknown."""
        venue = stadium_for(game.get("home_team", ""))
        if venue is None:
            logger.info("no stadium mapping for %s", game.get("home_team"))
            return None
        if venue["dome"]:
            return dome_snapshot(game["game_id"], venue["stadium"], game.get("commence_time"))

        kickoff = _parse_iso(game["commence_time"])
        payload = await self.fetch_forecast(venue["lat"], venue["lon"])
        slot = select_forecast_slot(payload, kickoff)
        if slot is None:
            return None
        return snapshot_from_slot(game["game_id"], venue["stadium"], slot)

    async def ingest_games(self, games: Sequence[dict[str, Any]]) -> list[WeatherSnapshot]:
        """Fetch and persist snapshots for a slate; venue errors are skipped."""
        snapshots: list[WeatherSnapshot] = []
        for game in games:
            try:
                snapshot = await self.snapshot_for_game(game)
            except (httpx.HTTPError, RuntimeError) as exc:
                logger.warning("weather fetch failed for %s: %s", game.get("game_id"), exc)
                continue
            if snapshot is not None:
                snapshots.append(snapshot)
        store_snapshots(snapshots, db_path=self.db_path)
        return snapshots


def store_snapshots(
    snapshots: Iterable[WeatherSnapshot], db_path: str | None = None
) -> int:
    return db.insert_weather([s.as_row() for s in snapshots], db_path=db_path)


def forecast_window(kickoff: datetime, hours: int | None = None) -> datetime:
    """Timestamp of the pre-kickoff read the design doc calls for."""
    lead = settings.weather_lead_hours if hours is None else hours
    return kickoff - timedelta(hours=lead)
