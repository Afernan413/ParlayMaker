"""Stadium weather ingestion.

Forecasts come from AccuWeather, and the reading nearest kickoff is kept. Domes
short-circuit the network call entirely -- there is no weather story indoors,
and it saves quota.

Three things about AccuWeather shape this module:

* **A venue is addressed by an opaque location key**, not by coordinates, and
  looking one up costs a call. The keys are stable, so they are cached in
  ``weather_locations`` and a venue is resolved once rather than every run.
* **The hourly forecast only reaches twelve hours out.** A slate built on
  Saturday morning for a Sunday afternoon kickoff is outside it, so the daily
  forecast is used for anything further away and the hourly one when kickoff is
  close. The hourly reading is sharper, so it is preferred whenever it covers
  the game.
* **The free tier allows fifty calls a day** and answers 503 once that is
  spent. That is a clean stop, not a failure: whatever has been fetched is
  kept and the rest of the slate is left without a forecast.

The run also carries its own budget (``settings.weather_call_budget``), because
the location-key cache lives in SQLite and a hosted build starts with an empty
database -- so every run there pays two calls per venue, and a 29-game slate
would spend 58 and exhaust the day's allowance in one go. The budget is the only
hard stop on a slow provider stalling a build, too.

The API key is read from ``OPENWEATHER_API_KEY``. The name is historical -- the
deployed secret is called that, and renaming it would mean re-adding it
everywhere.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

import httpx

from config.settings import settings
from src.ingestion import db
from src.models.schedule import EASTERN_SHIFT

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


#: AccuWeather's hourly forecast reaches this far ahead. Beyond it the daily
#: forecast is the only thing that covers the game.
HOURLY_HORIZON = timedelta(hours=12)

#: What a 503 from AccuWeather means when the body says so.
QUOTA_MESSAGE = "allowed number of requests has been exceeded"


class WeatherQuotaExhausted(RuntimeError):
    """The provider's daily call allowance is spent."""


def select_forecast_slot(
    payload: Any, kickoff: datetime
) -> dict[str, Any] | None:
    """Pick the hourly reading closest to kickoff.

    AccuWeather returns a bare list of hours. The design doc asks for a read a
    few hours before kickoff; the closest hour to kickoff itself is the sharpest
    available signal for game conditions.
    """
    slots = payload if isinstance(payload, list) else (payload or {}).get("list") or []
    dated = [slot for slot in slots if slot.get("DateTime")]
    if not dated:
        return None
    return min(dated, key=lambda slot: abs(_parse_iso(slot["DateTime"]) - kickoff))


def select_daily_slot(payload: Any, kickoff: datetime) -> dict[str, Any] | None:
    """Pick the day covering kickoff from the 5-day forecast.

    Compared in local terms, not UTC. A Sunday night kickoff at 8:20pm Eastern
    is 01:20 the next day in UTC, so matching UTC dates would ask for tomorrow's
    forecast for tonight's game. AccuWeather dates its days in the venue's own
    offset, which is what the shift lines up with.
    """
    days = (payload or {}).get("DailyForecasts") or []
    dated = [day for day in days if day.get("Date")]
    if not dated:
        return None
    local_kickoff = (kickoff.astimezone(timezone.utc) - EASTERN_SHIFT).date()
    return min(
        dated,
        key=lambda day: abs((_parse_iso(day["Date"]) - EASTERN_SHIFT).date() - local_kickoff),
    )


def _value(block: Any) -> float | None:
    """AccuWeather wraps every measurement as ``{"Value": x, "Unit": "F"}``."""
    if not isinstance(block, dict):
        return None
    value = block.get("Value")
    return float(value) if value is not None else None


def _probability(percent: Any) -> float | None:
    """A 0-100 chance as the 0-1 fraction the rest of the engine stores."""
    if percent is None:
        return None
    return max(0.0, min(float(percent) / 100.0, 1.0))


def _flags(temp_f: float | None, wind_mph: float | None) -> tuple[int, int]:
    return (
        int(wind_mph is not None and wind_mph > settings.high_wind_mph),
        int(temp_f is not None and temp_f <= settings.freezing_temp_f),
    )


def snapshot_from_slot(
    game_id: str, stadium: str | None, slot: dict[str, Any]
) -> WeatherSnapshot:
    """Convert one AccuWeather hourly reading into a :class:`WeatherSnapshot`."""
    wind = slot.get("Wind") or {}
    temp_f = _value(slot.get("Temperature"))
    wind_mph = _value(wind.get("Speed"))
    high_wind, freezing = _flags(temp_f, wind_mph)
    return WeatherSnapshot(
        game_id=game_id,
        stadium=stadium,
        is_dome=0,
        temperature_f=temp_f,
        wind_speed_mph=wind_mph,
        wind_deg=(wind.get("Direction") or {}).get("Degrees"),
        precip_chance=_probability(slot.get("PrecipitationProbability")),
        conditions=slot.get("IconPhrase"),
        high_wind=high_wind,
        freezing=freezing,
        forecast_for=slot.get("DateTime"),
    )


def snapshot_from_day(
    game_id: str, stadium: str | None, day: dict[str, Any]
) -> WeatherSnapshot:
    """Convert a daily forecast into a snapshot, for a kickoff too far out.

    Coarser than the hourly reading by nature: the temperature is the day's
    high rather than the temperature at kickoff, and the wind is the daytime
    figure. It is still much better than no forecast, and the conditions string
    says which it is so nothing downstream mistakes one for the other.
    """
    daytime = day.get("Day") or {}
    temperature = day.get("Temperature") or {}
    temp_f = _value(temperature.get("Maximum"))
    wind_mph = _value((daytime.get("Wind") or {}).get("Speed"))
    high_wind, freezing = _flags(temp_f, wind_mph)
    phrase = daytime.get("IconPhrase") or day.get("Headline", {}).get("Text")
    return WeatherSnapshot(
        game_id=game_id,
        stadium=stadium,
        is_dome=0,
        temperature_f=temp_f,
        wind_speed_mph=wind_mph,
        wind_deg=((daytime.get("Wind") or {}).get("Direction") or {}).get("Degrees"),
        precip_chance=_probability(daytime.get("PrecipitationProbability")),
        conditions=f"{phrase} (daily outlook)" if phrase else "daily outlook",
        high_wind=high_wind,
        freezing=freezing,
        forecast_for=day.get("Date"),
    )


class WeatherClient:
    """AccuWeather forecast fetcher with dome short-circuiting and key caching."""

    api_name = "accuweather"
    provider = "accuweather"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
        db_path: str | None = None,
        call_budget: int | None = None,
    ) -> None:
        # Read from the OpenWeather-named setting on purpose: that is what the
        # deployed secret is called. See the module docstring.
        self.api_key = api_key if api_key is not None else settings.openweather_api_key
        self.base_url = (base_url or settings.accuweather_base_url).rstrip("/")
        self.db_path = db_path
        self._owns_client = client is None
        self._client = client
        self._quota_spent = False
        self.call_budget = (
            settings.weather_call_budget if call_budget is None else call_budget
        )
        self.calls_made = 0

    async def __aenter__(self) -> "WeatherClient":
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url, timeout=settings.weather_timeout_seconds
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
                base_url=self.base_url, timeout=settings.weather_timeout_seconds
            )
        return self._client

    # -- plumbing -----------------------------------------------------
    async def _get(self, path: str, params: dict[str, Any]) -> Any:
        """One authenticated call, with the quota answer recognised as such."""
        if not self.api_key:
            raise RuntimeError("OPENWEATHER_API_KEY is not set")
        if self._quota_spent:
            raise WeatherQuotaExhausted("daily forecast allowance already spent")
        if self.calls_made >= self.call_budget:
            raise WeatherQuotaExhausted(
                f"this run's forecast budget of {self.call_budget} calls is spent"
            )

        self.calls_made += 1
        response = await self.client.get(path, params={**params, "apikey": self.api_key})
        db.log_quota(
            self.api_name, path, status_code=response.status_code, db_path=self.db_path
        )
        if response.status_code == 503 and QUOTA_MESSAGE in response.text.lower():
            # Not a provider outage: the day's allowance is gone. Remember it so
            # the rest of the slate does not spend a call each to find out.
            self._quota_spent = True
            raise WeatherQuotaExhausted(
                "AccuWeather daily request allowance exhausted; "
                "the rest of this slate has no forecast"
            )
        response.raise_for_status()
        return response.json()

    async def location_key(self, lat: float, lon: float) -> str | None:
        """AccuWeather's key for a venue, from the cache where possible."""
        venue_key = f"{round(float(lat), 4)},{round(float(lon), 4)}"
        cached = db.cached_location_key(venue_key, self.provider, db_path=self.db_path)
        if cached:
            return cached

        payload = await self._get(
            "/locations/v1/cities/geoposition/search", {"q": venue_key}
        )
        key = (payload or {}).get("Key")
        if not key:
            logger.warning("no AccuWeather location for %s", venue_key)
            return None
        db.store_location_key(
            venue_key, self.provider, str(key),
            name=(payload or {}).get("LocalizedName"), db_path=self.db_path,
        )
        return str(key)

    # -- forecasts ----------------------------------------------------
    async def fetch_hourly(self, location_key: str) -> Any:
        """The next twelve hours, in imperial units."""
        return await self._get(
            f"/forecasts/v1/hourly/12hour/{location_key}",
            {"details": "true", "metric": "false"},
        )

    async def fetch_daily(self, location_key: str) -> Any:
        """The next five days, for a kickoff beyond the hourly horizon."""
        return await self._get(
            f"/forecasts/v1/daily/5day/{location_key}",
            {"details": "true", "metric": "false"},
        )

    async def fetch_forecast(self, lat: float, lon: float) -> Any:
        """Raw hourly forecast for a venue. Kept for callers that want it."""
        key = await self.location_key(lat, lon)
        if key is None:
            raise RuntimeError(f"no forecast location for {lat},{lon}")
        return await self.fetch_hourly(key)

    async def snapshot_for_game(self, game: dict[str, Any]) -> WeatherSnapshot | None:
        """Forecast for one ``games`` row, or ``None`` if the venue is unknown."""
        venue = stadium_for(game.get("home_team", ""))
        if venue is None:
            logger.info("no stadium mapping for %s", game.get("home_team"))
            return None
        if venue["dome"]:
            return dome_snapshot(game["game_id"], venue["stadium"], game.get("commence_time"))

        kickoff = _parse_iso(game["commence_time"])
        key = await self.location_key(venue["lat"], venue["lon"])
        if key is None:
            return None

        # The hourly reading is sharper, so use it whenever it reaches the game.
        within_hourly = kickoff - datetime.now(timezone.utc) <= HOURLY_HORIZON
        if within_hourly:
            slot = select_forecast_slot(await self.fetch_hourly(key), kickoff)
            if slot is not None:
                return snapshot_from_slot(game["game_id"], venue["stadium"], slot)

        day = select_daily_slot(await self.fetch_daily(key), kickoff)
        if day is None:
            return None
        return snapshot_from_day(game["game_id"], venue["stadium"], day)

    async def ingest_games(self, games: Sequence[dict[str, Any]]) -> list[WeatherSnapshot]:
        """Fetch and persist snapshots for a slate; venue errors are skipped."""
        snapshots: list[WeatherSnapshot] = []
        for game in games:
            try:
                snapshot = await self.snapshot_for_game(game)
            except WeatherQuotaExhausted as exc:
                # Stop rather than hammer: every further call would answer 503.
                logger.warning("%s", exc)
                break
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
