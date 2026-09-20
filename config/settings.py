"""Central configuration for the parlay engine.

Everything tunable lives here so that no module hard-codes a threshold.
Values are read from the environment (or a local ``.env``) via
pydantic-settings; the defaults encode the risk rules from the design doc.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
MOCK_DIR = DATA_DIR / "mock"

# The Odds API sport keys.
SPORT_KEYS: dict[str, str] = {
    "nfl": "americanfootball_nfl",
    "ncaaf": "americanfootball_ncaaf",
    "nba": "basketball_nba",
}

#: Sports that share the NFL's markets, models and distribution families.
FOOTBALL_SPORTS = frozenset({"nfl", "ncaaf"})

# Weather only matters for outdoor football.
WEATHER_SPORTS = frozenset({"nfl"})

#: Minutes before start time that official inactives are published.
INACTIVE_LEAD_MINUTES: dict[str, int] = {"nfl": 90, "ncaaf": 90, "nba": 30}


class Settings(BaseSettings):
    """Runtime settings. Secrets stay in the environment, never in code."""

    # The env file is resolved against the project root, not the working
    # directory: running a script from elsewhere would otherwise read an empty
    # key and silently fall back to mock data.
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- credentials -------------------------------------------------
    odds_api_key: str = ""
    odds_api_base_url: str = "https://api.the-odds-api.com"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-5"
    #: Forecast provider key. The name says OpenWeather for continuity -- the
    #: deployed secret is called OPENWEATHER_API_KEY -- but the value is an
    #: AccuWeather key and the client talks to AccuWeather. Renaming it would
    #: mean re-adding the secret everywhere it is already set.
    openweather_api_key: str = ""
    accuweather_base_url: str = "https://dataservice.accuweather.com"
    #: Calls a single run may spend on forecasts. The free tier allows 50 a day
    #: and a hosted build starts with an empty location-key cache, so an
    #: unbounded slate spends two per venue and blows the allowance in one run.
    #: It is also the only hard stop on a slow provider stalling a build.
    weather_call_budget: int = 24
    #: Forecasts are not worth waiting 20 seconds for.
    weather_timeout_seconds: float = 8.0
    nfl_injury_feed_url: str = (
        "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
    )
    nba_injury_feed_url: str = (
        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
    )
    #: College football. The NCAA mandates no injury report, so coverage is
    #: patchy by conference -- the Big Ten and SEC publish availability reports,
    #: many programmes publish nothing. A thin result here is the sport, not a
    #: bug, and the run says how many designations it actually got.
    ncaaf_injury_feed_url: str = (
        "https://site.api.espn.com/apis/site/v2/sports/football/college-football/injuries"
    )
    discord_webhook_url: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # --- storage -----------------------------------------------------
    db_path: Path = DATA_DIR / "sports_data.db"

    # --- web app -----------------------------------------------------
    #: Shared access token. Empty means the app may only bind to loopback.
    web_access_token: str = ""
    web_session_hours: int = 336  # 14 days
    #: Floor between slate rebuilds, so a refresh loop cannot drain the quota.
    min_refresh_seconds: int = 300

    # --- API hygiene -------------------------------------------------
    min_quota_remaining: int = 50
    http_timeout_seconds: float = 20.0
    http_max_retries: int = 4
    http_backoff_seconds: float = 2.0

    # --- modelling ---------------------------------------------------
    copula_iterations: int = 10_000
    random_seed: int = 1_337
    rolling_weeks: int = 4
    #: Most-recent-week-first weights for the 4-week rolling window.
    rolling_weights: tuple[float, ...] = (0.4, 0.3, 0.2, 0.1)

    # --- reasoning layer --------------------------------------------
    max_context_adjustment: float = 0.20  # hard +/- ceiling on Claude shifts
    min_context_confidence: float = 0.35  # below this an adjustment is ignored

    # --- weather triggers -------------------------------------------
    high_wind_mph: float = 15.0
    freezing_temp_f: float = 32.0
    weather_lead_hours: int = 3

    # --- edge / staking ---------------------------------------------
    min_leg_ev: float = 0.04
    min_sgp_correlation: float = 0.25
    leg_odds_min: int = -140
    leg_odds_max: int = 130
    parlay_odds_min: int = 200
    parlay_odds_max: int = 650
    min_legs: int = 2
    max_legs: int = 4
    max_tickets: int = 5
    kelly_fraction: float = 0.25
    bankroll: float = Field(default=1_000.0, gt=0)

    @property
    def rolling_weight_list(self) -> list[float]:
        return list(self.rolling_weights[: self.rolling_weeks])


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor (so ``.env`` is parsed once per process)."""
    return Settings()


@lru_cache(maxsize=1)
def bookmaker_config() -> dict:
    """FanDuel market-id mappings loaded from ``config/bookmaker_keys.json``."""
    with (CONFIG_DIR / "bookmaker_keys.json").open() as fh:
        return json.load(fh)


def market_meta(market_key: str) -> dict:
    """Return the canonical metadata for an Odds API market key.

    Unknown markets fall back to a continuous family so callers never crash
    on a newly added FanDuel market.
    """
    markets = bookmaker_config()["markets"]
    return markets.get(
        market_key,
        {"stat": market_key, "family": "lognormal", "discrete": False, "sport": "any"},
    )


settings = get_settings()
