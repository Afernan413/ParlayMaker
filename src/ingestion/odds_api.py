"""The Odds API client, scoped to FanDuel.

Two endpoints are used:

* ``/v4/sports/{sport}/odds``                    -- full-slate game lines
* ``/v4/sports/{sport}/events/{eventId}/odds``   -- per-event player props

Every response's quota headers are persisted to ``api_quota_log`` and the
client refuses to issue another request once the remaining allowance drops
below :attr:`~config.settings.Settings.min_quota_remaining`.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import httpx

from config.settings import SPORT_KEYS, bookmaker_config, settings
from src.ingestion import db

logger = logging.getLogger(__name__)

RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})
FATAL_STATUS = frozenset({401, 403, 404, 422})

HEADER_REMAINING = "x-requests-remaining"
HEADER_USED = "x-requests-used"
HEADER_LAST = "x-requests-last"


class OddsAPIError(RuntimeError):
    """Non-retryable failure from The Odds API."""


class QuotaExhaustedError(OddsAPIError):
    """Raised when the remaining request allowance is below the safety floor."""


@dataclass
class QuotaState:
    """Rolling view of the API allowance, updated from response headers."""

    remaining: int | None = None
    used: int | None = None
    last_cost: int | None = None

    def update(self, headers: httpx.Headers) -> None:
        self.remaining = _to_int(headers.get(HEADER_REMAINING), self.remaining)
        self.used = _to_int(headers.get(HEADER_USED), self.used)
        self.last_cost = _to_int(headers.get(HEADER_LAST), self.last_cost)


@dataclass
class IngestSummary:
    """What a slate ingestion actually wrote."""

    sport: str
    games: int = 0
    lines: int = 0
    props: int = 0
    events_polled: int = 0
    requests_made: int = 0
    quota_remaining: int | None = None
    skipped_events: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sport": self.sport,
            "games": self.games,
            "lines": self.lines,
            "props": self.props,
            "events_polled": self.events_polled,
            "requests_made": self.requests_made,
            "quota_remaining": self.quota_remaining,
            "skipped_events": list(self.skipped_events),
        }


def _to_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def sport_key(sport: str) -> str:
    """Map ``nfl``/``nba`` to The Odds API sport key."""
    try:
        return SPORT_KEYS[sport.lower()]
    except KeyError as exc:  # pragma: no cover - guarded by CLI choices
        raise OddsAPIError(f"unsupported sport: {sport}") from exc


class OddsAPIClient:
    """Async FanDuel odds fetcher with retry, quota tracking and persistence."""

    api_name = "odds_api"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
        db_path: str | None = None,
        min_quota_remaining: int | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.odds_api_key
        self.base_url = (base_url or settings.odds_api_base_url).rstrip("/")
        self.db_path = db_path
        self.min_quota_remaining = (
            settings.min_quota_remaining
            if min_quota_remaining is None
            else min_quota_remaining
        )
        self._owns_client = client is None
        self._client = client
        self.quota = QuotaState()
        self.requests_made = 0
        self.config = bookmaker_config()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    async def __aenter__(self) -> "OddsAPIClient":
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url, timeout=settings.http_timeout_seconds
            )
        self._prime_quota_from_db()
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

    def _prime_quota_from_db(self) -> None:
        """Carry the last known allowance over from a previous run."""
        row = db.latest_quota(self.api_name, db_path=self.db_path)
        if row and self.quota.remaining is None:
            self.quota.remaining = row.get("requests_remaining")
            self.quota.used = row.get("requests_used")

    # ------------------------------------------------------------------
    # transport
    # ------------------------------------------------------------------
    def assert_quota(self) -> None:
        """Abort before spending a request we cannot afford."""
        if self.quota.remaining is None:
            return
        if self.quota.remaining < self.min_quota_remaining:
            raise QuotaExhaustedError(
                f"Odds API allowance too low to continue: "
                f"{self.quota.remaining} remaining < floor {self.min_quota_remaining}"
            )

    async def _get(self, path: str, params: dict[str, Any]) -> Any:
        """GET with quota guard, retry/backoff and quota logging."""
        if not self.api_key:
            raise OddsAPIError(
                "ODDS_API_KEY is not set; run with --mock for offline pipelines"
            )
        self.assert_quota()
        query = {"apiKey": self.api_key, **params}
        delay = settings.http_backoff_seconds
        last_error: Exception | None = None

        for attempt in range(1, settings.http_max_retries + 1):
            try:
                response = await self.client.get(path, params=query)
            except httpx.HTTPError as exc:  # transport-level failure
                last_error = exc
                logger.warning("odds api transport error on %s (try %s): %s", path, attempt, exc)
            else:
                self.requests_made += 1
                self.quota.update(response.headers)
                db.log_quota(
                    self.api_name,
                    path,
                    requests_used=self.quota.used,
                    requests_remaining=self.quota.remaining,
                    last_cost=self.quota.last_cost,
                    status_code=response.status_code,
                    db_path=self.db_path,
                )
                if response.status_code in FATAL_STATUS:
                    raise OddsAPIError(
                        f"{response.status_code} from {path}: {response.text[:200]}"
                    )
                if response.status_code in RETRY_STATUS:
                    last_error = OddsAPIError(
                        f"{response.status_code} from {path}: {response.text[:200]}"
                    )
                    logger.warning(
                        "odds api %s on %s (try %s)", response.status_code, path, attempt
                    )
                else:
                    response.raise_for_status()
                    return response.json()

            if attempt < settings.http_max_retries:
                await asyncio.sleep(delay)
                delay *= 2

        raise OddsAPIError(f"exhausted retries for {path}") from last_error

    # ------------------------------------------------------------------
    # endpoints
    # ------------------------------------------------------------------
    async def fetch_game_odds(
        self, sport: str, markets: Sequence[str] | None = None
    ) -> list[dict[str, Any]]:
        """Full slate of FanDuel game lines (moneyline / spread / total)."""
        markets = list(markets or self.config["game_markets"])
        return await self._get(
            f"/v4/sports/{sport_key(sport)}/odds",
            {
                "regions": self.config["regions"],
                "markets": ",".join(markets),
                "oddsFormat": self.config["odds_format"],
                "bookmakers": self.config["bookmaker_key"],
            },
        )

    async def fetch_events(self, sport: str) -> list[dict[str, Any]]:
        """Upcoming events (free call -- used to enumerate prop event ids)."""
        return await self._get(f"/v4/sports/{sport_key(sport)}/events", {})

    async def fetch_event_props(
        self, sport: str, event_id: str, markets: Sequence[str] | None = None
    ) -> dict[str, Any]:
        """FanDuel player props for one event."""
        markets = list(markets or self.config["prop_markets"].get(sport.lower(), []))
        return await self._get(
            f"/v4/sports/{sport_key(sport)}/events/{event_id}/odds",
            {
                "regions": self.config["regions"],
                "markets": ",".join(markets),
                "oddsFormat": self.config["odds_format"],
                "bookmakers": self.config["bookmaker_key"],
            },
        )

    # ------------------------------------------------------------------
    # orchestration
    # ------------------------------------------------------------------
    def estimated_credits(
        self,
        sport: str,
        *,
        events: int,
        include_props: bool = True,
        prop_markets: Sequence[str] | None = None,
    ) -> int:
        """Credits a slate ingestion will cost.

        The Odds API bills one credit per market per region, so the game-lines
        call costs ``len(game_markets)`` and each event's props cost
        ``len(prop_markets)``. The exact figure the API charged for the last
        call comes back in the ``x-requests-last`` header and is logged to
        ``api_quota_log``, so this is a forecast, not the source of truth.
        """
        regions = len(self.config["regions"].split(","))
        cost = len(self.config["game_markets"]) * regions
        if include_props:
            markets = prop_markets or self.config["prop_markets"].get(sport.lower(), [])
            cost += events * len(markets) * regions
        return cost

    async def ingest_slate(
        self,
        sport: str,
        *,
        include_props: bool = True,
        max_events: int | None = None,
        prop_markets: Sequence[str] | None = None,
    ) -> IngestSummary:
        """Pull lines (and optionally props) for a slate straight into SQLite.

        Stops cleanly -- rather than raising -- once the quota floor is hit
        mid-slate, so whatever was already written stays usable. ``max_events``
        caps how many events get prop requests, which is the expensive part.
        """
        db.init_db(self.db_path)
        summary = IngestSummary(sport=sport.lower())

        events = await self.fetch_game_odds(sport)
        games = [parse_game(ev, sport) for ev in events]
        summary.games = db.upsert_games(games, db_path=self.db_path)
        line_rows = [row for ev in events for row in parse_lines(ev)]
        summary.lines = db.insert_lines(line_rows, db_path=self.db_path)

        if include_props:
            event_ids = [ev["id"] for ev in events][: max_events or len(events)]
            logger.info(
                "props for %s/%s events will cost about %s credits (%s remaining)",
                len(event_ids),
                len(events),
                self.estimated_credits(
                    sport, events=len(event_ids), prop_markets=prop_markets
                )
                - len(self.config["game_markets"]),
                self.quota.remaining if self.quota.remaining is not None else "unknown",
            )
            for event_id in event_ids:
                try:
                    self.assert_quota()
                except QuotaExhaustedError as exc:
                    logger.warning("halting prop ingestion: %s", exc)
                    summary.skipped_events.extend(
                        event_ids[event_ids.index(event_id) :]
                    )
                    break
                payload = await self.fetch_event_props(sport, event_id, prop_markets)
                prop_rows = parse_props(payload)
                summary.props += db.insert_props(prop_rows, db_path=self.db_path)
                summary.events_polled += 1

        summary.requests_made = self.requests_made
        summary.quota_remaining = self.quota.remaining
        return summary


# ----------------------------------------------------------------------
# pure parsers (no I/O -- directly unit testable)
# ----------------------------------------------------------------------
def parse_game(event: dict[str, Any], sport: str) -> dict[str, Any]:
    """Normalise one Odds API event into a ``games`` row."""
    return {
        "game_id": event["id"],
        "sport": sport.lower(),
        "commence_time": event["commence_time"],
        "home_team": event.get("home_team") or "",
        "away_team": event.get("away_team") or "",
    }


def _fanduel_markets(event: dict[str, Any]) -> Iterable[dict[str, Any]]:
    target = bookmaker_config()["bookmaker_key"]
    for book in event.get("bookmakers", []) or []:
        if book.get("key") != target:
            continue
        yield from book.get("markets", []) or []


def parse_lines(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract FanDuel game lines from one event payload."""
    rows: list[dict[str, Any]] = []
    captured = db.utcnow()
    for market in _fanduel_markets(event):
        for outcome in market.get("outcomes", []) or []:
            price = _to_int(outcome.get("price"))
            if price is None:
                continue
            rows.append(
                {
                    "game_id": event["id"],
                    "market": market["key"],
                    "selection": outcome.get("name", ""),
                    "line": outcome.get("point"),
                    "american_odds": price,
                    "captured_at": captured,
                }
            )
    return rows


def parse_props(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract FanDuel player props from one event-odds payload.

    Player name lives in ``description``; ``name`` carries Over/Under/Yes.
    """
    rows: list[dict[str, Any]] = []
    captured = db.utcnow()
    game_id = payload.get("id")
    if not game_id:
        return rows
    for market in _fanduel_markets(payload):
        for outcome in market.get("outcomes", []) or []:
            price = _to_int(outcome.get("price"))
            player = outcome.get("description") or outcome.get("participant")
            if price is None or not player:
                continue
            rows.append(
                {
                    "game_id": game_id,
                    "market": market["key"],
                    "player_name": player,
                    "selection": outcome.get("name", ""),
                    "line": outcome.get("point"),
                    "american_odds": price,
                    "captured_at": captured,
                }
            )
    return rows
