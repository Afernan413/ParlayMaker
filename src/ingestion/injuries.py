"""Injury / inactive report ingestion.

The parsers are pure functions over the feed payload so the vocabulary
normalisation (``Game-Time Decision`` -> ``QUESTIONABLE`` etc.) is testable
without touching the network. NFL inactives publish ~90 minutes before
kickoff, NBA ~30 minutes before tip -- :func:`inactives_window` computes when a
run should look.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable

import httpx

from config.settings import INACTIVE_LEAD_MINUTES, settings
from src.ingestion import db

logger = logging.getLogger(__name__)

#: Canonical statuses, most severe first. Order matters for `worst_status`.
#:
#: ``OUT_LAST_WEEK`` is not a league designation. It is what the model knows
#: about a player ruled out last week when this week's game designations have
#: not been published yet -- they come on Friday, and a build on any other day
#: would otherwise read the gap as "nobody is hurt".
STATUS_ORDER = ("OUT", "DOUBTFUL", "OUT_LAST_WEEK", "QUESTIONABLE", "PROBABLE", "ACTIVE")

STATUS_MAP: dict[str, str] = {
    "out": "OUT",
    "inactive": "OUT",
    "suspension": "OUT",
    "injured reserve": "OUT",
    "ir": "OUT",
    "physically unable to perform": "OUT",
    "pup": "OUT",
    "doubtful": "DOUBTFUL",
    "questionable": "QUESTIONABLE",
    "game-time decision": "QUESTIONABLE",
    "game time decision": "QUESTIONABLE",
    "gtd": "QUESTIONABLE",
    "probable": "PROBABLE",
    "active": "ACTIVE",
    "available": "ACTIVE",
    "day-to-day": "QUESTIONABLE",
}

#: Multiplier applied to a player's projection given their status. ``OUT`` is
#: absolute; the rest are volume haircuts consistent with historical snap data.
STATUS_MULTIPLIER: dict[str, float] = {
    "OUT": 0.0,
    "DOUBTFUL": 0.25,
    # Measured, not guessed: of 3,413 players ruled out in a week of the
    # 2023-25 regular seasons, 33% were out again the next week, 2% doubtful,
    # 17% questionable and 48% off the report. Weighting each by the multipliers
    # here gives 0.63 -- and the same 0.63 whether the player had been out one
    # week or several. (Some of the 48% went to injured reserve, which the
    # weekly roster removes separately, so for a player still active 0.63 is if
    # anything cautious.)
    "OUT_LAST_WEEK": 0.63,
    "QUESTIONABLE": 0.88,
    "PROBABLE": 0.97,
    "ACTIVE": 1.0,
}


def normalize_status(raw: str | None) -> str:
    """Map a feed's free-text status onto :data:`STATUS_ORDER`."""
    if not raw:
        return "ACTIVE"
    # A canonical value passes straight through. Without this, "OUT_LAST_WEEK"
    # would fall to the substring search below and come out as "OUT".
    canonical = str(raw).strip().upper()
    if canonical in STATUS_ORDER:
        return canonical
    key = str(raw).strip().lower()
    if key in STATUS_MAP:
        return STATUS_MAP[key]
    for needle, value in STATUS_MAP.items():
        if needle in key:
            return value
    return "ACTIVE"


def status_multiplier(status: str | None) -> float:
    return STATUS_MULTIPLIER.get(normalize_status(status), 1.0)


def worst_status(statuses: Iterable[str]) -> str:
    """Most severe status in a collection (used when feeds disagree)."""
    ranked = [normalize_status(s) for s in statuses]
    for candidate in STATUS_ORDER:
        if candidate in ranked:
            return candidate
    return "ACTIVE"


@dataclass
class InjuryRecord:
    """One normalised player availability row."""

    sport: str
    team: str | None
    player_name: str
    position: str | None
    status: str
    practice: str | None
    detail: str | None
    source: str
    report_date: str | None

    def as_row(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def multiplier(self) -> float:
        return status_multiplier(self.status)


def inactives_window(kickoff: datetime, sport: str) -> datetime:
    """When official inactives for ``sport`` become available."""
    lead = INACTIVE_LEAD_MINUTES.get(sport.lower(), 60)
    return kickoff - timedelta(minutes=lead)


def parse_espn_injuries(
    payload: dict[str, Any], sport: str, *, source: str = "espn"
) -> list[InjuryRecord]:
    """Parse the ESPN ``/injuries`` shape into :class:`InjuryRecord` rows.

    Shape: ``{"injuries": [{"displayName": team, "injuries": [{...}]}]}`` where
    each entry carries ``status`` plus a nested ``athlete``.
    """
    records: list[InjuryRecord] = []
    for team_block in payload.get("injuries", []) or []:
        team = team_block.get("displayName") or team_block.get("abbreviation")
        for entry in team_block.get("injuries", []) or []:
            athlete = entry.get("athlete") or {}
            name = athlete.get("displayName") or entry.get("displayName")
            if not name:
                continue
            position = (
                (athlete.get("position") or {}).get("abbreviation")
                if isinstance(athlete.get("position"), dict)
                else athlete.get("position")
            )
            details = entry.get("details") or {}
            records.append(
                InjuryRecord(
                    sport=sport.lower(),
                    team=team,
                    player_name=name,
                    position=position,
                    status=normalize_status(entry.get("status")),
                    practice=details.get("returnDate") or entry.get("practice"),
                    detail=entry.get("longComment")
                    or entry.get("shortComment")
                    or details.get("type"),
                    source=source,
                    report_date=entry.get("date") or details.get("returnDate"),
                )
            )
    return records


def parse_inactive_list(
    rows: Iterable[dict[str, Any]], sport: str, *, source: str = "official"
) -> list[InjuryRecord]:
    """Parse a flat official inactives list.

    Accepts the minimal ``{"player": ..., "team": ..., "status": ...}`` shape
    that league inactive feeds publish pregame.
    """
    records: list[InjuryRecord] = []
    for row in rows:
        name = row.get("player") or row.get("player_name") or row.get("displayName")
        if not name:
            continue
        records.append(
            InjuryRecord(
                sport=sport.lower(),
                team=row.get("team"),
                player_name=name,
                position=row.get("position"),
                status=normalize_status(row.get("status", "OUT")),
                practice=row.get("practice"),
                detail=row.get("detail") or row.get("reason"),
                source=source,
                report_date=row.get("report_date") or row.get("date"),
            )
        )
    return records


class InjuryClient:
    """Fetches and persists availability reports for a sport."""

    api_name = "injury_feed"

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        db_path: str | None = None,
        feed_urls: dict[str, str] | None = None,
    ) -> None:
        self.db_path = db_path
        # `or` would treat an explicitly empty mapping as "not supplied" and
        # silently fall back to the live feeds.
        self.feed_urls = (
            feed_urls
            if feed_urls is not None
            else {
                "nfl": settings.nfl_injury_feed_url,
                "nba": settings.nba_injury_feed_url,
                "ncaaf": settings.ncaaf_injury_feed_url,
            }
        )
        self._owns_client = client is None
        self._client = client

    async def __aenter__(self) -> "InjuryClient":
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=settings.http_timeout_seconds)
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=settings.http_timeout_seconds)
        return self._client

    async def fetch(self, sport: str) -> list[InjuryRecord]:
        """Pull the configured feed for ``sport`` and normalise it."""
        url = self.feed_urls.get(sport.lower())
        if not url:
            raise RuntimeError(f"no injury feed configured for {sport}")
        response = await self.client.get(url)
        db.log_quota(
            self.api_name, url, status_code=response.status_code, db_path=self.db_path
        )
        response.raise_for_status()
        return parse_espn_injuries(response.json(), sport)

    async def ingest(self, sport: str) -> list[InjuryRecord]:
        """Fetch + persist; network failures degrade to an empty report."""
        try:
            records = await self.fetch(sport)
        except (httpx.HTTPError, RuntimeError) as exc:
            logger.warning("injury fetch failed for %s: %s", sport, exc)
            return []
        store_records(records, db_path=self.db_path)
        return records


def store_records(records: Iterable[InjuryRecord], db_path: str | None = None) -> int:
    return db.insert_injuries([r.as_row() for r in records], db_path=db_path)


def status_index(records: Iterable[InjuryRecord]) -> dict[str, str]:
    """``{player_name: worst_status}`` lookup for the projection layer."""
    index: dict[str, list[str]] = {}
    for record in records:
        index.setdefault(record.player_name, []).append(record.status)
    return {name: worst_status(values) for name, values in index.items()}


# ----------------------------------------------------------------------
# the league's own report, via nflverse
# ----------------------------------------------------------------------
#: nflverse ``report_status`` values, which are the league's own wording.
#: ``normalize_status`` already understands them; this only names the column.
NFLVERSE_STATUS_COLUMN = "report_status"
NFLVERSE_PRACTICE_COLUMN = "practice_status"


def parse_nflverse_injuries(frame, *, week: int | None = None) -> list[InjuryRecord]:
    """Normalise nflverse's weekly injury report into :class:`InjuryRecord` rows.

    This is the league's own Wednesday-to-Friday report -- practice
    participation plus the Friday game status -- rather than a scrape of a
    site's summary. It carries the team and the position, which the
    reallocation model in :mod:`src.models.roles` needs to know *which*
    position group lost a player, and it is published per week going back
    through history, so the learning loop can see what the model would have
    known at the time.

    A row with no status is a player who appeared on the practice report but
    was not given a game designation, which means available.
    """
    import pandas as pd

    if frame is None or len(frame) == 0:
        return []
    rows = frame if isinstance(frame, pd.DataFrame) else frame.to_pandas()
    if week is not None and "week" in rows.columns:
        rows = rows[rows["week"] == week]

    records: list[InjuryRecord] = []
    for row in rows.to_dict("records"):
        name = row.get("full_name") or " ".join(
            part for part in (row.get("first_name"), row.get("last_name")) if part
        )
        if not name:
            continue
        status = normalize_status(row.get(NFLVERSE_STATUS_COLUMN))
        detail = row.get("report_primary_injury") or row.get("practice_primary_injury")
        records.append(
            InjuryRecord(
                sport="nfl",
                team=row.get("team"),
                player_name=str(name).strip(),
                position=row.get("position"),
                status=status,
                practice=row.get(NFLVERSE_PRACTICE_COLUMN),
                detail=str(detail) if detail else None,
                source="nflverse",
                report_date=None,
            )
        )
    return records


def load_nflverse_injuries(
    seasons: Iterable[int], *, week: int | None = None
) -> list[InjuryRecord]:
    """Fetch and normalise the league injury report for the given seasons."""
    try:
        import nflreadpy
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "nflreadpy is not installed; `uv pip install -e '.[stats]'`"
        ) from exc
    frame = nflreadpy.load_injuries(seasons=sorted(set(seasons))).to_pandas()
    return parse_nflverse_injuries(frame, week=week)


def latest_week(records: Iterable[InjuryRecord]) -> list[InjuryRecord]:
    """Only the most severe designation per player, for one slate."""
    worst: dict[str, InjuryRecord] = {}
    for record in records:
        seen = worst.get(record.player_name)
        if seen is None or STATUS_ORDER.index(record.status) < STATUS_ORDER.index(seen.status):
            worst[record.player_name] = record
    return list(worst.values())
