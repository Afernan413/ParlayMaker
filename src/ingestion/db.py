"""SQLite storage layer.

All DDL is idempotent (``CREATE TABLE IF NOT EXISTS``) so the module can be
imported and ``init_db()`` called on every run without migrations. Schema
changes belong in :data:`MIGRATIONS` as additional idempotent statements.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import settings

BET_LOG_SCHEMA = """
    CREATE TABLE IF NOT EXISTS bet_log (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id        TEXT NOT NULL,
        sport         TEXT NOT NULL,
        ticket_index  INTEGER NOT NULL,
        ticket_type   TEXT,
        ticket_odds   INTEGER,
        game_id       TEXT NOT NULL,
        market        TEXT NOT NULL,
        player_name   TEXT,
        selection     TEXT NOT NULL,
        line          REAL,
        taken_odds    INTEGER NOT NULL,
        p_model       REAL,
        p_implied     REAL,
        ev            REAL,
        stake         REAL,
        captured_at   TEXT NOT NULL,
        UNIQUE (run_id, ticket_index, market, player_name, selection, line)
    )
"""

SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS games (
        game_id       TEXT PRIMARY KEY,
        sport         TEXT NOT NULL,
        commence_time TEXT NOT NULL,
        home_team     TEXT NOT NULL,
        away_team     TEXT NOT NULL,
        season        INTEGER,
        week          INTEGER,
        updated_at    TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fanduel_lines (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id       TEXT NOT NULL REFERENCES games(game_id),
        market        TEXT NOT NULL,          -- h2h | spreads | totals
        selection     TEXT NOT NULL,          -- team name, Over, Under
        line          REAL,                   -- point/handicap (NULL for h2h)
        american_odds INTEGER NOT NULL,
        captured_at   TEXT NOT NULL,
        UNIQUE (game_id, market, selection, line, captured_at)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fanduel_props (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id       TEXT NOT NULL REFERENCES games(game_id),
        market        TEXT NOT NULL,          -- player_pass_yds, player_points, ...
        player_name   TEXT NOT NULL,
        selection     TEXT NOT NULL,          -- Over | Under | Yes | No
        line          REAL,
        american_odds INTEGER NOT NULL,
        captured_at   TEXT NOT NULL,
        UNIQUE (game_id, market, player_name, selection, line, captured_at)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS weather_snapshots (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id        TEXT NOT NULL REFERENCES games(game_id),
        stadium        TEXT,
        is_dome        INTEGER NOT NULL DEFAULT 0,
        temperature_f  REAL,
        wind_speed_mph REAL,
        wind_deg       REAL,
        precip_chance  REAL,
        conditions     TEXT,
        high_wind      INTEGER NOT NULL DEFAULT 0,
        freezing       INTEGER NOT NULL DEFAULT 0,
        forecast_for   TEXT,
        captured_at    TEXT NOT NULL,
        UNIQUE (game_id, forecast_for, captured_at)
    )
    """,
    # AccuWeather addresses a venue by an opaque location key, which costs a
    # call to look up. The keys are stable, so they are cached here: 30-odd rows
    # resolved once turn a two-call-per-venue forecast into one.
    """
    CREATE TABLE IF NOT EXISTS weather_locations (
        venue_key    TEXT PRIMARY KEY,       -- "lat,lon" rounded, provider-agnostic
        provider     TEXT NOT NULL,
        location_key TEXT NOT NULL,
        name         TEXT,
        resolved_at  TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS injury_reports (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        sport         TEXT NOT NULL,
        team          TEXT,
        player_name   TEXT NOT NULL,
        position      TEXT,
        status        TEXT NOT NULL,          -- OUT|DOUBTFUL|QUESTIONABLE|PROBABLE|ACTIVE
        practice      TEXT,
        detail        TEXT,
        source        TEXT,
        report_date   TEXT,
        captured_at   TEXT NOT NULL,
        UNIQUE (sport, player_name, report_date, captured_at)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS api_quota_log (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        api           TEXT NOT NULL,
        endpoint      TEXT NOT NULL,
        requests_used INTEGER,
        requests_remaining INTEGER,
        last_cost     INTEGER,
        status_code   INTEGER,
        captured_at   TEXT NOT NULL
    )
    """,
    # Recommendations are logged so their prices can later be benchmarked
    # against the closing line (see src/optimizer/clv.py).
    BET_LOG_SCHEMA,
    # Every leg the model priced, graded once the game is played. This is what
    # the learning loop trains on going forward: history from nflverse only
    # covers what the model *would* have said, this covers what it did say.
    """
    CREATE TABLE IF NOT EXISTS projection_log (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id        TEXT NOT NULL,
        sport         TEXT NOT NULL,
        game_id       TEXT NOT NULL,
        commence_time TEXT,
        season        INTEGER,
        week          INTEGER,
        player_name   TEXT,
        team          TEXT,
        market        TEXT NOT NULL,
        selection     TEXT NOT NULL,
        line          REAL,
        projected     REAL,
        p_model       REAL,
        p_implied     REAL,
        american_odds INTEGER,
        actual        REAL,
        hit           INTEGER,
        graded_at     TEXT,
        captured_at   TEXT NOT NULL,
        UNIQUE (run_id, game_id, market, player_name, selection, line)
    )
    """,
)

INDEXES: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_games_sport_time ON games(sport, commence_time)",
    "CREATE INDEX IF NOT EXISTS idx_lines_game_market ON fanduel_lines(game_id, market)",
    "CREATE INDEX IF NOT EXISTS idx_lines_captured ON fanduel_lines(captured_at)",
    "CREATE INDEX IF NOT EXISTS idx_props_game_market ON fanduel_props(game_id, market)",
    "CREATE INDEX IF NOT EXISTS idx_props_player ON fanduel_props(player_name)",
    "CREATE INDEX IF NOT EXISTS idx_weather_game ON weather_snapshots(game_id)",
    "CREATE INDEX IF NOT EXISTS idx_injury_player ON injury_reports(sport, player_name)",
    "CREATE INDEX IF NOT EXISTS idx_quota_api_time ON api_quota_log(api, captured_at)",
    "CREATE INDEX IF NOT EXISTS idx_betlog_run ON bet_log(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_betlog_market ON bet_log(sport, market, player_name)",
    "CREATE INDEX IF NOT EXISTS idx_projlog_pending ON projection_log(sport, graded_at)",
    "CREATE INDEX IF NOT EXISTS idx_projlog_market ON projection_log(sport, market, player_name)",
)

#: Additive, idempotent schema patches applied after :data:`SCHEMA`.
MIGRATIONS: tuple[str, ...] = ()


def utcnow() -> str:
    """ISO-8601 UTC timestamp used for every ``captured_at`` column."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(db_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a WAL-mode connection with row access by name.

    Commits on clean exit, rolls back on exception.
    """
    path = Path(db_path or settings.db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | Path | None = None) -> None:
    """Create every table, index and migration. Safe to call repeatedly."""
    with connect(db_path) as conn:
        for statement in (*SCHEMA, *INDEXES, *MIGRATIONS):
            conn.execute(statement)


def _insert(
    conn: sqlite3.Connection,
    table: str,
    rows: Sequence[dict[str, Any]],
    *,
    ignore_conflict: bool = True,
) -> int:
    if not rows:
        return 0
    columns = list(rows[0])
    verb = "INSERT OR IGNORE" if ignore_conflict else "INSERT"
    sql = (
        f"{verb} INTO {table} ({', '.join(columns)}) "
        f"VALUES ({', '.join('?' for _ in columns)})"
    )
    cur = conn.executemany(sql, [tuple(row[col] for col in columns) for row in rows])
    return cur.rowcount or 0


def upsert_games(rows: Iterable[dict[str, Any]], db_path=None) -> int:
    """Insert-or-replace schedule rows keyed on ``game_id``."""
    rows = [dict(r) for r in rows]
    if not rows:
        return 0
    for row in rows:
        row.setdefault("updated_at", utcnow())
        row.setdefault("season", None)
        row.setdefault("week", None)
    with connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO games (game_id, sport, commence_time, home_team, away_team,
                               season, week, updated_at)
            VALUES (:game_id, :sport, :commence_time, :home_team, :away_team,
                    :season, :week, :updated_at)
            ON CONFLICT(game_id) DO UPDATE SET
                commence_time = excluded.commence_time,
                home_team     = excluded.home_team,
                away_team     = excluded.away_team,
                season        = COALESCE(excluded.season, games.season),
                week          = COALESCE(excluded.week, games.week),
                updated_at    = excluded.updated_at
            """,
            rows,
        )
    return len(rows)


def insert_lines(rows: Iterable[dict[str, Any]], db_path=None) -> int:
    return _bulk("fanduel_lines", rows, db_path)


def insert_props(rows: Iterable[dict[str, Any]], db_path=None) -> int:
    return _bulk("fanduel_props", rows, db_path)


def insert_weather(rows: Iterable[dict[str, Any]], db_path=None) -> int:
    return _bulk("weather_snapshots", rows, db_path)


def insert_injuries(rows: Iterable[dict[str, Any]], db_path=None) -> int:
    return _bulk("injury_reports", rows, db_path)


def cached_location_key(venue_key: str, provider: str, db_path=None) -> str | None:
    """A forecast provider's key for a venue, if it has been looked up before."""
    rows = fetch_all(
        "SELECT location_key FROM weather_locations WHERE venue_key = ? AND provider = ?",
        (venue_key, provider),
        db_path=db_path,
    )
    return rows[0]["location_key"] if rows else None


def store_location_key(
    venue_key: str, provider: str, location_key: str, name: str | None = None, db_path=None
) -> None:
    """Remember a venue's provider key so it is never looked up twice."""
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO weather_locations "
            "(venue_key, provider, location_key, name, resolved_at) VALUES (?, ?, ?, ?, ?)",
            (venue_key, provider, location_key, name, utcnow()),
        )


def insert_projections(rows: Iterable[dict[str, Any]], db_path=None) -> int:
    """Record priced legs for later grading. Re-running a run is a no-op."""
    return _bulk("projection_log", rows, db_path)


def ungraded_projections(
    sport: str | None = None, before: str | None = None, db_path=None
) -> list[dict[str, Any]]:
    """Logged legs that have no result yet.

    ``before`` limits it to games that have already kicked off, so a slate
    still hours away is not reported as missing its grades.
    """
    clauses = ["graded_at IS NULL"]
    params: list[Any] = []
    if sport:
        clauses.append("sport = ?")
        params.append(sport)
    if before:
        clauses.append("(commence_time IS NULL OR commence_time < ?)")
        params.append(before)
    sql = f"SELECT * FROM projection_log WHERE {' AND '.join(clauses)} ORDER BY id"
    return fetch_all(sql, params, db_path=db_path)


def grade_projections(grades: Iterable[tuple[int, float, int]], db_path=None) -> int:
    """Attach ``(row id, actual, hit)`` results to logged legs."""
    rows = [(float(actual), int(hit), utcnow(), int(row_id)) for row_id, actual, hit in grades]
    if not rows:
        return 0
    with connect(db_path) as conn:
        cur = conn.executemany(
            "UPDATE projection_log SET actual = ?, hit = ?, graded_at = ? WHERE id = ?", rows
        )
        return cur.rowcount or 0


def graded_projections(sport: str | None = None, db_path=None) -> list[dict[str, Any]]:
    """Every logged leg that has a result, oldest first."""
    sql = "SELECT * FROM projection_log WHERE graded_at IS NOT NULL"
    params: list[Any] = []
    if sport:
        sql += " AND sport = ?"
        params.append(sport)
    return fetch_all(sql + " ORDER BY id", params, db_path=db_path)


def insert_bets(rows: Iterable[dict[str, Any]], db_path=None) -> int:
    return _bulk("bet_log", rows, db_path)


def log_quota(
    api: str,
    endpoint: str,
    *,
    requests_used: int | None = None,
    requests_remaining: int | None = None,
    last_cost: int | None = None,
    status_code: int | None = None,
    db_path=None,
) -> None:
    """Record one API call's quota headers."""
    _bulk(
        "api_quota_log",
        [
            {
                "api": api,
                "endpoint": endpoint,
                "requests_used": requests_used,
                "requests_remaining": requests_remaining,
                "last_cost": last_cost,
                "status_code": status_code,
                "captured_at": utcnow(),
            }
        ],
        db_path,
    )


def latest_quota(api: str = "odds_api", db_path=None) -> dict[str, Any] | None:
    """Most recent quota row for ``api``, or ``None`` if nothing logged yet."""
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM api_quota_log
            WHERE api = ? AND requests_remaining IS NOT NULL
            ORDER BY id DESC LIMIT 1
            """,
            (api,),
        ).fetchone()
    return dict(row) if row else None


def _bulk(table: str, rows: Iterable[dict[str, Any]], db_path) -> int:
    rows = [dict(r) for r in rows]
    for row in rows:
        row.setdefault("captured_at", utcnow())
    with connect(db_path) as conn:
        return _insert(conn, table, rows)


def fetch_all(sql: str, params: Sequence[Any] = (), db_path=None) -> list[dict[str, Any]]:
    """Run a read-only query and return plain dicts."""
    with connect(db_path) as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def latest_props(game_id: str, db_path=None) -> list[dict[str, Any]]:
    """Newest capture of every (market, player, selection, line) for a game."""
    return fetch_all(
        """
        SELECT p.* FROM fanduel_props p
        JOIN (
            SELECT market, player_name, selection, line, MAX(captured_at) AS ts
            FROM fanduel_props WHERE game_id = ?
            GROUP BY market, player_name, selection, line
        ) latest
          ON p.market = latest.market
         AND p.player_name = latest.player_name
         AND p.selection = latest.selection
         AND IFNULL(p.line, -999) = IFNULL(latest.line, -999)
         AND p.captured_at = latest.ts
        WHERE p.game_id = ?
        """,
        (game_id, game_id),
        db_path=db_path,
    )


def latest_lines(game_id: str, db_path=None) -> list[dict[str, Any]]:
    """Newest capture of every (market, selection, line) for a game."""
    return fetch_all(
        """
        SELECT l.* FROM fanduel_lines l
        JOIN (
            SELECT market, selection, line, MAX(captured_at) AS ts
            FROM fanduel_lines WHERE game_id = ?
            GROUP BY market, selection, line
        ) latest
          ON l.market = latest.market
         AND l.selection = latest.selection
         AND IFNULL(l.line, -999) = IFNULL(latest.line, -999)
         AND l.captured_at = latest.ts
        WHERE l.game_id = ?
        """,
        (game_id, game_id),
        db_path=db_path,
    )
