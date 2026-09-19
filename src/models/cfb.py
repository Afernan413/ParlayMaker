"""College football data, normalised into the shapes the NFL model already uses.

cfbfastR publishes ESPN-derived play-by-play to GitHub releases, which is the
only college source reachable without an API key. It has no pre-aggregated
player stat table, so the weekly frame is built here by grouping the plays --
the same thing nflverse does for the NFL.

Everything is renamed into the nflverse column names on the way out
(``player_display_name``, ``recent_team``, ``posteam``, ``epa``, ...), so
:mod:`src.models.baseline` projects college football without knowing it exists.
"""

from __future__ import annotations

import logging
from typing import BinaryIO, Iterable

import pandas as pd

logger = logging.getLogger(__name__)

PBP_RELEASE = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
    "/espn_cfb_pbp/play_by_play_{season}.parquet"
)

#: The columns the projection model cannot do without. A season's file missing
#: one of these is unusable.
PBP_REQUIRED_COLUMNS = [
    "season", "week", "game_id", "pos_team", "def_pos_team", "EPA",
    "pass", "rush", "pass_attempt", "completion", "pass_td", "rush_td",
    "passer_player_name", "rusher_player_name", "receiver_player_name",
    "yds_receiving", "yds_rushed",
]

#: Columns worth having when they are there. The release's schema is not stable
#: across seasons -- the final-score columns only appear from 2026, and earlier
#: files carry the running score per play instead -- so asking for these
#: unconditionally silently drops every earlier season.
PBP_OPTIONAL_COLUMNS = [
    "EPA_success",
    "homeTeamName", "awayTeamName",
    "homeFinalScore", "awayFinalScore",
    "end.homeScore", "end.awayScore", "homeScore", "awayScore",
]

PBP_COLUMNS = PBP_REQUIRED_COLUMNS + PBP_OPTIONAL_COLUMNS


def fetch_season(url: str) -> tuple[set[str], "BinaryIO"]:
    """One season's parquet: the columns it carries, and its bytes.

    Both come from one download. The schema has to be read before the frame,
    because it differs between seasons and ``read_parquet(columns=...)`` fails
    the whole read on the first name it cannot find -- but downloading a
    several-hundred-megabyte file twice to learn that is not worth it.
    """
    import io

    import pyarrow.parquet as pq
    import requests

    response = requests.get(url, timeout=300)
    response.raise_for_status()
    body = io.BytesIO(response.content)
    return set(pq.read_schema(body).names), body


def load_cfb_pbp(seasons: Iterable[int]) -> pd.DataFrame:
    """Raw ESPN college play-by-play for the given seasons.

    Each season is read with whichever of :data:`PBP_COLUMNS` it has. A season
    missing something in :data:`PBP_REQUIRED_COLUMNS` is skipped and said so;
    one merely missing an optional column is used as-is.
    """
    frames: list[pd.DataFrame] = []
    problems: list[str] = []
    for season in sorted(set(seasons)):
        url = PBP_RELEASE.format(season=season)
        try:
            available, body = fetch_season(url)
        except Exception as exc:  # not published yet, or no route to github.com
            problems.append(f"{season}: could not be read ({type(exc).__name__}: {exc})")
            logger.warning("college play-by-play unreadable for %s: %s", season, exc)
            continue

        missing = [name for name in PBP_REQUIRED_COLUMNS if name not in available]
        if missing:
            problems.append(f"{season}: missing {', '.join(missing)}")
            logger.warning(
                "college play-by-play for %s is missing %s; skipping that season",
                season, ", ".join(missing),
            )
            continue

        wanted = [name for name in PBP_COLUMNS if name in available]
        absent = [name for name in PBP_OPTIONAL_COLUMNS if name not in available]
        if absent:
            logger.info("%s has no %s; working without them", season, ", ".join(absent))
        frame = pd.read_parquet(body, columns=wanted)
        frames.append(frame)

    if not frames:
        raise RuntimeError(
            "no college play-by-play could be loaded -- "
            + "; ".join(problems or ["no seasons were requested"])
        )
    return pd.concat(frames, ignore_index=True)


def cfb_player_weekly(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-player, per-game volume, in nflverse's weekly column names.

    Built from three groupings of the same plays -- passing, rushing and
    receiving -- merged into one row per player-week.
    """
    keys = ["season", "week", "pos_team"]
    plays = pbp.copy()
    for column in ("pass", "rush", "completion", "pass_attempt", "pass_td", "rush_td"):
        if column in plays.columns:
            plays[column] = plays[column].fillna(False).astype(bool)
    for column in ("yds_receiving", "yds_rushed", "EPA"):
        if column in plays.columns:
            plays[column] = pd.to_numeric(plays[column], errors="coerce").fillna(0.0)

    passing = (
        plays[plays["passer_player_name"].notna()]
        .groupby([*keys, "passer_player_name"], dropna=False)
        .agg(
            passing_yards=("yds_receiving", "sum"),
            passing_tds=("pass_td", "sum"),
            attempts=("pass_attempt", "sum"),
        )
        .reset_index()
        .rename(columns={"passer_player_name": "player"})
    )

    rushing = (
        plays[plays["rusher_player_name"].notna()]
        .groupby([*keys, "rusher_player_name"], dropna=False)
        .agg(
            rushing_yards=("yds_rushed", "sum"),
            carries=("rush", "sum"),
            rushing_tds=("rush_td", "sum"),
        )
        .reset_index()
        .rename(columns={"rusher_player_name": "player"})
    )

    receiving = (
        plays[plays["receiver_player_name"].notna()]
        .groupby([*keys, "receiver_player_name"], dropna=False)
        .agg(
            receiving_yards=("yds_receiving", "sum"),
            receptions=("completion", "sum"),
            targets=("pass_attempt", "sum"),
        )
        .reset_index()
        .rename(columns={"receiver_player_name": "player"})
    )

    weekly = passing.merge(rushing, on=[*keys, "player"], how="outer")
    weekly = weekly.merge(receiving, on=[*keys, "player"], how="outer")
    weekly = weekly.fillna(0.0)

    weekly = weekly.rename(
        columns={"player": "player_display_name", "pos_team": "recent_team"}
    )
    # target_share is what the NFL usage model reaches for; derive it per team.
    team_targets = weekly.groupby(["season", "week", "recent_team"])["targets"].transform("sum")
    weekly["target_share"] = (weekly["targets"] / team_targets.replace(0, pd.NA)).fillna(0.0)
    return weekly


def cfb_pbp_normalised(pbp: pd.DataFrame) -> pd.DataFrame:
    """Play-by-play in nflverse's column names, for the efficiency model."""
    frame = pd.DataFrame(
        {
            "season": pbp["season"],
            "week": pbp["week"],
            "posteam": pbp["pos_team"],
            "defteam": pbp["def_pos_team"],
            "epa": pd.to_numeric(pbp["EPA"], errors="coerce"),
        }
    )
    frame["play_type"] = "other"
    frame.loc[pbp["pass"].fillna(False).astype(bool), "play_type"] = "pass"
    frame.loc[pbp["rush"].fillna(False).astype(bool), "play_type"] = "run"
    # EPA_success is one of the optional columns; without it, a play succeeded
    # if it gained expected points.
    from_epa = (frame["epa"] > 0).astype(float)
    if "EPA_success" in pbp.columns:
        frame["success"] = pd.to_numeric(pbp["EPA_success"], errors="coerce").fillna(from_epa)
    else:
        frame["success"] = from_epa
    return frame.dropna(subset=["posteam", "defteam", "epa"])


def load_cfb_frames(seasons: Iterable[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(weekly, pbp)`` for college football, matching ``load_nfl_frames``."""
    raw = load_cfb_pbp(seasons)
    return cfb_player_weekly(raw), cfb_pbp_normalised(raw)


#: Where a final score can be found, best first. Only the 2026 release carries
#: an explicit final; earlier files carry the running score, whose maximum over
#: a game is the same number.
SCORE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("homeFinalScore", "awayFinalScore"),
    ("end.homeScore", "end.awayScore"),
    ("homeScore", "awayScore"),
)


def score_columns(frame: pd.DataFrame) -> tuple[str, str] | None:
    """The best pair of score columns this frame has, or ``None``."""
    for home, away in SCORE_COLUMNS:
        if home in frame.columns and away in frame.columns:
            return home, away
    return None


def cfb_results(pbp: pd.DataFrame) -> pd.DataFrame:
    """Final scores per game, for grading past predictions.

    Returns an empty frame rather than raising when the release carries no
    score columns at all: grading is a nice-to-have, and the projection model
    does not depend on it.
    """
    columns = score_columns(pbp)
    if columns is None or "homeTeamName" not in pbp.columns:
        logger.warning("college play-by-play carries no final scores; cannot grade games")
        return pd.DataFrame(
            columns=["game_id", "season", "week", "home_team", "away_team",
                     "home_score", "away_score"]
        )
    home_column, away_column = columns

    games = (
        pbp.dropna(subset=["game_id"])
        .groupby("game_id")
        .agg(
            season=("season", "first"),
            week=("week", "first"),
            home_team=("homeTeamName", "first"),
            away_team=("awayTeamName", "first"),
            home_score=(home_column, "max"),
            away_score=(away_column, "max"),
        )
        .reset_index()
    )
    for column in ("home_score", "away_score"):
        games[column] = pd.to_numeric(games[column], errors="coerce")
    return games.dropna(subset=["home_score", "away_score"])


def normalise_team(name: str | None) -> str:
    """Loose key for matching an odds-feed school name to a stats one.

    Feeds disagree on mascots and abbreviations ("USC Trojans" against
    "Southern California"), so comparison happens on a reduced form rather
    than the raw string.
    """
    if not name:
        return ""
    text = str(name).lower()
    # Dropped outright: "St. John's" and "St Johns" have to agree.
    for gone in (".", "'", "\u2019"):
        text = text.replace(gone, "")
    # Turned into a boundary: "Texas A&M" and "Texas A M" have to agree.
    for spacer in ("&", "-", "(", ")", "/"):
        text = text.replace(spacer, " ")
    return " ".join(text.split())
