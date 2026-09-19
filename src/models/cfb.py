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
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

PBP_RELEASE = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
    "/espn_cfb_pbp/play_by_play_{season}.parquet"
)

#: Only the columns the model needs; the release carries 500+.
PBP_COLUMNS = [
    "season", "week", "game_id", "pos_team", "def_pos_team", "EPA", "EPA_success",
    "pass", "rush", "pass_attempt", "completion", "pass_td", "rush_td",
    "passer_player_name", "rusher_player_name", "receiver_player_name",
    "yds_receiving", "yds_rushed",
    "homeTeamName", "awayTeamName", "homeFinalScore", "awayFinalScore",
]


def load_cfb_pbp(seasons: Iterable[int]) -> pd.DataFrame:
    """Raw ESPN college play-by-play for the given seasons."""
    frames = []
    for season in sorted(set(seasons)):
        url = PBP_RELEASE.format(season=season)
        try:
            frame = pd.read_parquet(url, columns=PBP_COLUMNS)
        except Exception as exc:  # a season that has not been published yet
            logger.warning("college play-by-play unavailable for %s: %s", season, exc)
            continue
        frames.append(frame)
    if not frames:
        raise RuntimeError(
            "no college play-by-play could be loaded; the release may be "
            "rebuilding, or this machine cannot reach github.com"
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
    success = pd.to_numeric(pbp.get("EPA_success"), errors="coerce")
    frame["success"] = success.fillna((frame["epa"] > 0).astype(float))
    return frame.dropna(subset=["posteam", "defteam", "epa"])


def load_cfb_frames(seasons: Iterable[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(weekly, pbp)`` for college football, matching ``load_nfl_frames``."""
    raw = load_cfb_pbp(seasons)
    return cfb_player_weekly(raw), cfb_pbp_normalised(raw)


def cfb_results(pbp: pd.DataFrame) -> pd.DataFrame:
    """Final scores per game, for grading past predictions."""
    games = (
        pbp.dropna(subset=["game_id"])
        .groupby("game_id")
        .agg(
            season=("season", "first"),
            week=("week", "first"),
            home_team=("homeTeamName", "first"),
            away_team=("awayTeamName", "first"),
            home_score=("homeFinalScore", "max"),
            away_score=("awayFinalScore", "max"),
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
    for noise in (" state", "&", ".", "'", "-"):
        text = text.replace(noise, " state " if noise == " state" else " ")
    return " ".join(text.split())
