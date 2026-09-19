"""Deterministic baseline projections (no LLM involved).

NFL inputs come from ``nfl_data_py`` (weekly box scores + play-by-play), NBA
inputs from ``nba_api``. Both libraries are optional extras and imported
lazily, so this module -- and its tests -- work with synthetic frames offline.

The projection recipe is deliberately transparent:

    mean = weighted_4wk_volume x opponent_factor x availability_factor

Opponent factors are derived from defensive efficiency z-scores and clipped, so
one extreme defence cannot swing a projection by more than
``OPPONENT_CLIP``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from config.settings import settings
from src.ingestion.injuries import status_multiplier
from src.models.calibration import calibration_for
from src.models.legs import Projection

logger = logging.getLogger(__name__)

#: NFL weekly stat column -> Odds API market key.
NFL_STAT_MARKETS: dict[str, str] = {
    "passing_yards": "player_pass_yds",
    "passing_tds": "player_pass_tds",
    "rushing_yards": "player_rush_yds",
    "receiving_yards": "player_reception_yds",
    "receptions": "player_receptions",
}

#: NBA per-game stat -> Odds API market key.
NBA_STAT_MARKETS: dict[str, str] = {
    "PTS": "player_points",
    "REB": "player_rebounds",
    "AST": "player_assists",
    "FG3M": "player_threes",
}

#: Which side of the ball a market's volume depends on.
NFL_MARKET_UNIT: dict[str, str] = {
    "player_pass_yds": "pass",
    "player_pass_tds": "pass",
    "player_rush_yds": "rush",
    "player_reception_yds": "pass",
    "player_receptions": "pass",
}

#: NBA club name (as The Odds API reports it) -> abbreviation used by nba_api.
NBA_TEAM_ALIASES: dict[str, str] = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}


def team_lookup(sport: str) -> dict[str, str]:
    """Club name -> abbreviation map for the sport's stat feed.

    College football has hundreds of schools and no stable abbreviation set, so
    it uses no lookup at all -- :func:`same_team` compares reduced names there.
    """
    from src.ingestion.weather import TEAM_ALIASES as NFL_TEAM_ALIASES

    sport = sport.lower()
    if sport == "nfl":
        return dict(NFL_TEAM_ALIASES)
    if sport == "ncaaf":
        return {}
    return dict(NBA_TEAM_ALIASES)


def resolve_in_index(index, name: str, sport: str) -> str | None:
    """Find ``name`` in a team-indexed frame, tolerating naming differences."""
    if name in index:
        return name
    return next((candidate for candidate in index if same_team(candidate, name, sport)), None)


def resolve_team(name: str | None, sport: str) -> str:
    """Normalise a club name or abbreviation to the stat feed's abbreviation."""
    if not name:
        return ""
    candidate = name.strip()
    return team_lookup(sport).get(candidate, candidate).upper()


def same_team(first: str | None, second: str | None, sport: str) -> bool:
    """Do two team references point at the same club?

    Feeds disagree on naming -- The Odds API says ``Buffalo Bills`` while
    nflverse and ESPN say ``BUF`` -- so both sides are resolved first. College
    names carry mascots inconsistently ("USC Trojans" against "USC"), so they
    are compared on a reduced form and by containment.
    """
    if sport.lower() == "ncaaf":
        from src.models.cfb import normalise_team

        left, right = normalise_team(first), normalise_team(second)
        if not left or not right:
            return False
        return left == right or left.startswith(right) or right.startswith(left)

    left, right = resolve_team(first, sport), resolve_team(second, sport)
    return bool(left) and left == right


LEAGUE_AVG_PPG_NFL = 22.5
NFL_PLAYS_PER_GAME = 63.0
#: Standard deviation of a final score / margin, per code.
NFL_SCORE_SD = 13.2
COLLEGE_SCORE_SD = 16.5
#: College scoring baseline: more possessions and wider talent gaps.
LEAGUE_AVG_PPG_NCAAF = 27.5
OPPONENT_BETA = 0.35  # how strongly a 1-sd defence moves a projection
OPPONENT_CLIP = 0.15  # +/- ceiling on the opponent factor


def score_spread(sport: str, market: str) -> float:
    """How far a final score lands from its projection, in points.

    The constants above are priors. Once ``src.learning.train`` has measured
    the real residual around a closing line, that measurement is used instead
    -- separately for the total and the margin, which are not equally noisy.
    """
    fallback = COLLEGE_SCORE_SD if sport == "ncaaf" else NFL_SCORE_SD
    fitted = calibration_for(sport, market)
    if fitted is not None and fitted.dispersion:
        return float(fitted.dispersion)
    return fallback


# ----------------------------------------------------------------------
# generic helpers
# ----------------------------------------------------------------------
def rolling_weighted_mean(
    values: Sequence[float], weights: Sequence[float] | None = None
) -> float:
    """Recency-weighted mean of ``values`` (most recent first).

    Shorter samples reuse the leading weights and renormalise, so a player
    with two games played is not penalised for missing history.
    """
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not clean:
        return 0.0
    weights = list(weights or settings.rolling_weight_list)
    used = weights[: len(clean)] or [1.0] * len(clean)
    if len(used) < len(clean):
        clean = clean[: len(used)]
    total = sum(used)
    if total <= 0:
        return float(np.mean(clean))
    return float(sum(v * w for v, w in zip(clean, used)) / total)


def _recent_weeks(frame: pd.DataFrame, weeks: int) -> pd.DataFrame:
    """Most recent ``weeks`` rows, newest first.

    Sorted by season as well as week, so a window that reaches back over a
    season boundary still orders week 18 of last year before week 1 of this.
    """
    keys = [key for key in ("season", "week") if key in frame.columns]
    if keys:
        return frame.sort_values(keys, ascending=False).head(weeks)
    return frame.tail(weeks).iloc[::-1]


def opponent_factor(
    value: float, league_mean: float, league_sd: float, *, beta: float = OPPONENT_BETA
) -> float:
    """Clipped multiplier from a defensive efficiency reading.

    ``value`` is EPA allowed per play (NFL) or defensive rating (NBA): higher
    means a weaker defence, so the factor rises above 1.0.
    """
    if league_sd is None or league_sd <= 0 or pd.isna(league_sd):
        return 1.0
    z = (value - league_mean) / league_sd
    return float(np.clip(1.0 + beta * z, 1.0 - OPPONENT_CLIP, 1.0 + OPPONENT_CLIP))


@dataclass(frozen=True)
class GameProjection:
    """Model view of a game's scoring environment."""

    game_id: str
    sport: str
    home_team: str
    away_team: str
    total_mean: float
    total_sd: float
    home_margin_mean: float
    margin_sd: float

    @property
    def home_points(self) -> float:
        return (self.total_mean + self.home_margin_mean) / 2.0

    @property
    def away_points(self) -> float:
        return (self.total_mean - self.home_margin_mean) / 2.0


# ----------------------------------------------------------------------
# NFL
# ----------------------------------------------------------------------
def nfl_team_efficiency(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-team EPA/play, success rates and pass rate from play-by-play.

    Expects the ``nfl_data_py`` play-by-play columns ``posteam``, ``defteam``,
    ``epa``, ``success``, ``play_type``.
    """
    plays = pbp.dropna(subset=["posteam", "defteam", "epa"]).copy()
    if plays.empty:
        return pd.DataFrame(
            columns=[
                "off_epa", "def_epa_allowed", "pass_rate",
                "pass_success", "rush_success", "plays",
            ]
        )
    plays["is_pass"] = plays.get("play_type", "").eq("pass")
    plays["is_rush"] = plays.get("play_type", "").eq("run")
    if "success" not in plays.columns:
        plays["success"] = (plays["epa"] > 0).astype(float)

    offense = plays.groupby("posteam").agg(
        off_epa=("epa", "mean"),
        pass_rate=("is_pass", "mean"),
        plays=("epa", "size"),
    )
    pass_success = (
        plays[plays["is_pass"]].groupby("posteam")["success"].mean().rename("pass_success")
    )
    rush_success = (
        plays[plays["is_rush"]].groupby("posteam")["success"].mean().rename("rush_success")
    )
    defense = plays.groupby("defteam")["epa"].mean().rename("def_epa_allowed")

    table = offense.join([pass_success, rush_success]).join(defense, how="outer")
    return table.fillna({"pass_success": 0.0, "rush_success": 0.0})


def nfl_player_volume(
    weekly: pd.DataFrame, weeks: int | None = None
) -> pd.DataFrame:
    """Recency-weighted per-game volume and usage for every player.

    Expects ``nfl_data_py.import_weekly_data`` columns; missing optional
    columns (``target_share``, ``air_yards_share``) are tolerated.
    """
    weeks = weeks or settings.rolling_weeks
    stat_cols = [c for c in NFL_STAT_MARKETS if c in weekly.columns]
    usage_cols = [
        c for c in ("targets", "carries", "attempts", "target_share", "air_yards_share")
        if c in weekly.columns
    ]
    name_col = "player_display_name" if "player_display_name" in weekly.columns else "player_name"
    team_col = "recent_team" if "recent_team" in weekly.columns else "team"

    records: list[dict[str, Any]] = []
    for (player, team), group in weekly.groupby([name_col, team_col]):
        recent = _recent_weeks(group, weeks)
        row: dict[str, Any] = {
            "player_name": player,
            "team": team,
            "games": int(len(recent)),
        }
        for col in (*stat_cols, *usage_cols):
            row[col] = rolling_weighted_mean(recent[col].tolist())
        records.append(row)
    return pd.DataFrame(records)


def build_nfl_projections(
    weekly: pd.DataFrame,
    pbp: pd.DataFrame,
    games: Sequence[Mapping[str, Any]],
    *,
    team_lookup: Mapping[str, str] | None = None,
    injury_index: Mapping[str, str] | None = None,
    markets: Sequence[str] | None = None,
    sport: str = "nfl",
) -> list[Projection]:
    """Baseline football player projections for a slate.

    Used for both NFL and college football: the college loader normalises its
    play-by-play into these same columns, so the model does not need to know
    which code it is looking at.

    ``games`` rows need ``game_id``/``home_team``/``away_team``; team names are
    resolved to the abbreviations used by ``weekly`` via ``team_lookup``.
    """
    volume = nfl_player_volume(weekly)
    efficiency = nfl_team_efficiency(pbp)
    injury_index = injury_index or {}
    wanted = set(markets or NFL_STAT_MARKETS.values())

    def_mean = float(efficiency["def_epa_allowed"].mean()) if not efficiency.empty else 0.0
    def_sd = float(efficiency["def_epa_allowed"].std(ddof=0)) if len(efficiency) > 1 else 0.0

    # College has no stable abbreviation table, so the slate's school name has
    # to be matched against the stat feed's tolerantly. Resolving once per
    # distinct name keeps it out of the inner loop.
    known_teams = list(volume["team"].dropna().unique()) if not volume.empty else []
    resolved: dict[str, str | None] = {}

    def resolve(name: str) -> str | None:
        if name not in resolved:
            match = next(
                (candidate for candidate in known_teams if same_team(candidate, name, sport)),
                None,
            )
            resolved[name] = match
        return resolved[name]

    projections: list[Projection] = []
    for game in games:
        matchups = _matchups(game, team_lookup)
        for team, opponent in matchups:
            stat_team = resolve(team)
            if stat_team is None:
                continue
            side = volume[volume["team"] == stat_team]
            stat_opponent = resolve_in_index(efficiency.index, opponent, sport)
            opp_def = (
                float(efficiency.loc[stat_opponent, "def_epa_allowed"])
                if stat_opponent is not None
                else def_mean
            )
            factor = opponent_factor(opp_def, def_mean, def_sd)
            for _, row in side.iterrows():
                availability = status_multiplier(injury_index.get(row["player_name"]))
                if availability <= 0:
                    continue
                for stat, market in NFL_STAT_MARKETS.items():
                    if market not in wanted or stat not in row:
                        continue
                    base = float(row[stat])
                    if base <= 0:
                        continue
                    projections.append(
                        Projection(
                            sport=sport,
                            game_id=str(game["game_id"]),
                            player_name=row["player_name"],
                            team=team,
                            opponent=opponent,
                            market=market,
                            mean=base * factor * availability,
                            notes=(
                                f"4wk weighted {stat}={base:.2f}",
                                f"opp factor={factor:.3f}",
                                f"availability={availability:.2f}",
                            ),
                        )
                    )
    return projections


def project_nfl_game(
    game: Mapping[str, Any],
    efficiency: pd.DataFrame,
    *,
    team_lookup: Mapping[str, str] | None = None,
    sport: str = "nfl",
) -> GameProjection | None:
    """Total/margin projection from EPA differentials.

    College games are higher scoring and far more variable than the NFL, so the
    baseline and the spread of outcomes both widen.
    """
    matchups = _matchups(game, team_lookup)
    if len(matchups) != 2:
        return None
    (home, away), (away_again, home_again) = matchups
    del away_again, home_again

    def points(team: str, opponent: str) -> float:
        own = resolve_in_index(efficiency.index, team, sport)
        against = resolve_in_index(efficiency.index, opponent, sport)
        off = float(efficiency.loc[own, "off_epa"]) if own is not None else 0.0
        opp_def = (
            float(efficiency.loc[against, "def_epa_allowed"])
            if against is not None
            else 0.0
        )
        edge = (off + opp_def) / 2.0
        baseline = LEAGUE_AVG_PPG_NCAAF if sport == "ncaaf" else LEAGUE_AVG_PPG_NFL
        return baseline + edge * NFL_PLAYS_PER_GAME

    home_points = points(home, away)
    away_points = points(away, home)
    # College games swing far wider than the NFL: bigger talent gaps, more
    # possessions, less roster parity.
    return GameProjection(
        game_id=str(game["game_id"]),
        sport=sport,
        home_team=home,
        away_team=away,
        total_mean=home_points + away_points,
        total_sd=score_spread(sport, "totals"),
        home_margin_mean=home_points - away_points + 1.4,  # home-field advantage
        margin_sd=score_spread(sport, "spreads"),
    )


# ----------------------------------------------------------------------
# NBA
# ----------------------------------------------------------------------
def nba_player_rates(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Per-minute production and usage from a league dashboard frame.

    Expects ``PLAYER_NAME``, ``TEAM_ABBREVIATION``, ``MIN`` and the counting
    stats in :data:`NBA_STAT_MARKETS`; ``USG_PCT`` is used when present.
    """
    frame = player_stats.copy()
    minutes = frame["MIN"].replace(0, np.nan)
    for stat in NBA_STAT_MARKETS:
        if stat in frame.columns:
            frame[f"{stat}_per_min"] = frame[stat] / minutes
    frame["projected_minutes"] = frame["MIN"]
    if "USG_PCT" not in frame.columns:
        frame["USG_PCT"] = np.nan
    return frame


def nba_team_context(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Pace and efficiency indexed by team abbreviation."""
    frame = team_stats.copy()
    index = "TEAM_ABBREVIATION" if "TEAM_ABBREVIATION" in frame.columns else "TEAM_NAME"
    return frame.set_index(index)[
        [c for c in ("PACE", "OFF_RATING", "DEF_RATING") if c in frame.columns]
    ]


def build_nba_projections(
    player_stats: pd.DataFrame,
    team_stats: pd.DataFrame,
    games: Sequence[Mapping[str, Any]],
    *,
    team_lookup: Mapping[str, str] | None = None,
    injury_index: Mapping[str, str] | None = None,
    markets: Sequence[str] | None = None,
) -> list[Projection]:
    """Baseline NBA player projections: per-minute rate x minutes x pace x matchup."""
    rates = nba_player_rates(player_stats)
    context = nba_team_context(team_stats)
    injury_index = injury_index or {}
    wanted = set(markets or NBA_STAT_MARKETS.values())

    league_pace = float(context["PACE"].mean()) if "PACE" in context.columns else 100.0
    def_mean = float(context["DEF_RATING"].mean()) if "DEF_RATING" in context.columns else 112.0
    def_sd = (
        float(context["DEF_RATING"].std(ddof=0))
        if "DEF_RATING" in context.columns and len(context) > 1
        else 0.0
    )

    projections: list[Projection] = []
    for game in games:
        for team, opponent in _matchups(game, team_lookup):
            pace_factor = 1.0
            if "PACE" in context.columns and team in context.index and opponent in context.index:
                game_pace = (
                    float(context.loc[team, "PACE"]) + float(context.loc[opponent, "PACE"])
                ) / 2.0
                pace_factor = game_pace / league_pace if league_pace else 1.0
            opp_def = (
                float(context.loc[opponent, "DEF_RATING"])
                if "DEF_RATING" in context.columns and opponent in context.index
                else def_mean
            )
            matchup_factor = opponent_factor(opp_def, def_mean, def_sd)

            side = rates[rates["TEAM_ABBREVIATION"] == team]
            for _, row in side.iterrows():
                player = row["PLAYER_NAME"]
                availability = status_multiplier(injury_index.get(player))
                if availability <= 0:
                    continue
                minutes = float(row.get("projected_minutes") or 0.0)
                if minutes <= 0:
                    continue
                for stat, market in NBA_STAT_MARKETS.items():
                    per_min = row.get(f"{stat}_per_min")
                    if market not in wanted or per_min is None or pd.isna(per_min):
                        continue
                    mean = float(per_min) * minutes * pace_factor * matchup_factor * availability
                    if mean <= 0:
                        continue
                    projections.append(
                        Projection(
                            sport="nba",
                            game_id=str(game["game_id"]),
                            player_name=player,
                            team=team,
                            opponent=opponent,
                            market=market,
                            mean=mean,
                            notes=(
                                f"{stat}/min={float(per_min):.3f} x {minutes:.1f} min",
                                f"pace factor={pace_factor:.3f}",
                                f"matchup factor={matchup_factor:.3f}",
                            ),
                        )
                    )
    return projections


def project_nba_game(
    game: Mapping[str, Any],
    team_stats: pd.DataFrame,
    *,
    team_lookup: Mapping[str, str] | None = None,
) -> GameProjection | None:
    """Total/margin from possessions x efficiency."""
    context = nba_team_context(team_stats)
    matchups = _matchups(game, team_lookup)
    if len(matchups) != 2 or "OFF_RATING" not in context.columns:
        return None
    home, away = matchups[0]
    if home not in context.index or away not in context.index:
        return None

    pace = (
        (float(context.loc[home, "PACE"]) + float(context.loc[away, "PACE"])) / 2.0
        if "PACE" in context.columns
        else 100.0
    )
    def_col = "DEF_RATING" if "DEF_RATING" in context.columns else "OFF_RATING"

    def points(team: str, opponent: str) -> float:
        off = float(context.loc[team, "OFF_RATING"])
        opp_def = float(context.loc[opponent, def_col])
        return pace * ((off + opp_def) / 2.0) / 100.0

    home_points = points(home, away) + 1.1  # home court
    away_points = points(away, home)
    return GameProjection(
        game_id=str(game["game_id"]),
        sport="nba",
        home_team=home,
        away_team=away,
        total_mean=home_points + away_points,
        total_sd=12.5,
        home_margin_mean=home_points - away_points,
        margin_sd=12.5,
    )


def _matchups(
    game: Mapping[str, Any], team_lookup: Mapping[str, str] | None
) -> list[tuple[str, str]]:
    """``[(home, away), (away, home)]`` with names resolved to abbreviations."""
    lookup = team_lookup or {}
    home = lookup.get(game.get("home_team", ""), game.get("home_team", ""))
    away = lookup.get(game.get("away_team", ""), game.get("away_team", ""))
    if not home or not away:
        return []
    return [(home, away), (away, home)]


# ----------------------------------------------------------------------
# live data loaders (optional dependencies, imported lazily)
# ----------------------------------------------------------------------
def load_nfl_frames(seasons: Iterable[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(weekly, pbp)`` from nflverse. Requires the ``stats`` extra.

    Uses ``nflreadpy``, the maintained nflverse client. Its predecessor
    ``nfl_data_py`` still installs but 404s on every season after 2024, which
    fails as a plain HTTP error rather than anything that reads like "this
    library is out of date".
    """
    try:
        import nflreadpy
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "nflreadpy is not installed; `uv pip install -e '.[stats]'` or run --mock"
        ) from exc

    years = sorted(set(seasons))
    weekly = nflreadpy.load_player_stats(seasons=years).to_pandas()
    pbp = nflreadpy.load_pbp(seasons=years).to_pandas()
    return weekly, pbp


def seasons_to_load(season: int, *, weeks_needed: int | None = None) -> list[int]:
    """Which seasons to pull so the rolling window can be filled.

    Early in a season there are not yet four games to average, so the previous
    one is loaded as well and the window simply runs back across the boundary.
    """
    del weeks_needed  # the window itself decides how far back to reach
    return [season - 1, season]


def latest_season_plays(pbp: pd.DataFrame, *, min_plays: int = 1_000) -> pd.DataFrame:
    """Team efficiency from the current season once it has enough plays.

    Blending last season's EPA into this season's ratings would smear over
    roster and scheme turnover, so the prior season is only kept while the
    current one is too thin to measure.
    """
    if "season" not in pbp.columns or pbp.empty:
        return pbp
    latest = pbp[pbp["season"] == pbp["season"].max()]
    return latest if len(latest) >= min_plays else pbp


def load_nba_frames(season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(player_stats, team_stats)`` from ``nba_api``. Requires ``stats`` extra."""
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "nba_api is not installed; `uv pip install -e '.[stats]'` or run --mock"
        ) from exc
    players = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season, per_mode_detailed="PerGame", measure_type_detailed_defense="Base"
    ).get_data_frames()[0]
    teams = leaguedashteamstats.LeagueDashTeamStats(
        season=season, per_mode_detailed="PerGame", measure_type_detailed_defense="Advanced"
    ).get_data_frames()[0]
    return players, teams
