"""Baseline projection tests. Synthetic frames only -- no nfl_data_py/nba_api."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import baseline
from src.models.baseline import (
    build_nba_projections,
    build_nfl_projections,
    nfl_player_volume,
    nfl_team_efficiency,
    opponent_factor,
    project_nba_game,
    project_nfl_game,
    rolling_weighted_mean,
)


@pytest.fixture
def weekly() -> pd.DataFrame:
    rows = []
    for week, yards in ((1, 270.0), (2, 280.0), (3, 290.0), (4, 300.0)):
        rows.append({
            "player_display_name": "QB One", "recent_team": "KC", "week": week,
            "passing_yards": yards, "passing_tds": 2.0, "attempts": 34.0,
            "rushing_yards": 8.0, "receiving_yards": 0.0, "receptions": 0.0,
            "targets": 0.0, "carries": 3.0,
        })
        rows.append({
            "player_display_name": "WR One", "recent_team": "BUF", "week": week,
            "passing_yards": 0.0, "passing_tds": 0.0, "attempts": 0.0,
            "rushing_yards": 0.0, "receiving_yards": 60.0 + 5 * week,
            "receptions": 5.0, "targets": 9.0, "carries": 0.0, "target_share": 0.26,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def pbp() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({
        "posteam": ["KC"] * 60 + ["BUF"] * 60,
        "defteam": ["BUF"] * 60 + ["KC"] * 60,
        "epa": np.concatenate([rng.normal(0.15, 0.4, 60), rng.normal(-0.05, 0.4, 60)]),
        "play_type": (["pass"] * 36 + ["run"] * 24) * 2,
    })
    frame["success"] = (frame["epa"] > 0).astype(float)
    return frame


NFL_GAMES = [{"game_id": "g1", "home_team": "KC", "away_team": "BUF"}]


# ------------------------------------------------------------------ helpers
def test_rolling_mean_weights_recent_games_more():
    recent_hot = rolling_weighted_mean([300, 200, 200, 200])
    recent_cold = rolling_weighted_mean([200, 300, 300, 300])
    assert recent_hot > np.mean([300, 200, 200, 200])
    assert recent_cold < np.mean([200, 300, 300, 300])


def test_rolling_mean_renormalises_short_samples():
    assert rolling_weighted_mean([100, 100]) == pytest.approx(100.0)
    assert rolling_weighted_mean([]) == 0.0
    assert rolling_weighted_mean([np.nan, 50.0]) == pytest.approx(50.0)


def test_opponent_factor_is_clipped_both_ways():
    assert opponent_factor(0.5, 0.0, 0.05) == pytest.approx(1.15)
    assert opponent_factor(-0.5, 0.0, 0.05) == pytest.approx(0.85)
    assert opponent_factor(0.1, 0.1, 0.05) == pytest.approx(1.0)
    assert opponent_factor(0.1, 0.0, 0.0) == 1.0  # no spread -> no adjustment


# --------------------------------------------------------------------- NFL
def test_team_efficiency_splits_offence_and_defence(pbp):
    table = nfl_team_efficiency(pbp)
    assert table.loc["KC", "off_epa"] > table.loc["BUF", "off_epa"]
    # KC's offence is what BUF's defence allowed
    assert table.loc["BUF", "def_epa_allowed"] == pytest.approx(table.loc["KC", "off_epa"])
    assert 0.0 <= table.loc["KC", "pass_success"] <= 1.0
    assert table.loc["KC", "pass_rate"] == pytest.approx(0.6)


def test_team_efficiency_handles_an_empty_frame():
    assert nfl_team_efficiency(pd.DataFrame(columns=["posteam", "defteam", "epa"])).empty


def test_player_volume_is_recency_weighted(weekly):
    volume = nfl_player_volume(weekly)
    qb = volume[volume["player_name"] == "QB One"].iloc[0]
    assert qb["passing_yards"] == pytest.approx(290.0)  # .4*300+.3*290+.2*280+.1*270
    assert qb["games"] == 4


def test_nfl_projections_cover_every_market(weekly, pbp):
    projections = build_nfl_projections(weekly, pbp, NFL_GAMES)
    markets = {p.market for p in projections}
    assert markets == {
        "player_pass_yds", "player_pass_tds", "player_rush_yds",
        "player_reception_yds", "player_receptions",
    }
    assert {p.game_id for p in projections} == {"g1"}
    qb = next(p for p in projections if p.market == "player_pass_yds")
    assert qb.team == "KC" and qb.opponent == "BUF"
    assert qb.mean > 0 and any("opp factor" in note for note in qb.notes)


def test_ruled_out_players_get_no_projections(weekly, pbp):
    projections = build_nfl_projections(
        weekly, pbp, NFL_GAMES, injury_index={"QB One": "OUT"}
    )
    assert not [p for p in projections if p.player_name == "QB One"]
    assert [p for p in projections if p.player_name == "WR One"]


def test_questionable_status_trims_the_projection(weekly, pbp):
    healthy = build_nfl_projections(weekly, pbp, NFL_GAMES)
    limited = build_nfl_projections(
        weekly, pbp, NFL_GAMES, injury_index={"WR One": "QUESTIONABLE"}
    )

    def mean_for(projections):
        return next(
            p.mean for p in projections
            if p.player_name == "WR One" and p.market == "player_reception_yds"
        )

    assert mean_for(limited) < mean_for(healthy)


def test_team_lookup_resolves_full_club_names(weekly, pbp):
    games = [{"game_id": "g1", "home_team": "Kansas City Chiefs",
              "away_team": "Buffalo Bills"}]
    lookup = {"Kansas City Chiefs": "KC", "Buffalo Bills": "BUF"}
    assert build_nfl_projections(weekly, pbp, games, team_lookup=lookup)
    assert not build_nfl_projections(weekly, pbp, games)  # unresolved names


def test_nfl_game_projection_favours_the_better_offence(weekly, pbp):
    projection = project_nfl_game(NFL_GAMES[0], nfl_team_efficiency(pbp))
    assert projection.home_margin_mean > 0  # KC is the stronger side
    assert projection.total_mean == pytest.approx(
        projection.home_points + projection.away_points
    )


def test_market_filter_restricts_projections(weekly, pbp):
    projections = build_nfl_projections(
        weekly, pbp, NFL_GAMES, markets=["player_receptions"]
    )
    assert {p.market for p in projections} == {"player_receptions"}


# --------------------------------------------------------------------- NBA
@pytest.fixture
def nba_players() -> pd.DataFrame:
    return pd.DataFrame([
        {"PLAYER_NAME": "Guard A", "TEAM_ABBREVIATION": "BOS", "MIN": 34.0,
         "PTS": 27.0, "REB": 5.0, "AST": 6.0, "FG3M": 3.4, "USG_PCT": 0.31},
        {"PLAYER_NAME": "Wing B", "TEAM_ABBREVIATION": "LAL", "MIN": 30.0,
         "PTS": 18.0, "REB": 7.0, "AST": 3.0, "FG3M": 1.8, "USG_PCT": 0.24},
        {"PLAYER_NAME": "Bench C", "TEAM_ABBREVIATION": "LAL", "MIN": 0.0,
         "PTS": 0.0, "REB": 0.0, "AST": 0.0, "FG3M": 0.0, "USG_PCT": 0.0},
    ])


@pytest.fixture
def nba_teams() -> pd.DataFrame:
    return pd.DataFrame([
        {"TEAM_ABBREVIATION": "BOS", "PACE": 101.5, "OFF_RATING": 119.0,
         "DEF_RATING": 110.0},
        {"TEAM_ABBREVIATION": "LAL", "PACE": 98.0, "OFF_RATING": 113.0,
         "DEF_RATING": 114.0},
    ])


NBA_GAMES = [{"game_id": "n1", "home_team": "BOS", "away_team": "LAL"}]


def test_nba_projections_scale_per_minute_rates(nba_players, nba_teams):
    projections = build_nba_projections(nba_players, nba_teams, NBA_GAMES)
    points = next(
        p for p in projections
        if p.player_name == "Guard A" and p.market == "player_points"
    )
    assert points.mean > 27.0  # weak LAL defence lifts the baseline
    assert points.sport == "nba" and points.opponent == "LAL"
    assert {p.market for p in projections} == {
        "player_points", "player_rebounds", "player_assists", "player_threes"
    }


def test_zero_minute_players_are_dropped(nba_players, nba_teams):
    projections = build_nba_projections(nba_players, nba_teams, NBA_GAMES)
    assert not [p for p in projections if p.player_name == "Bench C"]


def test_nba_game_projection_uses_pace_and_ratings(nba_players, nba_teams):
    projection = project_nba_game(NBA_GAMES[0], nba_teams)
    assert 180 < projection.total_mean < 280
    assert projection.home_margin_mean > 0  # BOS is the stronger side


def test_live_loaders_explain_the_missing_extra(monkeypatch):
    monkeypatch.setattr(baseline, "__name__", baseline.__name__)  # no-op guard
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith(("nfl_data_py", "nba_api")):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match="nfl_data_py"):
        baseline.load_nfl_frames([2025])
    with pytest.raises(RuntimeError, match="nba_api"):
        baseline.load_nba_frames("2025-26")
