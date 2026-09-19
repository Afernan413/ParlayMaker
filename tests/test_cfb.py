"""College football data, normalised into the shapes the NFL model uses.

The release's schema is not stable across seasons, which is what these tests
are mostly about: a file missing a column the model does not need must not take
the whole season down with it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.models import cfb


def plays(**overrides) -> pd.DataFrame:
    """A few college plays, in the release's own column names."""
    rows = [
        # Ohio State: a completed 30-yard pass to Receiver A for a TD
        dict(season=2025, week=3, game_id=1, pos_team="Ohio State", def_pos_team="Michigan",
             EPA=1.8, EPA_success=1.0, **{"pass": True}, rush=False, pass_attempt=True,
             completion=True, pass_td=True, rush_td=False,
             passer_player_name="QB One", rusher_player_name=None,
             receiver_player_name="Receiver A", yds_receiving=30.0, yds_rushed=0.0),
        # an incompletion to Receiver B
        dict(season=2025, week=3, game_id=1, pos_team="Ohio State", def_pos_team="Michigan",
             EPA=-0.6, EPA_success=0.0, **{"pass": True}, rush=False, pass_attempt=True,
             completion=False, pass_td=False, rush_td=False,
             passer_player_name="QB One", rusher_player_name=None,
             receiver_player_name="Receiver B", yds_receiving=0.0, yds_rushed=0.0),
        # a 12-yard run
        dict(season=2025, week=3, game_id=1, pos_team="Ohio State", def_pos_team="Michigan",
             EPA=0.4, EPA_success=1.0, **{"pass": False}, rush=True, pass_attempt=False,
             completion=False, pass_td=False, rush_td=False,
             passer_player_name=None, rusher_player_name="Back One",
             receiver_player_name=None, yds_receiving=0.0, yds_rushed=12.0),
    ]
    frame = pd.DataFrame(rows)
    for column, value in overrides.items():
        frame[column] = value
    return frame


# ----------------------------------------------------------------------
# the weekly frame
# ----------------------------------------------------------------------
def test_passing_rushing_and_receiving_merge_into_one_row_per_player():
    weekly = cfb.cfb_player_weekly(plays())
    assert set(weekly["player_display_name"]) == {"QB One", "Back One", "Receiver A", "Receiver B"}
    qb = weekly.set_index("player_display_name").loc["QB One"]
    assert qb["passing_yards"] == pytest.approx(30.0)
    assert qb["passing_tds"] == pytest.approx(1.0)
    assert qb["attempts"] == pytest.approx(2.0)


def test_columns_come_out_in_nflverse_names():
    """The whole point: baseline.py projects college without knowing it exists."""
    weekly = cfb.cfb_player_weekly(plays())
    for column in ("player_display_name", "recent_team", "season", "week",
                   "passing_yards", "rushing_yards", "receiving_yards", "receptions"):
        assert column in weekly.columns


def test_a_receiver_is_credited_yards_but_not_the_incompletion():
    weekly = cfb.cfb_player_weekly(plays()).set_index("player_display_name")
    assert weekly.loc["Receiver A", "receiving_yards"] == pytest.approx(30.0)
    assert weekly.loc["Receiver A", "receptions"] == pytest.approx(1.0)
    assert weekly.loc["Receiver B", "receptions"] == pytest.approx(0.0)


def test_target_share_is_derived_per_team():
    weekly = cfb.cfb_player_weekly(plays()).set_index("player_display_name")
    # Two targets on the team, one each to A and B.
    assert weekly.loc["Receiver A", "target_share"] == pytest.approx(0.5)
    assert weekly["target_share"].sum() == pytest.approx(1.0)


# ----------------------------------------------------------------------
# the efficiency frame
# ----------------------------------------------------------------------
def test_play_types_are_labelled_the_way_the_efficiency_model_expects():
    frame = cfb.cfb_pbp_normalised(plays())
    assert sorted(frame["play_type"].unique()) == ["pass", "run"]
    assert list(frame.columns) >= ["season", "week", "posteam", "defteam", "epa"]


def test_success_falls_back_to_the_sign_of_epa():
    """EPA_success is optional in the release; the model still needs success."""
    without = plays().drop(columns=["EPA_success"])
    frame = cfb.cfb_pbp_normalised(without)
    assert frame["success"].tolist() == [1.0, 0.0, 1.0]


def test_plays_with_no_team_or_no_epa_are_dropped():
    frame = plays()
    frame.loc[0, "EPA"] = None
    assert len(cfb.cfb_pbp_normalised(frame)) == 2


# ----------------------------------------------------------------------
# final scores: the schema that moves between seasons
# ----------------------------------------------------------------------
def test_an_explicit_final_score_is_preferred():
    frame = plays()
    frame["homeTeamName"] = "Ohio State"
    frame["awayTeamName"] = "Michigan"
    frame["homeFinalScore"] = 31
    frame["awayFinalScore"] = 24
    frame["end.homeScore"] = 17          # a mid-game value, deliberately wrong
    assert cfb.score_columns(frame) == ("homeFinalScore", "awayFinalScore")
    results = cfb.cfb_results(frame)
    assert results.iloc[0]["home_score"] == pytest.approx(31)
    assert results.iloc[0]["away_score"] == pytest.approx(24)


def test_a_season_without_a_final_score_uses_the_running_one():
    """2024 and 2025 carry no final score, only the score at each play."""
    frame = plays()
    frame["homeTeamName"] = "Ohio State"
    frame["awayTeamName"] = "Michigan"
    frame["end.homeScore"] = [7, 14, 21]
    frame["end.awayScore"] = [0, 3, 10]
    assert cfb.score_columns(frame) == ("end.homeScore", "end.awayScore")
    results = cfb.cfb_results(frame)
    assert results.iloc[0]["home_score"] == pytest.approx(21)
    assert results.iloc[0]["away_score"] == pytest.approx(10)


def test_the_per_play_score_is_the_last_resort():
    frame = plays()
    frame["homeTeamName"] = "Ohio State"
    frame["awayTeamName"] = "Michigan"
    frame["homeScore"] = [7, 14, 28]
    frame["awayScore"] = [0, 3, 3]
    assert cfb.score_columns(frame) == ("homeScore", "awayScore")
    assert cfb.cfb_results(frame).iloc[0]["home_score"] == pytest.approx(28)


def test_no_score_columns_is_an_empty_frame_not_an_exception():
    """Grading is a nice-to-have; projecting must not depend on it."""
    results = cfb.cfb_results(plays())
    assert results.empty
    assert "home_score" in results.columns


# ----------------------------------------------------------------------
# loading, and the seasons whose schema differs
# ----------------------------------------------------------------------
def fake_release(monkeypatch, schemas: dict[int, set[str]]):
    """Stand in for the GitHub release, one column set per season."""
    requested: dict[int, list[str]] = {}

    def fetch(url: str):
        season = int(url.rsplit("_", 1)[-1].split(".")[0])
        if season not in schemas:
            raise RuntimeError("404 not found")
        return schemas[season], season      # the "body" is just the season here

    def read_parquet(body, columns=None):
        requested[body] = list(columns or [])
        frame = plays()
        frame["season"] = body
        return frame[[c for c in columns if c in frame.columns]]

    monkeypatch.setattr(cfb, "fetch_season", fetch)
    monkeypatch.setattr(cfb.pd, "read_parquet", read_parquet)
    return requested


FULL = set(cfb.PBP_COLUMNS)
NO_FINAL_SCORE = FULL - {"homeFinalScore", "awayFinalScore"}


def test_a_season_missing_only_an_optional_column_is_still_loaded(monkeypatch):
    """The bug this guards: 2024 and 2025 have no final-score column, and
    asking for it unconditionally dropped both seasons entirely."""
    requested = fake_release(monkeypatch, {2025: NO_FINAL_SCORE})
    frame = cfb.load_cfb_pbp([2025])
    assert not frame.empty
    assert "homeFinalScore" not in requested[2025]
    assert "yds_rushed" in requested[2025]


def test_a_season_missing_a_required_column_is_skipped(monkeypatch):
    fake_release(monkeypatch, {2025: FULL - {"yds_rushed"}})
    with pytest.raises(RuntimeError, match="missing yds_rushed"):
        cfb.load_cfb_pbp([2025])


def test_one_bad_season_does_not_take_a_good_one_with_it(monkeypatch):
    fake_release(monkeypatch, {2024: FULL - {"EPA"}, 2025: NO_FINAL_SCORE})
    frame = cfb.load_cfb_pbp([2024, 2025])
    assert set(frame["season"]) == {2025}


def test_an_unpublished_season_says_which_one(monkeypatch):
    fake_release(monkeypatch, {})
    with pytest.raises(RuntimeError, match="2027"):
        cfb.load_cfb_pbp([2027])


def test_the_required_columns_are_what_the_model_actually_reads():
    """A column the projection path needs must be required, not optional."""
    assert "EPA" in cfb.PBP_REQUIRED_COLUMNS
    assert "yds_receiving" in cfb.PBP_REQUIRED_COLUMNS
    # ...and a column only grading needs must not be, or a season is lost to it.
    for optional in ("homeFinalScore", "awayFinalScore", "EPA_success"):
        assert optional in cfb.PBP_OPTIONAL_COLUMNS


# ----------------------------------------------------------------------
# team matching
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Ohio State", "ohio state"),
        ("Texas A&M", "Texas A M"),
        ("Miami (FL)", "Miami (FL)"),
        ("St. John's", "St Johns"),
    ],
)
def test_school_names_reduce_to_the_same_key(left, right):
    assert cfb.normalise_team(left) == cfb.normalise_team(right)


def test_nothing_reduces_to_an_empty_key():
    assert cfb.normalise_team(None) == ""
    assert cfb.normalise_team("") == ""
