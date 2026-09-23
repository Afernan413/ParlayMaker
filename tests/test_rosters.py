"""Who is on which team, who can play, and how turnover is weighted."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.baseline import build_nfl_projections, history_weights, nfl_player_volume
from src.models.rosters import Roster, RosterEntry, build_roster


def roster_frame(rows):
    return pd.DataFrame(
        [
            {"season": 2026, "week": week, "team": team, "full_name": name,
             "position": pos, "status": status}
            for week, team, name, pos, status in rows
        ]
    )


def roster(**entries) -> Roster:
    """``roster(AJ_Brown=("NE", "RES"))`` -- underscores become spaces."""
    return Roster(entries={
        name.replace("_", " ").lower().replace(" ", ""): RosterEntry(
            player=name.replace("_", " "), team=team, position="WR", status=status
        )
        for name, (team, status) in entries.items()
    })


def weekly(rows, *, ids=True):
    """Box scores: ``(player_id, name, season, week, team, receiving_yards)``."""
    frame = pd.DataFrame(
        [
            {"player_id": pid, "player_display_name": name, "season": season, "week": week,
             "team": team, "receiving_yards": yards, "receptions": yards / 12.0}
            for pid, name, season, week, team, yards in rows
        ]
    )
    return frame if ids else frame.drop(columns=["player_id"])


# ----------------------------------------------------------------------
# the roster itself
# ----------------------------------------------------------------------
def test_the_latest_week_is_used_by_default():
    frame = roster_frame([
        (2, "PHI", "Wide Receiver", "WR", "ACT"),
        (3, "NE", "Wide Receiver", "WR", "RES"),
    ])
    built = build_roster(frame)
    assert built.week == 3
    assert built.team_of("Wide Receiver") == "NE"
    assert not built.can_play("Wide Receiver")


def test_an_active_listing_wins_when_a_player_appears_twice_in_a_week():
    frame = roster_frame([
        (3, "NE", "Moved Midweek", "WR", "CUT"),
        (3, "BUF", "Moved Midweek", "WR", "ACT"),
    ])
    built = build_roster(frame)
    assert built.team_of("Moved Midweek") == "BUF" and built.can_play("Moved Midweek")


@pytest.mark.parametrize(
    ("status", "reason"),
    [("RES", "reserve"), ("DEV", "practice squad"), ("RET", "retired"), ("CUT", "released")],
)
def test_only_the_active_roster_can_play(status, reason):
    built = roster(Some_One=("NE", status))
    assert not built.can_play("Some One")
    assert reason in built.why_not("Some One")


def test_nobody_is_filtered_without_a_roster():
    """College and mock slates have no roster; that must not empty the slate."""
    assert Roster().can_play("Anyone At All")


def test_a_player_on_no_roster_cannot_play():
    built = roster(Some_One=("NE", "ACT"))
    assert not built.can_play("Somebody Else")
    assert built.why_not("Somebody Else") == "not on any roster"


def test_a_generational_suffix_still_matches():
    built = build_roster(roster_frame([(3, "ATL", "Michael Penix Jr.", "QB", "ACT")]))
    assert built.team_of("Michael Penix") == "ATL"


# ----------------------------------------------------------------------
# weighting a player's history
# ----------------------------------------------------------------------
def test_default_weights_are_the_models_usual_ones():
    assert history_weights(
        [2026, 2026, 2026, 2026], ["NE"] * 4, target_season=2026, target_team="NE"
    ) == pytest.approx([0.4, 0.3, 0.2, 0.1])


def test_last_seasons_games_can_count_for_less():
    weights = history_weights(
        [2026, 2026, 2025, 2025], ["NE"] * 4,
        target_season=2026, target_team="NE", season_decay=0.5,
    )
    assert weights == pytest.approx([0.4, 0.3, 0.1, 0.05])


def test_games_for_another_team_can_count_for_less():
    weights = history_weights(
        [2026] * 4, ["NE", "NE", "PHI", "PHI"],
        target_season=2026, target_team="NE", team_decay=0.25,
    )
    assert weights == pytest.approx([0.4, 0.3, 0.05, 0.025])


def test_a_player_with_no_games_for_his_new_team_keeps_his_old_ones():
    """Traded this week: the old team is a weak guide, but the only one."""
    weights = history_weights(
        [2026] * 3, ["PHI"] * 3, target_season=2026, target_team="NE", team_decay=0.0,
    )
    assert weights == pytest.approx([0.4, 0.3, 0.2])


# ----------------------------------------------------------------------
# volume: one row per player, on the team he plays for now
# ----------------------------------------------------------------------
TRADED = [
    ("p1", "Wide Receiver", 2026, 1, "PHI", 90.0),
    ("p1", "Wide Receiver", 2026, 2, "PHI", 80.0),
    ("p1", "Wide Receiver", 2026, 3, "NE", 40.0),
]


def test_a_traded_player_gets_one_row_not_a_ghost_for_his_old_team():
    """The bug: grouping by (player, team) left a PHI row for a player on NE."""
    volume = nfl_player_volume(weekly(TRADED), roster=roster(Wide_Receiver=("NE", "ACT")))
    assert len(volume) == 1
    assert volume.iloc[0]["team"] == "NE"
    assert volume.iloc[0]["changed_team"]


def test_a_traded_player_joins_his_new_team_before_playing_for_it():
    """Traded on Tuesday: the roster knows before any box score does."""
    rows = [r for r in TRADED if r[4] == "PHI"]
    volume = nfl_player_volume(weekly(rows), roster=roster(Wide_Receiver=("NE", "ACT")))
    assert volume.iloc[0]["team"] == "NE"


def test_injured_reserve_retired_and_released_players_get_no_row():
    rows = [
        ("p1", "On Reserve", 2026, 1, "NE", 50.0),
        ("p2", "Retired Vet", 2026, 1, "MIN", 50.0),
        ("p3", "Healthy Starter", 2026, 1, "NE", 50.0),
        ("p4", "Released Guy", 2026, 1, "LV", 50.0),
    ]
    built = roster(
        On_Reserve=("NE", "RES"), Retired_Vet=("MIN", "RET"), Healthy_Starter=("NE", "ACT"),
    )
    volume = nfl_player_volume(weekly(rows), roster=built)
    assert volume["player_name"].tolist() == ["Healthy Starter"]


def test_namesakes_with_different_ids_stay_separate():
    """2025 had two Byron Youngs. Merging them would average two careers."""
    rows = [
        ("a", "Byron Young", 2025, 1, "LA", 10.0),
        ("b", "Byron Young", 2025, 1, "PHI", 70.0),
    ]
    volume = nfl_player_volume(weekly(rows))
    assert len(volume) == 2
    assert sorted(volume["receiving_yards"]) == [10.0, 70.0]


def test_a_frame_without_ids_is_grouped_by_name_and_team_as_before():
    """College carries no ids; a transfer keeping an old row beats merging namesakes."""
    rows = [
        (None, "Chris Johnson", 2025, 1, "Ohio State", 10.0),
        (None, "Chris Johnson", 2025, 1, "Oregon", 70.0),
    ]
    volume = nfl_player_volume(weekly(rows, ids=False))
    assert len(volume) == 2


def test_the_team_decay_reaches_the_average():
    rows = [
        ("p1", "Wide Receiver", 2026, 1, "PHI", 100.0),
        ("p1", "Wide Receiver", 2026, 2, "NE", 20.0),
    ]
    plain = nfl_player_volume(weekly(rows), team_decay=1.0).iloc[0]["receiving_yards"]
    decayed = nfl_player_volume(weekly(rows), team_decay=0.25).iloc[0]["receiving_yards"]
    assert decayed < plain          # the big PHI game counts for less on NE


def test_a_player_facing_his_old_team_is_projected_once():
    """When both rows claimed the same game and market, the last one won."""
    rows = TRADED + [
        ("p9", "Other Receiver", 2026, 1, "PHI", 50.0),
        ("p9", "Other Receiver", 2026, 2, "PHI", 50.0),
    ]
    frame = weekly(rows)
    pbp = pd.DataFrame(columns=["posteam", "defteam", "epa", "success", "play_type"])
    games = [{"game_id": "g1", "home_team": "NE", "away_team": "PHI"}]
    projections = build_nfl_projections(
        frame, pbp, games,
        roster=roster(Wide_Receiver=("NE", "ACT"), Other_Receiver=("PHI", "ACT")),
        markets=["player_reception_yds"],
    )
    mine = [p for p in projections if p.player_name == "Wide Receiver"]
    assert len(mine) == 1 and mine[0].team == "NE"


# ----------------------------------------------------------------------
# the live model and the replay must weight history identically
# ----------------------------------------------------------------------
@pytest.mark.parametrize("window", [4, 6, 8])
def test_the_live_model_and_the_replay_average_the_same_games(window, monkeypatch):
    """They used to take weights from two places. At an eight-game window one
    would have extended the weights and the other truncated them back to four,
    and the calibration would have been fitted on a model that never runs."""
    from config.settings import settings
    from src.learning.backtest import observations

    monkeypatch.setattr(settings, "volume_window_nfl", window)
    history = [("p1", "Wide Receiver", 2026, week, "NE", 20.0 + 10 * week) for week in range(1, 12)]
    frame = weekly(history)

    live = nfl_player_volume(frame.iloc[:-1]).iloc[0]["receiving_yards"]
    replayed = [
        obs for obs in observations(frame, min_history=2)
        if obs.week == 11 and obs.market == "player_reception_yds"
    ]
    assert replayed, "the replay should project week 11"
    assert replayed[0].projected == pytest.approx(live)


def test_a_per_snap_rate_divides_production_and_snaps_from_the_same_games():
    """Volume averages eight games; snaps are counted over the team's last
    four. Dividing the one by the other would mix two windows, so the per-snap
    path takes production from the matching short window."""
    from src.models.baseline import _volume_for
    from src.models.roles import Role

    promoted = Role(player="WR", team="NE", position="WR", snaps=30.0, share=0.5,
                    expected_share=0.9)           # a vacancy ahead of him this week
    group = (300.0, 200.0)                         # the position group's production and snaps
    long_avg, recent = 40.0, 70.0                  # he has been producing more lately

    from_average, _ = _volume_for(long_avg, "receiving_yards", promoted, group, 60.0)
    from_recent, _ = _volume_for(long_avg, "receiving_yards", promoted, group, 60.0, recent=recent)
    assert from_recent > from_average


def test_a_settled_role_keeps_the_long_average():
    """Only a changed role is repriced per snap; the eight-game average is
    what the calibration was fitted on."""
    from src.models.baseline import _volume_for
    from src.models.roles import Role

    settled = Role(player="WR", team="NE", position="WR", snaps=50.0, share=0.8,
                   expected_share=0.8)
    mean, _ = _volume_for(40.0, "receiving_yards", settled, (300.0, 200.0), 60.0, recent=70.0)
    assert mean == 40.0
