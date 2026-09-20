"""Who is playing, how much, and what that does to a projection."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.injuries import InjuryRecord
from src.models import roles
from src.models.roles import (
    MIN_WINDOW_GAMES,
    Role,
    RoleModel,
    build_role_model,
    next_week,
    normalise_name,
    per_snap_projection,
    redistribute,
    snap_window,
)


def snap_rows(spec: dict[tuple[str, str], list[tuple[int, float]]], season: int = 2026):
    """Build a snap-count frame from ``{(team, player): [(week, snaps), ...]}``."""
    rows = []
    positions = {"QB1": "QB", "QB2": "QB", "WR1": "WR", "WR2": "WR", "WR3": "WR", "LT": "T"}
    for (team, player), games in spec.items():
        for week, snaps in games:
            rows.append(
                {
                    "season": season, "week": week, "team": team, "player": player,
                    "position": positions.get(player, "WR"),
                    "offense_snaps": snaps, "offense_pct": snaps / 60.0,
                }
            )
    return pd.DataFrame(rows)


FOUR_WEEKS = [(1, 60.0), (2, 60.0), (3, 60.0), (4, 60.0)]


# ----------------------------------------------------------------------
# joining a feed's spelling to a box score's
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Michael Penix Jr.", "Michael Penix"),
        ("Odell Beckham Jr", "Odell Beckham"),
        ("Robert Griffin III", "Robert Griffin"),
        ("De'Von Achane", "DeVon Achane"),
        ("Amon-Ra St. Brown", "Amon Ra St Brown"),
    ],
)
def test_a_generational_suffix_does_not_break_the_join(left, right):
    """The bug this guards: the injury report says Jr., the snap data does not,
    and a ruled-out starter reads as available."""
    assert normalise_name(left) == normalise_name(right)


def test_a_two_token_name_keeps_both_tokens():
    """Never strip so far that a first and last name stop being distinct."""
    assert normalise_name("Jalen Milroe") == "jalenmilroe"
    assert normalise_name("Anthony Sr") == "anthonysr"


# ----------------------------------------------------------------------
# the snap window
# ----------------------------------------------------------------------
def test_share_is_measured_over_the_teams_games_not_the_players():
    """A quarterback who started two of the team's four games has a share of a
    half, not of one. Measured over his own appearances every starter who ever
    played reads as full-time, and a position group sums to well over its slots."""
    frame = snap_rows({
        ("SEA", "QB1"): [(1, 60.0), (2, 60.0)],
        ("SEA", "QB2"): [(3, 60.0), (4, 60.0)],
    })
    window = snap_window(frame, season=2026, week=5).set_index("player")
    assert window.loc["QB1", "share"] + window.loc["QB2", "share"] == pytest.approx(1.0, abs=0.01)


def test_snaps_are_measured_over_appearances_so_a_rate_has_the_right_denominator():
    """`snaps` divides the games played, because the production it will be
    divided into is averaged the same way. `share` divides the team's window."""
    frame = snap_rows({
        ("SEA", "QB1"): FOUR_WEEKS,                   # gives the team four games
        ("SEA", "WR1"): [(3, 50.0), (4, 50.0)],       # played only two of them
    })
    row = snap_window(frame, season=2026, week=5).set_index("player").loc["WR1"]
    assert row["snaps"] == pytest.approx(50.0)        # per appearance
    assert row["share"] < 0.6                         # but only two of four games


def test_the_window_does_not_reach_into_last_season():
    """Rosters change; a share mixing two squads is worse than none."""
    last = snap_rows({("SEA", "WR1"): FOUR_WEEKS}, season=2025)
    frame = pd.concat([last, snap_rows({("SEA", "WR2"): [(1, 60.0), (2, 60.0)]})])
    window = snap_window(frame, season=2026, week=3)
    assert set(window["player"]) == {"WR2"}


def test_one_game_is_not_enough_to_measure_a_share():
    frame = snap_rows({("SEA", "WR1"): [(1, 60.0)]})
    assert snap_window(frame, season=2026, week=2).empty


def test_two_games_is_enough():
    frame = snap_rows({("SEA", "WR1"): [(1, 60.0), (2, 60.0)]})
    window = snap_window(frame, season=2026, week=3)
    assert len(window) == 1 and window.iloc[0]["share"] == pytest.approx(1.0)


def test_team_snaps_per_game_comes_out_of_the_window():
    frame = snap_rows({
        ("SEA", "QB1"): [(1, 62.0), (2, 62.0)],
        ("SEA", "WR1"): [(1, 40.0), (2, 40.0)],
    })
    assert snap_window(frame, season=2026, week=3)["team_snaps"].iloc[0] == pytest.approx(62.0)


def test_no_snap_data_is_an_empty_window_not_an_exception():
    assert snap_window(None, season=2026, week=3).empty
    assert snap_window(pd.DataFrame(), season=2026, week=3).empty


# ----------------------------------------------------------------------
# which week is being priced
# ----------------------------------------------------------------------
def test_the_week_being_priced_is_the_first_incomplete_one():
    """A Thursday night game lands in the release days before the Sunday slate.
    Counting its week as finished points the model at the week after the one it
    is pricing, which then finds no injury report and concludes nobody is hurt."""
    played = pd.DataFrame(
        [{"season": 2026, "week": 1, "team": f"T{i}"} for i in range(32)]
        + [{"season": 2026, "week": 2, "team": "T0"}, {"season": 2026, "week": 2, "team": "T1"}]
    )
    assert next_week(played, 2026) == 2


def test_a_fully_played_week_advances():
    played = pd.DataFrame(
        [{"season": 2026, "week": week, "team": f"T{i}"} for week in (1, 2) for i in range(32)]
    )
    assert next_week(played, 2026) == 3


def test_no_snaps_at_all_is_week_one():
    assert next_week(None, 2026) == 1
    assert next_week(pd.DataFrame(columns=["season", "week", "team"]), 2026) == 1


# ----------------------------------------------------------------------
# redistribution
# ----------------------------------------------------------------------
def role(player, share, snaps, status="ACTIVE", position="WR", team="SEA"):
    return Role(player=player, team=team, position=position, share=share,
                snaps=snaps, status=status)


def test_nothing_moves_when_nobody_is_out():
    group = [role("WR1", 0.8, 48.0), role("WR2", 0.5, 30.0)]
    expected = redistribute(group, 60.0)
    assert expected["wr1"] == pytest.approx(0.8, abs=0.02)


def test_the_only_available_quarterback_takes_the_whole_slot():
    """The case the model was blind to: a backup with a 0.37 share inherits the
    starter's workload, not a five per cent nudge."""
    group = [
        role("QB1", 0.63, 42.0, status="OUT", position="QB"),
        role("QB2", 0.37, 37.0, position="QB"),
    ]
    expected = redistribute(group, 60.0)
    assert expected["qb2"] == pytest.approx(1.0, abs=0.01)
    assert "qb1" not in expected


def test_a_full_time_player_who_missed_games_is_not_priced_as_part_time():
    """A receiver who plays four snaps in five when healthy but sat out two
    games has a four-week share of 0.3 and a role of 0.8. He is available now."""
    group = [role("WR1", 0.30, 49.0), role("WR2", 0.60, 36.0)]
    expected = redistribute(group, 60.0)
    assert expected["wr1"] > 0.30


def test_expected_shares_never_exceed_the_slots_the_team_fields():
    group = [
        role("WR1", 0.8, 48.0), role("WR2", 0.6, 36.0),
        role("WR3", 0.4, 24.0, status="OUT"),
    ]
    expected = redistribute(group, 60.0)
    assert sum(expected.values()) <= sum(r.share for r in group) + 1e-9


def test_nobody_is_given_more_than_every_snap():
    group = [role("WR1", 0.95, 59.0), role("WR2", 0.9, 58.0, status="OUT")]
    assert max(redistribute(group, 60.0).values()) <= 1.0


def test_a_group_with_nobody_left_is_left_alone():
    group = [role("QB1", 1.0, 60.0, status="OUT", position="QB")]
    assert redistribute(group, 60.0) == {}


# ----------------------------------------------------------------------
# the assembled model
# ----------------------------------------------------------------------
def out(team, player, position="QB"):
    return InjuryRecord(
        sport="nfl", team=team, player_name=player, position=position,
        status="OUT", practice=None, detail="Knee", source="test", report_date=None,
    )


def test_a_ruled_out_player_expects_no_snaps():
    frame = snap_rows({
        ("SEA", "QB1"): [(1, 60.0), (2, 60.0)],
        ("SEA", "QB2"): [(1, 5.0), (2, 5.0)],
    })
    model = build_role_model(frame, [out("SEA", "QB1")], season=2026, week=3)
    assert model.get("SEA", "QB1").expected_share == 0.0
    assert model.get("SEA", "QB2").expected_share > model.get("SEA", "QB2").share


def test_only_skill_positions_are_redistributed():
    """A blanket 'somebody on this team is out' rule bumped every receiver for a
    cornerback's absence, which with an injury on every team is a constant."""
    frame = snap_rows({
        ("SEA", "WR1"): [(1, 50.0), (2, 50.0)],
        ("SEA", "LT"): [(1, 60.0), (2, 60.0)],
    })
    model = build_role_model(
        frame,
        [InjuryRecord(sport="nfl", team="SEA", player_name="CB1", position="CB",
                      status="OUT", practice=None, detail=None, source="test",
                      report_date=None)],
        season=2026, week=3,
    )
    receiver = model.get("SEA", "WR1")
    assert receiver.snap_factor == pytest.approx(1.0, abs=0.01)


def test_availability_survives_a_window_too_thin_to_measure():
    """Who is out is known in week one, when no share is worth measuring yet."""
    frame = snap_rows({("SEA", "QB1"): [(1, 60.0)]})
    model = build_role_model(frame, [out("SEA", "QB1")], season=2026, week=2)
    assert model.empty                          # no roles
    assert model.status_for("QB1") == "OUT"     # but the report still stands
    assert "availability still applies" in model.reason


def test_a_designation_is_not_applied_to_a_namesake_on_another_team():
    frame = snap_rows({("SEA", "WR1"): [(1, 50.0), (2, 50.0)]})
    model = build_role_model(frame, [out("MIA", "WR1", position="WR")], season=2026, week=3)
    assert model.status_for("WR1", "MIA") == "OUT"
    assert model.status_for("WR1", "SEA") is None


def test_the_report_week_and_how_stale_it_is_are_recorded():
    frame = snap_rows({("SEA", "WR1"): [(1, 50.0), (2, 50.0)]})
    model = build_role_model(
        frame, [out("SEA", "WR1", "WR")], season=2026, week=4, report_week=2, report_lag=2
    )
    assert model.coverage()["report_week"] == 2
    assert model.coverage()["report_lag_weeks"] == 2


def test_an_empty_model_covers_nothing_and_says_so():
    model = RoleModel()
    assert model.empty
    assert model.get("SEA", "WR1") is None
    assert model.snaps_for("SEA") == 0.0


# ----------------------------------------------------------------------
# what it does to a number
# ----------------------------------------------------------------------
def test_a_settled_starter_is_projected_at_his_own_rate():
    """A starter plays 55 snaps a game, so his own rate should dominate."""
    projected = per_snap_projection(
        own_total=110.0, own_snaps=55.0,       # 2.0 per snap, full-time
        group_total=60.0, group_snaps=120.0,   # 0.5 per snap across the group
        expected_snaps=55.0,
    )
    # Keeps about three quarters of his own rate: 88 of a possible 110, against
    # the 27 the group's rate alone would give.
    assert 80.0 < projected < 110.0


def test_a_player_with_no_snaps_is_priced_at_the_groups_rate():
    """The honest answer for a backup nobody has seen play."""
    projected = per_snap_projection(
        own_total=0.0, own_snaps=0.0,
        group_total=120.0, group_snaps=60.0,   # 2.0 per snap
        expected_snaps=30.0,
    )
    assert projected == pytest.approx(60.0)


def test_a_thin_sample_is_shrunk_toward_the_group():
    """10 snaps of a freak rate must not be extrapolated to a full game."""
    own_rate_only = 10.0 * 60.0 / 10.0        # 60, what no shrinkage would give
    projected = per_snap_projection(
        own_total=10.0, own_snaps=10.0,       # 1.0 per snap on 10 snaps
        group_total=30.0, group_snaps=300.0,  # 0.1 per snap
        expected_snaps=60.0,
    )
    # Keeps a third of its own rate, so well under half the naive extrapolation.
    assert projected < own_rate_only * 0.5


def test_more_expected_snaps_means_more_production():
    kwargs = dict(own_total=50.0, own_snaps=40.0, group_total=200.0, group_snaps=200.0)
    assert per_snap_projection(**kwargs, expected_snaps=60.0) > per_snap_projection(
        **kwargs, expected_snaps=30.0
    )


def test_the_factor_is_bounded_both_ways():
    """A share measured over two games cannot quarter or triple a projection."""
    low, high = roles.SNAP_FACTOR_BOUNDS
    tiny = role("WR1", 0.9, 54.0)
    tiny = Role(**{**tiny.__dict__, "expected_share": 0.01})
    assert tiny.snap_factor == pytest.approx(low)
    big = Role(**{**role("WR2", 0.05, 3.0).__dict__, "expected_share": 1.0})
    assert big.snap_factor == pytest.approx(high)
