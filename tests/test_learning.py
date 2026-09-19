"""Walk-forward backtesting, the fits it feeds, and the forward journal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ingestion import db
from src.learning import calibrate, journal
from src.learning.backtest import (
    LINE_OFFSETS,
    brier_score,
    calibration_table,
    log_loss,
    observations,
    to_frame,
    with_derived_stats,
)
from src.models.calibration import Calibration, MarketCalibration, SportCalibration
from src.models.legs import Leg


def weekly_frame(
    *, players: int = 12, weeks: int = 10, season: int = 2026, seed: int = 7
) -> pd.DataFrame:
    """A synthetic nflverse-shaped frame with known per-player rates."""
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(players):
        rushing = 40.0 + 8.0 * index
        receiving = 55.0 + 5.0 * index
        for week in range(1, weeks + 1):
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "player_display_name": f"Player {index}",
                    "recent_team": "KC" if index % 2 else "BUF",
                    "rushing_yards": float(max(rng.normal(rushing, rushing * 0.5), 0.0)),
                    "receiving_yards": float(max(rng.normal(receiving, receiving * 0.5), 0.0)),
                    "receptions": float(rng.poisson(4.5)),
                    "rushing_tds": float(rng.poisson(0.4)),
                    "receiving_tds": float(rng.poisson(0.3)),
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# walk-forward pairs
# ----------------------------------------------------------------------
def test_observations_never_see_the_week_they_grade():
    """The whole point: a week's projection is built from earlier weeks only."""
    frame = weekly_frame(players=1, weeks=6)
    frame.loc[frame["week"] == 6, "rushing_yards"] = 999.0
    rows = [row for row in observations(frame, min_history=2) if row.market == "player_rush_yds"]
    week_six = [row for row in rows if row.week == 6]
    assert week_six, "week 6 should be graded"
    assert all(row.actual == 999.0 for row in week_six)
    assert all(row.projected < 400.0 for row in week_six), "the spike leaked into its own projection"


def test_early_weeks_are_skipped_until_there_is_history():
    rows = observations(weekly_frame(players=1, weeks=5), min_history=3)
    assert rows
    assert min(row.week for row in rows) == 4


def test_every_line_offset_is_probed():
    rows = [
        row
        for row in observations(weekly_frame(players=1, weeks=6), min_history=2)
        if row.market == "player_rush_yds" and row.week == 6
    ]
    assert len(rows) == len(LINE_OFFSETS)
    assert len({row.line for row in rows}) == len(LINE_OFFSETS)


def test_anytime_td_is_derived_and_asked_once():
    frame = weekly_frame(players=2, weeks=5)
    derived = with_derived_stats(frame)
    assert "anytime_td" in derived.columns
    expected = frame["rushing_tds"] + frame["receiving_tds"]
    assert derived["anytime_td"].tolist() == pytest.approx(expected.tolist())

    td_rows = [row for row in observations(frame, min_history=2) if row.market == "player_anytime_td"]
    assert td_rows
    assert {row.line for row in td_rows} == {0.5}


def test_the_batched_price_is_the_price_the_engine_would_give():
    """Observations are priced a market at a time, so pin it to the per-row path."""
    from src.models.distributions import DistributionSpec

    rows = observations(weekly_frame(players=10, weeks=10), min_history=2)
    assert rows
    for row in rows:
        expected = DistributionSpec.for_market(row.market, row.projected).prob_over(row.line)
        assert row.p_over == pytest.approx(expected, abs=1e-12)


def test_observations_come_back_in_the_order_the_weeks_were_played():
    """Pricing groups by market, which scrambles the order; it is put back."""
    rows = observations(weekly_frame(players=6, weeks=8), min_history=2)
    stamps = [(row.season, row.week) for row in rows]
    assert stamps == sorted(stamps)


def test_a_certainty_is_dropped_rather_than_recorded():
    """A probability of exactly 0 or 1 measures floating point, not the model."""
    rows = observations(weekly_frame(players=8, weeks=10), min_history=2)
    assert all(0.0 < row.p_over < 1.0 for row in rows)


def test_hit_matches_the_line():
    rows = observations(weekly_frame(players=3, weeks=6), min_history=2)
    assert all(row.hit == int(row.actual > row.line) for row in rows)


def test_lines_are_never_integers():
    """A half-point line cannot push, which keeps the grading unambiguous."""
    rows = observations(weekly_frame(players=4, weeks=6), min_history=2)
    assert all(abs(row.line - round(row.line)) > 1e-9 for row in rows)


# ----------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------
def test_brier_rewards_being_right():
    assert brier_score([1.0, 0.0], [1, 0]) == pytest.approx(0.0)
    assert brier_score([0.0, 1.0], [1, 0]) == pytest.approx(1.0)
    assert brier_score([0.5, 0.5], [1, 0]) == pytest.approx(0.25)


def test_log_loss_punishes_confident_mistakes_harder_than_brier():
    assert log_loss([0.5], [1]) == pytest.approx(0.693147, abs=1e-5)
    assert log_loss([0.01], [1]) > 4.0


def test_calibration_table_finds_a_known_gap():
    probabilities = [0.1] * 100
    outcomes = [1] * 30 + [0] * 70
    table = calibration_table(probabilities, outcomes, bins=10)
    row = table.iloc[0]
    assert row["claimed"] == pytest.approx(0.1)
    assert row["observed"] == pytest.approx(0.3)
    assert row["gap"] == pytest.approx(0.2)


# ----------------------------------------------------------------------
# the fits
# ----------------------------------------------------------------------
def test_residuals_drop_the_repeated_probe_lines():
    frame = to_frame(observations(weekly_frame(players=3, weeks=6), min_history=2))
    spread = calibrate.residuals(frame)
    assert len(spread) < len(frame)
    assert not spread.duplicated(subset=["season", "week", "player", "market"]).any()


def test_mean_factor_recovers_a_known_bias():
    projected = np.array([50.0, 100.0, 150.0])
    assert calibrate.fit_mean_factor(projected, projected * 0.9) == pytest.approx(0.9)


def test_mean_factor_weights_by_volume():
    """A 3-yard week must not outvote a 300-yard one."""
    projected = np.array([3.0, 300.0])
    actual = np.array([6.0, 300.0])  # one 2x miss on a tiny number
    assert calibrate.fit_mean_factor(projected, actual) < 1.02


def test_lognormal_dispersion_recovers_a_known_coefficient_of_variation():
    rng = np.random.default_rng(3)
    mu = np.full(20_000, 60.0)
    actual = rng.normal(60.0, 60.0 * 0.6, size=mu.size)
    fitted = calibrate.fit_dispersion("lognormal", mu, actual)
    assert fitted == pytest.approx(0.6, abs=0.02)


def test_count_dispersion_recovers_a_known_variance_multiple():
    rng = np.random.default_rng(4)
    mu = np.full(40_000, 5.0)
    actual = rng.poisson(5.0 * 1.0, size=mu.size) * 1.0
    assert calibrate.fit_dispersion("negative_binomial", mu, actual) == pytest.approx(1.0, abs=0.05)


def test_normal_dispersion_is_a_standard_deviation():
    rng = np.random.default_rng(5)
    mu = np.zeros(40_000)
    assert calibrate.fit_dispersion("normal", mu, rng.normal(0, 12.0, mu.size)) == pytest.approx(
        12.0, abs=0.2
    )


def test_poisson_stays_poisson_when_it_is_not_overdispersed():
    rng = np.random.default_rng(6)
    rows = pd.DataFrame({"projected": np.full(4000, 1.8), "actual": rng.poisson(1.8, 4000) * 1.0})
    fitted = calibrate.fit_shape("nfl", "player_pass_tds", rows)
    assert fitted.family is None
    assert fitted.dispersion is None


def test_an_overdispersed_poisson_is_promoted_to_negative_binomial():
    rng = np.random.default_rng(6)
    # A mixture of rates is wider than any single Poisson can be.
    rates = rng.choice([0.4, 3.2], size=4000)
    rows = pd.DataFrame({"projected": np.full(4000, 1.8), "actual": rng.poisson(rates) * 1.0})
    fitted = calibrate.fit_shape("nfl", "player_pass_tds", rows)
    assert fitted.family == "negative_binomial"
    assert fitted.dispersion > calibrate.POISSON_OVERDISPERSION_LIMIT


def test_platt_leaves_a_calibrated_model_alone():
    rng = np.random.default_rng(11)
    probabilities = rng.uniform(0.05, 0.95, 40_000)
    outcomes = (rng.uniform(size=probabilities.size) < probabilities).astype(int)
    a, b = calibrate.fit_platt(probabilities, outcomes)
    assert a == pytest.approx(1.0, abs=0.08)
    assert b == pytest.approx(0.0, abs=0.08)


def test_platt_recovers_a_known_distortion():
    rng = np.random.default_rng(12)
    truth = rng.uniform(0.05, 0.95, 60_000)
    outcomes = (rng.uniform(size=truth.size) < truth).astype(int)
    # The model states log-odds twice as extreme as the truth.
    claimed = 1.0 / (1.0 + np.exp(-2.0 * np.log(truth / (1 - truth))))
    a, b = calibrate.fit_platt(claimed, outcomes)
    assert a == pytest.approx(0.5, abs=0.05)


def test_platt_is_identity_when_nothing_ever_happens():
    a, b = calibrate.fit_platt([0.2, 0.3, 0.4], [0, 0, 0])
    assert (a, b) == (1.0, 0.0)


def test_game_markets_recover_the_residual_around_a_closing_line():
    rng = np.random.default_rng(13)
    lines = rng.uniform(-10, 10, 2000)
    schedule = pd.DataFrame(
        {
            "spread_line": lines,
            "result": lines + rng.normal(0, 13.5, lines.size),
            "total_line": np.full(lines.size, 45.0),
            "total": 45.0 + rng.normal(0, 10.5, lines.size),
        }
    )
    fitted = calibrate.fit_game_markets(schedule, sport="nfl")
    assert fitted["spreads"].dispersion == pytest.approx(13.5, abs=0.6)
    assert fitted["totals"].dispersion == pytest.approx(10.5, abs=0.6)


def test_game_markets_need_enough_played_games():
    schedule = pd.DataFrame({"spread_line": [1.0, 2.0], "result": [3.0, 4.0]})
    assert calibrate.fit_game_markets(schedule, sport="nfl", min_samples=60) == {}


# ----------------------------------------------------------------------
# the split, and whether the fit actually helps
# ----------------------------------------------------------------------
def test_split_is_chronological_and_does_not_overlap():
    frame = to_frame(observations(weekly_frame(players=4, weeks=10), min_history=2))
    train, test = calibrate.split_by_week(frame, holdout_weeks=3)
    assert not train.empty and not test.empty
    assert train["week"].max() < test["week"].min()
    assert len(train) + len(test) == len(frame)
    assert sorted(test["week"].unique().tolist()) == [8, 9, 10]


def test_split_falls_back_when_there_is_not_enough_history():
    frame = to_frame(observations(weekly_frame(players=2, weeks=4), min_history=2))
    train, test = calibrate.split_by_week(frame, holdout_weeks=10)
    assert test.empty
    assert len(train) == len(frame)


def test_training_improves_the_weeks_it_was_not_shown():
    """Fitted on the earlier weeks, scored on the later ones."""
    frame = to_frame(observations(weekly_frame(players=24, weeks=14), min_history=3))
    report = calibrate.train(frame, sport="nfl", holdout_weeks=3, min_samples=100)
    assert report.calibration.markets
    assert report.improved, f"brier went {report.before.brier} -> {report.after.brier}"
    assert report.brier_gain > 0
    # The stored metrics are rounded for the file they are written to.
    assert report.calibration.metrics["before"]["brier"] == pytest.approx(
        report.before.brier, abs=1e-6
    )


def test_probabilities_under_a_blank_calibration_match_the_recorded_ones():
    frame = to_frame(observations(weekly_frame(players=4, weeks=8), min_history=2))
    unchanged = calibrate.probabilities_under(frame, Calibration.blank(), sport="nfl")
    assert unchanged == pytest.approx(frame["p_over"].to_numpy(), abs=1e-9)


def test_probabilities_under_a_fit_actually_move():
    frame = to_frame(observations(weekly_frame(players=4, weeks=8), min_history=2))
    fit = Calibration(
        sports={
            "nfl": SportCalibration(
                sport="nfl",
                markets={
                    "player_rush_yds": MarketCalibration(
                        sport="nfl", market="player_rush_yds", dispersion=1.2
                    )
                },
            )
        }
    )
    moved = calibrate.probabilities_under(frame, fit, sport="nfl")
    rush = (frame["market"] == "player_rush_yds").to_numpy()
    assert not np.allclose(moved[rush], frame.loc[rush, "p_over"].to_numpy())
    others = ~rush
    assert moved[others] == pytest.approx(frame.loc[others, "p_over"].to_numpy(), abs=1e-9)


def test_comparison_table_shows_both_sides():
    frame = to_frame(observations(weekly_frame(players=6, weeks=9), min_history=2))
    report = calibrate.train(frame, sport="nfl", holdout_weeks=2, min_samples=100)
    table = calibrate.calibration_comparison(
        frame, Calibration(sports={"nfl": report.calibration}), sport="nfl", bins=5
    )
    assert {"claimed_before", "observed_before", "claimed_after", "observed_after"} <= set(table.columns)


def test_comparison_can_reuse_a_computed_pass():
    frame = to_frame(observations(weekly_frame(players=4, weeks=8), min_history=2))
    corrected = frame["p_over"].to_numpy()
    table = calibrate.calibration_comparison(frame, sport="nfl", bins=4, corrected=corrected)
    assert table["gap_before"].tolist() == pytest.approx(table["gap_after"].tolist())


def test_comparison_needs_something_to_compare():
    frame = to_frame(observations(weekly_frame(players=2, weeks=5), min_history=2))
    with pytest.raises(ValueError):
        calibrate.calibration_comparison(frame, sport="nfl")


# ----------------------------------------------------------------------
# the forward journal
# ----------------------------------------------------------------------
def leg(**overrides) -> Leg:
    fields = {
        "game_id": "g1",
        "sport": "nfl",
        "market": "player_rush_yds",
        "selection": "Over",
        "american_odds": -110,
        "line": 60.5,
        "player_name": "Player 0",
        "team": "KC",
        "p_model": 0.55,
        "p_implied": 0.52,
        "projection_mean": 64.0,
    }
    fields.update(overrides)
    return Leg(**fields)


GAMES = {"g1": {"commence_time": "2026-09-20T17:00:00Z", "season": 2026, "week": 3}}


def test_recording_skips_game_markets(db_path):
    written = journal.record(
        [leg(), leg(market="totals", player_name=None, selection="Over", line=47.5)],
        run_id="run-1",
        sport="nfl",
        games=GAMES,
        db_path=db_path,
    )
    assert written == 1


def test_recording_twice_is_a_no_op(db_path):
    for _ in range(2):
        journal.record([leg()], run_id="run-1", sport="nfl", games=GAMES, db_path=db_path)
    assert len(db.ungraded_projections("nfl", db_path=db_path)) == 1


def test_grading_settles_an_over_from_the_box_score(db_path):
    journal.record([leg()], run_id="run-1", sport="nfl", games=GAMES, db_path=db_path)
    weekly = pd.DataFrame(
        [{"season": 2026, "week": 3, "player_display_name": "Player 0", "rushing_yards": 88.0}]
    )
    assert journal.grade(weekly, sport="nfl", db_path=db_path) == 1
    graded = db.graded_projections("nfl", db_path=db_path)[0]
    assert graded["actual"] == pytest.approx(88.0)
    assert graded["hit"] == 1


def test_grading_settles_an_under_the_other_way(db_path):
    journal.record(
        [leg(selection="Under")], run_id="run-1", sport="nfl", games=GAMES, db_path=db_path
    )
    weekly = pd.DataFrame(
        [{"season": 2026, "week": 3, "player_display_name": "Player 0", "rushing_yards": 12.0}]
    )
    journal.grade(weekly, sport="nfl", db_path=db_path)
    assert db.graded_projections("nfl", db_path=db_path)[0]["hit"] == 1


def test_grading_matches_names_loosely(db_path):
    journal.record(
        [leg(player_name="A.J. Brown")], run_id="run-1", sport="nfl", games=GAMES, db_path=db_path
    )
    weekly = pd.DataFrame(
        [{"season": 2026, "week": 3, "player_display_name": "AJ Brown", "rushing_yards": 70.0}]
    )
    assert journal.grade(weekly, sport="nfl", db_path=db_path) == 1


def test_an_unplayed_game_stays_pending(db_path):
    journal.record([leg()], run_id="run-1", sport="nfl", games=GAMES, db_path=db_path)
    empty = pd.DataFrame(columns=["season", "week", "player_display_name", "rushing_yards"])
    assert journal.grade(empty, sport="nfl", db_path=db_path) == 0
    assert len(db.ungraded_projections("nfl", db_path=db_path)) == 1


def test_a_slate_that_has_not_kicked_off_is_not_reported_as_missing(db_path):
    journal.record([leg()], run_id="run-1", sport="nfl", games=GAMES, db_path=db_path)
    assert db.ungraded_projections("nfl", before="2026-09-19T00:00:00Z", db_path=db_path) == []
    assert len(db.ungraded_projections("nfl", before="2026-09-21T00:00:00Z", db_path=db_path)) == 1


def test_an_anytime_td_leg_settles_on_scoring_at_all(db_path):
    journal.record(
        [leg(market="player_anytime_td", selection="Yes", line=None)],
        run_id="run-1", sport="nfl", games=GAMES, db_path=db_path,
    )
    weekly = pd.DataFrame(
        [{
            "season": 2026, "week": 3, "player_display_name": "Player 0",
            "rushing_tds": 1.0, "receiving_tds": 0.0,
        }]
    )
    journal.grade(weekly, sport="nfl", db_path=db_path)
    graded = db.graded_projections("nfl", db_path=db_path)[0]
    assert graded["actual"] == pytest.approx(1.0)
    assert graded["hit"] == 1


def test_graded_rows_come_back_in_the_shape_the_fits_want(db_path):
    journal.record(
        [leg(), leg(selection="Under", line=70.5)],
        run_id="run-1", sport="nfl", games=GAMES, db_path=db_path,
    )
    weekly = pd.DataFrame(
        [{"season": 2026, "week": 3, "player_display_name": "Player 0", "rushing_yards": 88.0}]
    )
    journal.grade(weekly, sport="nfl", db_path=db_path)
    frame = journal.graded_frame("nfl", db_path=db_path)
    assert set(["sport", "season", "week", "player", "market", "projected", "actual", "line",
                "p_over", "hit"]) <= set(frame.columns)
    assert len(frame) == 2
    # Both rows are stated as the over, whichever side was priced.
    under = frame[frame["line"] == 70.5].iloc[0]
    assert under["p_over"] == pytest.approx(1.0 - 0.55)
    assert under["hit"] == 1
