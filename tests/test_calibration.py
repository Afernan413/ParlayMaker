"""The learned-correction record, and how it changes what the engine prices."""

from __future__ import annotations

import json

import pytest

from src.models.calibration import (
    CALIBRATION_PATH,
    MEAN_FACTOR_BOUNDS,
    Calibration,
    MarketCalibration,
    SportCalibration,
    active_calibration,
    calibration_for,
    set_active_calibration,
    using_calibration,
)
from src.models.distributions import DistributionSpec


def wrap(*fits: MarketCalibration) -> Calibration:
    """A calibration holding exactly these market fits."""
    sports: dict[str, SportCalibration] = {}
    for fit in fits:
        markets = sports.setdefault(
            fit.sport, SportCalibration(sport=fit.sport, markets={})
        ).markets
        markets[fit.market] = fit
    return Calibration(sports=sports)


# ----------------------------------------------------------------------
# the record itself
# ----------------------------------------------------------------------
def test_identity_calibration_changes_nothing():
    fit = MarketCalibration(sport="nfl", market="player_rush_yds")
    assert not fit.shifts_mean
    assert not fit.shifts_probability
    assert fit.adjust_mean(60.0) == 60.0
    assert fit.adjust_probability(0.37) == 0.37


def test_mean_factor_scales_the_projection():
    fit = MarketCalibration(sport="nfl", market="player_rush_yds", mean_factor=0.9)
    assert fit.adjust_mean(60.0) == pytest.approx(54.0)


def test_platt_flattens_towards_the_base_rate():
    """A slope below one pulls both ends in: longshots up, favourites down."""
    fit = MarketCalibration(sport="nfl", market="player_rush_yds", platt_a=0.5, platt_b=0.0)
    assert fit.adjust_probability(0.05) > 0.05
    assert fit.adjust_probability(0.95) < 0.95
    assert fit.adjust_probability(0.5) == pytest.approx(0.5)


def test_platt_is_monotone():
    fit = MarketCalibration(sport="nfl", market="player_rush_yds", platt_a=0.4, platt_b=-0.3)
    probabilities = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    corrected = [fit.adjust_probability(p) for p in probabilities]
    assert corrected == sorted(corrected)


def test_out_of_range_fits_are_clipped():
    wild = MarketCalibration(
        sport="nfl", market="player_rush_yds", mean_factor=4.0, platt_a=99.0, platt_b=-40.0
    ).clipped()
    assert wild.mean_factor == MEAN_FACTOR_BOUNDS[1]
    assert wild.platt_a == pytest.approx(4.0)
    assert wild.platt_b == pytest.approx(-3.0)


def test_round_trip_through_json(tmp_path):
    fit = MarketCalibration(
        sport="nfl", market="player_pass_tds", samples=900, mean_factor=0.96,
        dispersion=1.4, family="negative_binomial", platt_a=0.8, platt_b=0.05,
        variance_multiple=1.4,
    )
    path = tmp_path / "calibration.json"
    wrap(fit).save(path)
    restored = Calibration.load(path).get("nfl", "player_pass_tds")
    assert restored == fit


def test_a_future_format_version_is_ignored(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"version": 99, "sports": {"nfl": {"markets": {}}}}))
    assert Calibration.load(path).empty


def test_unreadable_file_is_ignored(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text("{ not json")
    assert Calibration.load(path).empty


def test_missing_file_is_blank(tmp_path):
    assert Calibration.load(tmp_path / "nothing.json").empty


# ----------------------------------------------------------------------
# lookup
# ----------------------------------------------------------------------
def test_college_borrows_the_nfl_player_fit():
    calibration = wrap(
        MarketCalibration(sport="nfl", market="player_rush_yds", mean_factor=0.9)
    )
    assert calibration.get("ncaaf", "player_rush_yds").mean_factor == pytest.approx(0.9)


def test_college_never_borrows_the_nfl_score_spread():
    """A college final score swings far wider; borrowing 13 points is worse."""
    calibration = wrap(MarketCalibration(sport="nfl", market="spreads", dispersion=13.0))
    assert calibration.get("ncaaf", "spreads") is None
    assert calibration.get("nfl", "spreads").dispersion == pytest.approx(13.0)


def test_no_sport_means_no_correction():
    calibration = wrap(MarketCalibration(sport="nfl", market="player_rush_yds", mean_factor=0.5))
    assert calibration.get(None, "player_rush_yds") is None


def test_merging_keeps_sports_the_newer_fit_does_not_mention():
    old = wrap(
        MarketCalibration(sport="nfl", market="player_rush_yds", mean_factor=0.9),
        MarketCalibration(sport="ncaaf", market="player_rush_yds", mean_factor=1.1),
    )
    new = wrap(MarketCalibration(sport="nfl", market="player_rush_yds", mean_factor=0.8))
    merged = old.merged_with(new)
    assert merged.get("nfl", "player_rush_yds").mean_factor == pytest.approx(0.8)
    assert merged.sports["ncaaf"].markets["player_rush_yds"].mean_factor == pytest.approx(1.1)


def test_using_calibration_restores_what_was_there():
    before = active_calibration()
    with using_calibration(wrap(MarketCalibration(sport="nfl", market="totals", dispersion=11.0))):
        assert calibration_for("nfl", "totals").dispersion == pytest.approx(11.0)
    assert active_calibration() is before


# ----------------------------------------------------------------------
# how it reaches a price
# ----------------------------------------------------------------------
def test_spec_ignores_calibration_without_a_sport():
    with using_calibration(
        wrap(MarketCalibration(sport="nfl", market="player_rush_yds", mean_factor=0.5))
    ):
        assert DistributionSpec.for_market("player_rush_yds", 60.0).mean == pytest.approx(60.0)


def test_spec_applies_the_mean_factor_for_its_sport():
    with using_calibration(
        wrap(MarketCalibration(sport="nfl", market="player_rush_yds", mean_factor=0.5))
    ):
        spec = DistributionSpec.for_market("player_rush_yds", 60.0, sport="nfl")
    assert spec.mean == pytest.approx(30.0)


def test_a_wider_fitted_dispersion_fattens_the_tail():
    """The point of the dispersion fit: longshots stop looking impossible."""
    tight = DistributionSpec.for_market("player_rush_yds", 60.0, sport="nfl")
    with using_calibration(
        wrap(MarketCalibration(sport="nfl", market="player_rush_yds", dispersion=0.9))
    ):
        wide = DistributionSpec.for_market("player_rush_yds", 60.0, sport="nfl")
    assert wide.prob_over(150.5) > tight.prob_over(150.5) * 1.5


def test_an_explicit_dispersion_beats_the_fitted_one():
    with using_calibration(
        wrap(MarketCalibration(sport="nfl", market="player_rush_yds", dispersion=0.9))
    ):
        spec = DistributionSpec.for_market("player_rush_yds", 60.0, 0.3, sport="nfl")
    assert spec.dispersion == pytest.approx(0.3)


def test_an_overdispersed_count_market_is_promoted():
    with using_calibration(
        wrap(MarketCalibration(
            sport="nfl", market="player_pass_tds", family="negative_binomial", dispersion=1.6
        ))
    ):
        spec = DistributionSpec.for_market("player_pass_tds", 1.8, sport="nfl")
    assert spec.family == "negative_binomial"
    assert spec.variance > spec.mean


def test_platt_reaches_the_priced_probability():
    raw = DistributionSpec.for_market("player_rush_yds", 60.0, sport="nfl").prob_over(90.5)
    with using_calibration(
        wrap(MarketCalibration(sport="nfl", market="player_rush_yds", platt_a=0.5, platt_b=0.0))
    ):
        corrected = DistributionSpec.for_market("player_rush_yds", 60.0, sport="nfl").prob_over(90.5)
    assert corrected > raw  # a long shot, pulled up towards the base rate


def test_recalibration_leaves_the_push_alone():
    """An integer line refunds a push, so only the two live sides are rescaled."""
    with using_calibration(
        wrap(MarketCalibration(sport="nfl", market="player_receptions", platt_a=0.6, platt_b=0.2))
    ):
        spec = DistributionSpec.for_market("player_receptions", 5.0, sport="nfl")
        raw_push = spec._raw_probability(5.0).prob_push
        probability = spec.probability(5.0)
    assert probability.prob_push == pytest.approx(raw_push)
    total = probability.prob_over + probability.prob_under + probability.prob_push
    assert total == pytest.approx(1.0)


def test_under_and_over_still_sum_to_one():
    with using_calibration(
        wrap(MarketCalibration(sport="nfl", market="player_rush_yds", platt_a=0.4, platt_b=-0.2))
    ):
        spec = DistributionSpec.for_market("player_rush_yds", 60.0, sport="nfl")
    assert spec.prob_over(70.5) + spec.prob_over(70.5, "Under") == pytest.approx(1.0)


# ----------------------------------------------------------------------
# the artifact that actually ships
# ----------------------------------------------------------------------
@pytest.mark.skipif(not CALIBRATION_PATH.exists(), reason="no calibration has been fitted yet")
def test_the_committed_calibration_parses_and_is_in_bounds():
    calibration = Calibration.load()
    assert not calibration.empty
    for sport in calibration.sports.values():
        for fit in sport.markets.values():
            assert fit.clipped() == fit, f"{fit.market} was saved outside its guard rails"
            assert fit.samples > 0
