"""Correlation priors and Gaussian copula tests."""

from __future__ import annotations

import numpy as np
import pytest

from config.settings import settings
from src.models.correlation import (
    GaussianCopulaSimulator,
    average_correlation,
    build_correlation_matrix,
    correlation_report,
    joint_probability,
    min_correlation,
    nearest_psd,
    pairwise_correlation,
)
from src.models.legs import Leg


def leg(**kwargs) -> Leg:
    base = dict(
        game_id="g1",
        sport="nfl",
        market="player_pass_yds",
        selection="Over",
        american_odds=-110,
        line=274.5,
        p_model=0.55,
    )
    base.update(kwargs)
    return Leg(**base)


QB = leg(player_name="QB One", team="KC", market="player_pass_yds", p_model=0.56)
WR1 = leg(player_name="WR One", team="KC", market="player_reception_yds",
          line=64.5, p_model=0.58)
OPP_WR = leg(player_name="WR Two", team="BUF", market="player_reception_yds",
             line=58.5, p_model=0.54)
TOTAL_UNDER = leg(market="totals", selection="Under", line=47.5, team=None,
                  player_name=None, p_model=0.53)
PASS_TD_OVER = leg(player_name="QB One", team="KC", market="player_pass_tds",
                   line=2.5, p_model=0.44)
CROSS_GAME = leg(game_id="g2", sport="nba", market="player_points",
                 player_name="Guard A", team="BOS", line=25.5, p_model=0.57)


# --------------------------------------------------------------- priors
def test_qb_to_wr_same_team_clears_the_sgp_threshold():
    rho = pairwise_correlation(QB, WR1)
    assert rho >= settings.min_sgp_correlation
    assert rho > pairwise_correlation(QB, OPP_WR)


def test_game_total_under_is_negatively_correlated_with_passing_tds():
    # the plan's canonical rejection case: Under 47.5 with Over 2.5 pass TDs
    assert pairwise_correlation(TOTAL_UNDER, PASS_TD_OVER) < 0


def test_flipping_a_side_flips_the_sign():
    total_over = leg(market="totals", selection="Over", line=47.5, team=None,
                     player_name=None, p_model=0.47)
    assert pairwise_correlation(total_over, PASS_TD_OVER) == pytest.approx(
        -pairwise_correlation(TOTAL_UNDER, PASS_TD_OVER)
    )


def test_cross_game_legs_are_independent():
    assert pairwise_correlation(QB, CROSS_GAME) == 0.0


def test_same_player_same_market_opposite_sides_is_perfectly_negative():
    over = leg(player_name="QB One", team="KC", p_model=0.56)
    under = leg(player_name="QB One", team="KC", selection="Under", p_model=0.44)
    assert pairwise_correlation(over, under) == pytest.approx(-0.95)  # clamped


def test_same_player_across_markets_is_tightly_coupled():
    rho = pairwise_correlation(QB, PASS_TD_OVER)
    assert rho > pairwise_correlation(QB, WR1)


def test_unknown_team_falls_back_to_a_mild_same_game_prior():
    unknown = leg(player_name="Mystery Man", team=None,
                  market="player_receptions", line=4.5)
    assert 0 < pairwise_correlation(unknown, QB) < 0.55


# --------------------------------------------------------------- matrix
def test_matrix_is_symmetric_unit_diagonal_and_psd():
    matrix = build_correlation_matrix([QB, WR1, TOTAL_UNDER, CROSS_GAME])
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0)
    assert np.min(np.linalg.eigvalsh(matrix)) > -1e-9


def test_nearest_psd_repairs_an_invalid_matrix():
    broken = np.array([[1.0, 0.95, -0.95], [0.95, 1.0, 0.95], [-0.95, 0.95, 1.0]])
    assert np.min(np.linalg.eigvalsh(broken)) < 0
    repaired = nearest_psd(broken)
    assert np.min(np.linalg.eigvalsh(repaired)) >= -1e-9
    assert np.allclose(np.diag(repaired), 1.0)


def test_summary_statistics_follow_the_pairs():
    legs = [TOTAL_UNDER, PASS_TD_OVER, WR1]
    assert min_correlation(legs) < 0
    assert min_correlation([QB, WR1]) == pytest.approx(pairwise_correlation(QB, WR1))
    assert average_correlation([QB]) == 0.0
    report = correlation_report([QB, WR1])
    assert report[0]["relation"] == "same_team"
    assert report[0]["same_game"] is True


# --------------------------------------------------------------- copula
def test_copula_reproduces_the_marginals():
    result = GaussianCopulaSimulator([QB, WR1], iterations=20_000, seed=3).simulate()
    assert result.marginal_probabilities[0] == pytest.approx(QB.p_model, abs=0.01)
    assert result.marginal_probabilities[1] == pytest.approx(WR1.p_model, abs=0.01)


def test_positive_correlation_lifts_the_joint_probability():
    result = GaussianCopulaSimulator([QB, WR1], iterations=20_000, seed=3).simulate()
    assert result.joint_probability > result.independent_probability
    assert result.correlation_lift > 1.05
    # a correlated pair still cannot beat its weakest leg
    assert result.joint_probability < min(QB.p_model, WR1.p_model)


def test_negative_correlation_depresses_the_joint_probability():
    result = GaussianCopulaSimulator(
        [TOTAL_UNDER, PASS_TD_OVER], iterations=20_000, seed=3
    ).simulate()
    assert result.joint_probability < result.independent_probability
    assert result.correlation_lift < 0.95


def test_independent_legs_match_the_product_of_probabilities():
    result = GaussianCopulaSimulator([QB, CROSS_GAME], iterations=40_000, seed=5).simulate()
    assert result.joint_probability == pytest.approx(
        QB.p_model * CROSS_GAME.p_model, abs=0.01
    )


def test_realised_correlation_tracks_the_latent_input():
    result = GaussianCopulaSimulator([QB, WR1], iterations=20_000, seed=3).simulate()
    realised = result.empirical_correlation[0, 1]
    assert realised > 0
    assert realised < result.correlation_matrix[0, 1]  # binarising attenuates it


def test_simulation_is_deterministic_for_a_seed():
    first = joint_probability([QB, WR1], iterations=5_000, seed=42)
    second = joint_probability([QB, WR1], iterations=5_000, seed=42)
    assert first == second


def test_single_leg_joint_probability_is_the_leg_probability():
    assert joint_probability([QB]) == QB.p_model


def test_simulator_rejects_empty_leg_lists():
    with pytest.raises(ValueError):
        GaussianCopulaSimulator([])
