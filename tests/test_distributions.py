"""Distribution fitting and probability conversion tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.distributions import (
    DistributionSpec,
    baseline_probability,
    over_probabilities,
    scale_to_team_total,
    team_td_consistency,
)


def test_market_families_follow_config():
    assert DistributionSpec.for_market("player_pass_yds", 250).family == "lognormal"
    assert DistributionSpec.for_market("player_receptions", 5).family == "negative_binomial"
    assert DistributionSpec.for_market("player_pass_tds", 2).family == "poisson"
    assert DistributionSpec.for_market("player_points", 22).family == "negative_binomial"
    assert DistributionSpec.for_market("totals", 47).family == "normal"


def test_over_under_probabilities_sum_to_one():
    spec = DistributionSpec.for_market("player_reception_yds", 68.0)
    prob = spec.probability(64.5)
    assert prob.prob_over + prob.prob_under + prob.prob_push == pytest.approx(1.0, abs=1e-9)
    assert prob.prob_over_no_push + prob.prob_under_no_push == pytest.approx(1.0)


def test_integer_line_carves_out_the_push():
    spec = DistributionSpec.for_market("player_receptions", 6.0)
    prob = spec.probability(6.0)
    assert prob.prob_push > 0.05  # P(exactly 6) is material
    assert prob.prob_over_no_push == pytest.approx(
        prob.prob_over / (prob.prob_over + prob.prob_under)
    )
    # "over 6" and "over 6.5" both need 7+, but refunding the push makes the
    # integer line the better bet
    assert spec.probability(6.5).prob_over == pytest.approx(prob.prob_over)
    assert prob.prob_over_no_push > spec.probability(6.5).prob_over_no_push


def test_probability_rises_with_the_projection():
    low = baseline_probability("player_rush_yds", 55.0, 64.5, "Over")
    high = baseline_probability("player_rush_yds", 80.0, 64.5, "Over")
    assert 0.0 < low < high < 1.0


def test_under_is_the_complement_of_over():
    over = baseline_probability("player_points", 24.0, 25.5, "Over")
    under = baseline_probability("player_points", 24.0, 25.5, "Under")
    assert over + under == pytest.approx(1.0)


def test_negative_binomial_is_overdispersed_relative_to_poisson():
    nb = DistributionSpec(family="negative_binomial", mean=6.0, dispersion=1.45)
    poisson = DistributionSpec(family="poisson", mean=6.0)
    assert nb.variance > poisson.variance
    # fatter tails mean a distant over is likelier under the NB fit
    assert nb.prob_at_least(11) > poisson.prob_at_least(11)


def test_anytime_td_is_a_bernoulli_on_at_least_one():
    spec = DistributionSpec.for_market("player_anytime_td", 0.7)
    assert spec.prob_over(0.5) == pytest.approx(1 - np.exp(-0.7), abs=1e-6)
    assert spec.prob_over(None) == pytest.approx(spec.prob_over(0.5))


def test_continuous_samples_are_never_negative():
    rng = np.random.default_rng(11)
    for market, mean in (("player_rush_yds", 12.0), ("player_reception_yds", 8.0)):
        draws = DistributionSpec.for_market(market, mean).sample(20_000, rng)
        assert draws.min() >= 0.0


def test_inverse_cdf_transform_respects_support():
    spec = DistributionSpec.for_market("player_rush_yds", 60.0)
    uniforms = np.linspace(0.001, 0.999, 500)
    values = spec.from_uniform(uniforms)
    assert values.min() >= 0.0
    assert np.all(np.diff(values) >= -1e-9)  # monotone in the uniform


def test_zero_projection_cannot_clear_a_line():
    spec = DistributionSpec.for_market("player_rush_yds", 0.0)
    assert spec.prob_over(0.5) == 0.0
    assert spec.sample(10).tolist() == [0.0] * 10


def test_unsupported_selection_is_rejected():
    spec = DistributionSpec.for_market("player_points", 20.0)
    with pytest.raises(ValueError):
        spec.prob_over(19.5, "Maybe")


def test_td_rates_are_reconciled_with_the_team_total():
    # four scorers averaging 0.45 TDs each = 1.8 TDs vs a 24.5 team total (3.5)
    means = [0.45, 0.45, 0.45, 0.45]
    assert team_td_consistency(means, 24.5) < 1.0
    rich = [1.2, 1.0, 0.9, 0.8]
    assert team_td_consistency(rich, 24.5) > 1.0
    scaled = scale_to_team_total(rich, 24.5)
    assert team_td_consistency(scaled, 24.5) == pytest.approx(1.0)
    assert sum(scaled) == pytest.approx(24.5 / 7.0)


# ----------------------------------------------------------------------
# the batch path must agree with the per-row one
# ----------------------------------------------------------------------
ALL_MARKETS = (
    "player_rush_yds", "player_pass_yds", "player_reception_yds",
    "player_receptions", "player_pass_tds", "player_anytime_td",
    "totals", "spreads", "player_points", "player_rebounds", "player_threes",
)


def per_row(market, means, lines, *, sport=None):
    return np.array([
        DistributionSpec.for_market(market, mean, sport=sport).prob_over(line)
        for mean, line in zip(means, lines)
    ])


@pytest.mark.parametrize("market", ALL_MARKETS)
def test_the_batch_path_matches_the_per_row_path(market):
    """A second implementation of money maths, pinned rather than trusted."""
    rng = np.random.default_rng(4)
    means = rng.uniform(0.2, 300.0, 300)
    # Lines all over the distribution, snapped to halves as a book posts them.
    lines = np.round(rng.uniform(0.05, 4.0, means.size) * means * 2) / 2
    batch = over_probabilities(market, means, lines)
    assert batch == pytest.approx(per_row(market, means, lines), abs=1e-12)


@pytest.mark.parametrize("market", ("player_receptions", "player_pass_tds", "player_points"))
def test_the_batch_path_handles_a_push_line(market):
    """A whole-number line refunds a push, which the two paths must split alike."""
    means = np.array([1.0, 2.5, 5.0, 8.0, 12.0])
    lines = np.array([1.0, 2.0, 5.0, 8.0, 12.0])
    assert over_probabilities(market, means, lines) == pytest.approx(
        per_row(market, means, lines), abs=1e-12
    )


def test_the_batch_path_agrees_under_a_learned_correction():
    from src.models.calibration import (
        Calibration, MarketCalibration, SportCalibration, using_calibration,
    )

    fit = MarketCalibration(
        sport="nfl", market="player_rush_yds", mean_factor=0.94,
        dispersion=0.68, platt_a=0.47, platt_b=-0.32,
    )
    calibration = Calibration(
        sports={"nfl": SportCalibration(sport="nfl", markets={"player_rush_yds": fit})}
    )
    rng = np.random.default_rng(5)
    means = rng.uniform(5.0, 150.0, 200)
    lines = np.round(rng.uniform(0.2, 3.0, means.size) * means * 2) / 2
    with using_calibration(calibration):
        batch = over_probabilities("player_rush_yds", means, lines, sport="nfl")
        rows = per_row("player_rush_yds", means, lines, sport="nfl")
    assert batch == pytest.approx(rows, abs=1e-12)


def test_a_zero_projection_is_zero_either_way():
    """No projection is not the same as an impossible outcome, and a logistic
    correction of a hard zero would invent a probability out of nothing."""
    from src.models.calibration import (
        Calibration, MarketCalibration, SportCalibration, using_calibration,
    )

    fit = MarketCalibration(
        sport="nfl", market="player_rush_yds", platt_a=0.47, platt_b=-0.32
    )
    calibration = Calibration(
        sports={"nfl": SportCalibration(sport="nfl", markets={"player_rush_yds": fit})}
    )
    with using_calibration(calibration):
        spec = DistributionSpec.for_market("player_rush_yds", 0.0, sport="nfl")
        assert spec.prob_over(10.5) == 0.0
        assert over_probabilities("player_rush_yds", [0.0], [10.5], sport="nfl")[0] == 0.0


def test_the_batch_path_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        over_probabilities("player_rush_yds", [1.0, 2.0], [1.0])


def test_the_batch_path_handles_an_empty_slate():
    assert over_probabilities("player_rush_yds", [], []).size == 0


def test_the_batch_path_is_much_faster():
    """The reason it exists: training re-prices hundreds of thousands of rows."""
    import time

    rng = np.random.default_rng(6)
    means = rng.uniform(5.0, 120.0, 4_000)
    lines = np.round(means * 1.15 * 2) / 2

    start = time.perf_counter()
    over_probabilities("player_rush_yds", means, lines)
    batched = time.perf_counter() - start

    start = time.perf_counter()
    per_row("player_rush_yds", means[:400], lines[:400])
    one_at_a_time = (time.perf_counter() - start) * 10   # scaled to the same count

    assert batched * 10 < one_at_a_time, f"{batched:.3f}s batched vs {one_at_a_time:.3f}s"
