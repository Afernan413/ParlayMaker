"""Distribution fitting and probability conversion tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.distributions import (
    DistributionSpec,
    baseline_probability,
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
