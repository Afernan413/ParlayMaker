"""Distribution fitting and raw win-probability estimation.

Each market maps to a family (see ``config/bookmaker_keys.json``):

* discrete counts  -> Poisson or Negative Binomial (overdispersed counts)
* continuous yards -> Log-Normal (strictly non-negative, right-skewed)
* game totals/spreads -> Normal

Integer lines are handled explicitly: FanDuel refunds a push, so the bettable
probability is renormalised over the non-push outcomes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import stats

from config.settings import bookmaker_config, market_meta
from src.models.calibration import MarketCalibration, calibration_for

CONTINUOUS_FAMILIES = frozenset({"lognormal", "normal"})
DISCRETE_FAMILIES = frozenset({"poisson", "negative_binomial", "poisson_binary"})


def dispersion_for(stat: str) -> float | None:
    """Prior dispersion for a canonical stat (CV, variance multiple or SD)."""
    return bookmaker_config()["dispersion"].get(stat)


def family_for(market: str) -> str:
    return market_meta(market)["family"]


def default_dispersion(market: str) -> float | None:
    return dispersion_for(market_meta(market)["stat"])


@dataclass(frozen=True)
class MarketProbability:
    """Over/under split for one line, with the push carved out."""

    prob_over: float
    prob_under: float
    prob_push: float = 0.0

    @property
    def prob_over_no_push(self) -> float:
        """Over probability conditional on the bet resolving (push refunded)."""
        live = self.prob_over + self.prob_under
        return self.prob_over / live if live > 0 else 0.0

    @property
    def prob_under_no_push(self) -> float:
        live = self.prob_over + self.prob_under
        return self.prob_under / live if live > 0 else 0.0

    def for_selection(self, selection: str) -> float:
        """Bettable probability for ``Over``/``Under``/``Yes``/``No``."""
        side = (selection or "").strip().lower()
        if side in {"over", "yes"}:
            return self.prob_over_no_push
        if side in {"under", "no"}:
            return self.prob_under_no_push
        raise ValueError(f"unsupported selection: {selection!r}")


@dataclass(frozen=True)
class DistributionSpec:
    """A fitted distribution for one player-market.

    ``dispersion`` means:

    * ``lognormal`` -- coefficient of variation (sd / mean)
    * ``normal``    -- standard deviation in points
    * ``negative_binomial`` -- variance multiple (var = mean * dispersion)
    * ``poisson`` / ``poisson_binary`` -- unused

    ``calibration`` is the market's learned correction, when one has been
    fitted. It has already been folded into ``mean``, ``dispersion`` and
    ``family``; the spec keeps it so that :meth:`probability` can also apply
    the recalibration that lives on the probability scale.
    """

    family: str
    mean: float
    dispersion: float | None = None
    calibration: MarketCalibration | None = None

    # -- construction -------------------------------------------------
    @classmethod
    def for_market(
        cls,
        market: str,
        mean: float,
        dispersion: float | None = None,
        *,
        sport: str | None = None,
    ) -> "DistributionSpec":
        """Build the spec a market's family calls for.

        Pass ``sport`` to apply the corrections learned for that sport's
        market (see :mod:`src.models.calibration`). Without it the raw priors
        from ``config/bookmaker_keys.json`` are used, which is what the
        training code wants when it is measuring the uncorrected model.

        An explicitly supplied ``dispersion`` always wins: the caller has
        measured this particular player, the calibration only replaces the
        market-wide prior.
        """
        fitted = calibration_for(sport, market)
        family = family_for(market)
        centre = float(mean)
        spread = dispersion

        if fitted is not None:
            centre = fitted.adjust_mean(centre)
            if fitted.family:
                family = fitted.family
            if spread is None and fitted.dispersion is not None:
                spread = fitted.dispersion

        return cls(
            family=family,
            mean=centre,
            dispersion=spread if spread is not None else default_dispersion(market),
            calibration=fitted,
        )

    # -- helpers ------------------------------------------------------
    @property
    def discrete(self) -> bool:
        return self.family in DISCRETE_FAMILIES

    @property
    def variance(self) -> float:
        if self.mean <= 0:
            return 0.0
        if self.family == "poisson":
            return self.mean
        if self.family == "poisson_binary":
            return self.mean
        if self.family == "negative_binomial":
            return self.mean * max(self.dispersion or 1.0, 1.0 + 1e-9)
        if self.family == "normal":
            return float(self.dispersion or 1.0) ** 2
        if self.family == "lognormal":
            return (self.mean * float(self.dispersion or 0.5)) ** 2
        raise ValueError(f"unknown family: {self.family}")

    @property
    def sd(self) -> float:
        return math.sqrt(self.variance)

    def _frozen(self):
        """The scipy frozen distribution backing this spec."""
        if self.family == "poisson":
            return stats.poisson(mu=max(self.mean, 1e-9))
        if self.family == "negative_binomial":
            mu = max(self.mean, 1e-9)
            var = max(self.variance, mu * (1 + 1e-6))
            p = mu / var
            n = mu * p / (1.0 - p)
            return stats.nbinom(n=n, p=p)
        if self.family == "normal":
            return stats.norm(loc=self.mean, scale=max(self.sd, 1e-9))
        if self.family == "lognormal":
            cv = max(float(self.dispersion or 0.5), 1e-6)
            sigma = math.sqrt(math.log(1.0 + cv**2))
            scale = max(self.mean, 1e-9) / math.exp(sigma**2 / 2.0)
            return stats.lognorm(s=sigma, scale=scale)
        if self.family == "poisson_binary":
            # Bernoulli on "at least one", driven by an underlying Poisson rate.
            return stats.poisson(mu=max(self.mean, 1e-9))
        raise ValueError(f"unknown family: {self.family}")

    # -- probabilities ------------------------------------------------
    def probability(self, line: float | None) -> MarketProbability:
        """Over / under / push split against ``line``, recalibrated."""
        return self._recalibrated(self._raw_probability(line))

    def _recalibrated(self, raw: MarketProbability) -> MarketProbability:
        """Apply the learned probability correction, holding the push fixed.

        A push is refunded, so only the two bettable outcomes are rescaled;
        their share of the total is what the correction was fitted on.
        """
        fitted = self.calibration
        if fitted is None or not fitted.shifts_probability:
            return raw
        live = raw.prob_over + raw.prob_under
        if live <= 0:
            return raw
        corrected = fitted.adjust_probability(raw.prob_over / live)
        return MarketProbability(
            prob_over=corrected * live,
            prob_under=(1.0 - corrected) * live,
            prob_push=raw.prob_push,
        )

    def _raw_probability(self, line: float | None) -> MarketProbability:
        """The family's own split, before any learned correction."""
        if line is None:
            # Binary market (e.g. anytime TD) -- "over" is "at least one".
            p_yes = self.prob_at_least(1)
            return MarketProbability(prob_over=p_yes, prob_under=1.0 - p_yes)

        line = float(line)
        if self.mean <= 0:
            return MarketProbability(prob_over=0.0, prob_under=1.0)

        if self.family == "poisson_binary":
            threshold = math.ceil(line + 1e-9) if line % 1 else int(line) + 1
            p_yes = self.prob_at_least(max(threshold, 1))
            return MarketProbability(prob_over=p_yes, prob_under=1.0 - p_yes)

        if self.discrete:
            if abs(line - round(line)) < 1e-9:  # integer line -> push possible
                k = int(round(line))
                frozen = self._frozen()
                push = float(frozen.pmf(k))
                over = float(frozen.sf(k))
                return MarketProbability(
                    prob_over=over, prob_under=max(1.0 - over - push, 0.0), prob_push=push
                )
            threshold = math.ceil(line)
            over = self.prob_at_least(threshold)
            return MarketProbability(prob_over=over, prob_under=1.0 - over)

        over = float(self._frozen().sf(line))
        return MarketProbability(prob_over=over, prob_under=1.0 - over)

    def prob_at_least(self, k: int) -> float:
        """``P(X >= k)`` for the discrete families."""
        if self.mean <= 0:
            return 0.0
        frozen = self._frozen()
        if k <= 0:
            return 1.0
        return float(frozen.sf(k - 1))

    def prob_over(self, line: float | None, selection: str = "Over") -> float:
        """Bettable probability for one side of ``line``."""
        return self.probability(line).for_selection(selection)

    # -- simulation ---------------------------------------------------
    def sample(self, size: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw samples; continuous families are clipped at zero."""
        rng = rng or np.random.default_rng()
        if self.mean <= 0:
            return np.zeros(size)
        draws = self._frozen().rvs(size=size, random_state=rng)
        values = np.asarray(draws, dtype=float)
        return np.clip(values, 0.0, None) if not self.discrete else values

    def from_uniform(self, uniform: np.ndarray) -> np.ndarray:
        """Inverse-CDF transform of copula uniforms into this marginal."""
        if self.mean <= 0:
            return np.zeros_like(uniform, dtype=float)
        clipped = np.clip(uniform, 1e-12, 1 - 1e-12)
        values = np.asarray(self._frozen().ppf(clipped), dtype=float)
        return values if self.discrete else np.clip(values, 0.0, None)


def spec_from_projection(projection) -> DistributionSpec:
    """Build a spec from a :class:`~src.models.legs.Projection`."""
    return DistributionSpec.for_market(
        projection.market, projection.mean, projection.dispersion, sport=projection.sport
    )


def baseline_probability(
    market: str,
    mean: float,
    line: float | None,
    selection: str,
    dispersion: float | None = None,
    *,
    sport: str | None = None,
) -> float:
    """P_base: unadjusted fair probability of a selection against a line."""
    spec = DistributionSpec.for_market(market, mean, dispersion, sport=sport)
    return spec.prob_over(line, selection)


# ----------------------------------------------------------------------
# sanity guards (mirrors the "Distribution Check" item in the plan)
# ----------------------------------------------------------------------
def team_td_consistency(
    td_means: Iterable[float], implied_team_total: float, *, points_per_td: float = 7.0
) -> float:
    """Ratio of summed anytime-TD rates to the TDs the team total implies.

    ``1.0`` means the individual scorer rates exactly reconcile with the team
    total; above ``1.0`` the book's props are collectively richer than the
    game total supports.
    """
    implied_tds = max(implied_team_total, 1e-9) / points_per_td
    return float(sum(td_means)) / implied_tds


def scale_to_team_total(
    td_means: Sequence[float], implied_team_total: float, *, points_per_td: float = 7.0
) -> list[float]:
    """Rescale scorer rates so they reconcile with the implied team total."""
    ratio = team_td_consistency(td_means, implied_team_total, points_per_td=points_per_td)
    if ratio <= 0:
        return list(td_means)
    return [value / ratio for value in td_means]
