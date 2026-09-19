"""Fit the corrections in :mod:`src.models.calibration` from graded history.

The input is the walk-forward pairs from :mod:`src.learning.backtest`: for
every past player-week, what the model would have projected knowing only the
weeks before it, and what actually happened.

Three things are fitted per market, in the order they compose, because each
one changes what the next has left to explain:

1. **Mean bias.** ``sum(actual) / sum(projected)``. Summing rather than
   averaging ratios weights by volume, so a 3-yard week for a backup cannot
   outvote a 300-yard week for a starter.
2. **Dispersion.** Matched to the residual spread the family's ``variance``
   property is defined in terms of -- a coefficient of variation for the
   log-normals, a variance multiple for the counts, a standard deviation for
   the normals. This is the correction that matters most for longshots: a
   prior that is too tight makes the tail of the distribution too thin, and a
   thin tail underprices exactly the legs a big-payout parlay is built from.
3. **Platt scaling.** A logistic fit on the probability scale, mopping up
   whatever miscalibration the first two leave behind (a wrong family shape,
   the zero mass in a receiving-yards line, the fact that projections are
   shrunk estimates rather than true means).

Candidates are scored through the production code path -- the same
``DistributionSpec`` the engine prices with, under
:func:`~src.models.calibration.using_calibration` -- so there is no second
implementation to drift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import optimize

from src.learning.backtest import brier_score, calibration_table, log_loss
from src.models.calibration import (
    MIN_SAMPLES,
    Calibration,
    MarketCalibration,
    SportCalibration,
    expit,
    logit,
    utcnow,
    using_calibration,
)
from src.models.distributions import DistributionSpec, family_for

logger = logging.getLogger(__name__)

#: A Poisson market whose measured variance exceeds its mean by more than this
#: is promoted to a Negative Binomial. Poisson pins variance to the mean; when
#: reality is wider, keeping it throws away the tail.
POISSON_OVERDISPERSION_LIMIT = 1.15

#: Sane ranges for a fitted dispersion, by family. A fit outside these says
#: something is wrong with the data rather than with the prior.
DISPERSION_BOUNDS: dict[str, tuple[float, float]] = {
    "lognormal": (0.10, 2.50),
    "negative_binomial": (1.0 + 1e-6, 6.00),
    "normal": (1.0, 40.0),
}

#: Columns that identify one projection, regardless of how many lines were
#: probed against it.
RESIDUAL_KEYS = ("sport", "season", "week", "player", "market")

#: Game markets are fitted from a schedule rather than a box score: the
#: closing line is the projection, the played game is the outcome, and the
#: residual spread is the only free parameter.
GAME_MARKET_COLUMNS: dict[str, tuple[str, str]] = {
    "spreads": ("spread_line", "result"),
    "totals": ("total_line", "total"),
}


@dataclass(frozen=True)
class Scorecard:
    """How a set of probabilities did against what happened."""

    samples: int
    brier: float
    log_loss: float
    hit_rate: float
    claimed: float

    @property
    def bias(self) -> float:
        """Claimed minus observed. Positive means the model over-claims."""
        return self.claimed - self.hit_rate

    def as_row(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "brier": round(self.brier, 6),
            "log_loss": round(self.log_loss, 6),
            "hit_rate": round(self.hit_rate, 6),
            "claimed": round(self.claimed, 6),
        }


def score(probabilities: Sequence[float], outcomes: Sequence[int]) -> Scorecard:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    return Scorecard(
        samples=int(p.size),
        brier=brier_score(p, y),
        log_loss=log_loss(p, y),
        hit_rate=float(y.mean()) if y.size else 0.0,
        claimed=float(p.mean()) if p.size else 0.0,
    )


# ----------------------------------------------------------------------
# step 1 + 2: mean and spread
# ----------------------------------------------------------------------
def residuals(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per projection, dropping the repeated probe lines.

    ``backtest.observations`` emits five lines per projection so the
    distribution is tested across its body. For fitting a mean and a spread
    that is the same projection five times over, which would make the sample
    look five times bigger than it is.
    """
    keys = [key for key in RESIDUAL_KEYS if key in frame.columns]
    return frame.drop_duplicates(subset=keys)[keys + ["projected", "actual"]].copy()


def fit_mean_factor(projected: np.ndarray, actual: np.ndarray) -> float:
    """Volume-weighted ratio of what happened to what was projected."""
    total = float(np.sum(projected))
    return float(np.sum(actual)) / total if total > 0 else 1.0


def fit_dispersion(family: str, mu: np.ndarray, actual: np.ndarray) -> float | None:
    """Match the fitted spread to the residual spread, in the family's units.

    ``None`` for families that carry no dispersion parameter.
    """
    squared_error = np.sum((actual - mu) ** 2)
    if family == "lognormal":
        denominator = float(np.sum(mu**2))
        if denominator <= 0:
            return None
        return float(np.sqrt(squared_error / denominator))
    if family in {"negative_binomial", "poisson"}:
        denominator = float(np.sum(mu))
        if denominator <= 0:
            return None
        return float(squared_error / denominator)
    if family == "normal":
        return float(np.sqrt(squared_error / max(mu.size, 1)))
    return None


def _bounded(family: str, value: float | None) -> float | None:
    bounds = DISPERSION_BOUNDS.get(family)
    if value is None or bounds is None:
        return value
    return float(np.clip(value, *bounds))


def fit_shape(sport: str, market: str, rows: pd.DataFrame) -> MarketCalibration:
    """Fit the mean factor, dispersion and any family promotion for a market."""
    projected = rows["projected"].to_numpy(dtype=float)
    actual = rows["actual"].to_numpy(dtype=float)
    family = family_for(market)

    mean_factor = fit_mean_factor(projected, actual)
    mu = projected * mean_factor

    measured = fit_dispersion(family, mu, actual)
    promoted: str | None = None
    dispersion = measured
    variance_multiple: float | None = None

    if family in {"poisson", "negative_binomial"}:
        variance_multiple = measured
        if family == "poisson":
            if measured is not None and measured > POISSON_OVERDISPERSION_LIMIT:
                promoted = "negative_binomial"
                dispersion = _bounded("negative_binomial", measured)
            else:
                # Poisson has no dispersion parameter; nothing to store.
                dispersion = None
        else:
            dispersion = _bounded("negative_binomial", measured)
    elif family == "poisson_binary":
        # The mean is a scoring rate; its spread is not a free parameter.
        dispersion = None
    else:
        dispersion = _bounded(family, measured)

    return MarketCalibration(
        sport=sport,
        market=market,
        samples=int(len(rows)),
        mean_factor=mean_factor,
        dispersion=dispersion,
        family=promoted,
        variance_multiple=variance_multiple,
    ).clipped()


def fit_game_markets(
    schedule: pd.DataFrame, *, sport: str, min_samples: int = MIN_SAMPLES
) -> dict[str, MarketCalibration]:
    """How far a played game lands from the line the book closed at.

    ``NFL_SCORE_SD`` and ``COLLEGE_SCORE_SD`` in :mod:`src.models.baseline` are
    guesses at this number, and everything priced on a total or a spread is
    priced off it. Measuring it needs no projection model at all: the closing
    line already is the market's mean, so the residual around it is the spread
    the normal family should be given.
    """
    fitted: dict[str, MarketCalibration] = {}
    for market, (line_column, outcome_column) in GAME_MARKET_COLUMNS.items():
        if line_column not in schedule.columns or outcome_column not in schedule.columns:
            continue
        rows = schedule[[line_column, outcome_column]].dropna()
        if len(rows) < min_samples:
            logger.info("skipping %s/%s: only %d played games", sport, market, len(rows))
            continue
        residual = rows[outcome_column].to_numpy(dtype=float) - rows[line_column].to_numpy(dtype=float)
        fitted[market] = MarketCalibration(
            sport=sport,
            market=market,
            samples=int(len(rows)),
            dispersion=_bounded("normal", float(np.std(residual, ddof=1))),
        ).clipped()
    return fitted


# ----------------------------------------------------------------------
# step 3: probability recalibration
# ----------------------------------------------------------------------
def fit_platt(probabilities: Sequence[float], outcomes: Sequence[int]) -> tuple[float, float]:
    """Logistic regression of outcomes on the claimed log-odds.

    Returns ``(a, b)`` for ``p' = sigmoid(a * logit(p) + b)``. ``(1, 0)`` is
    the identity, which is what a perfectly calibrated model would fit.
    """
    x = logit(probabilities)
    y = np.asarray(outcomes, dtype=float)
    if x.size == 0 or len(np.unique(y)) < 2:
        return 1.0, 0.0

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        a, b = float(theta[0]), float(theta[1])
        z = a * x + b
        # log(1 + exp(z)) written so large |z| cannot overflow.
        loss = float(np.mean(np.logaddexp(0.0, z) - y * z))
        residual = expit(z) - y
        return loss, np.array([float(np.mean(residual * x)), float(np.mean(residual))])

    result = optimize.minimize(
        objective, x0=np.array([1.0, 0.0]), jac=True, method="L-BFGS-B",
        bounds=[(0.05, 8.0), (-5.0, 5.0)],
    )
    if not result.success:
        logger.warning("Platt fit did not converge (%s); leaving it as identity", result.message)
        return 1.0, 0.0
    return float(result.x[0]), float(result.x[1])


def probabilities_under(
    frame: pd.DataFrame, calibration: Calibration, *, sport: str
) -> np.ndarray:
    """Re-price every observation with ``calibration`` in force.

    Goes through ``DistributionSpec`` exactly as the engine does, so a fit is
    scored on the code that will actually use it.
    """
    out = np.empty(len(frame), dtype=float)
    with using_calibration(calibration):
        columns = frame[["market", "projected", "line"]].itertuples(index=False, name=None)
        for position, (market, projected, line) in enumerate(columns):
            spec = DistributionSpec.for_market(market, float(projected), sport=sport)
            out[position] = spec.prob_over(float(line))
    return np.clip(out, 1e-9, 1 - 1e-9)


# ----------------------------------------------------------------------
# the whole fit
# ----------------------------------------------------------------------
def fit_sport(
    frame: pd.DataFrame,
    *,
    sport: str,
    min_samples: int = MIN_SAMPLES,
    seasons: Iterable[int] = (),
    game_markets: dict[str, MarketCalibration] | None = None,
) -> SportCalibration:
    """Fit every market in ``frame``, shape first and probabilities second."""
    shapes: dict[str, MarketCalibration] = {}
    spread = residuals(frame)
    for market, rows in spread.groupby("market", sort=True):
        if len(rows) < min_samples:
            logger.info("skipping %s/%s: only %d residuals", sport, market, len(rows))
            continue
        shapes[market] = fit_shape(sport, market, rows)

    # With the shapes in place, whatever miscalibration is left is what Platt
    # has to explain -- so the probabilities are recomputed before it is fitted.
    shaped = Calibration(sports={sport: SportCalibration(sport=sport, markets=shapes)})
    corrected = probabilities_under(frame, shaped, sport=sport)

    markets: dict[str, MarketCalibration] = {}
    for market, calibrated in shapes.items():
        mask = (frame["market"] == market).to_numpy()
        if mask.sum() < min_samples:
            markets[market] = calibrated
            continue
        a, b = fit_platt(corrected[mask], frame.loc[mask, "hit"].to_numpy(dtype=float))
        markets[market] = MarketCalibration(
            sport=calibrated.sport,
            market=calibrated.market,
            samples=calibrated.samples,
            mean_factor=calibrated.mean_factor,
            dispersion=calibrated.dispersion,
            family=calibrated.family,
            platt_a=a,
            platt_b=b,
            variance_multiple=calibrated.variance_multiple,
        ).clipped()

    markets.update(game_markets or {})
    return SportCalibration(
        sport=sport,
        markets=markets,
        fitted_at=utcnow(),
        seasons=tuple(sorted(set(int(s) for s in seasons))),
    )


def split_by_week(frame: pd.DataFrame, holdout_weeks: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on the earlier weeks, test on the last ``holdout_weeks``.

    Splitting on time rather than at random is the point: a random split lets
    the fit see the same week it is scored on through another player.
    """
    if frame.empty or holdout_weeks <= 0:
        return frame, frame.iloc[0:0]
    stamps = frame[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
    if len(stamps) <= holdout_weeks:
        return frame, frame.iloc[0:0]
    cutoff = stamps.iloc[-holdout_weeks]
    later = (frame["season"] > cutoff["season"]) | (
        (frame["season"] == cutoff["season"]) & (frame["week"] >= cutoff["week"])
    )
    return frame[~later].copy(), frame[later].copy()


@dataclass(frozen=True)
class TrainingReport:
    """What a fit learned and whether it actually helped."""

    sport: str
    calibration: SportCalibration
    before: Scorecard
    after: Scorecard
    holdout_weeks: int
    train_samples: int

    @property
    def improved(self) -> bool:
        return self.after.brier < self.before.brier

    @property
    def brier_gain(self) -> float:
        """Fraction of the Brier score removed. Positive is better."""
        return (self.before.brier - self.after.brier) / self.before.brier if self.before.brier else 0.0

    def metrics(self) -> dict[str, Any]:
        return {
            "holdout_weeks": self.holdout_weeks,
            "train_samples": self.train_samples,
            "before": self.before.as_row(),
            "after": self.after.as_row(),
            "brier_gain": round(self.brier_gain, 6),
        }


def train(
    frame: pd.DataFrame,
    *,
    sport: str,
    holdout_weeks: int = 4,
    min_samples: int = MIN_SAMPLES,
    seasons: Iterable[int] = (),
    game_markets: dict[str, MarketCalibration] | None = None,
) -> TrainingReport:
    """Fit on the earlier weeks and score the fit on the weeks held back.

    The returned calibration is refitted on everything once the holdout has
    had its say, so no week of history is thrown away -- but the scorecards
    come from the split, and they are the only honest evidence that the
    corrections help.
    """
    train_frame, test_frame = split_by_week(frame, holdout_weeks)
    trial = fit_sport(train_frame, sport=sport, min_samples=min_samples, seasons=seasons)

    scored = test_frame if not test_frame.empty else train_frame
    outcomes = scored["hit"].to_numpy(dtype=float)
    before = score(scored["p_over"].to_numpy(dtype=float), outcomes)
    after = score(
        probabilities_under(
            scored, Calibration(sports={sport: trial}), sport=sport
        ),
        outcomes,
    )

    final = fit_sport(
        frame,
        sport=sport,
        min_samples=min_samples,
        seasons=seasons,
        game_markets=game_markets,
    )
    report = TrainingReport(
        sport=sport,
        calibration=final,
        before=before,
        after=after,
        holdout_weeks=holdout_weeks,
        train_samples=int(len(train_frame)),
    )
    return TrainingReport(
        sport=sport,
        calibration=SportCalibration(
            sport=sport,
            markets=final.markets,
            fitted_at=final.fitted_at,
            seasons=final.seasons,
            metrics=report.metrics(),
        ),
        before=before,
        after=after,
        holdout_weeks=holdout_weeks,
        train_samples=int(len(train_frame)),
    )


def calibration_comparison(
    frame: pd.DataFrame,
    calibration: Calibration | None = None,
    *,
    sport: str,
    bins: int = 10,
    corrected: np.ndarray | None = None,
) -> pd.DataFrame:
    """Claimed-vs-observed tables before and after, side by side.

    Pass ``corrected`` to reuse a pass of :func:`probabilities_under` that has
    already been done; re-pricing the frame is the expensive part.
    """
    if corrected is None:
        if calibration is None:
            raise ValueError("pass either a calibration or precomputed probabilities")
        corrected = probabilities_under(frame, calibration, sport=sport)
    outcomes = frame["hit"].to_numpy(dtype=float)
    raw = calibration_table(frame["p_over"].to_numpy(dtype=float), outcomes, bins=bins)
    fixed = calibration_table(corrected, outcomes, bins=bins)
    raw = raw.rename(columns={"claimed": "claimed_before", "observed": "observed_before", "gap": "gap_before"})
    fixed = fixed.rename(columns={"claimed": "claimed_after", "observed": "observed_after", "gap": "gap_after"})
    return raw.merge(fixed, on="bucket", how="outer", suffixes=("", "_after"))
