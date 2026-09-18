"""Intra-game correlation engine -- the foundation for same-game parlays.

Leg dependencies are expressed as latent (Gaussian) correlations drawn from
:data:`CORRELATION_PRIORS`, then pushed through a Gaussian copula:

1. Build a correlation matrix for the legs (repaired to nearest PSD).
2. Draw correlated standard normals.
3. A leg hits when its latent draw falls below ``Phi^-1(p_model)``.

Because the marginals come straight from each leg's model probability, the
simulation reproduces the single-leg probabilities exactly and only adds the
dependency structure -- which is what a parlay price needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
from scipy import stats

from config.settings import market_meta, settings
from src.models.legs import Leg

#: Mild positive default for two legs in the same game with no explicit prior
#: (shared game script, shared officiating, shared weather).
DEFAULT_SAME_GAME = 0.10

#: Latent correlation between two canonical stats, keyed by the sorted stat
#: pair, for legs taken in the same direction (both Over / both Yes). Signs are
#: flipped per leg direction by :func:`direction_sign`.
CORRELATION_PRIORS: dict[tuple[str, str], dict[str, float]] = {
    # --- NFL passing game: the classic correlated SGP core ---------------
    ("passing_yards", "receiving_yards"): {"same_team": 0.55, "opposing": 0.15, "unknown": 0.25},
    ("passing_yards", "receptions"): {"same_team": 0.45, "opposing": 0.10, "unknown": 0.20},
    ("passing_tds", "receiving_yards"): {"same_team": 0.40, "opposing": 0.10, "unknown": 0.18},
    ("anytime_td", "passing_tds"): {"same_team": 0.35, "opposing": -0.05, "unknown": 0.12},
    ("receiving_yards", "receptions"): {"same_team": 0.35, "opposing": 0.10, "unknown": 0.18},
    ("receiving_yards", "receiving_yards"): {"same_team": 0.20, "opposing": 0.15, "unknown": 0.15},
    # --- NFL rushing / game script --------------------------------------
    ("passing_yards", "rushing_yards"): {"same_team": -0.10, "opposing": 0.15, "unknown": 0.05},
    ("rushing_yards", "spread"): {"same_team": 0.30, "opposing": -0.25, "unknown": 0.05},
    ("passing_yards", "spread"): {"same_team": 0.20, "opposing": -0.15, "unknown": 0.05},
    ("anytime_td", "spread"): {"same_team": 0.25, "opposing": -0.20, "unknown": 0.05},
    ("moneyline", "rushing_yards"): {"same_team": 0.28, "opposing": -0.22, "unknown": 0.05},
    ("moneyline", "spread"): {"same_team": 0.80, "opposing": -0.80, "unknown": 0.00},
    # --- game total couplings -------------------------------------------
    ("game_total", "passing_yards"): {"same_team": 0.35, "opposing": 0.35, "unknown": 0.35},
    ("game_total", "receiving_yards"): {"same_team": 0.32, "opposing": 0.32, "unknown": 0.32},
    ("game_total", "receptions"): {"same_team": 0.22, "opposing": 0.22, "unknown": 0.22},
    ("anytime_td", "game_total"): {"same_team": 0.30, "opposing": 0.30, "unknown": 0.30},
    ("game_total", "passing_tds"): {"same_team": 0.38, "opposing": 0.38, "unknown": 0.38},
    ("game_total", "rushing_yards"): {"same_team": 0.12, "opposing": 0.12, "unknown": 0.12},
    ("game_total", "points"): {"same_team": 0.34, "opposing": 0.34, "unknown": 0.34},
    ("game_total", "threes"): {"same_team": 0.26, "opposing": 0.26, "unknown": 0.26},
    ("assists", "game_total"): {"same_team": 0.24, "opposing": 0.24, "unknown": 0.24},
    ("game_total", "rebounds"): {"same_team": -0.10, "opposing": -0.10, "unknown": -0.10},
    # --- NBA -------------------------------------------------------------
    ("points", "points"): {"same_team": -0.08, "opposing": 0.12, "unknown": 0.05},
    ("assists", "points"): {"same_team": 0.30, "opposing": 0.10, "unknown": 0.15},
    ("points", "threes"): {"same_team": 0.25, "opposing": 0.08, "unknown": 0.12},
    ("points", "rebounds"): {"same_team": 0.05, "opposing": 0.05, "unknown": 0.05},
    ("rebounds", "rebounds"): {"same_team": -0.05, "opposing": -0.10, "unknown": -0.05},
    ("assists", "assists"): {"same_team": -0.05, "opposing": 0.08, "unknown": 0.0},
}

#: Same player, two different markets -- much tighter than teammate pairs.
SAME_PLAYER_PRIORS: dict[tuple[str, str], float] = {
    ("receiving_yards", "receptions"): 0.78,
    ("passing_tds", "passing_yards"): 0.62,
    ("anytime_td", "rushing_yards"): 0.55,
    ("anytime_td", "receiving_yards"): 0.52,
    ("passing_yards", "rushing_yards"): 0.15,
    ("points", "threes"): 0.62,
    ("assists", "points"): 0.35,
    ("points", "rebounds"): 0.30,
    ("assists", "rebounds"): 0.22,
    ("rebounds", "threes"): 0.05,
    ("assists", "threes"): 0.20,
}

SAME_PLAYER_DEFAULT = 0.45
MAX_ABS_CORRELATION = 0.95

OVER_SIDES = frozenset({"over", "yes"})
UNDER_SIDES = frozenset({"under", "no"})


def stat_for(leg: Leg) -> str:
    """Canonical stat name behind a leg's market."""
    return market_meta(leg.market)["stat"]


def direction_sign(leg: Leg) -> int:
    """``+1`` for Over/Yes/team-to-win, ``-1`` for Under/No.

    Priors are stated for same-direction pairs; flipping a leg to the Under
    flips the sign of every correlation it participates in.
    """
    side = (leg.selection or "").strip().lower()
    if side in UNDER_SIDES:
        return -1
    return 1


def _relation(leg_a: Leg, leg_b: Leg) -> str:
    if leg_a.team and leg_b.team:
        return "same_team" if leg_a.team == leg_b.team else "opposing"
    return "unknown"


def _prior(stat_a: str, stat_b: str, relation: str) -> float:
    key = tuple(sorted((stat_a, stat_b)))
    entry = CORRELATION_PRIORS.get(key)
    if entry is None:
        return DEFAULT_SAME_GAME
    return entry.get(relation, entry.get("unknown", DEFAULT_SAME_GAME))


def pairwise_correlation(leg_a: Leg, leg_b: Leg) -> float:
    """Latent correlation between two legs.

    Cross-game legs are treated as independent -- the whole point of the
    diversified (non-SGP) ticket.
    """
    if leg_a.game_id != leg_b.game_id:
        return 0.0

    stat_a, stat_b = stat_for(leg_a), stat_for(leg_b)
    same_subject = (
        leg_a.player_name is not None and leg_a.player_name == leg_b.player_name
    )

    if same_subject:
        if leg_a.market == leg_b.market:
            # Same player, same market: one latent variable drives both legs,
            # so the direction signs below turn Over/Under into a -1 pair.
            base = 1.0
        else:
            base = SAME_PLAYER_PRIORS.get(
                tuple(sorted((stat_a, stat_b))), SAME_PLAYER_DEFAULT
            )
    else:
        base = _prior(stat_a, stat_b, _relation(leg_a, leg_b))

    signed = base * direction_sign(leg_a) * direction_sign(leg_b)
    return float(np.clip(signed, -MAX_ABS_CORRELATION, MAX_ABS_CORRELATION))


def nearest_psd(matrix: np.ndarray, *, epsilon: float = 1e-8) -> np.ndarray:
    """Nearest positive semi-definite correlation matrix (eigenvalue clip)."""
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.clip(eigenvalues, epsilon, None)
    repaired = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    scale = np.sqrt(np.clip(np.diag(repaired), epsilon, None))
    normalised = repaired / np.outer(scale, scale)
    np.fill_diagonal(normalised, 1.0)
    return normalised


def build_correlation_matrix(legs: Sequence[Leg]) -> np.ndarray:
    """Latent correlation matrix for ``legs``, repaired to be PSD."""
    size = len(legs)
    matrix = np.eye(size)
    for i, j in combinations(range(size), 2):
        rho = pairwise_correlation(legs[i], legs[j])
        matrix[i, j] = matrix[j, i] = rho
    if size <= 1:
        return matrix
    if np.min(np.linalg.eigvalsh(matrix)) < 1e-10:
        return nearest_psd(matrix)
    return matrix


def average_correlation(legs: Sequence[Leg]) -> float:
    """Mean off-diagonal correlation -- a ticket's overall couplinag strength."""
    if len(legs) < 2:
        return 0.0
    values = [pairwise_correlation(a, b) for a, b in combinations(legs, 2)]
    return float(np.mean(values))


def min_correlation(legs: Sequence[Leg]) -> float:
    """Weakest (most negative) pair in a ticket."""
    if len(legs) < 2:
        return 0.0
    return float(min(pairwise_correlation(a, b) for a, b in combinations(legs, 2)))


@dataclass
class SimulationResult:
    """Output of one copula run over a set of legs."""

    joint_probability: float
    independent_probability: float
    marginal_probabilities: list[float]
    empirical_correlation: np.ndarray
    correlation_matrix: np.ndarray
    iterations: int
    legs: list[str] = field(default_factory=list)

    @property
    def correlation_lift(self) -> float:
        """Joint probability relative to naive independence."""
        if self.independent_probability <= 0:
            return 0.0
        return self.joint_probability / self.independent_probability


class GaussianCopulaSimulator:
    """Monte-Carlo joint probabilities for a set of correlated legs."""

    def __init__(
        self,
        legs: Sequence[Leg],
        *,
        iterations: int | None = None,
        seed: int | None = None,
        correlation_matrix: np.ndarray | None = None,
    ) -> None:
        if not legs:
            raise ValueError("at least one leg is required")
        self.legs = list(legs)
        self.iterations = iterations or settings.copula_iterations
        self.seed = settings.random_seed if seed is None else seed
        self.correlation_matrix = (
            build_correlation_matrix(self.legs)
            if correlation_matrix is None
            else np.asarray(correlation_matrix, dtype=float)
        )

    def _latent_draws(self, rng: np.random.Generator) -> np.ndarray:
        """Correlated normals, drawn with antithetic pairs.

        Mirroring each draw about the origin removes most of the sampling
        error in the marginals, so the simulated single-leg probabilities stay
        faithful to ``p_model`` at 10k iterations.
        """
        matrix = self.correlation_matrix
        try:
            chol = np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            chol = np.linalg.cholesky(nearest_psd(matrix))
        half = (self.iterations + 1) // 2
        raw = rng.standard_normal((half, len(self.legs)))
        paired = np.concatenate([raw, -raw], axis=0)[: self.iterations]
        return paired @ chol.T

    def simulate(self) -> SimulationResult:
        """Run the copula and report joint / marginal / realised correlation."""
        rng = np.random.default_rng(self.seed)
        probabilities = np.array([max(min(leg.p_model, 1.0), 0.0) for leg in self.legs])
        thresholds = stats.norm.ppf(np.clip(probabilities, 1e-9, 1 - 1e-9))
        latent = self._latent_draws(rng)
        hits = latent <= thresholds  # (iterations, legs) boolean

        all_hit = hits.all(axis=1)
        empirical = (
            np.corrcoef(hits.astype(float), rowvar=False)
            if len(self.legs) > 1
            else np.array([[1.0]])
        )
        return SimulationResult(
            joint_probability=float(all_hit.mean()),
            independent_probability=float(np.prod(probabilities)),
            marginal_probabilities=[float(v) for v in hits.mean(axis=0)],
            empirical_correlation=np.nan_to_num(empirical),
            correlation_matrix=self.correlation_matrix,
            iterations=self.iterations,
            legs=[leg.leg_id for leg in self.legs],
        )


def joint_probability(
    legs: Sequence[Leg], *, iterations: int | None = None, seed: int | None = None
) -> float:
    """Copula joint probability that every leg in ``legs`` cashes."""
    if len(legs) == 1:
        return float(legs[0].p_model)
    return GaussianCopulaSimulator(
        legs, iterations=iterations, seed=seed
    ).simulate().joint_probability


def correlation_report(legs: Sequence[Leg]) -> list[dict[str, object]]:
    """Human-readable pair-by-pair correlation listing for a ticket."""
    report: list[dict[str, object]] = []
    for leg_a, leg_b in combinations(legs, 2):
        report.append(
            {
                "leg_a": leg_a.describe(),
                "leg_b": leg_b.describe(),
                "same_game": leg_a.game_id == leg_b.game_id,
                "relation": _relation(leg_a, leg_b),
                "correlation": round(pairwise_correlation(leg_a, leg_b), 4),
            }
        )
    return report


def flatten(legs: Iterable[Leg]) -> list[Leg]:
    """Utility: materialise an iterable of legs preserving order."""
    return list(legs)
