"""Vig removal, expected value and stake sizing.

FanDuel prices carry a margin: the two sides of a market sum to more than 100%
implied probability. Removing that margin gives the market's own view
(:math:`P_{implied}`), which is what our model probability has to beat before a
bet is worth making.

Three de-vig methods are provided. ``power`` is the default because it handles
favourite-longshot bias better than splitting the overround evenly, which
matters for the +200/+650 range parlays live in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from scipy import optimize

from config.settings import settings
from src.models.legs import Leg

DEVIG_METHODS = ("power", "multiplicative", "additive")


# ----------------------------------------------------------------------
# odds conversions
# ----------------------------------------------------------------------
def american_to_decimal(american: float) -> float:
    """-150 -> 1.667, +180 -> 2.80."""
    value = float(american)
    if value == 0:
        raise ValueError("american odds cannot be 0")
    return 1.0 + (value / 100.0 if value > 0 else 100.0 / abs(value))


def decimal_to_american(decimal_odds: float) -> int:
    """Inverse of :func:`american_to_decimal`, rounded to the nearest integer."""
    if decimal_odds <= 1.0:
        raise ValueError("decimal odds must exceed 1.0")
    if decimal_odds >= 2.0:
        return int(round((decimal_odds - 1.0) * 100.0))
    return int(round(-100.0 / (decimal_odds - 1.0)))


def implied_probability(american: float) -> float:
    """Raw (vig-inclusive) probability implied by a price."""
    return 1.0 / american_to_decimal(american)


def net_profit(american: float, stake: float = 1.0) -> float:
    """Profit (excluding the returned stake) on a winning bet."""
    return stake * (american_to_decimal(american) - 1.0)


# ----------------------------------------------------------------------
# vig removal
# ----------------------------------------------------------------------
def _power_exponent(raw: Sequence[float]) -> float:
    """Exponent ``k`` with ``sum(p_i ** k) == 1``."""

    def objective(k: float) -> float:
        return sum(p**k for p in raw) - 1.0

    lower, upper = 0.25, 8.0
    if objective(lower) * objective(upper) > 0:
        return 1.0  # no sign change: fall back to leaving the book alone
    return float(optimize.brentq(objective, lower, upper, maxiter=200))


def remove_vig(
    american_odds: Sequence[float], method: str = "power"
) -> list[float]:
    """Fair probabilities for a complete market.

    ``american_odds`` must cover every outcome of the market (both sides of a
    prop, all three of a moneyline with a draw, ...).
    """
    if not american_odds:
        return []
    raw = [implied_probability(o) for o in american_odds]
    total = sum(raw)
    if len(raw) == 1 or total <= 0:
        return raw

    if method == "multiplicative":
        return [p / total for p in raw]
    if method == "additive":
        overround = total - 1.0
        share = overround / len(raw)
        adjusted = [max(p - share, 1e-6) for p in raw]
        scale = sum(adjusted)
        return [p / scale for p in adjusted]
    if method == "power":
        k = _power_exponent(raw)
        powered = [p**k for p in raw]
        scale = sum(powered)
        return [p / scale for p in powered]
    raise ValueError(f"unknown de-vig method: {method!r} (choose from {DEVIG_METHODS})")


def market_overround(american_odds: Sequence[float]) -> float:
    """Book margin: ``sum(implied) - 1``."""
    return sum(implied_probability(o) for o in american_odds) - 1.0


def devig_two_way(
    over_odds: float, under_odds: float, method: str = "power"
) -> tuple[float, float]:
    """Fair (over, under) probabilities for a two-way market."""
    fair = remove_vig([over_odds, under_odds], method)
    return fair[0], fair[1]


def devig_selection(
    rows: Sequence[Mapping[str, object]], selection: str, method: str = "power"
) -> float | None:
    """Fair probability of ``selection`` given every row of one market.

    ``rows`` are ``fanduel_props``/``fanduel_lines`` records sharing a market
    and line. Returns ``None`` when the opposing side is missing, because a
    one-sided market cannot be de-vigged.
    """
    if len(rows) < 2:
        return None
    odds = [float(row["american_odds"]) for row in rows]
    fair = remove_vig(odds, method)
    for row, probability in zip(rows, fair):
        if str(row.get("selection", "")).lower() == selection.lower():
            return probability
    return None


# ----------------------------------------------------------------------
# expected value and staking
# ----------------------------------------------------------------------
def expected_value(p_true: float, american: float, stake: float = 1.0) -> float:
    """EV = P(win) x profit - P(lose) x stake."""
    profit = net_profit(american, stake)
    return p_true * profit - (1.0 - p_true) * stake


def expected_value_decimal(p_true: float, decimal_odds: float, stake: float = 1.0) -> float:
    """EV for a price already in decimal form (e.g. a parlay)."""
    return p_true * stake * (decimal_odds - 1.0) - (1.0 - p_true) * stake


def edge(p_true: float, p_implied: float) -> float:
    """Model probability minus the market's de-vigged probability."""
    return p_true - p_implied


def kelly_stake(
    p_true: float,
    decimal_odds: float,
    *,
    bankroll: float | None = None,
    fraction: float | None = None,
) -> float:
    """Fractional-Kelly stake (never negative).

    ``f* = (p*d - 1) / (d - 1)``, scaled by ``settings.kelly_fraction``.
    """
    bankroll = settings.bankroll if bankroll is None else bankroll
    fraction = settings.kelly_fraction if fraction is None else fraction
    if decimal_odds <= 1.0:
        return 0.0
    full_kelly = (p_true * decimal_odds - 1.0) / (decimal_odds - 1.0)
    return max(full_kelly, 0.0) * fraction * bankroll


def kelly_fraction_of_bankroll(p_true: float, decimal_odds: float, fraction: float | None = None) -> float:
    """Fractional-Kelly stake as a share of bankroll."""
    fraction = settings.kelly_fraction if fraction is None else fraction
    if decimal_odds <= 1.0:
        return 0.0
    return max((p_true * decimal_odds - 1.0) / (decimal_odds - 1.0), 0.0) * fraction


# ----------------------------------------------------------------------
# parlay pricing
# ----------------------------------------------------------------------
def parlay_decimal_odds(american_odds: Iterable[float]) -> float:
    """Multiply legs into a single decimal price."""
    product = 1.0
    for odds in american_odds:
        product *= american_to_decimal(odds)
    return product


def parlay_american_odds(american_odds: Iterable[float]) -> int:
    return decimal_to_american(parlay_decimal_odds(american_odds))


# ----------------------------------------------------------------------
# leg evaluation
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LegEvaluation:
    """Diagnostics for one candidate leg."""

    leg_id: str
    p_model: float
    p_implied: float
    edge: float
    ev: float
    decimal_odds: float
    passes: bool
    reason: str = ""


def evaluate_leg(
    leg: Leg,
    *,
    p_implied: float | None = None,
    min_ev: float | None = None,
    method: str = "power",
) -> LegEvaluation:
    """Populate ``p_implied``/``ev`` on a leg and judge it against the floor.

    ``p_implied`` should be the de-vigged market probability. When it is not
    supplied the raw implied probability is used, which is conservative (it
    overstates what the market thinks).
    """
    floor = settings.min_leg_ev if min_ev is None else min_ev
    decimal_odds = american_to_decimal(leg.american_odds)
    fair = implied_probability(leg.american_odds) if p_implied is None else p_implied
    ev = expected_value(leg.p_model, leg.american_odds)

    leg.p_implied = fair
    leg.ev = ev

    in_odds_band = settings.leg_odds_min <= leg.american_odds <= settings.leg_odds_max
    if not in_odds_band:
        reason = (
            f"odds {leg.american_odds} outside "
            f"[{settings.leg_odds_min}, {settings.leg_odds_max}]"
        )
    elif ev < floor:
        reason = f"EV {ev:.3f} below floor {floor:.3f}"
    else:
        reason = "ok"

    return LegEvaluation(
        leg_id=leg.leg_id,
        p_model=leg.p_model,
        p_implied=fair,
        edge=edge(leg.p_model, fair),
        ev=ev,
        decimal_odds=decimal_odds,
        passes=reason == "ok",
        reason=reason,
    )


def find_edges(
    legs: Sequence[Leg],
    *,
    min_ev: float | None = None,
    implied: Mapping[str, float] | None = None,
    method: str = "power",
) -> tuple[list[Leg], list[LegEvaluation]]:
    """Split legs into actionable edges and return every evaluation.

    ``implied`` optionally maps ``leg_id`` to a de-vigged probability.
    """
    implied = implied or {}
    keepers: list[Leg] = []
    evaluations: list[LegEvaluation] = []
    for leg in legs:
        evaluation = evaluate_leg(
            leg, p_implied=implied.get(leg.leg_id), min_ev=min_ev, method=method
        )
        evaluations.append(evaluation)
        if evaluation.passes:
            keepers.append(leg)
    return keepers, evaluations


def odds_within_band(american: float, low: int | None = None, high: int | None = None) -> bool:
    """Is a price inside the configured single-leg band?"""
    low = settings.leg_odds_min if low is None else low
    high = settings.leg_odds_max if high is None else high
    return low <= american <= high


def breakeven_probability(american: float) -> float:
    """Probability at which a price is exactly EV-neutral."""
    return 1.0 / american_to_decimal(american)


def clv_delta(closing_odds: float, taken_odds: float) -> float:
    """Closing line value: fair-probability gain from the price we took.

    Positive means our price was better than the close -- the standard
    benchmark for whether de-vigged model probabilities carry real signal.
    """
    return breakeven_probability(closing_odds) - breakeven_probability(taken_odds)


def _assert_finite(value: float) -> float:  # pragma: no cover - defensive
    if not math.isfinite(value):
        raise ValueError("non-finite value in EV computation")
    return value
