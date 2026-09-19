"""Turn historical weeks into (prediction, outcome) pairs.

This is the raw material for training. For each past week the model is rebuilt
using only the games before it, asked for a projection, and then compared with
what actually happened -- so nothing downstream ever sees the week it is being
scored on.

The pairs are what every correction is fitted from: how far the projected mean
sits from the real one, how wide the real spread is, and whether a stated
probability comes true as often as it claims.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from config.settings import settings
from src.models.baseline import NFL_STAT_MARKETS, rolling_weighted_mean
from src.models.distributions import DistributionSpec, over_probabilities

logger = logging.getLogger(__name__)

#: Lines are placed around the projection as multiples of it, so the whole
#: distribution is tested rather than only its middle.
#:
#: The range is deliberately wide. A book's line sits near the player's true
#: mean, and our projection is a noisy estimate of that mean, so a real line
#: routinely lands at half or double what we projected -- and those are the
#: legs a big-payout parlay is made of. Probing only near the median would
#: leave the corrections fitted in the body and extrapolated into the tails,
#: which is the one place they must not be guessed.
LINE_OFFSETS = (0.40, 0.55, 0.70, 0.85, 1.00, 1.15, 1.35, 1.70, 2.20)

#: Anytime-TD has no stat column of its own; it is the sum of the two ways a
#: non-kicker scores. Added here because a big-payout parlay leans on it.
TD_STAT = "anytime_td"
TD_SOURCE_COLUMNS = ("rushing_tds", "receiving_tds")
TD_MARKETS: dict[str, str] = {TD_STAT: "player_anytime_td"}


@dataclass(frozen=True)
class Observation:
    """One projection measured against what actually happened."""

    sport: str
    season: int
    week: int
    player: str
    team: str
    market: str
    projected: float
    actual: float
    line: float
    p_over: float
    hit: int
    games_of_history: int

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def with_derived_stats(weekly: pd.DataFrame) -> pd.DataFrame:
    """Add the stat columns that are sums of others, notably anytime TDs."""
    frame = weekly.copy()
    sources = [column for column in TD_SOURCE_COLUMNS if column in frame.columns]
    if sources and TD_STAT not in frame.columns:
        frame[TD_STAT] = frame[sources].fillna(0.0).sum(axis=1)
    return frame


def _history_before(frame: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Every game a player played before the given week."""
    return frame[(frame["season"] < season) | ((frame["season"] == season) & (frame["week"] < week))]


def observations(
    weekly: pd.DataFrame,
    *,
    sport: str = "nfl",
    markets: dict[str, str] | None = None,
    min_history: int = 2,
    stat_markets: dict[str, str] | None = None,
) -> list[Observation]:
    """Walk every player-week forward, projecting from the past only.

    ``weekly`` is an nflverse-shaped frame: one row per player per game, with
    ``season``, ``week``, ``player_display_name``/``player_name`` and the stat
    columns.
    """
    stat_markets = {**(stat_markets or markets or NFL_STAT_MARKETS), **TD_MARKETS}
    name_col = "player_display_name" if "player_display_name" in weekly.columns else "player_name"
    team_col = "recent_team" if "recent_team" in weekly.columns else "team"

    frame = with_derived_stats(weekly)
    frame = frame.dropna(subset=[name_col, "season", "week"])
    frame["season"] = frame["season"].astype(int)
    frame["week"] = frame["week"].astype(int)

    # The family and whether it is discrete are properties of the market, not
    # of the row, so resolve them once rather than per projection. Deliberately
    # without a sport: the point of a backtest is to measure the model before
    # any correction, whatever happens to be installed.
    shapes = {
        market: DistributionSpec.for_market(market, 1.0)
        for stat, market in stat_markets.items()
        if stat in frame.columns
    }

    # Phase one: walk the weeks forward and record the projections. No
    # probabilities yet -- pricing a row at a time means building a scipy
    # distribution 685,000 times over on a college season.
    rows: list[dict[str, Any]] = []
    for player, played in frame.groupby(name_col, sort=False):
        played = played.sort_values(["season", "week"])
        for position in range(len(played)):
            row = played.iloc[position]
            history = played.iloc[:position]
            if len(history) < min_history:
                continue
            # Most recent games first, exactly as the live model sees them.
            recent = history.iloc[::-1].head(settings.rolling_weeks)

            for stat, market in stat_markets.items():
                if market not in shapes:
                    continue
                projected = rolling_weighted_mean(recent[stat].tolist())
                actual = float(row[stat]) if pd.notna(row[stat]) else 0.0
                if projected <= 0:
                    continue

                for line in _lines_for(shapes[market], projected):
                    rows.append(
                        {
                            "sport": sport,
                            "season": int(row["season"]),
                            "week": int(row["week"]),
                            "player": str(player),
                            "team": str(row.get(team_col, "")),
                            "market": market,
                            "projected": float(projected),
                            "actual": actual,
                            "line": float(line),
                            "hit": int(actual > line),
                            "games_of_history": int(len(recent)),
                        }
                    )

    # Phase two: price them, one market at a time.
    return _priced(rows)


def _priced(rows: Sequence[dict[str, Any]]) -> list[Observation]:
    """Attach the model's probability to each recorded projection.

    Batched by market, because within a market only the mean and the line
    differ. A probability of exactly 0 or 1 is dropped: it says the line is
    beyond what the distribution can represent, which measures floating point
    rather than the model.
    """
    if not rows:
        return []
    by_market: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_market.setdefault(row["market"], []).append(index)

    results: list[Observation] = []
    for market, indices in by_market.items():
        means = np.fromiter((rows[i]["projected"] for i in indices), float, len(indices))
        lines = np.fromiter((rows[i]["line"] for i in indices), float, len(indices))
        probabilities = over_probabilities(market, means, lines)
        for index, probability in zip(indices, probabilities):
            if not 0.0 < probability < 1.0:
                continue
            results.append(Observation(p_over=float(probability), **rows[index]))
    # Grouping by market scrambled the order; put it back so the caller sees
    # the weeks in the order they were played.
    results.sort(key=lambda row: (row.season, row.week, row.player, row.market, row.line))
    return results


def _lines_for(spec: DistributionSpec, projected: float) -> list[float]:
    """The lines to probe this projection at.

    A binary market has only one line worth asking about -- did they score --
    so it gets 0.5 and nothing else.
    """
    if spec.family == "poisson_binary":
        return [0.5]
    lines = (_snap(projected * offset, spec.discrete) for offset in LINE_OFFSETS)
    return [line for line in lines if line > 0]


def _snap(value: float, discrete: bool) -> float:
    """Put a line where a book would: on a half point."""
    return float(np.floor(value) + 0.5) if discrete else round(value * 2) / 2 + 0.25


def to_frame(rows: Iterable[Observation]) -> pd.DataFrame:
    return pd.DataFrame([row.as_row() for row in rows])


# ----------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------
def brier_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mean squared error of a probability forecast. Lower is better."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    return float(np.mean((p - y) ** 2))


def log_loss(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    """Penalises confident mistakes much harder than Brier does."""
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1 - 1e-9)
    y = np.asarray(outcomes, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_table(
    probabilities: Sequence[float], outcomes: Sequence[int], bins: int = 10
) -> pd.DataFrame:
    """Claimed probability against observed frequency, bucketed."""
    frame = pd.DataFrame({"p": probabilities, "hit": outcomes})
    frame["bucket"] = pd.cut(frame["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    table = (
        frame.groupby("bucket", observed=True)
        .agg(claimed=("p", "mean"), observed=("hit", "mean"), n=("hit", "size"))
        .reset_index()
    )
    table["gap"] = table["observed"] - table["claimed"]
    return table
