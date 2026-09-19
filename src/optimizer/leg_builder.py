"""Turn stored FanDuel prices plus projections into evaluated legs.

This is the join point of the pipeline: a market row supplies the price, a
projection supplies the model mean, the distribution layer converts the two
into a probability, and the de-vigged opposite side supplies what the market
itself thinks.

A prop with only one side stored cannot be de-vigged, so its raw implied
probability is used -- conservative, because it overstates the market's view.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from src.models.baseline import GameProjection
from src.models.distributions import DistributionSpec
from src.models.legs import Leg, Projection
from src.optimizer.ev_calculator import devig_selection, evaluate_leg

logger = logging.getLogger(__name__)

PLAYER_SIDES = ("Over", "Under", "Yes", "No")


def _line_key(row: Mapping[str, Any]) -> tuple:
    return (row.get("market"), row.get("player_name"), row.get("line"))


def group_prop_rows(
    rows: Iterable[Mapping[str, Any]]
) -> dict[tuple, list[Mapping[str, Any]]]:
    """Group prop rows into complete markets (same market, player and line)."""
    grouped: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_line_key(row)].append(row)
    return dict(grouped)


def projection_index(
    projections: Iterable[Projection],
) -> dict[tuple[str, str, str], Projection]:
    """``{(game_id, player_name, market): projection}``."""
    return {p.key: p for p in projections}


def legs_from_props(
    prop_rows: Sequence[Mapping[str, Any]],
    projections: Sequence[Projection],
    *,
    sport: str,
    devig_method: str = "power",
) -> list[Leg]:
    """Build one leg per priced side that we have a projection for."""
    index = projection_index(projections)
    legs: list[Leg] = []

    for (market, player, line), rows in group_prop_rows(prop_rows).items():
        if not player or not market:
            continue
        game_id = str(rows[0].get("game_id"))
        projection = index.get((game_id, player, market))
        if projection is None:
            continue

        spec = DistributionSpec.for_market(market, projection.mean, projection.dispersion)
        probability = spec.probability(line)

        for row in rows:
            selection = str(row.get("selection") or "")
            if selection not in PLAYER_SIDES:
                continue
            try:
                p_model = probability.for_selection(selection)
            except ValueError:
                continue
            p_implied = devig_selection(rows, selection, devig_method)
            leg = Leg(
                game_id=game_id,
                sport=sport,
                market=market,
                selection=selection,
                american_odds=int(row["american_odds"]),
                line=line,
                player_name=player,
                team=projection.team,
                p_model=p_model,
                projection_mean=projection.mean,
                baseline_mean=projection.mean,
                meta={"push_probability": probability.prob_push},
            )
            evaluate_leg(leg, p_implied=p_implied, method=devig_method)
            legs.append(leg)
    return legs


def legs_from_lines(
    line_rows: Sequence[Mapping[str, Any]],
    game_projection: GameProjection | None,
    *,
    sport: str,
    devig_method: str = "power",
) -> list[Leg]:
    """Build game-market legs (total, spread, moneyline) from a game projection."""
    if game_projection is None:
        return []

    grouped: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for row in line_rows:
        grouped[(row.get("market"), row.get("line"))].append(row)

    total_spec = DistributionSpec(
        family="normal", mean=game_projection.total_mean, dispersion=game_projection.total_sd
    )
    margin_spec = DistributionSpec(
        family="normal",
        mean=game_projection.home_margin_mean,
        dispersion=game_projection.margin_sd,
    )

    legs: list[Leg] = []
    for (market, line), rows in grouped.items():
        for row in rows:
            selection = str(row.get("selection") or "")
            p_model = _game_market_probability(
                market, selection, line, game_projection, total_spec, margin_spec
            )
            if p_model is None:
                continue
            leg = Leg(
                game_id=str(row.get("game_id")),
                sport=sport,
                market=str(market),
                selection=selection,
                american_odds=int(row["american_odds"]),
                line=line,
                team=_team_for_selection(selection, game_projection),
                p_model=p_model,
                projection_mean=(
                    game_projection.total_mean if market == "totals"
                    else game_projection.home_margin_mean
                ),
            )
            evaluate_leg(
                leg,
                p_implied=devig_selection(rows, selection, devig_method),
                method=devig_method,
            )
            legs.append(leg)
    return legs


def _game_market_probability(
    market: str | None,
    selection: str,
    line: float | None,
    projection: GameProjection,
    total_spec: DistributionSpec,
    margin_spec: DistributionSpec,
) -> float | None:
    """Model probability for a total / spread / moneyline selection."""
    if market == "totals" and line is not None:
        probability = total_spec.probability(line)
        try:
            return probability.for_selection(selection)
        except ValueError:
            return None
    if market == "spreads" and line is not None:
        # A spread is stated from the selected team's perspective: they cover
        # when their margin beats -line.
        if selection == projection.home_team:
            return float(margin_spec.probability(-line).prob_over)
        if selection == projection.away_team:
            return float(margin_spec.probability(line).prob_under)
        return None
    if market == "h2h":
        if selection == projection.home_team:
            return float(margin_spec.probability(0.0).prob_over)
        if selection == projection.away_team:
            return float(margin_spec.probability(0.0).prob_under)
        return None
    return None


def _team_for_selection(selection: str, projection: GameProjection) -> str | None:
    if selection in (projection.home_team, projection.away_team):
        return selection
    return None


def attach_rationale(legs: Sequence[Leg], context_results: Iterable[Any]) -> None:
    """Copy the reasoning layer's audit notes onto the affected legs."""
    notes: dict[tuple[str, str], list[str]] = defaultdict(list)
    for result in context_results:
        for applied in getattr(result, "applied", []):
            notes[(applied.player_name, applied.market)].append(
                f"{applied.reasoning} (x{applied.applied_factor:.2f}"
                + (", clamped" if applied.clamped else "")
                + ")"
            )
    for leg in legs:
        if leg.player_name is None:
            continue
        for note in notes.get((leg.player_name, leg.market), []):
            if note not in leg.rationale:
                leg.rationale.append(note)


def summarise_legs(legs: Sequence[Leg]) -> list[dict[str, Any]]:
    """Compact, loggable view of a leg pool."""
    return [
        {
            "leg": leg.describe(),
            "model": round(leg.p_model, 4),
            "implied": round(leg.p_implied, 4),
            "ev": round(leg.ev, 4),
        }
        for leg in legs
    ]
