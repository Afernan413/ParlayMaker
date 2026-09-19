"""Service layer behind the web UI.

Holds one built slate per sport in memory and prices arbitrary leg combinations
against it. Every number the UI shows comes from the same engine the CLI uses --
the copula for joint probability, the de-vig for the market's own view, and
fractional Kelly for staking -- so the browser never re-implements the maths.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Sequence

from config.settings import SPORT_KEYS, market_meta, settings
from src.models.correlation import pairwise_correlation
from src.models.legs import MARKET_LABELS, Leg, format_odds
from src.optimizer.ev_calculator import (
    american_to_decimal,
    breakeven_probability,
    decimal_to_american,
    expected_value_decimal,
    kelly_fraction_of_bankroll,
    kelly_stake,
    parlay_decimal_odds,
)
from src.optimizer.parlay_builder import (
    ParlayTicket,
    build_parlays,
    describe_ticket_type,
)
from src.models.correlation import GaussianCopulaSimulator
from run_pipeline import Slate, build_slate

#: Copula iterations for interactive pricing. Lower than the CLI default so a
#: keystroke-speed request stays responsive; still +/-0.5% on a joint probability.
INTERACTIVE_ITERATIONS = 4_000

logger = logging.getLogger("parlay_engine.web")


#: Display names for markets whose canonical stat does not title-case cleanly.
MARKET_DISPLAY: dict[str, str] = {
    "h2h": "Moneyline",
    "spreads": "Spread",
    "totals": "Game Total",
    "player_pass_tds": "Passing TDs",
    "player_anytime_td": "Anytime TD",
    "player_threes": "3-Pointers Made",
}


def market_display(market: str) -> str:
    """Readable market name, derived from the canonical stat where possible."""
    if market in MARKET_DISPLAY:
        return MARKET_DISPLAY[market]
    stat = market_meta(market)["stat"]
    return stat.replace("_", " ").title()


class SlateNotFound(LookupError):
    """No slate has been built for that sport yet."""


class UnknownLeg(LookupError):
    """A leg id the current slate does not contain."""


@dataclass
class SlateMeta:
    """Headline facts about a built slate, for the UI status bar."""

    sport: str
    mock: bool
    built_at: str
    games: int
    legs: int
    edges: int
    projections: int
    adjustments: int
    seconds: float


class SlateStore:
    """Builds and caches slates, one per sport, guarded by a lock."""

    def __init__(
        self,
        *,
        db_path: str | None = None,
        use_mock: bool = True,
        iterations: int = INTERACTIVE_ITERATIONS,
        max_events: int | None = None,
        min_refresh_seconds: int | None = None,
    ) -> None:
        self.db_path = db_path
        self.use_mock = use_mock
        self.iterations = iterations
        self.max_events = max_events
        self.min_refresh_seconds = (
            settings.min_refresh_seconds
            if min_refresh_seconds is None
            else min_refresh_seconds
        )
        self._slates: dict[str, Slate] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._built_at: dict[str, float] = {}

    def _lock(self, sport: str) -> asyncio.Lock:
        return self._locks.setdefault(sport, asyncio.Lock())

    def cached(self, sport: str) -> Slate | None:
        return self._slates.get(sport.lower())

    def refresh_allowed(self, sport: str, *, now: float | None = None) -> bool:
        """Has enough time passed since the last build to spend credits again?

        Live rebuilds cost API credits, so a refresh that arrives inside the
        floor is served from cache rather than refused -- the caller still gets
        a slate, just not a new bill.
        """
        built = self._built_at.get(sport.lower())
        if built is None:
            return True
        now = time.monotonic() if now is None else now
        return (now - built) >= self.min_refresh_seconds

    async def get(
        self, sport: str, *, refresh: bool = False, use_mock: bool | None = None
    ) -> Slate:
        """Return a slate, building it on first use (or when refreshed)."""
        sport = sport.lower()
        if sport not in SPORT_KEYS:
            raise SlateNotFound(f"unsupported sport: {sport}")
        if refresh and not self.refresh_allowed(sport):
            logger.info(
                "refresh for %s ignored: inside the %ss floor", sport, self.min_refresh_seconds
            )
            refresh = False
        async with self._lock(sport):
            if refresh or sport not in self._slates:
                self._slates[sport] = await build_slate(
                    sport=sport,
                    use_mock=self.use_mock if use_mock is None else use_mock,
                    db_path=self.db_path,
                    max_events=self.max_events,
                )
                self._built_at[sport] = time.monotonic()
            return self._slates[sport]

    def require(self, sport: str) -> Slate:
        slate = self.cached(sport)
        if slate is None:
            raise SlateNotFound(f"no slate built for {sport}")
        return slate

    def legs(self, sport: str, leg_ids: Sequence[str]) -> list[Leg]:
        """Resolve leg ids against the cached slate, preserving slip order."""
        index = self.require(sport).legs_by_id
        resolved: list[Leg] = []
        for leg_id in leg_ids:
            leg = index.get(leg_id)
            if leg is None:
                raise UnknownLeg(leg_id)
            resolved.append(leg)
        return resolved


# ----------------------------------------------------------------------
# serialisation
# ----------------------------------------------------------------------
def game_label(game: dict[str, Any] | None) -> str:
    if not game:
        return ""
    return f"{game.get('away_team', '?')} @ {game.get('home_team', '?')}"


def leg_payload(leg: Leg, game: dict[str, Any] | None = None) -> dict[str, Any]:
    """One selectable leg, as the browser sees it."""
    return {
        "leg_id": leg.leg_id,
        "game_id": leg.game_id,
        "game": game_label(game),
        "kickoff": (game or {}).get("commence_time"),
        "sport": leg.sport,
        "market": leg.market,
        "market_label": market_display(leg.market),
        "subject": leg.subject,
        "label": leg.label,
        "player_name": leg.player_name,
        "team": leg.team,
        "selection": leg.selection,
        "line": leg.line,
        "american_odds": leg.american_odds,
        "odds_display": format_odds(leg.american_odds),
        "decimal_odds": round(american_to_decimal(leg.american_odds), 4),
        "p_model": round(leg.p_model, 4),
        "p_implied": round(leg.p_implied, 4),
        "edge": round(leg.p_model - leg.p_implied, 4),
        "ev": round(leg.ev, 4),
        "has_edge": leg.ev >= settings.min_leg_ev
        and settings.leg_odds_min <= leg.american_odds <= settings.leg_odds_max,
        "projection_mean": leg.projection_mean,
        "description": leg.describe(),
        "rationale": list(leg.rationale),
    }


def slate_meta(slate: Slate) -> dict[str, Any]:
    adjustments = sum(len(result.applied) for result in slate.context_results)
    return {
        "sport": slate.sport,
        "mock": slate.mock,
        "built_at": slate.built_at,
        "games": len(slate.games),
        "legs": len(slate.legs),
        "edges": len(slate.edges),
        "projections": slate.projections,
        "adjustments": adjustments,
        "seconds": round(sum(slate.timings.values()), 2),
    }


def slate_payload(slate: Slate) -> dict[str, Any]:
    """Games, legs and per-game context for the leg browser."""
    games = slate.games_by_id
    scripts = {
        result.game_id: result.game_script
        for result in slate.context_results
        if result.game_script
    }
    return {
        "meta": slate_meta(slate),
        "games": [
            {
                "game_id": game["game_id"],
                "label": game_label(game),
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "kickoff": game["commence_time"],
                "game_script": scripts.get(game["game_id"], ""),
            }
            for game in slate.games
        ],
        "legs": [leg_payload(leg, games.get(leg.game_id)) for leg in slate.legs],
    }


def ticket_payload(ticket: ParlayTicket, games: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticket_type": ticket.ticket_type,
        "american_odds": ticket.american_odds,
        "odds_display": format_odds(ticket.american_odds),
        "decimal_odds": round(ticket.decimal_odds, 4),
        "p_model": round(ticket.joint_probability, 4),
        "p_implied": round(ticket.implied_probability, 4),
        "edge": round(ticket.edge, 4),
        "ev": round(ticket.ev, 4),
        "correlation_lift": round(ticket.correlation_lift, 3),
        "stake": ticket.stake,
        "to_win": round(ticket.to_win, 2),
        "leg_ids": [leg.leg_id for leg in ticket.legs],
        "legs": [leg_payload(leg, games.get(leg.game_id)) for leg in ticket.legs],
    }


# ----------------------------------------------------------------------
# pricing a slip
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Advisory:
    """A rule the slip breaks, with the engine's own threshold attached."""

    level: str  # "block" | "warn" | "info"
    code: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"level": self.level, "code": self.code, "message": self.message}


def correlation_pairs(legs: Sequence[Leg]) -> list[dict[str, Any]]:
    """Pairwise correlations, so the UI can explain a same-game price."""
    pairs: list[dict[str, Any]] = []
    for first, second in combinations(range(len(legs)), 2):
        leg_a, leg_b = legs[first], legs[second]
        pairs.append(
            {
                "a": leg_a.leg_id,
                "b": leg_b.leg_id,
                "a_label": leg_a.describe(),
                "b_label": leg_b.describe(),
                "same_game": leg_a.game_id == leg_b.game_id,
                "correlation": round(pairwise_correlation(leg_a, leg_b), 3),
            }
        )
    return pairs


def review_slip(legs: Sequence[Leg], pairs: Sequence[dict[str, Any]]) -> list[Advisory]:
    """House rules, reported rather than enforced.

    The UI lets someone build any slip they like -- it just has to be honest
    about which of the engine's guardrails that slip is outside.
    """
    advisories: list[Advisory] = []
    if len(legs) < settings.min_legs:
        advisories.append(
            Advisory("info", "too_few_legs", f"Add at least {settings.min_legs} legs to price a parlay.")
        )
    if len(legs) > settings.max_legs:
        advisories.append(
            Advisory(
                "warn",
                "too_many_legs",
                f"{len(legs)} legs is past the {settings.max_legs}-leg ceiling; "
                "the house edge compounds faster than the payout.",
            )
        )
    for leg in legs:
        if not (settings.leg_odds_min <= leg.american_odds <= settings.leg_odds_max):
            advisories.append(
                Advisory(
                    "warn",
                    "leg_price_band",
                    f"{leg.describe()} is outside the "
                    f"{format_odds(settings.leg_odds_min)}/"
                    f"{format_odds(settings.leg_odds_max)} single-leg band.",
                )
            )
        if leg.ev < settings.min_leg_ev:
            advisories.append(
                Advisory(
                    "warn",
                    "leg_ev_floor",
                    f"{leg.describe()} has {leg.ev:+.1%} EV, under the "
                    f"{settings.min_leg_ev:.0%} floor.",
                )
            )
    seen: set[tuple[str, str]] = set()
    for leg in legs:
        key = (leg.subject, leg.market)
        if key in seen:
            advisories.append(
                Advisory(
                    "block",
                    "duplicate_selection",
                    f"{leg.subject} {leg.market} is on the slip twice.",
                )
            )
        seen.add(key)
    for pair in pairs:
        if pair["same_game"] and pair["correlation"] < settings.min_sgp_correlation:
            advisories.append(
                Advisory(
                    "warn",
                    "sgp_correlation",
                    f"{pair['a_label']} and {pair['b_label']} are same-game at "
                    f"r={pair['correlation']:.2f}, under the "
                    f"{settings.min_sgp_correlation:.2f} floor"
                    + (" (they work against each other)" if pair["correlation"] < 0 else ""),
                )
            )
    return advisories


def price_slip(
    legs: Sequence[Leg],
    *,
    stake: float = 10.0,
    bankroll: float | None = None,
    iterations: int = INTERACTIVE_ITERATIONS,
    seed: int | None = None,
    preset_stakes: Iterable[float] = (5, 10, 25, 50, 100),
) -> dict[str, Any]:
    """Price a slip: payout, model probability, edge, EV and staking advice."""
    bankroll = settings.bankroll if bankroll is None else bankroll
    stake = max(float(stake), 0.0)
    pairs = correlation_pairs(legs)
    advisories = [a.as_dict() for a in review_slip(legs, pairs)]

    if not legs:
        return {
            "legs": [],
            "priceable": False,
            "advisories": advisories,
            "correlation_pairs": [],
        }

    decimal_odds = parlay_decimal_odds(leg.american_odds for leg in legs)
    american = decimal_to_american(decimal_odds)
    independent = 1.0
    for leg in legs:
        independent *= leg.p_model

    if len(legs) == 1:
        joint = legs[0].p_model
        lift = 1.0
    else:
        simulation = GaussianCopulaSimulator(
            legs, iterations=iterations, seed=seed
        ).simulate()
        joint = simulation.joint_probability
        lift = simulation.correlation_lift

    implied = 1.0 / decimal_odds
    profit = stake * (decimal_odds - 1.0)
    ev_per_unit = expected_value_decimal(joint, decimal_odds)
    kelly_share = kelly_fraction_of_bankroll(joint, decimal_odds)
    recommended = round(kelly_stake(joint, decimal_odds, bankroll=bankroll), 2)

    return {
        "legs": [leg.leg_id for leg in legs],
        "priceable": True,
        "leg_count": len(legs),
        "ticket_type": describe_ticket_type(legs),
        "is_sgp": len({leg.game_id for leg in legs}) == 1,
        "american_odds": american,
        "odds_display": format_odds(american),
        "decimal_odds": round(decimal_odds, 4),
        "p_model": round(joint, 4),
        "p_implied": round(implied, 4),
        "edge": round(joint - implied, 4),
        "independent_probability": round(independent, 4),
        "correlation_lift": round(lift, 3),
        "fair_odds": decimal_to_american(1.0 / joint) if joint > 0 else None,
        "breakeven_probability": round(breakeven_probability(american), 4),
        "stake": round(stake, 2),
        "profit": round(profit, 2),
        "payout": round(stake + profit, 2),
        "ev_per_unit": round(ev_per_unit, 4),
        "ev_dollars": round(ev_per_unit * stake, 2),
        "kelly_share": round(kelly_share, 4),
        "recommended_stake": recommended,
        "bankroll": bankroll,
        "iterations": iterations if len(legs) > 1 else 0,
        "payout_table": [
            {
                "stake": float(amount),
                "profit": round(amount * (decimal_odds - 1.0), 2),
                "payout": round(amount * decimal_odds, 2),
                "ev": round(ev_per_unit * amount, 2),
            }
            for amount in preset_stakes
        ],
        "correlation_pairs": pairs,
        "advisories": advisories,
        "rationale": sorted(
            {note for leg in legs for note in leg.rationale}
        ),
    }


def auto_build(
    slate: Slate,
    *,
    legs: int | None = None,
    max_tickets: int | None = None,
    bankroll: float | None = None,
    iterations: int = INTERACTIVE_ITERATIONS,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run the ILP optimizer over the slate's qualifying legs."""
    tickets, report = build_parlays(
        slate.edges,
        min_legs=legs,
        max_legs=legs,
        max_tickets=max_tickets,
        iterations=iterations,
        seed=seed,
        bankroll=bankroll,
    )
    games = slate.games_by_id
    return {
        "tickets": [ticket_payload(ticket, games) for ticket in tickets],
        "report": {
            "considered": report.considered,
            "candidates": report.candidates,
            "selected": report.selected,
            "solver_status": report.solver_status,
            "rejected": report.rejected,
        },
    }


def engine_config() -> dict[str, Any]:
    """Thresholds the UI displays so the rules are never hidden from the user."""
    return {
        "sports": sorted(SPORT_KEYS),
        "min_legs": settings.min_legs,
        "max_legs": settings.max_legs,
        "leg_odds_min": settings.leg_odds_min,
        "leg_odds_max": settings.leg_odds_max,
        "parlay_odds_min": settings.parlay_odds_min,
        "parlay_odds_max": settings.parlay_odds_max,
        "min_leg_ev": settings.min_leg_ev,
        "min_sgp_correlation": settings.min_sgp_correlation,
        "kelly_fraction": settings.kelly_fraction,
        "bankroll": settings.bankroll,
        "max_context_adjustment": settings.max_context_adjustment,
        "max_tickets": settings.max_tickets,
    }
