"""Parlay assembly as an integer linear program.

Two stages:

1. **Enumerate** every feasible 2-4 leg combination. Feasibility is where the
   house edge is fought: single-leg price band, ticket price band, and the SGP
   rule that same-game legs must be positively correlated (``r >= 0.25``).
   Joint probabilities come from the Gaussian copula, so correlation is priced
   rather than assumed away.
2. **Select** a portfolio of tickets with PuLP: maximise total expected value
   subject to a ticket cap and the diversification rule that a subject (player
   or team) may appear on at most one ticket.

Stage 2 is a set-packing ILP; if no solver is available it degrades to a greedy
pick over the same candidates so the pipeline still produces a card.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Sequence

import pulp

from config.settings import settings
from src.models.correlation import (
    average_correlation,
    min_correlation,
    pairwise_correlation,
)
from src.models.correlation import GaussianCopulaSimulator
from src.models.legs import Leg, format_odds
from src.optimizer.ev_calculator import (
    american_to_decimal,
    decimal_to_american,
    expected_value_decimal,
    kelly_fraction_of_bankroll,
    kelly_stake,
    parlay_decimal_odds,
)

logger = logging.getLogger(__name__)


@dataclass
class ParlayTicket:
    """A priced, staked parlay candidate."""

    legs: list[Leg]
    joint_probability: float
    independent_probability: float
    decimal_odds: float
    american_odds: int
    ev: float
    stake: float
    kelly_share: float
    average_correlation: float
    weakest_correlation: float
    ticket_type: str

    @property
    def leg_count(self) -> int:
        return len(self.legs)

    @property
    def game_ids(self) -> set[str]:
        return {leg.game_id for leg in self.legs}

    @property
    def is_sgp(self) -> bool:
        return len(self.game_ids) == 1

    @property
    def subjects(self) -> set[str]:
        return {leg.subject for leg in self.legs}

    @property
    def implied_probability(self) -> float:
        return 1.0 / self.decimal_odds

    @property
    def edge(self) -> float:
        return self.joint_probability - self.implied_probability

    @property
    def correlation_lift(self) -> float:
        if self.independent_probability <= 0:
            return 0.0
        return self.joint_probability / self.independent_probability

    @property
    def to_win(self) -> float:
        return self.stake * (self.decimal_odds - 1.0)

    def rationale(self) -> list[str]:
        """Flattened per-leg reasoning, for the notification card."""
        notes: list[str] = []
        for leg in self.legs:
            for note in leg.rationale:
                entry = f"{leg.subject}: {note}"
                if entry not in notes:
                    notes.append(entry)
        return notes

    def as_dict(self) -> dict[str, object]:
        return {
            "ticket_type": self.ticket_type,
            "legs": [leg.describe() for leg in self.legs],
            "american_odds": self.american_odds,
            "decimal_odds": round(self.decimal_odds, 4),
            "model_probability": round(self.joint_probability, 4),
            "implied_probability": round(self.implied_probability, 4),
            "edge": round(self.edge, 4),
            "ev_per_unit": round(self.ev, 4),
            "correlation_lift": round(self.correlation_lift, 4),
            "average_correlation": round(self.average_correlation, 4),
            "stake": round(self.stake, 2),
            "to_win": round(self.to_win, 2),
        }


@dataclass
class BuildReport:
    """Why the candidate set ended up the size it did."""

    considered: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    candidates: int = 0
    selected: int = 0
    solver_status: str = "not_run"

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1


def describe_ticket_type(legs: Sequence[Leg]) -> str:
    """e.g. ``2-Leg Correlated NFL SGP`` / ``3-Leg Cross-Game NBA Parlay``."""
    sports = {leg.sport.upper() for leg in legs}
    sport = sports.pop() if len(sports) == 1 else "MULTI"
    if len({leg.game_id for leg in legs}) == 1:
        return f"{len(legs)}-Leg Correlated {sport} SGP"
    return f"{len(legs)}-Leg Cross-Game {sport} Parlay"


def same_game_pairs_are_correlated(
    legs: Sequence[Leg], threshold: float | None = None
) -> bool:
    """Every same-game pair must clear the SGP correlation floor.

    Cross-game pairs are exempt: they are independent by construction, which is
    exactly why they are allowed on a diversified ticket.
    """
    floor = settings.min_sgp_correlation if threshold is None else threshold
    for leg_a, leg_b in combinations(legs, 2):
        if leg_a.game_id != leg_b.game_id:
            continue
        if pairwise_correlation(leg_a, leg_b) < floor:
            return False
    return True


def _has_duplicate_selection(legs: Sequence[Leg]) -> bool:
    """Reject the same subject+market appearing twice on one ticket."""
    seen: set[tuple[str, str]] = set()
    for leg in legs:
        key = (leg.subject, leg.market)
        if key in seen:
            return True
        seen.add(key)
    return False


def price_ticket(
    legs: Sequence[Leg],
    *,
    iterations: int | None = None,
    seed: int | None = None,
    bankroll: float | None = None,
    kelly: float | None = None,
) -> ParlayTicket:
    """Price a leg combination through the copula and size the stake."""
    simulation = GaussianCopulaSimulator(
        legs, iterations=iterations, seed=seed
    ).simulate()
    decimal_odds = parlay_decimal_odds(leg.american_odds for leg in legs)
    joint = simulation.joint_probability
    return ParlayTicket(
        legs=list(legs),
        joint_probability=joint,
        independent_probability=simulation.independent_probability,
        decimal_odds=decimal_odds,
        american_odds=decimal_to_american(decimal_odds),
        ev=expected_value_decimal(joint, decimal_odds),
        stake=round(kelly_stake(joint, decimal_odds, bankroll=bankroll, fraction=kelly), 2),
        kelly_share=kelly_fraction_of_bankroll(joint, decimal_odds, kelly),
        average_correlation=average_correlation(legs),
        weakest_correlation=min_correlation(legs),
        ticket_type=describe_ticket_type(legs),
    )


#: Hard cap on the leg pool fed to the enumerator. Combinations grow as the
#: fourth power, and a slate rarely offers more genuinely distinct edges.
MAX_POOL = 40


def enumerate_tickets(
    legs: Sequence[Leg],
    *,
    min_legs: int | None = None,
    max_legs: int | None = None,
    min_ticket_ev: float = 0.0,
    iterations: int | None = None,
    seed: int | None = None,
    bankroll: float | None = None,
    kelly: float | None = None,
    max_pool: int | None = None,
    report: BuildReport | None = None,
) -> list[ParlayTicket]:
    """Feasible, positive-EV tickets from a pool of legs.

    Cheap filters (duplicate subject, correlation floor, ticket price band) run
    before the copula, so a Monte-Carlo simulation is only spent on a
    combination that could actually be recommended.
    """
    low = settings.min_legs if min_legs is None else min_legs
    high = settings.max_legs if max_legs is None else max_legs
    report = report if report is not None else BuildReport()

    eligible = [
        leg for leg in legs
        if settings.leg_odds_min <= leg.american_odds <= settings.leg_odds_max
    ]
    skipped = len(legs) - len(eligible)
    for _ in range(skipped):
        report.reject("leg odds outside band")

    cap = MAX_POOL if max_pool is None else max_pool
    if len(eligible) > cap:
        eligible = sorted(eligible, key=lambda leg: leg.ev, reverse=True)[:cap]

    candidates: list[ParlayTicket] = []
    for size in range(low, high + 1):
        for combination in combinations(eligible, size):
            report.considered += 1
            if _has_duplicate_selection(combination):
                report.reject("duplicate subject/market on ticket")
                continue
            if not same_game_pairs_are_correlated(combination):
                report.reject("same-game pair below correlation floor")
                continue
            price = parlay_decimal_odds(leg.american_odds for leg in combination)
            american = decimal_to_american(price)
            if not (settings.parlay_odds_min <= american <= settings.parlay_odds_max):
                report.reject("ticket odds outside band")
                continue

            ticket = price_ticket(
                combination,
                iterations=iterations,
                seed=seed,
                bankroll=bankroll,
                kelly=kelly,
            )
            if ticket.ev < min_ticket_ev:
                report.reject("ticket EV below floor")
                continue
            candidates.append(ticket)

    candidates.sort(key=lambda t: t.ev, reverse=True)
    report.candidates = len(candidates)
    return candidates


def select_portfolio(
    candidates: Sequence[ParlayTicket],
    *,
    max_tickets: int | None = None,
    report: BuildReport | None = None,
) -> list[ParlayTicket]:
    """Pick the EV-maximising set of tickets with unique subjects (ILP).

    Constraints:
      * at most ``max_tickets`` tickets;
      * each subject (player or team) appears on at most one ticket.
    """
    cap = settings.max_tickets if max_tickets is None else max_tickets
    report = report if report is not None else BuildReport()
    if not candidates or cap <= 0:
        report.solver_status = "empty"
        return []

    problem = pulp.LpProblem("parlay_portfolio", pulp.LpMaximize)
    choices = [
        pulp.LpVariable(f"ticket_{index}", cat=pulp.LpBinary)
        for index in range(len(candidates))
    ]
    problem += pulp.lpSum(
        ticket.ev * choice for ticket, choice in zip(candidates, choices)
    )
    problem += pulp.lpSum(choices) <= cap, "ticket_cap"

    subjects: dict[str, list[pulp.LpVariable]] = {}
    for ticket, choice in zip(candidates, choices):
        for subject in ticket.subjects:
            subjects.setdefault(subject, []).append(choice)
    for subject, variables in subjects.items():
        if len(variables) > 1:
            problem += pulp.lpSum(variables) <= 1, f"unique_{_slug(subject)}"

    try:
        problem.solve(pulp.PULP_CBC_CMD(msg=False))
        status = pulp.LpStatus[problem.status]
    except pulp.PulpSolverError as exc:  # pragma: no cover - solver availability
        logger.warning("ILP solver unavailable (%s); falling back to greedy", exc)
        report.solver_status = "greedy_fallback"
        return _greedy_portfolio(candidates, cap)

    report.solver_status = status
    if status != "Optimal":
        logger.warning("solver returned %s; falling back to greedy", status)
        return _greedy_portfolio(candidates, cap)

    selected = [
        ticket
        for ticket, choice in zip(candidates, choices)
        if choice.value() is not None and choice.value() > 0.5
    ]
    selected.sort(key=lambda t: t.ev, reverse=True)
    report.selected = len(selected)
    return selected


def _greedy_portfolio(
    candidates: Sequence[ParlayTicket], cap: int
) -> list[ParlayTicket]:
    """EV-ordered greedy pick honouring the unique-subject rule."""
    chosen: list[ParlayTicket] = []
    used: set[str] = set()
    for ticket in sorted(candidates, key=lambda t: t.ev, reverse=True):
        if len(chosen) >= cap:
            break
        if ticket.subjects & used:
            continue
        chosen.append(ticket)
        used |= ticket.subjects
    return chosen


def build_parlays(
    legs: Sequence[Leg],
    *,
    min_legs: int | None = None,
    max_legs: int | None = None,
    max_tickets: int | None = None,
    min_ticket_ev: float = 0.0,
    iterations: int | None = None,
    seed: int | None = None,
    bankroll: float | None = None,
    kelly: float | None = None,
    max_pool: int | None = None,
) -> tuple[list[ParlayTicket], BuildReport]:
    """Enumerate, then select. Returns the card and a diagnostic report."""
    report = BuildReport()
    candidates = enumerate_tickets(
        legs,
        min_legs=min_legs,
        max_legs=max_legs,
        min_ticket_ev=min_ticket_ev,
        iterations=iterations,
        seed=seed,
        bankroll=bankroll,
        kelly=kelly,
        max_pool=max_pool,
        report=report,
    )
    selected = select_portfolio(candidates, max_tickets=max_tickets, report=report)
    report.selected = len(selected)
    return selected, report


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)


def summarise(tickets: Iterable[ParlayTicket]) -> str:
    """One-line-per-ticket text summary (used by the CLI)."""
    lines: list[str] = []
    for ticket in tickets:
        legs = " + ".join(leg.describe() for leg in ticket.legs)
        lines.append(
            f"{ticket.ticket_type} {format_odds(ticket.american_odds)} | "
            f"model {ticket.joint_probability:.1%} vs implied "
            f"{ticket.implied_probability:.1%} | EV {ticket.ev:+.1%} | "
            f"stake {ticket.stake:.2f}\n    {legs}"
        )
    return "\n".join(lines) or "no qualifying tickets"
