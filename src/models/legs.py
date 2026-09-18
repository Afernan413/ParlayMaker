"""Shared vocabulary passed between the model, reasoning and optimizer layers.

Keeping :class:`Projection` and :class:`Leg` in one leaf module avoids circular
imports: ``baseline`` produces projections, ``distributions`` turns them into
probabilities, ``reasoning`` shifts them, and the optimizer consumes legs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class Projection:
    """A baseline (pre-context) expectation for one player-market."""

    sport: str
    game_id: str
    player_name: str
    team: str
    market: str
    mean: float
    dispersion: float | None = None
    opponent: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.game_id, self.player_name, self.market)

    def scaled(self, factor: float, note: str | None = None) -> "Projection":
        """Return a copy with the mean multiplied by ``factor``."""
        notes = self.notes + ((note,) if note else ())
        return replace(self, mean=max(self.mean * factor, 0.0), notes=notes)


@dataclass
class Leg:
    """A single bettable selection with model and market probabilities."""

    game_id: str
    sport: str
    market: str
    selection: str  # Over | Under | Yes | team name
    american_odds: int
    line: float | None = None
    player_name: str | None = None
    team: str | None = None
    p_model: float = 0.0  # model ("true") probability
    p_implied: float = 0.0  # de-vigged market probability
    ev: float = 0.0  # expected value per 1 unit staked
    projection_mean: float | None = None
    baseline_mean: float | None = None
    rationale: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def leg_id(self) -> str:
        subject = self.player_name or self.team or self.selection
        line = "" if self.line is None else f" {self.line:g}"
        return f"{self.game_id}|{self.market}|{subject}|{self.selection}{line}"

    @property
    def subject(self) -> str:
        """Player when it is a prop, otherwise the team/selection."""
        return self.player_name or self.team or self.selection

    def describe(self) -> str:
        line = "" if self.line is None else f" {self.line:g}"
        label = f"{self.subject} {self.selection}{line}".strip()
        return f"{label} ({format_odds(self.american_odds)})"


def format_odds(american: int | float) -> str:
    """American odds with an explicit sign, the way a bet slip shows them."""
    value = int(round(american))
    return f"+{value}" if value > 0 else str(value)
