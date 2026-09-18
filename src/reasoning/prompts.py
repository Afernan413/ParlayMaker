"""System prompts and structured-output schemas for the reasoning layer.

Claude's job here is narrow on purpose: it may nudge projection *parameters*
based on late-breaking context, and nothing else. It never sees a chance to
invent a line, a price or a player.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, Field, field_validator

from config.settings import settings

MAX_FACTOR = 1.0 + settings.max_context_adjustment
MIN_FACTOR = 1.0 - settings.max_context_adjustment


class PropAdjustment(BaseModel):
    """One bounded shift to a player's projected mean."""

    player_name: str
    market: str
    original_projection: float
    adjusted_projection: float
    adjustment_factor: float = Field(
        description=(
            "Multiplier applied to the baseline projection, e.g. 0.85 for a 15% "
            f"downgrade. Must stay within [{MIN_FACTOR:.2f}, {MAX_FACTOR:.2f}]."
        )
    )
    confidence_score: float = Field(ge=0.0, le=1.0)
    primary_reasoning: str = Field(
        description="One sentence citing the specific evidence, e.g. "
        "'Wind sustained at 22mph; deep ball volume downgraded'."
    )

    @field_validator("adjustment_factor")
    @classmethod
    def _factor_must_be_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("adjustment_factor must be > 0")
        return value


class GameContextReport(BaseModel):
    """Claude's full read on one game."""

    game_id: str
    projected_game_script: str = Field(
        description="Two sentences max on the expected flow of the game."
    )
    adjustments: list[PropAdjustment] = Field(default_factory=list)


SYSTEM_PROMPT = f"""You are a sports analytics risk profiler embedded in a \
quantitative betting pipeline. A statistical model has already produced \
baseline player projections from historical usage and efficiency data. Your \
only job is to adjust those projections for late-breaking context the model \
cannot see: weather, injuries and inactives, personnel reallocation, rest, and \
scheme mismatches.

Hard rules:
1. You may only adjust projections that are given to you. Never introduce a \
player, market, line or price that is not in the payload.
2. Every adjustment_factor must stay within \
[{MIN_FACTOR:.2f}, {MAX_FACTOR:.2f}]. Context is a nudge, not a rewrite. If the \
evidence seems to justify more, cap it at the bound and say so in the reasoning.
3. adjusted_projection must equal original_projection * adjustment_factor.
4. Only report an adjustment when concrete evidence in the payload supports it. \
A projection you would leave alone must be omitted entirely -- do not pad the \
list with 1.0 factors.
5. confidence_score reflects how directly the evidence bears on that player's \
volume or efficiency: 0.8+ for a ruled-out teammate or extreme wind, 0.4-0.6 \
for a scheme read, below 0.4 for speculation (which will be discarded).
6. primary_reasoning must cite the specific fact from the payload (wind speed, \
who is out, which defender). No generic filler.

Calibration guidance:
- Sustained wind above 15 mph suppresses deep passing volume and kicking; it \
mildly helps rushing volume.
- Freezing temperatures modestly suppress passing efficiency.
- A ruled-out primary receiver reallocates target share to the remaining \
starters; a ruled-out lead back concentrates carries.
- A questionable player already carries a volume haircut in the baseline -- do \
not double-count it.
- Elite defensive matchups are already partly priced into the baseline via \
efficiency data; adjust only for a specific mismatch."""


def summarise_weather(weather: Mapping[str, Any] | None) -> str:
    if not weather:
        return "No forecast available."
    if weather.get("is_dome"):
        return f"Indoor game at {weather.get('stadium') or 'a dome'}; weather is a non-factor."
    parts = [
        f"{weather.get('temperature_f')}F" if weather.get("temperature_f") is not None else None,
        f"wind {weather.get('wind_speed_mph')} mph" if weather.get("wind_speed_mph") is not None else None,
        f"precip chance {weather.get('precip_chance')}" if weather.get("precip_chance") is not None else None,
        weather.get("conditions"),
    ]
    summary = ", ".join(p for p in parts if p)
    flags = []
    if weather.get("high_wind"):
        flags.append("HIGH WIND")
    if weather.get("freezing"):
        flags.append("FREEZING")
    return summary + (f" [{', '.join(flags)}]" if flags else "")


def summarise_injuries(injuries: Sequence[Mapping[str, Any]] | None) -> list[str]:
    if not injuries:
        return []
    return [
        f"{row.get('player_name')} ({row.get('position') or '?'}, "
        f"{row.get('team') or '?'}): {row.get('status')}"
        + (f" -- {row['detail']}" if row.get("detail") else "")
        for row in injuries
    ]


def build_context_payload(
    game: Mapping[str, Any],
    projections: Iterable[Any],
    *,
    weather: Mapping[str, Any] | None = None,
    injuries: Sequence[Mapping[str, Any]] | None = None,
    lines: Sequence[Mapping[str, Any]] | None = None,
    notes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Assemble everything Claude is allowed to reason over for one game."""
    return {
        "game": {
            "game_id": game.get("game_id"),
            "sport": game.get("sport"),
            "kickoff": game.get("commence_time"),
            "home_team": game.get("home_team"),
            "away_team": game.get("away_team"),
        },
        "weather": summarise_weather(weather),
        "injuries": summarise_injuries(injuries),
        "market_lines": [
            {
                "market": row.get("market"),
                "selection": row.get("selection"),
                "line": row.get("line"),
                "odds": row.get("american_odds"),
            }
            for row in (lines or [])
        ],
        "baseline_projections": [
            {
                "player_name": projection.player_name,
                "team": projection.team,
                "opponent": projection.opponent,
                "market": projection.market,
                "projected_mean": round(projection.mean, 3),
                "derivation": list(projection.notes),
            }
            for projection in projections
        ],
        "analyst_notes": list(notes or []),
    }


def render_user_prompt(payload: Mapping[str, Any]) -> str:
    """Render the payload as the user turn (JSON keeps it unambiguous)."""
    import json

    game = payload.get("game", {})
    header = (
        f"Game {game.get('game_id')}: {game.get('away_team')} at "
        f"{game.get('home_team')} ({game.get('sport')}, kickoff {game.get('kickoff')})."
    )
    return (
        f"{header}\n\n"
        "Context payload:\n"
        f"{json.dumps(payload, indent=2, default=str)}\n\n"
        "Return one game_context_report describing the projected game script and "
        "only the adjustments the evidence above supports."
    )


def adjustment_tool_schema() -> dict[str, Any]:
    """Anthropic tool definition derived from :class:`GameContextReport`."""
    # $defs is kept: the nested PropAdjustment definition is part of the schema.
    schema = GameContextReport.model_json_schema()
    return {
        "name": "game_context_report",
        "description": (
            "Report the projected game script and every bounded projection "
            "adjustment the supplied context justifies."
        ),
        "input_schema": schema,
    }


#: Ready-made tool definition for the Anthropic Messages API.
ADJUSTMENT_TOOL: dict[str, Any] = adjustment_tool_schema()
