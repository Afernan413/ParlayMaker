"""Claude-powered context evaluation.

The agent receives baseline projections plus weather, injuries and market
lines, and returns bounded multiplicative shifts. Three guardrails make the
output safe to feed into money maths:

* the projection set is fixed -- adjustments for unknown player/market pairs
  are discarded;
* every factor is clamped to +/- ``settings.max_context_adjustment``;
* low-confidence adjustments are ignored.

With no API key configured the agent reports itself unavailable and the
pipeline runs on pure baseline projections.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from pydantic import ValidationError

from config.settings import settings
from src.models.legs import Projection
from src.reasoning.prompts import (
    ADJUSTMENT_TOOL,
    SYSTEM_PROMPT,
    GameContextReport,
    PropAdjustment,
    build_context_payload,
    render_user_prompt,
)

logger = logging.getLogger(__name__)


def clamp_factor(factor: float, max_adjustment: float | None = None) -> float:
    """Clamp an adjustment factor into the permitted band."""
    ceiling = settings.max_context_adjustment if max_adjustment is None else max_adjustment
    return float(min(max(factor, 1.0 - ceiling), 1.0 + ceiling))


@dataclass
class AppliedAdjustment:
    """Audit record for one adjustment that actually changed a projection."""

    player_name: str
    market: str
    requested_factor: float
    applied_factor: float
    baseline_mean: float
    adjusted_mean: float
    confidence: float
    reasoning: str
    clamped: bool

    def note(self) -> str:
        suffix = " (clamped to bound)" if self.clamped else ""
        return (
            f"{self.player_name} {self.market}: x{self.applied_factor:.2f}{suffix} "
            f"-- {self.reasoning}"
        )


@dataclass
class ContextResult:
    """Outcome of reasoning over one game."""

    game_id: str
    projections: list[Projection]
    report: GameContextReport | None = None
    applied: list[AppliedAdjustment] = field(default_factory=list)
    discarded: list[str] = field(default_factory=list)

    @property
    def game_script(self) -> str:
        return self.report.projected_game_script if self.report else ""

    @property
    def rationale(self) -> list[str]:
        return [item.note() for item in self.applied]


def apply_report(
    projections: Sequence[Projection],
    report: GameContextReport | None,
    *,
    max_adjustment: float | None = None,
    min_confidence: float | None = None,
) -> ContextResult:
    """Apply a report to baseline projections under the hard guardrails.

    Returns adjusted projections in the original order; anything the report
    does not touch (or touches invalidly) passes through untouched.
    """
    game_id = report.game_id if report else (projections[0].game_id if projections else "")
    result = ContextResult(game_id=game_id, projections=list(projections))
    if report is None:
        return result

    floor = settings.min_context_confidence if min_confidence is None else min_confidence
    index = {(p.player_name, p.market): i for i, p in enumerate(projections)}
    updated = list(projections)

    for adjustment in report.adjustments:
        key = (adjustment.player_name, adjustment.market)
        position = index.get(key)
        if position is None:
            result.discarded.append(
                f"unknown projection {adjustment.player_name}/{adjustment.market}"
            )
            continue
        if adjustment.confidence_score < floor:
            result.discarded.append(
                f"low confidence {adjustment.confidence_score:.2f} for "
                f"{adjustment.player_name}/{adjustment.market}"
            )
            continue

        applied_factor = clamp_factor(adjustment.adjustment_factor, max_adjustment)
        if applied_factor == 1.0:
            continue
        baseline = updated[position]
        adjusted = baseline.scaled(applied_factor, note=adjustment.primary_reasoning)
        updated[position] = adjusted
        result.applied.append(
            AppliedAdjustment(
                player_name=adjustment.player_name,
                market=adjustment.market,
                requested_factor=adjustment.adjustment_factor,
                applied_factor=applied_factor,
                baseline_mean=baseline.mean,
                adjusted_mean=adjusted.mean,
                confidence=adjustment.confidence_score,
                reasoning=adjustment.primary_reasoning,
                clamped=abs(applied_factor - adjustment.adjustment_factor) > 1e-9,
            )
        )

    result.projections = updated
    result.report = report
    return result


def parse_tool_response(message: Any) -> GameContextReport | None:
    """Pull a :class:`GameContextReport` out of an Anthropic message.

    Tolerates a JSON text block as a fallback, and returns ``None`` when the
    payload cannot be validated -- the caller then keeps the baseline.
    """
    blocks = getattr(message, "content", None) or []
    for block in blocks:
        if getattr(block, "type", None) == "tool_use":
            try:
                return GameContextReport.model_validate(block.input)
            except ValidationError as exc:
                logger.warning("context report failed validation: %s", exc)
                return None
    for block in blocks:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            return GameContextReport.model_validate(json.loads(text))
        except (ValidationError, json.JSONDecodeError):
            continue
    logger.warning("no usable game_context_report in model response")
    return None


class ContextAgent:
    """Thin, testable wrapper around one Anthropic tool call per game."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_adjustment: float | None = None,
        min_confidence: float | None = None,
        max_tokens: int = 2048,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.anthropic_api_key
        self.model = model or settings.anthropic_model
        self.max_adjustment = (
            settings.max_context_adjustment if max_adjustment is None else max_adjustment
        )
        self.min_confidence = (
            settings.min_context_confidence if min_confidence is None else min_confidence
        )
        self.max_tokens = max_tokens
        self._client = client

    @property
    def available(self) -> bool:
        """True when a real (or injected) client can be used."""
        return self._client is not None or bool(self.api_key)

    @property
    def client(self) -> Any:
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def request_report(self, payload: Mapping[str, Any]) -> GameContextReport | None:
        """One Claude call for one game's payload."""
        if not self.available:
            logger.info("context agent unavailable (no ANTHROPIC_API_KEY); skipping")
            return None
        try:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=[ADJUSTMENT_TOOL],
                tool_choice={"type": "tool", "name": ADJUSTMENT_TOOL["name"]},
                messages=[{"role": "user", "content": render_user_prompt(payload)}],
            )
        except Exception as exc:  # network/API failure must not kill the pipeline
            logger.warning("context agent call failed: %s", exc)
            return None
        return parse_tool_response(message)

    async def evaluate_game(
        self,
        game: Mapping[str, Any],
        projections: Sequence[Projection],
        *,
        weather: Mapping[str, Any] | None = None,
        injuries: Sequence[Mapping[str, Any]] | None = None,
        lines: Sequence[Mapping[str, Any]] | None = None,
        notes: Sequence[str] | None = None,
    ) -> ContextResult:
        """Adjust one game's projections; degrades to baseline on any failure."""
        if not projections:
            return ContextResult(game_id=str(game.get("game_id", "")), projections=[])
        payload = build_context_payload(
            game, projections, weather=weather, injuries=injuries, lines=lines, notes=notes
        )
        report = await self.request_report(payload)
        return apply_report(
            projections,
            report,
            max_adjustment=self.max_adjustment,
            min_confidence=self.min_confidence,
        )


def adjustment_from_factor(
    projection: Projection, factor: float, reasoning: str, confidence: float = 0.7
) -> PropAdjustment:
    """Build an adjustment record (used by mock runs and tests)."""
    return PropAdjustment(
        player_name=projection.player_name,
        market=projection.market,
        original_projection=projection.mean,
        adjusted_projection=projection.mean * factor,
        adjustment_factor=factor,
        confidence_score=confidence,
        primary_reasoning=reasoning,
    )
