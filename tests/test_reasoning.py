"""Reasoning-layer tests. The Anthropic client is always a stub."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.models.legs import Projection
from src.reasoning.context_agent import (
    ContextAgent,
    apply_report,
    clamp_factor,
    parse_tool_response,
)
from src.reasoning.prompts import (
    ADJUSTMENT_TOOL,
    SYSTEM_PROMPT,
    GameContextReport,
    build_context_payload,
    summarise_weather,
)

PROJECTIONS = [
    Projection(sport="nfl", game_id="g1", player_name="QB One", team="KC",
               opponent="BUF", market="player_pass_yds", mean=290.0),
    Projection(sport="nfl", game_id="g1", player_name="WR One", team="KC",
               opponent="BUF", market="player_reception_yds", mean=72.0),
]


def report(**adjustment) -> GameContextReport:
    base = {
        "player_name": "QB One",
        "market": "player_pass_yds",
        "original_projection": 290.0,
        "adjusted_projection": 246.5,
        "adjustment_factor": 0.85,
        "confidence_score": 0.8,
        "primary_reasoning": "Wind sustained at 22mph; deep ball volume downgraded",
    }
    base.update(adjustment)
    return GameContextReport(
        game_id="g1", projected_game_script="Low-scoring, run-leaning.",
        adjustments=[base],
    )


class StubMessages:
    def __init__(self, response, recorder: dict):
        self._response = response
        self._recorder = recorder

    async def create(self, **kwargs):
        self._recorder.update(kwargs)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class StubClient:
    def __init__(self, response):
        self.calls: dict = {}
        self.messages = StubMessages(response, self.calls)


def tool_message(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", name="game_context_report", input=payload)]
    )


# ------------------------------------------------------------------ prompts
def test_system_prompt_states_the_hard_bounds():
    assert "0.80" in SYSTEM_PROMPT and "1.20" in SYSTEM_PROMPT
    assert "Never introduce a player" in SYSTEM_PROMPT


def test_tool_schema_exposes_the_report_shape():
    schema = ADJUSTMENT_TOOL["input_schema"]
    assert ADJUSTMENT_TOOL["name"] == "game_context_report"
    assert set(schema["properties"]) >= {"game_id", "projected_game_script", "adjustments"}
    assert "PropAdjustment" in schema["$defs"]


def test_payload_carries_only_supplied_facts():
    payload = build_context_payload(
        {"game_id": "g1", "sport": "nfl", "home_team": "KC", "away_team": "BUF",
         "commence_time": "2026-09-20T17:00:00Z"},
        PROJECTIONS,
        weather={"is_dome": 0, "wind_speed_mph": 22.0, "high_wind": 1,
                 "temperature_f": 30.0, "freezing": 1},
        injuries=[{"player_name": "WR Two", "position": "WR", "team": "KC",
                   "status": "OUT", "detail": "ankle"}],
        lines=[{"market": "totals", "selection": "Under", "line": 41.5,
                "american_odds": -110}],
    )
    assert [p["player_name"] for p in payload["baseline_projections"]] == ["QB One", "WR One"]
    assert "HIGH WIND" in payload["weather"] and "FREEZING" in payload["weather"]
    assert payload["injuries"] == ["WR Two (WR, KC): OUT -- ankle"]
    assert payload["market_lines"][0]["line"] == 41.5


def test_dome_weather_summary_is_explicit():
    assert "non-factor" in summarise_weather({"is_dome": 1, "stadium": "Ford Field"})
    assert summarise_weather(None) == "No forecast available."


# ------------------------------------------------------------------- bounds
@pytest.mark.parametrize(
    ("requested", "expected"),
    [(0.5, 0.8), (0.0001, 0.8), (1.9, 1.2), (10.0, 1.2), (0.93, 0.93), (1.07, 1.07)],
)
def test_clamp_factor_enforces_the_twenty_percent_ceiling(requested, expected):
    assert clamp_factor(requested) == pytest.approx(expected)


def test_extreme_adjustment_is_clamped_and_flagged():
    result = apply_report(PROJECTIONS, report(adjustment_factor=0.4))
    assert result.projections[0].mean == pytest.approx(290.0 * 0.8)
    applied = result.applied[0]
    assert applied.clamped is True
    assert applied.requested_factor == 0.4 and applied.applied_factor == 0.8
    assert "clamped" in applied.note()


def test_in_bounds_adjustment_is_applied_verbatim():
    result = apply_report(PROJECTIONS, report(adjustment_factor=0.85))
    assert result.projections[0].mean == pytest.approx(246.5)
    assert result.applied[0].clamped is False
    assert result.projections[1].mean == 72.0  # untouched
    assert "Wind sustained" in result.projections[0].notes[-1]


def test_unknown_player_market_pairs_are_discarded():
    result = apply_report(PROJECTIONS, report(player_name="Ghost Player"))
    assert result.applied == []
    assert "unknown projection Ghost Player" in result.discarded[0]
    assert [p.mean for p in result.projections] == [290.0, 72.0]


def test_low_confidence_adjustments_are_ignored():
    result = apply_report(PROJECTIONS, report(confidence_score=0.2))
    assert result.applied == []
    assert "low confidence" in result.discarded[0]


def test_no_report_leaves_projections_untouched():
    result = apply_report(PROJECTIONS, None)
    assert result.projections == list(PROJECTIONS)
    assert result.rationale == []


def test_neutral_factor_is_not_recorded_as_an_adjustment():
    result = apply_report(PROJECTIONS, report(adjustment_factor=1.0))
    assert result.applied == []
    assert result.projections[0].mean == 290.0


def test_non_positive_factor_is_rejected_by_the_schema():
    with pytest.raises(ValueError):
        report(adjustment_factor=0.0)


# -------------------------------------------------------------------- agent
async def test_agent_sends_the_tool_and_applies_the_report():
    payload = report(adjustment_factor=0.88).model_dump()
    client = StubClient(tool_message(payload))
    agent = ContextAgent(client=client, model="claude-opus-5")
    result = await agent.evaluate_game(
        {"game_id": "g1", "sport": "nfl", "home_team": "KC", "away_team": "BUF"},
        PROJECTIONS,
        weather={"is_dome": 0, "wind_speed_mph": 22.0, "high_wind": 1},
    )

    assert client.calls["tools"][0]["name"] == "game_context_report"
    assert client.calls["tool_choice"] == {"type": "tool", "name": "game_context_report"}
    assert client.calls["system"] == SYSTEM_PROMPT
    assert result.projections[0].mean == pytest.approx(290.0 * 0.88)
    assert result.game_script == "Low-scoring, run-leaning."
    assert result.rationale and "Wind sustained" in result.rationale[0]


async def test_agent_degrades_to_baseline_when_the_api_fails():
    agent = ContextAgent(client=StubClient(RuntimeError("503 overloaded")))
    result = await agent.evaluate_game({"game_id": "g1"}, PROJECTIONS)
    assert result.report is None
    assert [p.mean for p in result.projections] == [290.0, 72.0]


async def test_agent_without_a_key_is_unavailable():
    agent = ContextAgent(api_key="")
    assert agent.available is False
    result = await agent.evaluate_game({"game_id": "g1"}, PROJECTIONS)
    assert result.report is None
    assert result.projections == list(PROJECTIONS)


async def test_agent_handles_an_empty_projection_set():
    agent = ContextAgent(client=StubClient(tool_message(report().model_dump())))
    result = await agent.evaluate_game({"game_id": "g1"}, [])
    assert result.projections == []


def test_malformed_tool_input_is_rejected():
    assert parse_tool_response(tool_message({"game_id": "g1"})) is None
    assert parse_tool_response(SimpleNamespace(content=[])) is None


def test_json_text_block_is_accepted_as_a_fallback():
    payload = report().model_dump_json()
    message = SimpleNamespace(content=[SimpleNamespace(type="text", text=payload)])
    parsed = parse_tool_response(message)
    assert parsed is not None and parsed.game_id == "g1"
