"""End-to-end pipeline tests, driven entirely by the cached fixtures."""

from __future__ import annotations

import time

import pytest

import run_pipeline
from config.settings import settings
from src.ingestion import db, mock
from src.reasoning.context_agent import ContextAgent, RuleBasedContextAgent
from src.reasoning.prompts import GameContextReport, PropAdjustment


# --------------------------------------------------------------- fixtures
def test_fixtures_exist_for_both_sports():
    assert set(mock.available_sports()) >= {"nfl", "nba"}


def test_mock_ingestion_fills_every_table(db_path):
    summary = mock.ingest_mock_slate("nfl", db_path=db_path).as_dict()
    assert summary["games"] == 3
    assert summary["lines"] > 0 and summary["props"] > 0
    assert summary["requests_made"] == 0  # no API credit spent

    counts = {
        table: db.fetch_all(f"SELECT COUNT(*) c FROM {table}", db_path=db_path)[0]["c"]
        for table in ("games", "fanduel_lines", "fanduel_props",
                      "weather_snapshots", "injury_reports")
    }
    assert all(count > 0 for count in counts.values()), counts
    # the dome game is flagged without a forecast, the cold game carries flags
    weather = {row["game_id"]: row for row in
               db.fetch_all("SELECT * FROM weather_snapshots", db_path=db_path)}
    assert weather["nfl-mock-2"]["is_dome"] == 1
    assert weather["nfl-mock-1"]["high_wind"] == 1


def test_missing_fixture_raises_a_helpful_error():
    with pytest.raises(mock.MockDataMissing, match="generate_mock_data"):
        mock.load_fixture("nhl", "odds")


# --------------------------------------------------------------- pipeline
@pytest.mark.parametrize("sport", ["nfl", "nba"])
async def test_dry_run_produces_a_valid_card(sport, db_path):
    started = time.perf_counter()
    result = await run_pipeline.run_pipeline(
        sport=sport, mode="dry-run", use_mock=True, db_path=db_path,
        iterations=2_000, seed=11, bankroll=1_000, notify=False,
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 45.0  # the plan's dry-run budget
    assert result.games == 3
    assert result.projections > 0
    assert result.legs_considered > 0
    assert result.tickets, "expected at least one qualifying ticket"
    assert result.build_report.solver_status == "Optimal"

    seen_subjects: set[str] = set()
    for ticket in result.tickets:
        assert settings.min_legs <= ticket.leg_count <= settings.max_legs
        assert settings.parlay_odds_min <= ticket.american_odds <= settings.parlay_odds_max
        assert ticket.ev > 0 and ticket.stake > 0
        for leg in ticket.legs:
            assert settings.leg_odds_min <= leg.american_odds <= settings.leg_odds_max
            assert leg.ev >= settings.min_leg_ev
        assert not (ticket.subjects & seen_subjects)  # diversification
        seen_subjects |= ticket.subjects


async def test_same_game_tickets_only_pair_correlated_legs(db_path):
    result = await run_pipeline.run_pipeline(
        sport="nfl", use_mock=True, db_path=db_path, iterations=2_000,
        seed=11, notify=False,
    )
    for ticket in result.tickets:
        if ticket.is_sgp:
            assert ticket.weakest_correlation >= settings.min_sgp_correlation


async def test_leg_count_flag_pins_the_ticket_size(db_path):
    result = await run_pipeline.run_pipeline(
        sport="nfl", use_mock=True, db_path=db_path, legs=2,
        iterations=2_000, seed=11, notify=False,
    )
    assert result.tickets
    assert {ticket.leg_count for ticket in result.tickets} == {2}


async def test_context_adjustments_are_applied_and_bounded(db_path):
    result = await run_pipeline.run_pipeline(
        sport="nfl", use_mock=True, db_path=db_path, iterations=1_000,
        seed=11, notify=False,
    )
    assert result.adjustments > 0
    for context in result.context_results:
        for applied in context.applied:
            assert 1 - settings.max_context_adjustment - 1e-9 <= applied.applied_factor
            assert applied.applied_factor <= 1 + settings.max_context_adjustment + 1e-9
    # the rationale reaches the legs and therefore the card
    notes = [note for ticket in result.tickets for note in ticket.rationale()]
    assert any("wind" in note.lower() for note in notes)


async def test_extreme_agent_output_cannot_move_a_projection_more_than_20_percent(db_path):
    class RunawayAgent(ContextAgent):
        """Returns absurd factors to prove the clamp holds end to end."""

        @property
        def available(self) -> bool:
            return True

        async def request_report(self, payload):
            return GameContextReport(
                game_id=payload["game"]["game_id"],
                projected_game_script="Adversarial test.",
                adjustments=[
                    PropAdjustment(
                        player_name=row["player_name"],
                        market=row["market"],
                        original_projection=row["projected_mean"],
                        adjusted_projection=row["projected_mean"] * 5,
                        adjustment_factor=5.0,
                        confidence_score=1.0,
                        primary_reasoning="Adversarial input",
                    )
                    for row in payload["baseline_projections"]
                ],
            )

    result = await run_pipeline.run_pipeline(
        sport="nfl", use_mock=True, db_path=db_path, iterations=1_000,
        seed=11, notify=False, agent=RunawayAgent(),
    )
    assert result.adjustments > 0
    for context in result.context_results:
        for applied in context.applied:
            assert applied.applied_factor == pytest.approx(1.2)
            assert applied.clamped is True
            assert applied.adjusted_mean == pytest.approx(applied.baseline_mean * 1.2)


async def test_no_games_short_circuits_cleanly(db_path, monkeypatch):
    async def _no_op(*args, **kwargs):
        return {}

    monkeypatch.setattr(run_pipeline, "ingest_stage", _no_op)
    result = await run_pipeline.run_pipeline(
        sport="nba", db_path=db_path, notify=False, agent=RuleBasedContextAgent(),
    )
    assert result.games == 0 and result.tickets == []
    assert result.projections == 0


async def test_notify_stage_runs_in_dry_run(db_path, capsys):
    result = await run_pipeline.run_pipeline(
        sport="nfl", mode="dry-run", use_mock=True, db_path=db_path,
        iterations=1_000, seed=11,
    )
    assert "console" in result.dispatch
    assert "ticket(s)" in capsys.readouterr().out
    assert set(result.timings) >= {"ingest", "projections", "reasoning", "legs",
                                  "optimize", "notify"}


# -------------------------------------------------------------------- CLI
def test_cli_defaults():
    args = run_pipeline.parse_args([])
    assert args.sport == "nfl" and args.mode == "dry-run"
    assert args.legs is None and args.mock is False
    assert args.include_props is True and args.include_game_markets is True


def test_cli_rejects_five_leg_tickets():
    with pytest.raises(SystemExit):
        run_pipeline.parse_args(["--legs", "5"])


def test_cli_mock_run_exits_zero(tmp_path, capsys):
    code = run_pipeline.main(
        ["--sport", "nfl", "--mock", "--db", str(tmp_path / "cli.db"),
         "--iterations", "1000", "--seed", "7", "--max-tickets", "2"]
    )
    output = capsys.readouterr().out
    assert code == 0
    assert "tickets=" in output


def test_cli_json_summary(tmp_path, capsys):
    import json

    code = run_pipeline.main(
        ["--sport", "nba", "--mock", "--json", "--db", str(tmp_path / "cli.db"),
         "--iterations", "500", "--seed", "7", "--legs", "2", "--max-tickets", "1"]
    )
    output = capsys.readouterr().out
    summary = json.loads(output[output.index("{"):])
    assert code == 0
    assert summary["sport"] == "nba" and summary["mock"] is True
    assert summary["tickets"] <= 1 and summary["seconds"] > 0


def test_live_mode_without_a_key_exits_with_a_hint(monkeypatch, capsys):
    monkeypatch.setattr(settings, "odds_api_key", "")
    code = run_pipeline.main(["--sport", "nfl", "--mode", "live"])
    assert code == 2
    assert "ODDS_API_KEY is not set" in capsys.readouterr().err
