#!/usr/bin/env python3
"""Master orchestration CLI for the FanDuel parlay engine.

    python run_pipeline.py --sport nfl --mode dry-run --mock
    python run_pipeline.py --sport nba --mode live --legs 3

Stages: ingest -> project -> reason -> price -> optimise -> notify. ``--mock``
swaps every network call for cached fixtures so the whole pipeline can be
verified end to end without spending Odds API credits.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from config.settings import SPORT_KEYS, WEATHER_SPORTS, settings
from src.ingestion import db, mock
from src.ingestion.injuries import InjuryClient, status_index
from src.ingestion.odds_api import OddsAPIClient, QuotaExhaustedError
from src.ingestion.weather import WeatherClient
from src.models import baseline
from src.models.legs import Leg
from src.notifications.notifier import Notifier, render_console
from src.optimizer import clv
from src.optimizer.leg_builder import attach_rationale, legs_from_lines, legs_from_props
from src.optimizer.parlay_builder import BuildReport, ParlayTicket, build_parlays
from src.reasoning.context_agent import ContextAgent, ContextResult, RuleBasedContextAgent

logger = logging.getLogger("parlay_engine")


@dataclass
class PipelineResult:
    """Everything one run produced, for the CLI and for tests."""

    sport: str
    mode: str
    mock: bool
    tickets: list[ParlayTicket] = field(default_factory=list)
    build_report: BuildReport = field(default_factory=BuildReport)
    legs_considered: int = 0
    edges: int = 0
    projections: int = 0
    games: int = 0
    context_results: list[ContextResult] = field(default_factory=list)
    ingest: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    dispatch: str = ""
    run_id: str = ""

    @property
    def duration(self) -> float:
        return sum(self.timings.values())

    @property
    def adjustments(self) -> int:
        return sum(len(result.applied) for result in self.context_results)

    def summary(self) -> dict[str, Any]:
        return {
            "sport": self.sport,
            "mode": self.mode,
            "mock": self.mock,
            "games": self.games,
            "projections": self.projections,
            "context_adjustments": self.adjustments,
            "legs_considered": self.legs_considered,
            "legs_with_edge": self.edges,
            "ticket_candidates": self.build_report.candidates,
            "tickets": len(self.tickets),
            "solver_status": self.build_report.solver_status,
            "run_id": self.run_id,
            "rejections": self.build_report.rejected,
            "seconds": round(self.duration, 2),
        }


class _Stopwatch:
    """Records how long each stage took."""

    def __init__(self, timings: dict[str, float]) -> None:
        self.timings = timings

    def __call__(self, name: str) -> "_StageTimer":
        return _StageTimer(self.timings, name)


class _StageTimer:
    def __init__(self, timings: dict[str, float], name: str) -> None:
        self.timings = timings
        self.name = name

    def __enter__(self) -> "_StageTimer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc_info) -> None:
        self.timings[self.name] = round(time.perf_counter() - self.start, 3)


# ----------------------------------------------------------------------
# stages
# ----------------------------------------------------------------------
async def ingest_stage(
    sport: str, *, use_mock: bool, db_path: str | None, include_props: bool
) -> dict[str, Any]:
    """Load the slate, weather and injuries into SQLite."""
    if use_mock:
        return mock.ingest_mock_slate(sport, db_path=db_path).as_dict()

    async with OddsAPIClient(db_path=db_path) as client:
        try:
            summary = await client.ingest_slate(sport, include_props=include_props)
        except QuotaExhaustedError as exc:
            logger.error("aborting ingestion: %s", exc)
            raise

    games = db.fetch_all(
        "SELECT * FROM games WHERE sport = ?", (sport,), db_path=db_path
    )
    if sport in WEATHER_SPORTS and settings.openweather_api_key:
        async with WeatherClient(db_path=db_path) as weather_client:
            await weather_client.ingest_games(games)
    async with InjuryClient(db_path=db_path) as injury_client:
        await injury_client.ingest(sport)
    return summary.as_dict()


def projection_stage(
    sport: str, games: Sequence[dict[str, Any]], *, use_mock: bool, injuries: dict[str, str]
) -> tuple[list, dict[str, Any]]:
    """Baseline player projections plus one game projection per game."""
    lookup = baseline.team_lookup(sport)
    if use_mock:
        first, second = mock.mock_stat_frames(sport)
    elif sport == "nfl":
        season = datetime.now(timezone.utc).year
        first, second = baseline.load_nfl_frames([season])
    else:
        year = datetime.now(timezone.utc).year
        first, second = baseline.load_nba_frames(f"{year}-{str(year + 1)[-2:]}")

    if sport == "nfl":
        projections = baseline.build_nfl_projections(
            first, second, games, team_lookup=lookup, injury_index=injuries
        )
        efficiency = baseline.nfl_team_efficiency(second)
        game_projections = {
            game["game_id"]: baseline.project_nfl_game(game, efficiency, team_lookup=lookup)
            for game in games
        }
    else:
        projections = baseline.build_nba_projections(
            first, second, games, team_lookup=lookup, injury_index=injuries
        )
        game_projections = {
            game["game_id"]: baseline.project_nba_game(game, second, team_lookup=lookup)
            for game in games
        }
    return projections, game_projections


async def reasoning_stage(
    games: Sequence[dict[str, Any]],
    projections: Sequence[Any],
    *,
    agent: ContextAgent,
    db_path: str | None,
) -> tuple[list, list[ContextResult]]:
    """Apply bounded context adjustments game by game."""
    weather_rows = {
        row["game_id"]: row
        for row in db.fetch_all(
            "SELECT * FROM weather_snapshots ORDER BY id", db_path=db_path
        )
    }
    injury_rows = db.fetch_all("SELECT * FROM injury_reports ORDER BY id", db_path=db_path)

    adjusted: list = []
    results: list[ContextResult] = []
    for game in games:
        game_projections = [p for p in projections if p.game_id == game["game_id"]]
        if not game_projections:
            continue
        teams = {game.get("home_team"), game.get("away_team")}
        relevant_injuries = [
            row for row in injury_rows
            if any(
                baseline.same_team(row.get("team"), name, row.get("sport") or "nfl")
                for name in teams
            )
        ]
        result = await agent.evaluate_game(
            game,
            game_projections,
            weather=weather_rows.get(game["game_id"]),
            injuries=relevant_injuries,
            lines=db.latest_lines(game["game_id"], db_path=db_path),
        )
        results.append(result)
        adjusted.extend(result.projections)
    return adjusted, results


def leg_stage(
    sport: str,
    games: Sequence[dict[str, Any]],
    projections: Sequence[Any],
    game_projections: dict[str, Any],
    *,
    db_path: str | None,
    include_game_markets: bool = True,
) -> list[Leg]:
    """Join stored prices with projections into priced legs."""
    legs: list[Leg] = []
    for game in games:
        game_id = game["game_id"]
        props = db.latest_props(game_id, db_path=db_path)
        legs.extend(
            legs_from_props(
                props,
                [p for p in projections if p.game_id == game_id],
                sport=sport,
            )
        )
        if include_game_markets:
            legs.extend(
                legs_from_lines(
                    db.latest_lines(game_id, db_path=db_path),
                    game_projections.get(game_id),
                    sport=sport,
                )
            )
    return legs


# ----------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------
async def run_pipeline(
    *,
    sport: str,
    mode: str = "dry-run",
    use_mock: bool = False,
    legs: int | None = None,
    bankroll: float | None = None,
    max_tickets: int | None = None,
    db_path: str | None = None,
    iterations: int | None = None,
    seed: int | None = None,
    include_props: bool = True,
    include_game_markets: bool = True,
    agent: ContextAgent | None = None,
    notifier: Notifier | None = None,
    notify: bool = True,
    log_bets: bool = True,
) -> PipelineResult:
    """Run every stage and return the resulting card plus diagnostics."""
    sport = sport.lower()
    result = PipelineResult(sport=sport, mode=mode, mock=use_mock)
    clock = _Stopwatch(result.timings)

    with clock("ingest"):
        result.ingest = await ingest_stage(
            sport, use_mock=use_mock, db_path=db_path, include_props=include_props
        )

    games = db.fetch_all(
        "SELECT * FROM games WHERE sport = ? ORDER BY commence_time",
        (sport,),
        db_path=db_path,
    )
    result.games = len(games)
    if not games:
        logger.warning("no %s games available; nothing to do", sport)
        return result

    with clock("projections"):
        injury_rows = db.fetch_all(
            "SELECT * FROM injury_reports WHERE sport = ?", (sport,), db_path=db_path
        )
        from src.ingestion.injuries import InjuryRecord

        index = status_index(
            InjuryRecord(
                sport=row["sport"], team=row["team"], player_name=row["player_name"],
                position=row["position"], status=row["status"], practice=row["practice"],
                detail=row["detail"], source=row["source"], report_date=row["report_date"],
            )
            for row in injury_rows
        )
        projections, game_projections = projection_stage(
            sport, games, use_mock=use_mock, injuries=index
        )
        result.projections = len(projections)

    with clock("reasoning"):
        agent = agent or (RuleBasedContextAgent() if use_mock else ContextAgent())
        projections, context_results = await reasoning_stage(
            games, projections, agent=agent, db_path=db_path
        )
        result.context_results = context_results

    with clock("legs"):
        candidate_legs = leg_stage(
            sport, games, projections, game_projections,
            db_path=db_path, include_game_markets=include_game_markets,
        )
        attach_rationale(candidate_legs, context_results)
        result.legs_considered = len(candidate_legs)
        edges = [leg for leg in candidate_legs if leg.ev >= settings.min_leg_ev
                 and settings.leg_odds_min <= leg.american_odds <= settings.leg_odds_max]
        result.edges = len(edges)

    with clock("optimize"):
        tickets, build_report = build_parlays(
            edges,
            min_legs=legs,
            max_legs=legs,
            max_tickets=max_tickets,
            iterations=iterations,
            seed=seed,
            bankroll=bankroll,
        )
        result.tickets = tickets
        result.build_report = build_report
        if tickets and log_bets:
            result.run_id = clv.log_recommendations(
                tickets, sport=sport, db_path=db_path
            )

    if notify:
        with clock("notify"):
            dry_run = mode != "live"
            notifier = notifier or Notifier(dry_run=dry_run)
            async with notifier as dispatcher:
                dispatch = await dispatcher.send(
                    tickets, header=_card_header(sport, mode, len(tickets))
                )
            result.dispatch = f"{','.join(dispatch.channels)} ({dispatch.detail})"
    return result


def _card_header(sport: str, mode: str, count: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"{sport.upper()} card - {count} ticket(s) - {mode} - {stamp}"


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Build FanDuel parlay recommendations for an NFL or NBA slate.",
    )
    parser.add_argument("--sport", choices=sorted(SPORT_KEYS), default="nfl")
    parser.add_argument(
        "--mode", choices=("dry-run", "live"), default="dry-run",
        help="dry-run prints the card to stdout; live dispatches to the webhooks",
    )
    parser.add_argument(
        "--legs", type=int, choices=(2, 3, 4), default=None,
        help="build tickets of exactly this many legs (default: 2-4)",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="use cached fixtures instead of live APIs (spends no Odds API credits)",
    )
    parser.add_argument("--bankroll", type=float, default=None)
    parser.add_argument("--max-tickets", type=int, default=None)
    parser.add_argument("--db", dest="db_path", default=None, help="override DB_PATH")
    parser.add_argument(
        "--iterations", type=int, default=None,
        help=f"copula iterations per ticket (default {settings.copula_iterations})",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-props", dest="include_props", action="store_false",
        help="skip per-event prop requests (saves Odds API credits)",
    )
    parser.add_argument(
        "--no-game-markets", dest="include_game_markets", action="store_false",
        help="player props only; skip moneyline/spread/total legs",
    )
    parser.add_argument(
        "--clv-report", dest="clv_report", action="store_true",
        help="print a closing line value report from the logged bets and exit",
    )
    parser.add_argument(
        "--no-bet-log", dest="log_bets", action="store_false",
        help="do not record the recommended legs (skips CLV benchmarking)",
    )
    parser.add_argument(
        "--json", dest="as_json", action="store_true",
        help="print the run summary as JSON (in addition to the card)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.clv_report:
        db.init_db(args.db_path)
        entries, summary = clv.report(sport=args.sport, db_path=args.db_path)
        print(clv.render(entries, summary))
        return 0

    if args.mode == "live" and not args.mock and not settings.odds_api_key:
        print(
            "ODDS_API_KEY is not set. Copy .env.example to .env and fill it in, "
            "or re-run with --mock.",
            file=sys.stderr,
        )
        return 2

    try:
        result = asyncio.run(
            run_pipeline(
                sport=args.sport,
                mode=args.mode,
                use_mock=args.mock,
                legs=args.legs,
                bankroll=args.bankroll,
                max_tickets=args.max_tickets,
                db_path=args.db_path,
                iterations=args.iterations,
                seed=args.seed,
                include_props=args.include_props,
                include_game_markets=args.include_game_markets,
                log_bets=args.log_bets,
            )
        )
    except QuotaExhaustedError as exc:
        print(f"Aborted: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # surface the failure, keep the traceback for -v
        logger.exception("pipeline failed") if args.verbose else None
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1

    summary = result.summary()
    if args.as_json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            "\n"
            + " | ".join(
                f"{key}={value}"
                for key, value in summary.items()
                if key not in {"rejections"}
            )
        )
        if summary["rejections"]:
            print(f"rejections: {summary['rejections']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
