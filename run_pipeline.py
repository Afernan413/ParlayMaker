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
from typing import Mapping, Any, Sequence

from config.settings import FOOTBALL_SPORTS, SPORT_KEYS, WEATHER_SPORTS, settings
from src.ingestion import db, mock
from src.ingestion.injuries import InjuryClient, InjuryRecord, status_index
from src.ingestion.odds_api import OddsAPIClient, QuotaExhaustedError
from src.ingestion.weather import WeatherClient
from src.models import baseline
from src.models.legs import Leg
from src.models.inputs import ModelInputs, injury_status, starter_status, weather_status
from src.models.roles import RoleModel, load_nfl_roles
from src.notifications.notifier import Notifier, render_console
from src.optimizer import clv
from src.optimizer.ev_calculator import find_edges
from src.learning import journal
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
    journalled: int = 0
    #: One line naming what the projections could and could not see.
    inputs_summary: str = ""

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
            "journalled": self.journalled,
            "inputs": self.inputs_summary,
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
    sport: str,
    *,
    use_mock: bool,
    db_path: str | None,
    include_props: bool,
    max_events: int | None = None,
    inputs: ModelInputs | None = None,
) -> dict[str, Any]:
    """Load the slate, weather and injuries into SQLite.

    ``inputs`` records what was actually available, so a missing forecast key or
    a sport with no injury feed is reported rather than silently skipped.
    """
    if use_mock:
        if inputs is not None:
            inputs.record("weather", True, "bundled fixture forecast", 0)
        return mock.ingest_mock_slate(sport, db_path=db_path).as_dict()

    async with OddsAPIClient(db_path=db_path) as client:
        try:
            summary = await client.ingest_slate(
                sport, include_props=include_props, max_events=max_events
            )
        except QuotaExhaustedError as exc:
            logger.error("aborting ingestion: %s", exc)
            raise

    games = db.fetch_all(
        "SELECT * FROM games WHERE sport = ?", (sport,), db_path=db_path
    )
    snapshots = 0
    if sport in WEATHER_SPORTS and settings.openweather_api_key:
        async with WeatherClient(db_path=db_path) as weather_client:
            snapshots = len(await weather_client.ingest_games(games))
    if inputs is not None:
        inputs.statuses["weather"] = weather_status(
            sport,
            snapshots=snapshots,
            has_key=bool(settings.openweather_api_key),
            venues=sport in WEATHER_SPORTS,
        )
    try:
        async with InjuryClient(db_path=db_path) as injury_client:
            await injury_client.ingest(sport)
    except Exception as exc:  # a sport with no feed, or a feed that is down
        logger.warning("injury fetch failed for %s: %s", sport, exc)
    return summary.as_dict()


def projection_stage(
    sport: str,
    games: Sequence[dict[str, Any]],
    *,
    use_mock: bool,
    injuries: dict[str, str],
    inputs: ModelInputs | None = None,
) -> tuple[list, dict[str, Any]]:
    """Baseline player projections plus one game projection per game."""
    lookup = baseline.team_lookup(sport)
    role_model = RoleModel()
    if use_mock:
        first, second = mock.mock_stat_frames(sport)
    elif sport == "nfl":
        season = datetime.now(timezone.utc).year
        first, second = baseline.load_nfl_frames(baseline.seasons_to_load(season))
        second = baseline.latest_season_plays(second)
        # Who is starting, and who is about to start because the man ahead of
        # them is out. NFL only: nothing publishes college or basketball snaps
        # in a form this can read, so those keep the per-game average.
        try:
            role_model = load_nfl_roles(baseline.seasons_to_load(season), season=season)
        except Exception as exc:  # a release rebuilding must not stop the run
            logger.warning("snap/injury roles unavailable: %s", exc)

    if inputs is not None:
        _record_context(inputs, sport, role_model, injuries)
    elif sport == "ncaaf":
        from src.models import cfb

        season = datetime.now(timezone.utc).year
        first, second = cfb.load_cfb_frames(baseline.seasons_to_load(season))
        second = baseline.latest_season_plays(second)
    else:
        year = datetime.now(timezone.utc).year
        first, second = baseline.load_nba_frames(f"{year}-{str(year + 1)[-2:]}")

    if sport in FOOTBALL_SPORTS:
        projections = baseline.build_nfl_projections(
            first, second, games, team_lookup=lookup, injury_index=injuries,
            sport=sport, roles=role_model,
        )
        efficiency = baseline.nfl_team_efficiency(second)
        game_projections = {
            game["game_id"]: baseline.project_nfl_game(
                game, efficiency, team_lookup=lookup, sport=sport
            )
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


#: Why a sport has no snap data. Only the NFL publishes it in a readable form.
NO_SNAPS = {
    "ncaaf": "college football publishes no snap counts, so roles come from usage alone",
    "nba": "nba_api starting lineups need a residential connection",
}


def _record_context(
    inputs: ModelInputs, sport: str, role_model: RoleModel, injuries: Mapping[str, str]
) -> None:
    """Note what the projections were actually able to take into account."""
    coverage = role_model.coverage()
    inputs.statuses["starters"] = starter_status(
        players=coverage["players"],
        detail=(
            f"{coverage['players']} players across {coverage['teams']} teams, "
            f"{coverage['promoted']} with a changed role"
            if coverage["players"]
            # A sport with no snap source at all gets its own reason: the role
            # model's is about a thin window, which is not what is wrong here.
            else NO_SNAPS.get(sport) or role_model.reason
        ),
    )

    designations = coverage["designations"] or len(injuries)
    lag = coverage["report_lag_weeks"]
    if coverage["designations"]:
        detail = (
            f"league report, week {coverage['report_week']}"
            + (f" ({lag} week(s) stale)" if lag else " (current)")
            + f", {coverage['out']} ruled out"
        )
    elif injuries:
        detail = f"live feed only, {len(injuries)} designations"
    else:
        detail = ""
    inputs.statuses["injuries"] = injury_status(sport, designations=designations, detail=detail)


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
                commence_time=game.get("commence_time"),
            )
        )
        if include_game_markets:
            legs.extend(
                legs_from_lines(
                    db.latest_lines(game_id, db_path=db_path),
                    game_projections.get(game_id),
                    sport=sport,
                    commence_time=game.get("commence_time"),
                )
            )
    return legs


# ----------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------
@dataclass
class Slate:
    """Everything needed to price parlays for one slate.

    The web layer holds one of these in memory and prices arbitrary leg
    combinations against it, so a user assembling a slip never re-runs
    ingestion or projections.
    """

    sport: str
    mock: bool
    built_at: str
    #: What the projections were able to take into account -- starters, injuries
    #: and weather -- so a gap is reported rather than silently absent.
    inputs: ModelInputs | None = None
    games: list[dict[str, Any]] = field(default_factory=list)
    game_projections: dict[str, Any] = field(default_factory=dict)
    legs: list[Leg] = field(default_factory=list)
    edges: list[Leg] = field(default_factory=list)
    context_results: list[ContextResult] = field(default_factory=list)
    projections: int = 0
    ingest: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def legs_by_id(self) -> dict[str, Leg]:
        return {leg.leg_id: leg for leg in self.legs}

    @property
    def games_by_id(self) -> dict[str, dict[str, Any]]:
        return {game["game_id"]: game for game in self.games}


async def build_slate(
    *,
    sport: str,
    use_mock: bool = False,
    db_path: str | None = None,
    include_props: bool = True,
    include_game_markets: bool = True,
    max_events: int | None = None,
    agent: ContextAgent | None = None,
    timings: dict[str, float] | None = None,
) -> Slate:
    """Ingest, project, reason and price every leg on a slate.

    This is the shared half of the pipeline: the CLI hands the result to the
    optimizer, the web app hands it to a bet slip.
    """
    sport = sport.lower()
    clock = _Stopwatch(timings if timings is not None else {})
    slate = Slate(
        sport=sport, mock=use_mock, built_at=db.utcnow(),
        inputs=ModelInputs(sport=sport),
        timings=clock.timings,
    )

    with clock("ingest"):
        slate.ingest = await ingest_stage(
            sport,
            use_mock=use_mock,
            db_path=db_path,
            include_props=include_props,
            max_events=max_events,
            inputs=slate.inputs,
        )

    slate.games = db.fetch_all(
        "SELECT * FROM games WHERE sport = ? ORDER BY commence_time",
        (sport,),
        db_path=db_path,
    )
    if not slate.games:
        logger.warning("no %s games available; nothing to do", sport)
        return slate

    with clock("projections"):
        injury_rows = db.fetch_all(
            "SELECT * FROM injury_reports WHERE sport = ?", (sport,), db_path=db_path
        )
        index = status_index(
            InjuryRecord(
                sport=row["sport"], team=row["team"], player_name=row["player_name"],
                position=row["position"], status=row["status"], practice=row["practice"],
                detail=row["detail"], source=row["source"], report_date=row["report_date"],
            )
            for row in injury_rows
        )
        projections, game_projections = projection_stage(
            sport, slate.games, use_mock=use_mock, injuries=index, inputs=slate.inputs
        )
        slate.projections = len(projections)
        slate.game_projections = game_projections

    with clock("reasoning"):
        agent = agent or (RuleBasedContextAgent() if use_mock else ContextAgent())
        projections, context_results = await reasoning_stage(
            slate.games, projections, agent=agent, db_path=db_path
        )
        slate.context_results = context_results

    with clock("legs"):
        slate.legs = leg_stage(
            sport, slate.games, projections, slate.game_projections,
            db_path=db_path, include_game_markets=include_game_markets,
        )
        attach_rationale(slate.legs, slate.context_results)
        # Re-run the evaluator so the EV floor and price band live in one place.
        slate.edges, _ = find_edges(
            slate.legs,
            implied={leg.leg_id: leg.p_implied for leg in slate.legs},
        )
    return slate


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
    max_events: int | None = None,
    agent: ContextAgent | None = None,
    notifier: Notifier | None = None,
    notify: bool = True,
    log_bets: bool = True,
) -> PipelineResult:
    """Run every stage and return the resulting card plus diagnostics."""
    sport = sport.lower()
    result = PipelineResult(sport=sport, mode=mode, mock=use_mock)
    clock = _Stopwatch(result.timings)

    slate = await build_slate(
        sport=sport,
        use_mock=use_mock,
        db_path=db_path,
        include_props=include_props,
        include_game_markets=include_game_markets,
        max_events=max_events,
        agent=agent,
        timings=result.timings,
    )
    result.ingest = slate.ingest
    result.inputs_summary = slate.inputs.summary() if slate.inputs else ""
    result.games = len(slate.games)
    result.projections = slate.projections
    result.context_results = slate.context_results
    result.legs_considered = len(slate.legs)
    result.edges = len(slate.edges)
    if not slate.games:
        return result

    # Every priced leg is journalled, not just the ones that made the card, so
    # the learning loop trains on the model's whole opinion rather than on the
    # slice the optimizer happened to like. Mock runs are not journalled:
    # fictional players never appear in a box score, so they would sit in the
    # queue unresolved forever.
    if log_bets:
        result.run_id = clv.new_run_id()
        if not use_mock:
            with clock("journal"):
                result.journalled = journal.record(
                    slate.legs,
                    run_id=result.run_id,
                    sport=sport,
                    games=slate.games_by_id,
                    db_path=db_path,
                )

    with clock("optimize"):
        tickets, build_report = build_parlays(
            slate.edges,
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
            clv.log_recommendations(
                tickets, sport=sport, run_id=result.run_id or None, db_path=db_path
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
        "--max-events", type=int, default=None,
        help="only request player props for the first N games (saves API credits)",
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
                max_events=args.max_events,
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
