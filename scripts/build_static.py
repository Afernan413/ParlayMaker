#!/usr/bin/env python3
"""Build the static site: a self-contained parlay model you can open anywhere.

    python scripts/build_static.py                # cached fixtures
    python scripts/build_static.py --live         # real odds (needs ODDS_API_KEY)

The Python engine does everything expensive and judgement-laden -- projections,
distribution fits, the reasoning layer, de-vigging, and **every pairwise
correlation** -- and writes the result to ``site/data.js``. The browser then
only has to do arithmetic and a Monte-Carlo draw, so the correlation priors have
exactly one home (``src/models/correlation.py``) and cannot drift between the
two implementations.

Output is written as JavaScript rather than JSON on purpose: a page opened from
``file://`` cannot ``fetch()`` a local file, but it can load a script. The same
bundle therefore works by double-clicking the HTML, from a local server, and
from GitHub Pages.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import SPORT_KEYS, settings  # noqa: E402
from run_pipeline import Slate, build_slate  # noqa: E402
from src.models.calibration import active_calibration  # noqa: E402
from src.models.correlation import pairwise_correlation  # noqa: E402
from src.models.legs import Leg  # noqa: E402
from src.models.schedule import week_key, week_label, weeks_on  # noqa: E402
from src.optimizer.ev_calculator import american_to_decimal  # noqa: E402
from src.web.service import game_label, market_display  # noqa: E402

SITE_DIR = ROOT / "site"
SOURCE_DIR = ROOT / "src" / "web" / "static_site"


def leg_row(leg: Leg, game: dict[str, Any] | None, index: int) -> dict[str, Any]:
    """One selectable bet, already priced by the engine.

    ``i`` is this leg's position in the exported list. Correlation rows refer to
    legs by that index rather than by id: the ids are long strings and each pair
    would repeat two of them, which dominated the bundle size.
    """
    return {
        "i": index,
        "id": leg.leg_id,
        "game_id": leg.game_id,
        "game": game_label(game),
        "sport": leg.sport,
        "market": leg.market,
        "market_label": market_display(leg.market),
        "label": leg.label,
        "subject": leg.subject,
        "team": leg.team,
        "selection": leg.selection,
        "line": leg.line,
        "week": leg.slate_week,
        "odds": leg.american_odds,
        "decimal": round(american_to_decimal(leg.american_odds), 6),
        "p_model": round(leg.p_model, 6),
        "p_implied": round(leg.p_implied, 6),
        "ev": round(leg.ev, 6),
        "edge": round(leg.p_model - leg.p_implied, 6),
        "description": leg.describe(),
        "why": list(leg.rationale),
    }


def correlation_rows(legs: Sequence[Leg]) -> list[list[float]]:
    """Non-zero pairwise correlations, as ``[index_a, index_b, r]`` triples.

    Only same-game pairs can be non-zero, so this stays sparse -- a dense matrix
    would be n^2 and mostly zeros.
    """
    by_game: dict[str, list[tuple[int, Leg]]] = {}
    for index, leg in enumerate(legs):
        by_game.setdefault(leg.game_id, []).append((index, leg))

    rows: list[list[float]] = []
    for game_legs in by_game.values():
        for (index_a, leg_a), (index_b, leg_b) in combinations(game_legs, 2):
            rho = pairwise_correlation(leg_a, leg_b)
            if abs(rho) > 1e-9:
                rows.append([index_a, index_b, round(rho, 4)])
    return rows


def game_rows(slate: Slate) -> list[dict[str, Any]]:
    """Per-game model predictions next to the market's own line."""
    scripts = {
        result.game_id: result.game_script
        for result in slate.context_results
        if result.game_script
    }
    market = _market_lines(slate)

    rows: list[dict[str, Any]] = []
    for game in slate.games:
        projection = slate.game_projections.get(game["game_id"])
        row = {
            "game_id": game["game_id"],
            "label": game_label(game),
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "kickoff": game["commence_time"],
            "week": week_key(game["commence_time"]),
            "game_script": scripts.get(game["game_id"], ""),
            **market.get(game["game_id"], {}),
        }
        if projection is not None:
            row.update(
                {
                    "model_total": round(projection.total_mean, 1),
                    "model_margin": round(projection.home_margin_mean, 1),
                    "model_home_points": round(projection.home_points, 1),
                    "model_away_points": round(projection.away_points, 1),
                }
            )
        rows.append(row)
    return rows


def _market_lines(slate: Slate) -> dict[str, dict[str, Any]]:
    """The posted total and spread per game, pulled off the game-market legs."""
    lines: dict[str, dict[str, Any]] = {}
    for leg in slate.legs:
        entry = lines.setdefault(leg.game_id, {})
        if leg.market == "totals" and leg.selection == "Over":
            entry["market_total"] = leg.line
        elif leg.market == "spreads" and leg.team == entry.get("_home"):
            entry["market_spread"] = leg.line
    # Resolve the home-side spread now that we know the home team per game.
    for game in slate.games:
        entry = lines.setdefault(game["game_id"], {})
        for leg in slate.legs:
            if (
                leg.game_id == game["game_id"]
                and leg.market == "spreads"
                and leg.selection == game["home_team"]
            ):
                entry["market_spread"] = leg.line
        entry.pop("_home", None)
    return lines


def week_rows(slate: Slate) -> list[dict[str, Any]]:
    """The weeks this slate spans, earliest first.

    The odds feed returns every upcoming event, so a slate routinely holds two
    weeks. A parlay must settle together, so the page picks one week and builds
    within it -- this is the list it picks from.
    """
    counts: dict[str, int] = {}
    for game in slate.games:
        key = week_key(game["commence_time"])
        if key:
            counts[key] = counts.get(key, 0) + 1
    return [
        {"key": key, "label": week_label(key), "games": counts[key]}
        for key in weeks_on(slate.games)
    ]


def sport_bundle(slate: Slate) -> dict[str, Any]:
    """Everything the browser needs for one sport."""
    games = {game["game_id"]: game for game in slate.games}
    return {
        "sport": slate.sport,
        "mock": slate.mock,
        "built_at": slate.built_at,
        "games": game_rows(slate),
        "weeks": week_rows(slate),
        "legs": [
            leg_row(leg, games.get(leg.game_id), index)
            for index, leg in enumerate(slate.legs)
        ],
        "correlations": correlation_rows(slate.legs),
        "adjustments": sum(len(r.applied) for r in slate.context_results),
        "projections": slate.projections,
    }


def engine_settings() -> dict[str, Any]:
    """Thresholds the page shows, so its rules are never hidden from the user."""
    return {
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
    }


#: Why a sport is absent when the build never tried it. Without this the page
#: can only say "not in this build", which tells you nothing about what to do.
NOT_BUILT: dict[str, str] = {
    "nba": (
        "not built here -- stats.nba.com refuses datacenter IPs, so nba_api times "
        "out on a hosted runner. Build it from your own machine: "
        "python scripts/build_static.py --live --sports nba"
    ),
}
DEFAULT_NOT_BUILT = (
    "not included in this build. Add it: python scripts/build_static.py --live "
    "--sports nfl ncaaf"
)


def training_summary() -> dict[str, Any]:
    """What the model has learned, so the page can say when it last trained.

    The corrections themselves are already baked into every ``p_model`` in the
    bundle; this is only the provenance.
    """
    calibration = active_calibration()
    summary: dict[str, Any] = {}
    for sport, fitted in sorted(calibration.sports.items()):
        if not fitted.markets:
            continue
        metrics = fitted.metrics or {}
        summary[sport] = {
            "fitted_at": fitted.fitted_at,
            "seasons": list(fitted.seasons),
            "markets": len(fitted.markets),
            "observations": sum(fit.samples for fit in fitted.markets.values()),
            "brier_gain": metrics.get("brier_gain"),
        }
    return summary


async def build_bundle(
    sports: Sequence[str],
    *,
    use_mock: bool,
    db_path: str | None = None,
    max_events: int | None = None,
) -> dict[str, Any]:
    """Run the engine for each sport and assemble the data bundle.

    One sport failing must not take the whole site down: an out-of-season
    league, a rate-limited stats host, or a provider outage skips that sport
    and the rest still publishes. A skipped sport is recorded and reported --
    never quietly replaced with sample data, which would look like real odds.
    """
    bundle: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": engine_settings(),
        "training": training_summary(),
        "sports": {},
        "skipped": {},
    }
    for sport in sports:
        try:
            slate = await build_slate(
                sport=sport, use_mock=use_mock, db_path=db_path, max_events=max_events
            )
        except Exception as exc:  # provider outage, rate limit, off-season feed
            reason = f"{type(exc).__name__}: {exc}"
            bundle["skipped"][sport] = reason
            print(f"  {sport}: SKIPPED -- {reason}")
            continue

        if not slate.games:
            bundle["skipped"][sport] = "no games on the slate (off-season?)"
            print(f"  {sport}: SKIPPED -- no games on the slate")
            continue

        bundle["sports"][sport] = sport_bundle(slate)
        print(
            f"  {sport}: {len(slate.games)} games, {len(slate.legs)} bets, "
            f"{len(slate.edges)} clearing the EV floor"
        )

    # A sport the build was never asked for is also absent from the page, so say
    # why and what to run. The page has a button for every sport it knows about.
    for sport in SPORT_KEYS:
        if sport not in bundle["sports"] and sport not in bundle["skipped"]:
            bundle["skipped"][sport] = NOT_BUILT.get(sport, DEFAULT_NOT_BUILT)
    return bundle


def write_site(bundle: dict[str, Any], out_dir: Path) -> list[Path]:
    """Copy the page files and write the data bundle beside them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in ("index.html", "app.js", "engine.js", "styles.css"):
        source = SOURCE_DIR / name
        target = out_dir / name
        shutil.copyfile(source, target)
        written.append(target)

    data_file = out_dir / "data.js"
    data_file.write_text(
        "// Generated by scripts/build_static.py -- do not edit by hand.\n"
        "window.PARLAY_DATA = "
        + json.dumps(bundle, separators=(",", ":"))
        + ";\n"
    )
    written.append(data_file)

    # Stops GitHub Pages running the upload through Jekyll, which would drop
    # any file or directory beginning with an underscore.
    nojekyll = out_dir / ".nojekyll"
    nojekyll.write_text("")
    written.append(nojekyll)
    return written


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_static.py",
        description="Build the standalone browser version of the parlay model.",
    )
    parser.add_argument(
        "--sports", nargs="+", default=sorted(SPORT_KEYS), choices=sorted(SPORT_KEYS)
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--mock", dest="use_mock", action="store_true", default=True)
    source.add_argument(
        "--live", dest="use_mock", action="store_false",
        help="pull real odds (requires ODDS_API_KEY; spends credits)",
    )
    parser.add_argument("--out", type=Path, default=SITE_DIR)
    parser.add_argument("--db", dest="db_path", default=None)
    parser.add_argument(
        "--max-events", type=int, default=None,
        help="cap how many games get prop requests on a live build",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.use_mock and not settings.odds_api_key:
        print("ODDS_API_KEY is not set; building from cached fixtures instead.")
        args.use_mock = True

    source = "cached fixtures" if args.use_mock else "live odds"
    print(f"Building the static site from {source}…")
    bundle = asyncio.run(
        build_bundle(
            args.sports,
            use_mock=args.use_mock,
            db_path=args.db_path,
            max_events=args.max_events,
        )
    )
    if not bundle["sports"]:
        print("\nNo sport built, so there is nothing to publish:", file=sys.stderr)
        for sport, reason in bundle["skipped"].items():
            print(f"  {sport}: {reason}", file=sys.stderr)
        return 1

    written = write_site(bundle, args.out)
    size_kb = (args.out / "data.js").stat().st_size / 1024
    print(f"\nWrote {len(written)} files to {args.out}/ (data.js is {size_kb:.0f} KB)")
    if bundle["skipped"]:
        print("Skipped: " + ", ".join(bundle["skipped"]))
    print(f"Open it now:  file://{(args.out / 'index.html').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
