"""``python -m src.learning.train`` -- refit the model from real results.

    uv run python -m src.learning.train --sport nfl --seasons 2025 2026
    uv run python -m src.learning.train --sport nfl --write

Pulls the seasons' box scores, replays every week forward (projecting only
from the weeks before it), fits the corrections, and reports whether they
improve the weeks that were held back. ``--write`` then saves them to
``data/calibration.json``, which is what the engine and the static site read.

Nothing is written unless the holdout says the fit helped, so a bad season of
data cannot quietly make the model worse. ``--force`` overrides that.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Iterable

import pandas as pd

from config.settings import settings
from src.learning import calibrate
from src.learning.backtest import observations, to_frame
from src.models.baseline import NBA_STAT_MARKETS, NFL_STAT_MARKETS, load_nfl_frames, seasons_to_load
from src.models.calibration import CALIBRATION_PATH, Calibration, using_calibration

logger = logging.getLogger(__name__)

STAT_MARKETS = {
    "nfl": NFL_STAT_MARKETS,
    "ncaaf": NFL_STAT_MARKETS,
    "nba": NBA_STAT_MARKETS,
}


def load_schedule(sport: str, seasons: Iterable[int]) -> pd.DataFrame:
    """Played games with the line they closed at, for the game-market fit.

    Only the NFL publishes closing lines alongside results in nflverse, so
    every other sport keeps its prior spread.
    """
    if sport != "nfl":
        return pd.DataFrame()
    import nflreadpy

    years = set(int(season) for season in seasons)
    schedule = nflreadpy.load_schedules().to_pandas()
    return schedule[schedule["season"].isin(years)]


def load_weekly(sport: str, seasons: Iterable[int]) -> pd.DataFrame:
    """The box-score frame for a sport, from whichever source covers it."""
    years = sorted(set(seasons))
    if sport == "nfl":
        weekly, _ = load_nfl_frames(years)
        return weekly
    if sport == "ncaaf":
        from src.models.cfb import load_cfb_frames

        weekly, _ = load_cfb_frames(years)
        return weekly
    if sport == "nba":
        raise SystemExit(
            "NBA history needs stats.nba.com, which blocks datacenter IPs; "
            "run this one from a machine with a residential connection"
        )
    raise SystemExit(f"unknown sport: {sport}")


def report_markets(frame: pd.DataFrame, fitted, *, sport: str, after) -> pd.DataFrame:
    """Per-market before/after, so a market that got worse is visible."""
    rows = []
    for market, part in frame.groupby("market", sort=True):
        mask = (frame["market"] == market).to_numpy()
        outcomes = part["hit"].to_numpy(dtype=float)
        before = calibrate.score(part["p_over"].to_numpy(dtype=float), outcomes)
        corrected = calibrate.score(after[mask], outcomes)
        learned = fitted.markets.get(market)
        rows.append(
            {
                "market": market,
                "n": before.samples,
                "hit_rate": round(before.hit_rate, 4),
                "brier_before": round(before.brier, 4),
                "brier_after": round(corrected.brier, 4),
                "mean_factor": round(learned.mean_factor, 3) if learned else None,
                "dispersion": round(learned.dispersion, 3) if learned and learned.dispersion else None,
                "family": (learned.family if learned else None) or "",
                "platt_a": round(learned.platt_a, 3) if learned else None,
                "platt_b": round(learned.platt_b, 3) if learned else None,
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sport", default="nfl", choices=sorted(STAT_MARKETS))
    parser.add_argument(
        "--seasons", type=int, nargs="*", default=None,
        help="seasons to learn from (default: this season and the one before)",
    )
    parser.add_argument("--holdout-weeks", type=int, default=4)
    parser.add_argument("--min-history", type=int, default=2)
    parser.add_argument("--min-samples", type=int, default=60)
    parser.add_argument("--write", action="store_true", help="save to data/calibration.json")
    parser.add_argument("--force", action="store_true", help="write even if the holdout got worse")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--no-game-markets", action="store_true",
        help="skip the totals/spreads fit, which needs closing lines",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sport = args.sport
    seasons = args.seasons or seasons_to_load(settings.season)

    print(f"loading {sport} box scores for {seasons} ...")
    weekly = load_weekly(sport, seasons)
    print(f"  {len(weekly):,} player-week rows")

    # Measure the uncorrected model, whatever is already on disk. Fitting on
    # top of a previous fit would compound the corrections a little more each
    # week, and the weekly workflow would drift without anything looking wrong.
    with using_calibration(Calibration.blank()):
        rows = observations(
            weekly,
            sport=sport,
            stat_markets=STAT_MARKETS[sport],
            min_history=args.min_history,
        )
    frame = to_frame(rows)
    if frame.empty:
        print("no observations -- nothing to learn from")
        return 1
    print(f"  {len(frame):,} observations across {frame['market'].nunique()} markets")

    game_markets: dict[str, Any] = {}
    if not args.no_game_markets:
        schedule = load_schedule(sport, seasons)
        if not schedule.empty:
            game_markets = calibrate.fit_game_markets(
                schedule, sport=sport, min_samples=args.min_samples
            )
            for market, fitted in sorted(game_markets.items()):
                print(f"  {market}: sd {fitted.dispersion:.2f} points over {fitted.samples} games")

    report = calibrate.train(
        frame,
        sport=sport,
        holdout_weeks=args.holdout_weeks,
        min_samples=args.min_samples,
        seasons=seasons,
        game_markets=game_markets,
    )

    print(f"\nheld back the last {report.holdout_weeks} week(s); trained on {report.train_samples:,}")
    print(f"  before  brier {report.before.brier:.4f}  log loss {report.before.log_loss:.4f}  "
          f"claimed {report.before.claimed:.3f} vs actual {report.before.hit_rate:.3f}")
    print(f"  after   brier {report.after.brier:.4f}  log loss {report.after.log_loss:.4f}  "
          f"claimed {report.after.claimed:.3f} vs actual {report.after.hit_rate:.3f}")
    print(f"  brier improved by {report.brier_gain * 100:.2f}%")

    whole = Calibration(sports={sport: report.calibration})
    corrected = calibrate.probabilities_under(frame, whole, sport=sport)

    print("\nper market (scored on every observation):")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(
            report_markets(frame, report.calibration, sport=sport, after=corrected)
            .to_string(index=False)
        )

    print("\ncalibration, claimed vs observed:")
    comparison = calibrate.calibration_comparison(
        frame, sport=sport, bins=args.bins, corrected=corrected
    )
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(comparison.to_string(index=False))

    if not args.write:
        print(f"\ndry run -- pass --write to save to {CALIBRATION_PATH}")
        return 0

    if not report.improved and not args.force:
        print("\nthe holdout got worse, so nothing was written. --force overrides this.")
        return 1

    merged = Calibration.load().merged_with(
        Calibration(sports={sport: report.calibration})
    )
    path = merged.save()
    print(f"\nwrote {len(report.calibration.markets)} market(s) for {sport} to {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
