"""The forward half of the learning loop: log what was priced, grade it later.

:mod:`src.learning.backtest` reconstructs what the model *would* have said
about past weeks. That is the only way to get a big sample quickly, but it is
not the same thing as what the model *did* say: the live pipeline also has
injuries, weather and the reasoning layer's nudges in it, and none of that can
be replayed after the fact.

So every leg the pipeline prices is written to ``projection_log`` when it is
priced, and graded once the box score exists. Over a season that becomes a
second, honest training set -- one that measures the whole pipeline rather
than the projection recipe alone.

    uv run python -m src.learning.journal --sport nfl     # grade what is due
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from config.settings import settings
from src.ingestion import db
from src.learning.backtest import TD_MARKETS, TD_STAT, with_derived_stats
from src.models.baseline import NBA_STAT_MARKETS, NFL_STAT_MARKETS
from src.models.legs import Leg

logger = logging.getLogger(__name__)

#: Odds API market key -> the box-score column that settles it.
MARKET_STATS: dict[str, dict[str, str]] = {
    "nfl": {market: stat for stat, market in {**NFL_STAT_MARKETS, **TD_MARKETS}.items()},
    "ncaaf": {market: stat for stat, market in NFL_STAT_MARKETS.items()},
    "nba": {market: stat for stat, market in NBA_STAT_MARKETS.items()},
}

#: Sides that settle over the line; the rest settle under it.
OVER_SIDES = frozenset({"over", "yes"})


def normalise_name(name: str) -> str:
    """Lowercase, punctuation-free form used to match a log row to a box score."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def record(
    legs: Sequence[Leg],
    *,
    run_id: str,
    sport: str,
    games: Mapping[str, Mapping[str, Any]] | None = None,
    db_path=None,
) -> int:
    """Write every priced player leg to ``projection_log``.

    Game markets are skipped: they are settled from a final score rather than
    a player's box score, and the grader below only knows how to read the
    latter.
    """
    settleable = MARKET_STATS.get(sport, {})
    games = games or {}
    rows: list[dict[str, Any]] = []
    captured_at = db.utcnow()
    for leg in legs:
        if leg.market not in settleable or not leg.player_name:
            continue
        game = games.get(leg.game_id) or {}
        rows.append(
            {
                "run_id": run_id,
                "sport": sport,
                "game_id": leg.game_id,
                "commence_time": game.get("commence_time"),
                "season": game.get("season"),
                "week": game.get("week"),
                "player_name": leg.player_name,
                "team": leg.team,
                "market": leg.market,
                "selection": leg.selection,
                "line": leg.line,
                "projected": leg.projection_mean,
                "p_model": leg.p_model,
                "p_implied": leg.p_implied,
                "american_odds": leg.american_odds,
                "actual": None,
                "hit": None,
                "graded_at": None,
                "captured_at": captured_at,
            }
        )
    return db.insert_projections(rows, db_path=db_path)


def _settled(selection: str, actual: float, line: float | None) -> int:
    """Did this side cash? A missing line means a binary market (any TD)."""
    side = (selection or "").strip().lower()
    threshold = 0.5 if line is None else float(line)
    over = actual > threshold
    return int(over if side in OVER_SIDES else not over)


def grade(
    weekly: pd.DataFrame,
    *,
    sport: str,
    db_path=None,
    before: str | None = None,
) -> int:
    """Settle every ungraded log row that ``weekly`` has a box score for.

    Rows whose game has not been played, or whose player is missing from the
    frame, are simply left pending for the next run.
    """
    pending = db.ungraded_projections(sport, before=before, db_path=db_path)
    if not pending:
        return 0

    stats = MARKET_STATS.get(sport, {})
    frame = with_derived_stats(weekly)
    name_col = "player_display_name" if "player_display_name" in frame.columns else "player_name"
    box: dict[tuple[str, int, int], pd.Series] = {}
    for row in frame.dropna(subset=[name_col, "season", "week"]).itertuples(index=False):
        record_ = row._asdict()
        key = (normalise_name(record_[name_col]), int(record_["season"]), int(record_["week"]))
        box[key] = record_

    grades: list[tuple[int, float, int]] = []
    for row in pending:
        stat = stats.get(row["market"])
        if stat is None or row["season"] is None or row["week"] is None:
            continue
        found = box.get((normalise_name(row["player_name"] or ""), int(row["season"]), int(row["week"])))
        if found is None or stat not in found:
            continue
        value = found[stat]
        actual = 0.0 if pd.isna(value) else float(value)
        grades.append((int(row["id"]), actual, _settled(row["selection"], actual, row["line"])))

    graded = db.grade_projections(grades, db_path=db_path)
    logger.info("graded %d of %d pending %s legs", graded, len(pending), sport)
    return graded


def graded_frame(sport: str, db_path=None) -> pd.DataFrame:
    """Graded log rows in the shape :mod:`src.learning.calibrate` expects.

    An ``Under`` leg is flipped back to the over side, because that is the
    convention the fitted corrections are stated in.
    """
    rows = db.graded_projections(sport, db_path=db_path)
    records = []
    for row in rows:
        if row["projected"] is None or row["actual"] is None:
            continue
        line = 0.5 if row["line"] is None else float(row["line"])
        records.append(
            {
                "sport": sport,
                "season": row["season"] or 0,
                "week": row["week"] or 0,
                "player": row["player_name"] or "",
                "team": row["team"] or "",
                "market": row["market"],
                "projected": float(row["projected"]),
                "actual": float(row["actual"]),
                "line": line,
                "p_over": _over_probability(row),
                "hit": int(float(row["actual"]) > line),
                "games_of_history": 0,
            }
        )
    return pd.DataFrame(records)


def _over_probability(row: Mapping[str, Any]) -> float:
    """``p_model`` restated as the probability of the over."""
    probability = float(row["p_model"] or 0.0)
    side = (row["selection"] or "").strip().lower()
    return probability if side in OVER_SIDES else 1.0 - probability


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade logged projections against box scores.")
    parser.add_argument("--sport", default="nfl", choices=sorted(MARKET_STATS))
    parser.add_argument("--seasons", type=int, nargs="*", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from src.learning.train import load_weekly

    seasons = args.seasons or [settings.season]
    pending = db.ungraded_projections(args.sport)
    if not pending:
        print(f"no ungraded {args.sport} legs")
        return 0
    print(f"{len(pending)} ungraded {args.sport} legs; loading {seasons} box scores ...")
    graded = grade(load_weekly(args.sport, seasons), sport=args.sport)
    print(f"graded {graded}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
