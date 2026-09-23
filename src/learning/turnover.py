"""How much should a player's old season and old team count?

``python -m src.learning.turnover`` answers it with held-out data rather than
opinion. The league turns over between and during seasons -- trades, free
agency, new coordinators, a quarterback benched -- and the volume model's
four-game average reaches straight across all of it. Whether to discount those
games, and by how much, is an empirical question, so each candidate weighting
is fitted on earlier seasons and scored on later ones it never saw.

Three slices are reported, because an overall number would hide the point:

* **all** observations;
* **early season** (weeks 1-4), where the window reaches back into last season;
* **changed team**, where the window holds games for a different team than the
  one the player is playing for.

A weighting that only helps overall but hurts the players who changed teams is
not the improvement it looks like.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.learning import calibrate
from src.learning.backtest import observations, to_frame
from src.models.baseline import load_nfl_frames
from src.models.calibration import Calibration, using_calibration
from src.models.roles import normalise_name

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Variant:
    name: str
    season_decay: float = 1.0
    team_decay: float = 1.0
    window: int = 4

    def kwargs(self) -> dict:
        return {
            "season_decay": self.season_decay,
            "team_decay": self.team_decay,
            "window": self.window,
        }


#: The second sweep: the first showed the longer window was what helped, and
#: that the decays on their own made things worse. This separates the two.
WINDOW_VARIANTS = (
    Variant("current model (4 games)"),
    Variant("6 games", window=6),
    Variant("8 games", window=8),
    Variant("10 games", window=10),
    Variant("8 games, last season x0.5", window=8, season_decay=0.5),
    Variant("8 games, old team x0.5", window=8, team_decay=0.5),
)

DEFAULT_VARIANTS = (
    Variant("current model"),
    Variant("last season x0.5", season_decay=0.5),
    Variant("last season x0.25", season_decay=0.25),
    Variant("old team x0.5", team_decay=0.5),
    Variant("old team x0.25", team_decay=0.25),
    Variant("season x0.5, team x0.25", season_decay=0.5, team_decay=0.25),
    Variant("6 games, season x0.5, team x0.25", season_decay=0.5, team_decay=0.25, window=6),
)


def changed_team_keys(weekly: pd.DataFrame, window: int) -> set[tuple[str, int, int]]:
    """``(player, season, week)`` whose previous ``window`` games span another team."""
    name_col = "player_display_name" if "player_display_name" in weekly.columns else "player_name"
    team_col = "recent_team" if "recent_team" in weekly.columns else "team"
    keys: set[tuple[str, int, int]] = set()
    frame = weekly.dropna(subset=[name_col, "season", "week"])
    for player, played in frame.groupby(name_col, sort=False):
        played = played.sort_values(["season", "week"])
        teams = played[team_col].astype(str).tolist()
        seasons = played["season"].astype(int).tolist()
        weeks = played["week"].astype(int).tolist()
        for i in range(len(played)):
            history = teams[max(0, i - window):i]
            if history and any(team != teams[i] for team in history):
                keys.add((str(player), seasons[i], weeks[i]))
    return keys


def score_variant(
    weekly: pd.DataFrame,
    variant: Variant,
    *,
    test_from: int,
    changed: set[tuple[str, int, int]],
) -> dict:
    """Fit on seasons before ``test_from``, score on it and after."""
    with using_calibration(Calibration.blank()):
        frame = to_frame(observations(weekly, sport="nfl", **variant.kwargs()))
    train = frame[frame["season"] < test_from]
    test = frame[frame["season"] >= test_from].copy()

    fitted = calibrate.fit_sport(train, sport="nfl", seasons=sorted(train["season"].unique()))
    corrected = calibrate.probabilities_under(
        test, Calibration(sports={"nfl": fitted}), sport="nfl"
    )
    test["p_fit"] = corrected
    test["changed"] = [
        (player, season, week) in changed
        for player, season, week in zip(test["player"], test["season"], test["week"])
    ]

    def slice_score(mask) -> dict:
        part = test[mask]
        if part.empty:
            return {"n": 0, "brier": float("nan"), "log_loss": float("nan")}
        card = calibrate.score(part["p_fit"].to_numpy(), part["hit"].to_numpy())
        return {"n": card.samples, "brier": card.brier, "log_loss": card.log_loss}

    # Mean absolute error of the projection itself, before any probability is
    # involved: the plainest test of whether the weighting predicts better.
    resid = calibrate.residuals(test)
    mae = float(np.mean(np.abs(resid["projected"] - resid["actual"])))
    errors = resid.assign(error=(resid["projected"] - resid["actual"]).abs())
    return {
        "errors": errors.set_index(["player", "season", "week", "market"])["error"],
        "variant": variant.name,
        "all": slice_score(np.ones(len(test), dtype=bool)),
        "early": slice_score((test["week"] <= 4).to_numpy()),
        "changed": slice_score(test["changed"].to_numpy()),
        "mae": mae,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
    parser.add_argument("--test-from", type=int, default=2025)
    parser.add_argument(
        "--sweep", choices=("turnover", "window"), default="window",
        help="turnover: season/team decays; window: how many games to average",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    weekly, _ = load_nfl_frames(args.seasons)
    weekly = weekly[weekly.get("season_type", "REG") == "REG"] if "season_type" in weekly else weekly
    print(f"{len(weekly):,} player-weeks, {args.seasons}; scoring on {args.test_from}+")

    variants = WINDOW_VARIANTS if args.sweep == "window" else DEFAULT_VARIANTS
    # "Changed team" is judged over the longest window in the sweep, so every
    # variant is scored on the same players rather than on a slice that grows
    # with its own window.
    changed = changed_team_keys(weekly, max(v.window for v in variants))
    baseline_errors = None
    for variant in variants:
        started = time.perf_counter()
        result = score_variant(weekly, variant, test_from=args.test_from, changed=changed)
        paired = ""
        if baseline_errors is None:
            baseline_errors = result["errors"]
        else:
            # Paired on the same player-week-market, so the comparison is not
            # at the mercy of which weeks happened to be noisy.
            joined = pd.concat(
                [baseline_errors.rename("base"), result["errors"].rename("this")], axis=1
            ).dropna()
            diff = joined["this"] - joined["base"]
            se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
            paired = f"  vs current {diff.mean():+.3f} yds (se {se:.3f}, n={len(diff):,})"
        print(
            f"  {result['variant']:30} brier all {result['all']['brier']:.5f}  "
            f"early {result['early']['brier']:.5f}  changed {result['changed']['brier']:.5f}  "
            f"mae {result['mae']:.3f}{paired}  [{time.perf_counter() - started:.0f}s]",
            flush=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
