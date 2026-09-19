#!/usr/bin/env python3
"""Regenerate the offline fixtures used by ``run_pipeline.py --mock``.

The fixtures mimic the real payload shapes (The Odds API events, OpenWeather
forecasts, ESPN injury feeds, nfl_data_py / nba_api frames) so mock runs
exercise the same parsers as live runs. Player and team names are fictional.

Prop lines are derived from the projections the pipeline itself computes, with a
deliberate offset on a subset of players so that a dry run has real edges to
find. Re-run this script if the projection recipe changes:

    python scripts/generate_mock_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import MOCK_DIR  # noqa: E402
from src.models.distributions import DistributionSpec  # noqa: E402
from src.optimizer.ev_calculator import decimal_to_american  # noqa: E402
from src.models.baseline import (  # noqa: E402
    build_nba_projections,
    build_nfl_projections,
    nfl_team_efficiency,
    project_nba_game,
    project_nfl_game,
)

RNG = np.random.default_rng(20260918)
KICKOFFS = ("2026-09-20T17:00:00Z", "2026-09-20T20:05:00Z", "2026-09-21T00:20:00Z")

# --------------------------------------------------------------------------
# NFL fixture definition
# --------------------------------------------------------------------------
NFL_GAMES = [
    {"game_id": "nfl-mock-1", "home": "KC", "away": "BUF",
     "home_name": "Kansas City Chiefs", "away_name": "Buffalo Bills"},
    {"game_id": "nfl-mock-2", "home": "DET", "away": "GB",
     "home_name": "Detroit Lions", "away_name": "Green Bay Packers"},
    {"game_id": "nfl-mock-3", "home": "SF", "away": "SEA",
     "home_name": "San Francisco 49ers", "away_name": "Seattle Seahawks"},
]

# EPA/play strength per team. Kept in a tight band so the derived spreads land
# in a realistic 2-7 point range rather than the blowouts a wide band produces.
NFL_TEAM_STRENGTH = {
    "KC": 0.055, "BUF": 0.040, "DET": 0.030, "GB": 0.005, "SF": 0.045, "SEA": -0.005
}

# position -> (per-game volume template)
NFL_TEMPLATES = {
    "QB": {"passing_yards": 262.0, "passing_tds": 1.8, "attempts": 33.0, "rushing_yards": 14.0},
    "RB": {"rushing_yards": 74.0, "carries": 17.0, "receiving_yards": 15.0, "receptions": 2.4,
           "targets": 3.1},
    "WR1": {"receiving_yards": 79.0, "receptions": 5.6, "targets": 8.7, "target_share": 0.26},
    "WR2": {"receiving_yards": 47.0, "receptions": 3.7, "targets": 6.0, "target_share": 0.17},
    "TE": {"receiving_yards": 43.0, "receptions": 4.1, "targets": 5.4, "target_share": 0.16},
}

NFL_NAMES: dict[str, list[str]] = {
    "KC": ["Marcus Vale", "Deon Rush", "Trey Alder", "Cole Pryor", "Hank Osei"],
    "BUF": ["Ellis Ward", "Jamal Pike", "Reece Colton", "Nate Brody", "Owen Falk"],
    "DET": ["Silas Moreno", "Kip Dawson", "Andre Bellamy", "Zane Hollis", "Miles Corbin"],
    "GB": ["Luca Reyes", "Beau Tanner", "Xavier Lund", "Dorian Slate", "Rhys Calder"],
    "SF": ["Emil Navarro", "Tobias Crane", "Jonas Petrie", "Kai Ellington", "Rory Beckett"],
    "SEA": ["Aiden Cross", "Malik Sorenson", "Griffin Poole", "Devon Marsh", "Sol Ramirez"],
}

NFL_POSITIONS = list(NFL_TEMPLATES)

NFL_PROP_MARKETS = {
    "QB": [("player_pass_yds", "passing_yards"), ("player_pass_tds", "passing_tds")],
    "RB": [("player_rush_yds", "rushing_yards")],
    "WR1": [("player_reception_yds", "receiving_yards"), ("player_receptions", "receptions")],
    "WR2": [("player_reception_yds", "receiving_yards")],
    "TE": [("player_receptions", "receptions"), ("player_reception_yds", "receiving_yards")],
}

# --------------------------------------------------------------------------
# NBA fixture definition
# --------------------------------------------------------------------------
NBA_GAMES = [
    {"game_id": "nba-mock-1", "home": "BOS", "away": "LAL",
     "home_name": "Boston Celtics", "away_name": "Los Angeles Lakers"},
    {"game_id": "nba-mock-2", "home": "DEN", "away": "PHX",
     "home_name": "Denver Nuggets", "away_name": "Phoenix Suns"},
    {"game_id": "nba-mock-3", "home": "MIL", "away": "MIA",
     "home_name": "Milwaukee Bucks", "away_name": "Miami Heat"},
]

NBA_TEAM_CONTEXT = {
    "BOS": {"PACE": 100.8, "OFF_RATING": 120.1, "DEF_RATING": 110.4},
    "LAL": {"PACE": 99.1, "OFF_RATING": 114.6, "DEF_RATING": 114.2},
    "DEN": {"PACE": 98.4, "OFF_RATING": 118.7, "DEF_RATING": 112.9},
    "PHX": {"PACE": 100.2, "OFF_RATING": 116.3, "DEF_RATING": 115.1},
    "MIL": {"PACE": 101.6, "OFF_RATING": 117.4, "DEF_RATING": 113.3},
    "MIA": {"PACE": 96.9, "OFF_RATING": 112.8, "DEF_RATING": 111.6},
}

NBA_NAMES: dict[str, list[str]] = {
    "BOS": ["Dexter Ames", "Julian Poe", "Marco Vidal", "Eli Sandoval"],
    "LAL": ["Terrell Boone", "Ivan Keller", "Samir Haddad", "Nico Bassett"],
    "DEN": ["Aleks Novak", "Brant Willow", "Omar Farrah", "Casey Lindell"],
    "PHX": ["Jules Marchand", "Ty Ferrara", "Roman Duke", "Ike Salter"],
    "MIL": ["Gio Tavares", "Wes Lambert", "Ronan Fitch", "Dario Mensah"],
    "MIA": ["Curtis Vance", "Avi Stern", "Lonnie Grant", "Pierce Nolan"],
}

NBA_ROLES = [
    {"MIN": 35.0, "PTS": 27.5, "REB": 5.2, "AST": 6.1, "FG3M": 3.3, "USG_PCT": 0.31},
    {"MIN": 33.0, "PTS": 21.0, "REB": 7.4, "AST": 3.2, "FG3M": 2.1, "USG_PCT": 0.25},
    {"MIN": 30.0, "PTS": 15.4, "REB": 9.1, "AST": 2.4, "FG3M": 0.8, "USG_PCT": 0.21},
    {"MIN": 27.0, "PTS": 12.2, "REB": 3.6, "AST": 4.8, "FG3M": 1.9, "USG_PCT": 0.18},
]

NBA_PROP_MARKETS = [
    ("player_points", "PTS"),
    ("player_rebounds", "REB"),
    ("player_assists", "AST"),
    ("player_threes", "FG3M"),
]

# Players ruled out to give the reasoning layer something to reallocate.
NFL_INACTIVES = [("BUF", 2, "OUT", "hamstring"), ("GB", 3, "QUESTIONABLE", "ankle")]
NBA_INACTIVES = [("LAL", 1, "OUT", "left knee soreness"), ("MIA", 3, "QUESTIONABLE", "rest")]


def half_line(value: float) -> float:
    """Snap to a .5 line, the way FanDuel posts props and totals (no push)."""
    import math

    return math.floor(value) + 0.5


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=1)
    print(f"wrote {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------
# NFL
# --------------------------------------------------------------------------
def nfl_roster() -> list[dict[str, Any]]:
    roster = []
    for team, names in NFL_NAMES.items():
        for position, name in zip(NFL_POSITIONS, names):
            roster.append({"team": team, "position": position, "player_name": name})
    return roster


def nfl_weekly(roster: list[dict[str, Any]]) -> pd.DataFrame:
    stat_columns = [
        "passing_yards", "passing_tds", "attempts", "rushing_yards", "carries",
        "receiving_yards", "receptions", "targets", "target_share",
    ]
    rows = []
    for player in roster:
        template = NFL_TEMPLATES[player["position"]]
        for week in (1, 2, 3, 4):
            row = {
                "player_display_name": player["player_name"],
                "recent_team": player["team"],
                "position": player["position"],
                "week": week,
            }
            for column in stat_columns:
                base = template.get(column, 0.0)
                noise = RNG.normal(1.0, 0.10) if base else 0.0
                row[column] = round(max(base * noise, 0.0), 2)
            rows.append(row)
    return pd.DataFrame(rows)


def nfl_pbp() -> pd.DataFrame:
    rows = []
    for game in NFL_GAMES:
        for offence, defence in ((game["home"], game["away"]), (game["away"], game["home"])):
            strength = NFL_TEAM_STRENGTH[offence] - NFL_TEAM_STRENGTH[defence] / 2.0
            for index in range(64):
                play_type = "pass" if index % 5 < 3 else "run"
                epa = float(RNG.normal(strength, 0.45))
                rows.append({
                    "posteam": offence, "defteam": defence, "play_type": play_type,
                    "epa": round(epa, 4), "success": float(epa > 0),
                })
    return pd.DataFrame(rows)


def game_market_payload(
    game: dict[str, Any], projection: Any, bias: float
) -> list[dict[str, Any]]:
    """Moneyline / spread / total priced from the model view plus ``bias``."""
    margin = DistributionSpec(
        family="normal", mean=projection.home_margin_mean, dispersion=projection.margin_sd
    )
    total = DistributionSpec(
        family="normal", mean=projection.total_mean, dispersion=projection.total_sd
    )
    total_line = half_line(projection.total_mean)
    spread_line = round(-projection.home_margin_mean * 2) / 2

    p_home_win = margin.probability(0.0).prob_over
    p_home_cover = margin.probability(-spread_line).prob_over
    p_over = total.probability(total_line).prob_over

    home_ml, away_ml = two_way_prices(p_home_win, bias)
    home_spread, away_spread = two_way_prices(p_home_cover, 0.0)
    over_price, under_price = two_way_prices(p_over, bias)
    return [
        {"key": "h2h", "outcomes": [
            {"name": game["home_name"], "price": home_ml},
            {"name": game["away_name"], "price": away_ml},
        ]},
        {"key": "spreads", "outcomes": [
            {"name": game["home_name"], "price": home_spread, "point": spread_line},
            {"name": game["away_name"], "price": away_spread, "point": -spread_line},
        ]},
        {"key": "totals", "outcomes": [
            {"name": "Over", "price": over_price, "point": total_line},
            {"name": "Under", "price": under_price, "point": total_line},
        ]},
    ]


def nfl_odds_events(game_projections: dict[str, Any]) -> list[dict[str, Any]]:
    events = []
    for index, (game, kickoff) in enumerate(zip(NFL_GAMES, KICKOFFS)):
        projection = game_projections[game["game_id"]]
        events.append({
            "id": game["game_id"],
            "sport_key": "americanfootball_nfl",
            "commence_time": kickoff,
            "home_team": game["home_name"],
            "away_team": game["away_name"],
            "bookmakers": [{
                "key": "fanduel",
                "title": "FanDuel",
                "markets": game_market_payload(
                    game, projection, PRICE_BIASES[index % len(PRICE_BIASES)]
                ),
            }],
        })
    return events


#: Book margin applied to each side of a two-way market (4.5% overround).
VIG_PER_SIDE = 1.045

#: The book's probability error, cycled across markets: +0.05 means the book
#: underrates the first side by 5 points (a beatable Over), 0.0 means it is
#: priced correctly and nothing should be bet. Prices are then derived from
#: those probabilities, so the engine has to de-vig to find the edge -- which is
#: exactly the path a live run takes.
PRICE_BIASES = (0.05, 0.0, -0.05)


def american_from_probability(probability: float) -> int:
    """Price a probability, rounded the way a book quotes it."""
    probability = float(min(max(probability, 0.02), 0.98))
    return decimal_to_american(1.0 / probability)


def two_way_prices(p_first: float, bias: float) -> tuple[int, int]:
    """Prices for a two-way market whose first side is misvalued by ``bias``."""
    book_first = float(min(max(p_first - bias, 0.10), 0.90))
    return (
        american_from_probability(book_first * VIG_PER_SIDE),
        american_from_probability((1.0 - book_first) * VIG_PER_SIDE),
    )


def calibrated_line(spec: DistributionSpec, target_over: float) -> float:
    """Half-point line whose model over-probability is closest to ``target_over``.

    Snapping a quantile to the nearest .5 is fine for yardage, but for a
    low-mean count (passing TDs, receptions) half a unit moves the probability
    enormously -- so those are searched directly on the probability scale.
    """
    if spec.discrete:
        candidates = [k + 0.5 for k in range(0, int(spec.mean * 3) + 4)]
        return min(
            candidates,
            key=lambda line: abs(spec.probability(line).prob_over - target_over),
        )
    return max(half_line(float(spec._frozen().ppf(1.0 - target_over))), 0.5)


def prop_outcomes(
    player: str, projected_mean: float, market: str, bias: float
) -> list[dict[str, Any]]:
    """A two-sided prop posted near a coin flip and priced with ``bias``."""
    spec = DistributionSpec.for_market(market, projected_mean)
    line = calibrated_line(spec, 0.5)
    p_over = spec.probability(line).prob_over_no_push
    over_price, under_price = two_way_prices(p_over, bias)
    return [
        {"name": "Over", "description": player, "price": over_price, "point": line},
        {"name": "Under", "description": player, "price": under_price, "point": line},
    ]


def nfl_props(projections, roster) -> dict[str, Any]:
    position_by_name = {p["player_name"]: p["position"] for p in roster}
    by_game: dict[str, dict[str, list]] = {}
    index = {(p.game_id, p.player_name, p.market): p for p in projections}

    for order, (key, projection) in enumerate(index.items()):
        game_id, player, market = key
        position = position_by_name.get(player)
        if position is None:
            continue
        wanted = {m for m, _ in NFL_PROP_MARKETS[position]}
        if market not in wanted:
            continue
        markets = by_game.setdefault(game_id, {})
        markets.setdefault(market, []).extend(
            prop_outcomes(
                player, projection.mean, market,
                PRICE_BIASES[order % len(PRICE_BIASES)],
            )
        )

    payloads = {}
    for game, kickoff in zip(NFL_GAMES, KICKOFFS):
        markets = by_game.get(game["game_id"], {})
        payloads[game["game_id"]] = {
            "id": game["game_id"],
            "commence_time": kickoff,
            "home_team": game["home_name"],
            "away_team": game["away_name"],
            "bookmakers": [{
                "key": "fanduel",
                "title": "FanDuel",
                "markets": [
                    {"key": market, "outcomes": outcomes}
                    for market, outcomes in sorted(markets.items())
                ],
            }],
        }
    return payloads


def nfl_weather() -> dict[str, Any]:
    """OpenWeather-shaped forecasts: game 1 is windy and cold, game 3 is mild."""
    profiles = {
        "nfl-mock-1": {"temp": 29.0, "wind": 22.0, "deg": 310, "pop": 0.35,
                       "conditions": "light snow"},
        "nfl-mock-3": {"temp": 63.0, "wind": 7.0, "deg": 220, "pop": 0.05,
                       "conditions": "clear sky"},
    }
    payloads = {}
    for game, kickoff in zip(NFL_GAMES, KICKOFFS):
        profile = profiles.get(game["game_id"])
        if profile is None:
            continue  # dome game: the pipeline short-circuits it
        slot_time = kickoff.replace("T", " ").replace("Z", "")
        payloads[game["game_id"]] = {
            "city": {"name": game["home"]},
            "list": [
                {
                    "dt_txt": slot_time,
                    "main": {"temp": profile["temp"], "feels_like": profile["temp"] - 6},
                    "wind": {"speed": profile["wind"], "deg": profile["deg"]},
                    "pop": profile["pop"],
                    "weather": [{"description": profile["conditions"]}],
                }
            ],
        }
    return payloads


def injury_payload(
    inactives, names: dict[str, list[str]], positions: list[str]
) -> dict[str, Any]:
    blocks: dict[str, list] = {}
    for team, index, status, detail in inactives:
        blocks.setdefault(team, []).append({
            "status": status,
            "date": "2026-09-19",
            "shortComment": detail,
            "athlete": {
                "displayName": names[team][index],
                "position": {"abbreviation": positions[index % len(positions)]},
            },
        })
    return {
        "injuries": [
            {"displayName": team, "abbreviation": team, "injuries": entries}
            for team, entries in blocks.items()
        ]
    }


# --------------------------------------------------------------------------
# NBA
# --------------------------------------------------------------------------
def nba_player_frame() -> pd.DataFrame:
    rows = []
    for team, names in NBA_NAMES.items():
        for role, name in zip(NBA_ROLES, names):
            row = {"PLAYER_NAME": name, "TEAM_ABBREVIATION": team}
            for stat, value in role.items():
                jitter = RNG.normal(1.0, 0.06)
                row[stat] = round(value * jitter, 3) if stat != "MIN" else value
            rows.append(row)
    return pd.DataFrame(rows)


def nba_team_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"TEAM_ABBREVIATION": team, **context}
        for team, context in NBA_TEAM_CONTEXT.items()
    ])


def nba_odds_events(game_projections) -> list[dict[str, Any]]:
    events = []
    for index, (game, kickoff) in enumerate(zip(NBA_GAMES, KICKOFFS)):
        projection = game_projections[game["game_id"]]
        events.append({
            "id": game["game_id"],
            "sport_key": "basketball_nba",
            "commence_time": kickoff,
            "home_team": game["home_name"],
            "away_team": game["away_name"],
            "bookmakers": [{
                "key": "fanduel",
                "title": "FanDuel",
                "markets": game_market_payload(
                    game, projection, PRICE_BIASES[index % len(PRICE_BIASES)]
                ),
            }],
        })
    return events


def nba_props(projections) -> dict[str, Any]:
    by_game: dict[str, dict[str, list]] = {}
    wanted = {market for market, _ in NBA_PROP_MARKETS}
    for order, projection in enumerate(projections):
        if projection.market not in wanted:
            continue
        markets = by_game.setdefault(projection.game_id, {})
        markets.setdefault(projection.market, []).extend(
            prop_outcomes(
                projection.player_name, projection.mean, projection.market,
                PRICE_BIASES[order % len(PRICE_BIASES)],
            )
        )

    payloads = {}
    for game, kickoff in zip(NBA_GAMES, KICKOFFS):
        markets = by_game.get(game["game_id"], {})
        payloads[game["game_id"]] = {
            "id": game["game_id"],
            "commence_time": kickoff,
            "home_team": game["home_name"],
            "away_team": game["away_name"],
            "bookmakers": [{
                "key": "fanduel",
                "title": "FanDuel",
                "markets": [
                    {"key": market, "outcomes": outcomes}
                    for market, outcomes in sorted(markets.items())
                ],
            }],
        }
    return payloads


# --------------------------------------------------------------------------
def main() -> None:
    nfl_dir = MOCK_DIR / "nfl"
    nba_dir = MOCK_DIR / "nba"

    # --- NFL -------------------------------------------------------------
    roster = nfl_roster()
    weekly = nfl_weekly(roster)
    pbp = nfl_pbp()
    efficiency = nfl_team_efficiency(pbp)
    games = [
        {"game_id": g["game_id"], "home_team": g["home"], "away_team": g["away"]}
        for g in NFL_GAMES
    ]
    game_projections = {
        g["game_id"]: project_nfl_game(g, efficiency) for g in games
    }
    projections = build_nfl_projections(weekly, pbp, games)

    _write(nfl_dir / "weekly.json", weekly.to_dict(orient="records"))
    _write(nfl_dir / "pbp.json", pbp.to_dict(orient="records"))
    _write(nfl_dir / "odds.json", nfl_odds_events(game_projections))
    _write(nfl_dir / "props.json", nfl_props(projections, roster))
    _write(nfl_dir / "weather.json", nfl_weather())
    _write(
        nfl_dir / "injuries.json",
        injury_payload(NFL_INACTIVES, NFL_NAMES, ["QB", "RB", "WR", "WR", "TE"]),
    )

    # --- NBA -------------------------------------------------------------
    players = nba_player_frame()
    teams = nba_team_frame()
    nba_games = [
        {"game_id": g["game_id"], "home_team": g["home"], "away_team": g["away"]}
        for g in NBA_GAMES
    ]
    nba_game_projections = {
        g["game_id"]: project_nba_game(g, teams) for g in nba_games
    }
    nba_projections = build_nba_projections(players, teams, nba_games)

    _write(nba_dir / "players.json", players.to_dict(orient="records"))
    _write(nba_dir / "teams.json", teams.to_dict(orient="records"))
    _write(nba_dir / "odds.json", nba_odds_events(nba_game_projections))
    _write(nba_dir / "props.json", nba_props(nba_projections))
    _write(
        nba_dir / "injuries.json",
        injury_payload(NBA_INACTIVES, NBA_NAMES, ["G", "G", "F", "C"]),
    )
    print(
        f"\nNFL: {len(projections)} projections across {len(games)} games; "
        f"NBA: {len(nba_projections)} projections across {len(nba_games)} games"
    )


if __name__ == "__main__":
    main()
