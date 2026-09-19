"""Joining stored prices to projections."""

from __future__ import annotations

import pytest

from src.models.baseline import GameProjection
from src.optimizer.leg_builder import legs_from_lines, legs_from_props
from src.models.legs import Projection

PROJECTION = GameProjection(
    game_id="g1", sport="nfl",
    home_team="KC", away_team="BUF",        # stat-feed abbreviations
    total_mean=47.0, total_sd=13.2,
    home_margin_mean=3.0, margin_sd=13.2,
)

# The odds feed names teams in full -- the mismatch this module has to bridge.
LINE_ROWS = [
    {"game_id": "g1", "market": "h2h", "selection": "Kansas City Chiefs",
     "line": None, "american_odds": -150},
    {"game_id": "g1", "market": "h2h", "selection": "Buffalo Bills",
     "line": None, "american_odds": 130},
    {"game_id": "g1", "market": "spreads", "selection": "Kansas City Chiefs",
     "line": -3.0, "american_odds": -110},
    {"game_id": "g1", "market": "spreads", "selection": "Buffalo Bills",
     "line": 3.0, "american_odds": -110},
    {"game_id": "g1", "market": "totals", "selection": "Over",
     "line": 47.5, "american_odds": -110},
    {"game_id": "g1", "market": "totals", "selection": "Under",
     "line": 47.5, "american_odds": -110},
]


def by_market(legs):
    out = {}
    for leg in legs:
        out.setdefault(leg.market, []).append(leg)
    return out


def test_club_names_resolve_to_the_projection_abbreviations():
    """Regression: full club names used to match nothing, silently dropping
    every moneyline and spread leg from the whole pipeline."""
    legs = legs_from_lines(LINE_ROWS, PROJECTION, sport="nfl")
    markets = by_market(legs)
    assert set(markets) == {"h2h", "spreads", "totals"}
    assert len(markets["h2h"]) == 2 and len(markets["spreads"]) == 2


def test_moneyline_probabilities_follow_the_projected_margin():
    markets = by_market(legs_from_lines(LINE_ROWS, PROJECTION, sport="nfl"))
    home = next(leg for leg in markets["h2h"] if leg.selection == "Kansas City Chiefs")
    away = next(leg for leg in markets["h2h"] if leg.selection == "Buffalo Bills")
    assert home.p_model > 0.5 > away.p_model        # home is projected +3
    assert home.p_model + away.p_model == pytest.approx(1.0, abs=1e-6)
    assert home.team == "KC" and away.team == "BUF"  # normalised for correlation


def test_spread_probabilities_are_near_even_at_the_projected_number():
    markets = by_market(legs_from_lines(LINE_ROWS, PROJECTION, sport="nfl"))
    home = next(leg for leg in markets["spreads"] if leg.selection == "Kansas City Chiefs")
    away = next(leg for leg in markets["spreads"] if leg.selection == "Buffalo Bills")
    # The line (-3) equals the projected margin, so both sides sit at a coin flip
    assert home.p_model == pytest.approx(0.5, abs=0.01)
    assert home.p_model + away.p_model == pytest.approx(1.0, abs=1e-6)


def test_totals_use_the_projected_total():
    markets = by_market(legs_from_lines(LINE_ROWS, PROJECTION, sport="nfl"))
    over = next(leg for leg in markets["totals"] if leg.selection == "Over")
    assert over.p_model < 0.5           # 47.5 line against a 47.0 projection
    assert over.team is None            # a total belongs to no team


def test_unknown_team_selection_is_skipped():
    rows = [{"game_id": "g1", "market": "h2h", "selection": "Toronto Huskies",
             "line": None, "american_odds": -150}]
    assert legs_from_lines(rows, PROJECTION, sport="nfl") == []


def test_no_projection_means_no_game_market_legs():
    assert legs_from_lines(LINE_ROWS, None, sport="nfl") == []


def test_props_need_a_matching_projection():
    rows = [
        {"game_id": "g1", "market": "player_pass_yds", "player_name": "QB One",
         "selection": "Over", "line": 274.5, "american_odds": -110},
        {"game_id": "g1", "market": "player_pass_yds", "player_name": "QB One",
         "selection": "Under", "line": 274.5, "american_odds": -110},
        {"game_id": "g1", "market": "player_pass_yds", "player_name": "Ghost",
         "selection": "Over", "line": 200.5, "american_odds": -110},
    ]
    projections = [Projection(sport="nfl", game_id="g1", player_name="QB One",
                              team="KC", market="player_pass_yds", mean=290.0)]
    legs = legs_from_props(rows, projections, sport="nfl")
    assert {leg.selection for leg in legs} == {"Over", "Under"}
    assert all(leg.player_name == "QB One" for leg in legs)   # Ghost has no projection
    over = next(leg for leg in legs if leg.selection == "Over")
    assert over.p_model > 0.5 and over.team == "KC"
    assert over.p_implied == pytest.approx(0.5, abs=0.01)     # de-vigged -110/-110
