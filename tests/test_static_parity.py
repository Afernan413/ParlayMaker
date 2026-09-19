"""The browser engine must agree with the Python engine.

The page re-implements the numerics (odds maths and the copula draw) so it can
run without a server. That is a second implementation of money maths, so it is
pinned to the first one here rather than trusted.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.models.correlation import GaussianCopulaSimulator, pairwise_correlation
from src.models.legs import Leg
from src.optimizer.ev_calculator import (
    american_to_decimal,
    decimal_to_american,
    expected_value_decimal,
    kelly_fraction_of_bankroll,
    parlay_decimal_odds,
)

ENGINE = Path(__file__).resolve().parent.parent / "src" / "web" / "static_site" / "engine.js"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

#: Monte-Carlo tolerance. Both sides draw 40k antithetic samples, so the
#: standard error on a joint probability is well under half a point.
MC_TOLERANCE = 0.012
ITERATIONS = 40_000


def run_node(script: str, payload: dict) -> dict:
    """Execute a snippet against engine.js and return its JSON result."""
    program = (
        f"const engine = require({str(ENGINE)!r});\n"
        f"const input = {json.dumps(payload)};\n"
        f"{script}\n"
    )
    completed = subprocess.run(
        [NODE, "-e", program], capture_output=True, text=True, timeout=120
    )
    if completed.returncode != 0:
        raise AssertionError(f"node failed:\n{completed.stderr}")
    return json.loads(completed.stdout)


def leg(**kwargs) -> Leg:
    base = dict(
        game_id="g1", sport="nfl", market="player_pass_yds", selection="Over",
        american_odds=-110, line=274.5, p_model=0.58,
    )
    base.update(kwargs)
    return Leg(**base)


def as_payload(legs) -> tuple[list[dict], list[list]]:
    """Legs and their pairwise correlations, in the shape the page receives."""
    rows = [
        {
            "i": index,
            "id": item.leg_id,
            "game_id": item.game_id,
            "subject": item.subject,
            "market": item.market,
            "odds": item.american_odds,
            "p_model": item.p_model,
            "ev": item.ev,
            "description": item.describe(),
        }
        for index, item in enumerate(legs)
    ]
    pairs = []
    for i, first in enumerate(legs):
        for j, second in enumerate(legs[i + 1:], start=i + 1):
            rho = pairwise_correlation(first, second)
            if abs(rho) > 1e-9:
                pairs.append([i, j, round(rho, 4)])
    return rows, pairs


def joint_in_js(legs, *, seed: int = 7, iterations: int = ITERATIONS) -> float:
    rows, pairs = as_payload(legs)
    result = run_node(
        """
        const pairs = new Map();
        for (const [a, b, r] of input.pairs) { pairs.set(a * 1e5 + b, r); pairs.set(b * 1e5 + a, r); }
        const lookup = (x, y) => pairs.get(x.i * 1e5 + y.i) ?? 0;
        const out = engine.jointProbability(input.legs, lookup,
            { iterations: input.iterations, seed: input.seed });
        console.log(JSON.stringify(out));
        """,
        {"legs": rows, "pairs": pairs, "iterations": iterations, "seed": seed},
    )
    return result["probability"]


def joint_in_python(legs, *, iterations: int = ITERATIONS) -> float:
    return GaussianCopulaSimulator(legs, iterations=iterations, seed=7).simulate().joint_probability


# ------------------------------------------------------------ pure maths
@pytest.mark.parametrize("american", [-110, -140, 100, 130, -250, 375])
def test_odds_conversion_matches_exactly(american):
    result = run_node(
        "console.log(JSON.stringify({decimal: engine.americanToDecimal(input.odds)}));",
        {"odds": american},
    )
    assert result["decimal"] == pytest.approx(american_to_decimal(american), abs=1e-12)


@pytest.mark.parametrize("decimal", [1.5, 1.909090909, 2.0, 3.6446, 7.5])
def test_decimal_to_american_matches_exactly(decimal):
    result = run_node(
        "console.log(JSON.stringify({odds: engine.decimalToAmerican(input.decimal)}));",
        {"decimal": decimal},
    )
    assert result["odds"] == decimal_to_american(decimal)


def test_parlay_price_and_ev_match():
    legs = [leg(player_name="A", team="KC"), leg(player_name="B", team="KC", american_odds=-105)]
    rows, _ = as_payload(legs)
    result = run_node(
        """
        const decimal = engine.parlayDecimal(input.legs);
        console.log(JSON.stringify({
            decimal,
            american: engine.decimalToAmerican(decimal),
            ev: engine.expectedValue(input.p, decimal),
            kelly: engine.kellyShare(input.p, decimal, 0.25),
        }));
        """,
        {"legs": rows, "p": 0.34},
    )
    decimal = parlay_decimal_odds(item.american_odds for item in legs)
    assert result["decimal"] == pytest.approx(decimal, abs=1e-12)
    assert result["american"] == decimal_to_american(decimal)
    assert result["ev"] == pytest.approx(expected_value_decimal(0.34, decimal), abs=1e-12)
    assert result["kelly"] == pytest.approx(kelly_fraction_of_bankroll(0.34, decimal, 0.25), abs=1e-12)


def test_normal_quantile_matches_scipy():
    from scipy import stats

    probabilities = [0.01, 0.1, 0.25, 0.5, 0.58, 0.75, 0.9, 0.99]
    result = run_node(
        "console.log(JSON.stringify(input.ps.map(engine.normalQuantile)));",
        {"ps": probabilities},
    )
    for value, probability in zip(result, probabilities):
        assert value == pytest.approx(float(stats.norm.ppf(probability)), abs=1e-6)


# ---------------------------------------------------------------- copula
def test_independent_legs_agree():
    legs = [
        leg(player_name="A", team="KC", p_model=0.58),
        leg(game_id="g2", sport="nba", market="player_points", player_name="B",
            team="BOS", line=25.5, american_odds=-115, p_model=0.60),
    ]
    js = joint_in_js(legs)
    assert js == pytest.approx(joint_in_python(legs), abs=MC_TOLERANCE)
    assert js == pytest.approx(0.58 * 0.60, abs=MC_TOLERANCE)


def test_correlated_same_game_pair_agrees():
    legs = [
        leg(player_name="QB One", team="KC", p_model=0.58),
        leg(player_name="WR One", team="KC", market="player_reception_yds",
            line=64.5, american_odds=-105, p_model=0.61),
    ]
    js, python = joint_in_js(legs), joint_in_python(legs)
    assert js == pytest.approx(python, abs=MC_TOLERANCE)
    assert js > 0.58 * 0.61          # positive correlation lifts the joint
    assert js < 0.58                 # but never beats the weakest leg


def test_negatively_correlated_pair_agrees():
    legs = [
        leg(market="totals", selection="Under", line=47.5, player_name=None,
            team=None, p_model=0.56),
        leg(market="player_pass_tds", selection="Over", line=2.5,
            player_name="QB One", team="KC", american_odds=120, p_model=0.45),
    ]
    js = joint_in_js(legs)
    assert js == pytest.approx(joint_in_python(legs), abs=MC_TOLERANCE)
    assert js < 0.56 * 0.45          # pulling against each other


def test_three_and_four_leg_tickets_agree():
    legs = [
        leg(player_name="QB One", team="KC", p_model=0.58),
        leg(player_name="WR One", team="KC", market="player_reception_yds",
            line=64.5, american_odds=-105, p_model=0.60),
        leg(game_id="g2", sport="nba", market="player_points", player_name="G A",
            team="BOS", line=25.5, american_odds=-115, p_model=0.57),
        leg(game_id="g2", sport="nba", market="player_assists", player_name="G B",
            team="BOS", line=5.5, american_odds=110, p_model=0.55),
    ]
    for size in (3, 4):
        subset = legs[:size]
        assert joint_in_js(subset) == pytest.approx(
            joint_in_python(subset), abs=MC_TOLERANCE
        ), f"{size}-leg ticket diverged"


def test_marginals_are_reproduced():
    legs = [
        leg(player_name="QB One", team="KC", p_model=0.58),
        leg(player_name="WR One", team="KC", market="player_reception_yds",
            line=64.5, american_odds=-105, p_model=0.61),
    ]
    rows, pairs = as_payload(legs)
    result = run_node(
        """
        const pairs = new Map();
        for (const [a, b, r] of input.pairs) { pairs.set(a * 1e5 + b, r); pairs.set(b * 1e5 + a, r); }
        const out = engine.jointProbability(input.legs, (x, y) => pairs.get(x.i * 1e5 + y.i) ?? 0,
            { iterations: 40000, seed: 7 });
        console.log(JSON.stringify(out));
        """,
        {"legs": rows, "pairs": pairs},
    )
    assert result["marginals"][0] == pytest.approx(0.58, abs=0.01)
    assert result["marginals"][1] == pytest.approx(0.61, abs=0.01)


def test_pricing_is_deterministic():
    legs = [leg(player_name="A", team="KC"), leg(player_name="B", team="KC")]
    assert joint_in_js(legs, iterations=5_000) == joint_in_js(legs, iterations=5_000)
