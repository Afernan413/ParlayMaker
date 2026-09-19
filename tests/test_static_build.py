"""The static bundle must be complete, self-contained and internally consistent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config.settings import settings
from scripts import build_static


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("site")
    code = build_static.main(
        ["--out", str(out), "--mock", "--db", str(out / "build.db")]
    )
    assert code == 0
    raw = (out / "data.js").read_text()
    bundle = json.loads(raw[raw.index("{"): raw.rindex(";")])
    return out, bundle


def test_every_file_the_page_needs_is_written(built):
    out, _ = built
    for name in ("index.html", "app.js", "engine.js", "styles.css", "data.js", ".nojekyll"):
        assert (out / name).exists(), f"{name} missing"


def test_the_page_only_references_local_files(built):
    """No CDN, no API: it has to work from file:// and offline.

    An ``xmlns`` inside the inline favicon is a namespace identifier, not a
    resource load, so only real src/href targets are checked.
    """
    import re

    out, _ = built
    html = (out / "index.html").read_text()
    external = [
        match.group(0)
        for match in re.finditer(r'(?:src|href)="(https?:)?//[^"]+"', html)
    ]
    assert external == [], f"page loads something remote: {external}"
    for asset in ("data.js", "engine.js", "app.js", "styles.css"):
        assert asset in html


def test_data_is_javascript_not_json(built):
    """A page opened from file:// cannot fetch() JSON, but it can load a script."""
    out, _ = built
    assert (out / "data.js").read_text().lstrip().startswith("//")
    assert "window.PARLAY_DATA" in (out / "data.js").read_text()


def test_bundle_carries_both_sports(built):
    _, bundle = built
    assert set(bundle["sports"]) == {"nfl", "nba"}
    for sport in bundle["sports"].values():
        assert sport["games"] and sport["legs"]
        assert sport["mock"] is True


def test_settings_travel_with_the_bundle(built):
    _, bundle = built
    rules = bundle["settings"]
    assert rules["min_leg_ev"] == settings.min_leg_ev
    assert rules["min_sgp_correlation"] == settings.min_sgp_correlation
    assert rules["kelly_fraction"] == settings.kelly_fraction
    assert rules["min_legs"] == settings.min_legs and rules["max_legs"] == settings.max_legs


def test_every_leg_has_what_the_page_renders(built):
    _, bundle = built
    for sport in bundle["sports"].values():
        for index, leg in enumerate(sport["legs"]):
            assert leg["i"] == index          # correlations index into this list
            assert {"id", "game_id", "game", "market_label", "label", "selection",
                    "odds", "decimal", "p_model", "p_implied", "ev", "edge",
                    "description"} <= set(leg)
            assert 0.0 <= leg["p_model"] <= 1.0
            assert 0.0 <= leg["p_implied"] <= 1.0
            assert leg["decimal"] > 1.0


def test_game_and_player_markets_are_both_present(built):
    """Regression: game markets were silently dropped by a team-name mismatch."""
    markets = {leg["market"] for leg in built[1]["sports"]["nfl"]["legs"]}
    assert {"h2h", "spreads", "totals"} <= markets
    assert any(market.startswith("player_") for market in markets)


def test_correlations_are_sparse_valid_and_same_game(built):
    _, bundle = built
    for sport in bundle["sports"].values():
        legs = sport["legs"]
        seen = set()
        for first, second, rho in sport["correlations"]:
            assert 0 <= first < len(legs) and 0 <= second < len(legs)
            assert first != second
            assert -1.0 <= rho <= 1.0
            assert legs[first]["game_id"] == legs[second]["game_id"]  # cross-game is 0
            pair = (min(first, second), max(first, second))
            assert pair not in seen, "each pair should appear once"
            seen.add(pair)


def test_games_carry_model_predictions_and_the_market_line(built):
    for sport in built[1]["sports"].values():
        for game in sport["games"]:
            assert game["label"] and game["kickoff"]
            assert game["model_total"] > 0
            assert game["model_home_points"] + game["model_away_points"] == pytest.approx(
                game["model_total"], abs=0.2
            )
            assert game["market_total"] is not None
            assert game["market_spread"] is not None


def test_bundle_stays_small_enough_to_load_on_a_phone(built):
    out, _ = built
    size_kb = (out / "data.js").stat().st_size / 1024
    assert size_kb < 600, f"data.js grew to {size_kb:.0f} KB"


def test_live_build_without_a_key_falls_back_to_fixtures(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(build_static.settings, "odds_api_key", "")
    code = build_static.main(
        ["--out", str(tmp_path / "site"), "--live", "--sports", "nfl",
         "--db", str(tmp_path / "fallback.db")]
    )
    assert code == 0
    assert "building from cached fixtures" in capsys.readouterr().out
