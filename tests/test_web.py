"""Web API tests. The app runs on the cached fixtures, so no network is used."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from config.settings import settings
from src.web.app import create_app
from src.web.service import market_display


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> TestClient:
    db_path = tmp_path_factory.mktemp("web") / "web.db"
    with TestClient(create_app(db_path=str(db_path), use_mock=True)) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def slate(client: TestClient) -> dict:
    response = client.get("/api/slate/nfl")
    assert response.status_code == 200
    return response.json()


@pytest.fixture(scope="module")
def edge_legs(slate: dict) -> list[dict]:
    legs = [leg for leg in slate["legs"] if leg["has_edge"]]
    assert len(legs) >= 4
    return legs


def price(client: TestClient, leg_ids, **kwargs) -> dict:
    payload = {"sport": "nfl", "leg_ids": list(leg_ids), "stake": 25.0, **kwargs}
    response = client.post("/api/price", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


# ------------------------------------------------------------------ config
def test_config_exposes_the_engine_thresholds(client: TestClient):
    body = client.get("/api/config").json()
    engine = body["engine"]
    assert body["mock"] is True
    assert engine["min_legs"] == settings.min_legs
    assert engine["max_legs"] == settings.max_legs
    assert engine["min_leg_ev"] == settings.min_leg_ev
    assert engine["min_sgp_correlation"] == settings.min_sgp_correlation
    assert engine["sports"] == ["nba", "ncaaf", "nfl"]


def test_market_names_are_human_readable():
    assert market_display("player_pass_yds") == "Passing Yards"
    assert market_display("h2h") == "Moneyline"
    assert market_display("player_threes") == "3-Pointers Made"


# ------------------------------------------------------------------- slate
def test_slate_returns_games_and_priced_legs(slate: dict):
    assert slate["meta"]["games"] == 3
    assert slate["meta"]["mock"] is True
    assert len(slate["games"]) == 3
    assert "@" in slate["games"][0]["label"]
    assert len(slate["legs"]) > 50

    leg = slate["legs"][0]
    assert {
        "leg_id", "game", "market_label", "selection", "american_odds",
        "odds_display", "p_model", "p_implied", "ev", "has_edge", "description",
    } <= set(leg)
    assert 0.0 <= leg["p_model"] <= 1.0
    assert leg["odds_display"].startswith(("+", "-"))


def test_legs_flagged_with_an_edge_match_the_engine_rules(slate: dict):
    for leg in slate["legs"]:
        expected = (
            leg["ev"] >= settings.min_leg_ev
            and settings.leg_odds_min <= leg["american_odds"] <= settings.leg_odds_max
        )
        assert leg["has_edge"] is expected


def test_unknown_sport_is_a_404(client: TestClient):
    assert client.get("/api/slate/cricket").status_code == 404


# ------------------------------------------------------------------ pricing
def test_pricing_is_internally_consistent(client: TestClient, edge_legs):
    ids = [leg["leg_id"] for leg in edge_legs[:2]]
    body = price(client, ids, stake=25.0)

    assert body["priceable"] is True
    assert body["leg_count"] == 2
    assert body["payout"] == pytest.approx(25.0 * body["decimal_odds"], abs=0.02)
    assert body["profit"] == pytest.approx(body["payout"] - 25.0, abs=0.02)
    assert body["ev_dollars"] == pytest.approx(body["ev_per_unit"] * 25.0, abs=0.02)
    assert body["edge"] == pytest.approx(body["p_model"] - body["p_implied"], abs=1e-3)
    assert body["p_implied"] == pytest.approx(1 / body["decimal_odds"], abs=1e-3)
    assert body["breakeven_probability"] == pytest.approx(body["p_implied"], abs=1e-3)


def test_payout_scales_linearly_with_the_stake(client: TestClient, edge_legs):
    ids = [leg["leg_id"] for leg in edge_legs[:2]]
    small = price(client, ids, stake=10.0)
    large = price(client, ids, stake=100.0)
    # Payouts are rounded to the cent, so ten times a rounded profit can miss
    # the exact product by up to a few cents.
    assert large["profit"] == pytest.approx(small["profit"] * 10, abs=0.05)
    assert large["decimal_odds"] == small["decimal_odds"]


def test_payout_table_covers_the_preset_stakes(client: TestClient, edge_legs):
    body = price(client, [leg["leg_id"] for leg in edge_legs[:2]])
    stakes = [row["stake"] for row in body["payout_table"]]
    assert stakes == [5, 10, 25, 50, 100]
    for row in body["payout_table"]:
        assert row["payout"] == pytest.approx(row["stake"] * body["decimal_odds"], abs=0.02)


def test_kelly_stake_is_a_quarter_of_full_kelly(client: TestClient, edge_legs):
    body = price(client, [leg["leg_id"] for leg in edge_legs[:2]], bankroll=2_000)
    assert body["bankroll"] == 2_000
    # kelly_share is rounded to 4dp for display, so allow a cent or two of drift
    assert body["recommended_stake"] == pytest.approx(
        body["kelly_share"] * 2_000, abs=0.2
    )
    assert 0 < body["kelly_share"] < settings.kelly_fraction


def test_single_leg_is_priced_without_the_copula(client: TestClient, edge_legs):
    body = price(client, [edge_legs[0]["leg_id"]])
    assert body["leg_count"] == 1
    assert body["iterations"] == 0
    assert body["p_model"] == pytest.approx(edge_legs[0]["p_model"], abs=1e-3)
    assert any(a["code"] == "too_few_legs" for a in body["advisories"])


def test_empty_slip_is_not_priceable(client: TestClient):
    body = price(client, [])
    assert body["priceable"] is False
    assert body["advisories"][0]["code"] == "too_few_legs"


def test_stale_leg_id_is_rejected_with_a_hint(client: TestClient):
    response = client.post(
        "/api/price", json={"sport": "nfl", "leg_ids": ["not-a-leg"], "stake": 10}
    )
    assert response.status_code == 409
    assert response.json()["code"] == "stale_leg"


# -------------------------------------------------------------- advisories
def test_same_game_negative_pair_is_flagged(client: TestClient, slate: dict):
    """Two legs that pull against each other must be called out, not hidden."""
    by_game: dict[str, list[dict]] = {}
    for leg in slate["legs"]:
        by_game.setdefault(leg["game_id"], []).append(leg)

    pair = None
    for legs in by_game.values():
        for first in legs:
            for second in legs:
                if first["leg_id"] == second["leg_id"]:
                    continue
                body = price(client, [first["leg_id"], second["leg_id"]])
                codes = {a["code"] for a in body["advisories"]}
                if "sgp_correlation" in codes:
                    pair = body
                    break
            if pair:
                break
        if pair:
            break

    assert pair is not None, "expected at least one weakly correlated same-game pair"
    advisory = next(a for a in pair["advisories"] if a["code"] == "sgp_correlation")
    assert advisory["level"] == "warn"
    assert str(settings.min_sgp_correlation) in advisory["message"]


def test_oversized_slip_is_flagged(client: TestClient, edge_legs):
    body = price(client, [leg["leg_id"] for leg in edge_legs[:5]])
    codes = {a["code"] for a in body["advisories"]}
    assert "too_many_legs" in codes
    assert body["priceable"] is True  # still priced, just advised against


def test_correlation_pairs_are_reported_for_every_pair(client: TestClient, edge_legs):
    body = price(client, [leg["leg_id"] for leg in edge_legs[:3]])
    assert len(body["correlation_pairs"]) == 3  # C(3, 2)
    for pair in body["correlation_pairs"]:
        assert -1.0 <= pair["correlation"] <= 1.0
        assert pair["a_label"] and pair["b_label"]


# ------------------------------------------------------------- auto build
def test_build_returns_tickets_that_resolve_against_the_slate(client, slate):
    response = client.post(
        "/api/build", json={"sport": "nfl", "max_tickets": 2, "seed": 11}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["report"]["solver_status"] == "Optimal"
    assert 1 <= len(body["tickets"]) <= 2

    known = {leg["leg_id"] for leg in slate["legs"]}
    for ticket in body["tickets"]:
        assert set(ticket["leg_ids"]) <= known
        assert settings.min_legs <= len(ticket["leg_ids"]) <= settings.max_legs
        assert settings.parlay_odds_min <= ticket["american_odds"] <= settings.parlay_odds_max
        assert ticket["ev"] > 0 and ticket["stake"] > 0


def test_build_honours_a_fixed_leg_count(client: TestClient):
    body = client.post(
        "/api/build", json={"sport": "nfl", "legs": 2, "max_tickets": 3, "seed": 11}
    ).json()
    assert body["tickets"]
    assert {len(ticket["leg_ids"]) for ticket in body["tickets"]} == {2}


def test_built_ticket_reprices_to_the_same_numbers(client: TestClient):
    ticket = client.post(
        "/api/build", json={"sport": "nfl", "max_tickets": 1, "seed": 11}
    ).json()["tickets"][0]
    body = price(client, ticket["leg_ids"], stake=ticket["stake"])
    assert body["american_odds"] == ticket["american_odds"]
    assert body["p_model"] == pytest.approx(ticket["p_model"], abs=0.03)


# ------------------------------------------------------------------ static
def test_ui_and_assets_are_served(client: TestClient):
    page = client.get("/")
    assert page.status_code == 200
    assert "Parlay Crafter" in page.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/styles.css").status_code == 200


def test_health_reports_cached_slates(client: TestClient, slate: dict):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["slates"]["nfl"]["games"] == 3


# ----------------------------------------------------------------------
# one week per slip
# ----------------------------------------------------------------------
def test_the_slate_reports_the_weeks_it_spans(client: TestClient):
    body = client.get("/api/slate/nfl").json()
    assert body["weeks"], "a slate with games always spans at least one week"
    keys = [week["key"] for week in body["weeks"]]
    assert keys == sorted(keys)
    assert {game["week"] for game in body["games"]} <= set(keys)
    assert {leg["week"] for leg in body["legs"]} <= set(keys)


def test_a_one_week_slip_is_not_flagged(client: TestClient, edge_legs):
    ids = [leg["leg_id"] for leg in edge_legs[:2]]
    body = price(client, ids)
    codes = {advisory["code"] for advisory in body["advisories"]}
    assert "mixed_weeks" not in codes, "the fixture slate is one week"


def test_review_flags_two_weeks_on_one_slip():
    from src.models.legs import Leg
    from src.web.service import review_slip

    def leg(week: str, name: str) -> Leg:
        return Leg(
            game_id=f"g-{week}", sport="nfl", market="player_rush_yds", selection="Over",
            american_odds=-110, line=60.5, player_name=name, team="KC",
            slate_week=week, p_model=0.55, p_implied=0.52, ev=0.05,
        )

    advisories = review_slip([leg("2026-09-15", "A"), leg("2026-09-22", "B")], [])
    mixed = [a for a in advisories if a.code == "mixed_weeks"]
    assert mixed and mixed[0].level == "block"
    assert "Sep 15" in mixed[0].message and "Sep 22" in mixed[0].message


def test_review_accepts_one_week():
    from src.models.legs import Leg
    from src.web.service import review_slip

    legs = [
        Leg(game_id="g1", sport="nfl", market="player_rush_yds", selection="Over",
            american_odds=-110, line=60.5, player_name="A", team="KC",
            slate_week="2026-09-15", p_model=0.55, p_implied=0.52, ev=0.05),
        Leg(game_id="g2", sport="nfl", market="player_reception_yds", selection="Over",
            american_odds=-110, line=64.5, player_name="B", team="BUF",
            slate_week="2026-09-15", p_model=0.55, p_implied=0.52, ev=0.05),
    ]
    assert not [a for a in review_slip(legs, []) if a.code == "mixed_weeks"]
