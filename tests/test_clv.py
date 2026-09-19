"""Closing line value benchmarking tests."""

from __future__ import annotations

import pytest

from src.ingestion import db
from src.models.legs import Leg
from src.optimizer import clv
from src.optimizer.parlay_builder import price_ticket


def seed_game(db_path) -> None:
    db.upsert_games(
        [{"game_id": "g1", "sport": "nfl", "commence_time": "2026-09-20T17:00:00Z",
          "home_team": "Kansas City Chiefs", "away_team": "Buffalo Bills"}],
        db_path=db_path,
    )


def capture_prop(db_path, *, over: int, under: int, at: str) -> None:
    db.insert_props(
        [
            {"game_id": "g1", "market": "player_reception_yds",
             "player_name": "WR One", "selection": "Over", "line": 64.5,
             "american_odds": over, "captured_at": at},
            {"game_id": "g1", "market": "player_reception_yds",
             "player_name": "WR One", "selection": "Under", "line": 64.5,
             "american_odds": under, "captured_at": at},
        ],
        db_path=db_path,
    )


BET_TIME = "2026-09-20T12:00:00+00:00"
OPEN_TIME = "2026-09-20T10:00:00+00:00"
CLOSE_TIME = "2026-09-20T16:30:00+00:00"


def log(ticket, db_path):
    return clv.log_recommendations(
        [ticket], sport="nfl", captured_at=BET_TIME, db_path=db_path
    )


def make_ticket(taken_odds: int = 120):
    wr = Leg(game_id="g1", sport="nfl", market="player_reception_yds",
             selection="Over", american_odds=taken_odds, line=64.5,
             player_name="WR One", team="KC", p_model=0.55, p_implied=0.46)
    qb = Leg(game_id="g1", sport="nfl", market="player_pass_yds", selection="Over",
             american_odds=-110, line=274.5, player_name="QB One", team="KC",
             p_model=0.58, p_implied=0.52)
    return price_ticket([wr, qb], iterations=2_000, seed=4, bankroll=1_000)


def test_recommendations_are_logged_with_the_taken_price(db_path):
    seed_game(db_path)
    ticket = make_ticket()
    run_id = log(ticket, db_path)

    rows = db.fetch_all("SELECT * FROM bet_log ORDER BY id", db_path=db_path)
    assert len(rows) == 2
    assert {row["run_id"] for row in rows} == {run_id}
    assert rows[0]["ticket_type"] == ticket.ticket_type
    assert rows[0]["ticket_odds"] == ticket.american_odds
    assert rows[0]["taken_odds"] == 120 and rows[0]["stake"] == ticket.stake


def test_no_later_capture_means_no_closing_comparison(db_path):
    seed_game(db_path)
    capture_prop(db_path, over=120, under=-140, at=OPEN_TIME)
    log(make_ticket(), db_path)

    entries, summary = clv.report(sport="nfl", db_path=db_path)
    assert entries == []
    assert summary.logged == 2 and summary.priced == 0
    assert "no logged bet has a later capture" in clv.render(entries, summary)


def test_positive_clv_when_the_price_shortens(db_path):
    seed_game(db_path)
    capture_prop(db_path, over=120, under=-140, at=OPEN_TIME)
    log(make_ticket(120), db_path)
    # the market moves toward our side: +120 closes at -115
    capture_prop(db_path, over=-115, under=-105, at=CLOSE_TIME)

    entries, summary = clv.report(sport="nfl", db_path=db_path)
    assert len(entries) == 1  # only the prop has a later capture
    entry = entries[0]
    assert entry.taken_odds == 120 and entry.closing_odds == -115
    assert entry.closing_fair > entry.taken_fair
    assert entry.clv > 0 and entry.beat_close
    assert summary.beat_rate == 1.0
    assert summary.non_negative_drift is True
    assert "PASS" in clv.render(entries, summary)


def test_negative_clv_when_the_price_drifts_out(db_path):
    seed_game(db_path)
    capture_prop(db_path, over=-115, under=-105, at=OPEN_TIME)
    log(make_ticket(-115), db_path)
    capture_prop(db_path, over=135, under=-155, at=CLOSE_TIME)

    entries, summary = clv.report(sport="nfl", db_path=db_path)
    assert entries[0].clv < 0 and not entries[0].beat_close
    assert summary.non_negative_drift is False
    assert "FAIL" in clv.render(entries, summary)


def test_summary_reports_the_model_edge(db_path):
    seed_game(db_path)
    capture_prop(db_path, over=120, under=-140, at=OPEN_TIME)
    log(make_ticket(120), db_path)
    capture_prop(db_path, over=-115, under=-105, at=CLOSE_TIME)

    _, summary = clv.report(sport="nfl", db_path=db_path)
    assert summary.mean_model_edge == pytest.approx(0.55 - 0.46, abs=1e-6)


def test_empty_bet_log_summarises_cleanly(db_path):
    entries, summary = clv.report(sport="nfl", db_path=db_path)
    assert entries == [] and summary.logged == 0
    assert summary.mean_clv == 0.0 and summary.non_negative_drift is True


def test_run_ids_are_distinct():
    assert clv.new_run_id() != clv.new_run_id()
