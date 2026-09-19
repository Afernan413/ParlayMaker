"""EV math and parlay-assembly tests."""

from __future__ import annotations

import pytest

from config.settings import settings
from src.models.legs import Leg
from src.optimizer.ev_calculator import (
    DEVIG_METHODS,
    american_to_decimal,
    breakeven_probability,
    clv_delta,
    decimal_to_american,
    devig_selection,
    devig_two_way,
    edge,
    evaluate_leg,
    expected_value,
    find_edges,
    implied_probability,
    kelly_stake,
    market_overround,
    parlay_american_odds,
    parlay_decimal_odds,
    remove_vig,
)
from src.optimizer.parlay_builder import (
    BuildReport,
    build_parlays,
    describe_ticket_type,
    enumerate_tickets,
    price_ticket,
    same_game_pairs_are_correlated,
    select_portfolio,
    settles_in_one_week,
    summarise,
)

ITERATIONS = 4_000


def make_leg(**kwargs) -> Leg:
    base = dict(
        game_id="g1",
        sport="nfl",
        market="player_pass_yds",
        selection="Over",
        american_odds=-110,
        line=274.5,
        p_model=0.58,
    )
    base.update(kwargs)
    return Leg(**base)


# --------------------------------------------------------------- conversions
@pytest.mark.parametrize(
    ("american", "decimal"),
    [(-110, 1.909090909), (100, 2.0), (180, 2.8), (-200, 1.5), (-140, 1.714285714)],
)
def test_american_decimal_round_trip(american, decimal):
    assert american_to_decimal(american) == pytest.approx(decimal)
    assert decimal_to_american(decimal) == american


def test_zero_odds_are_rejected():
    with pytest.raises(ValueError):
        american_to_decimal(0)
    with pytest.raises(ValueError):
        decimal_to_american(1.0)


def test_implied_probability_and_breakeven_agree():
    assert implied_probability(-110) == pytest.approx(0.5238, abs=1e-4)
    assert breakeven_probability(-110) == pytest.approx(implied_probability(-110))


# ---------------------------------------------------------------- de-vigging
def test_devig_removes_the_whole_margin():
    odds = [-114, -106]
    assert market_overround(odds) > 0.04
    for method in DEVIG_METHODS:
        fair = remove_vig(odds, method)
        assert sum(fair) == pytest.approx(1.0)
        assert fair[0] > fair[1]  # the shorter price stays the favourite


def test_devig_preserves_the_favourite_ordering_on_a_lopsided_market():
    fair_over, fair_under = devig_two_way(-260, 200)
    assert fair_over > fair_under
    assert fair_over + fair_under == pytest.approx(1.0)
    assert fair_over < implied_probability(-260)  # margin stripped out


def test_power_method_differs_from_the_naive_split():
    odds = [-300, 240]
    power = remove_vig(odds, "power")
    multiplicative = remove_vig(odds, "multiplicative")
    assert power != multiplicative
    assert sum(power) == pytest.approx(1.0)


def test_unknown_devig_method_is_rejected():
    with pytest.raises(ValueError):
        remove_vig([-110, -110], "vibes")


def test_devig_selection_needs_both_sides():
    rows = [
        {"selection": "Over", "american_odds": -114},
        {"selection": "Under", "american_odds": -106},
    ]
    assert devig_selection(rows, "Over") == pytest.approx(remove_vig([-114, -106])[0])
    assert devig_selection(rows[:1], "Over") is None
    assert devig_selection(rows, "Yes") is None


def test_single_outcome_market_is_returned_untouched():
    assert remove_vig([-110]) == [pytest.approx(implied_probability(-110))]
    assert remove_vig([]) == []


# ------------------------------------------------------------ EV and staking
def test_expected_value_matches_the_formula():
    # 55% on -110: 0.55 * 0.909 - 0.45 * 1 = 0.05
    assert expected_value(0.55, -110) == pytest.approx(0.05, abs=1e-3)
    assert expected_value(breakeven_probability(-110), -110) == pytest.approx(0.0)
    assert expected_value(0.40, -110) < 0


def test_edge_is_model_minus_market():
    assert edge(0.58, 0.52) == pytest.approx(0.06)


def test_kelly_scales_with_the_edge_and_never_goes_negative():
    small = kelly_stake(0.55, american_to_decimal(-110), bankroll=1_000, fraction=0.25)
    large = kelly_stake(0.65, american_to_decimal(-110), bankroll=1_000, fraction=0.25)
    assert 0 < small < large
    assert kelly_stake(0.40, american_to_decimal(-110), bankroll=1_000) == 0.0
    # quarter Kelly is exactly a quarter of full Kelly
    full = kelly_stake(0.55, american_to_decimal(-110), bankroll=1_000, fraction=1.0)
    assert small == pytest.approx(full * 0.25)


def test_clv_is_positive_when_we_beat_the_close():
    assert clv_delta(-130, -110) > 0  # took -110, closed -130
    assert clv_delta(-110, -130) < 0


def test_parlay_pricing_multiplies_the_legs():
    assert parlay_decimal_odds([-110, -110]) == pytest.approx(1.909090909**2)
    assert parlay_american_odds([-110, -110]) == 264  # 1.909^2 = 3.645


# ------------------------------------------------------------ leg filtering
def test_leg_below_the_ev_floor_is_rejected():
    leg = make_leg(p_model=0.53)
    evaluation = evaluate_leg(leg, p_implied=0.52)
    assert evaluation.ev < settings.min_leg_ev
    assert not evaluation.passes and "below floor" in evaluation.reason


def test_leg_outside_the_odds_band_is_rejected():
    for odds in (-180, 250):
        evaluation = evaluate_leg(make_leg(american_odds=odds, p_model=0.9))
        assert not evaluation.passes and "outside" in evaluation.reason


def test_find_edges_splits_and_annotates():
    good = make_leg(player_name="QB One", p_model=0.62)
    thin = make_leg(player_name="WR One", market="player_reception_yds",
                    line=64.5, p_model=0.53)
    rich = make_leg(player_name="RB One", market="player_rush_yds",
                    line=48.5, american_odds=-190, p_model=0.80)
    keepers, evaluations = find_edges(
        [good, thin, rich], implied={good.leg_id: 0.55, thin.leg_id: 0.53}
    )
    assert keepers == [good]
    assert len(evaluations) == 3
    assert good.p_implied == 0.55 and good.ev > settings.min_leg_ev
    assert {e.passes for e in evaluations} == {True, False}


# ------------------------------------------------------------ ticket pricing
def test_correlated_sgp_prices_above_independence():
    qb = make_leg(player_name="QB One", team="KC", p_model=0.60)
    wr = make_leg(player_name="WR One", team="KC", market="player_reception_yds",
                  line=64.5, american_odds=-105, p_model=0.60)
    ticket = price_ticket([qb, wr], iterations=20_000, seed=9, bankroll=1_000)
    assert ticket.joint_probability > ticket.independent_probability
    assert ticket.correlation_lift > 1.0
    assert ticket.is_sgp and ticket.ticket_type == "2-Leg Correlated NFL SGP"
    assert ticket.stake > 0 and ticket.to_win > 0
    assert ticket.implied_probability == pytest.approx(1 / ticket.decimal_odds)


def test_cross_game_ticket_is_labelled_and_independent():
    a = make_leg(player_name="QB One", team="KC", p_model=0.58)
    b = make_leg(game_id="g2", sport="nba", market="player_points",
                 player_name="Guard A", team="BOS", line=25.5,
                 american_odds=-115, p_model=0.58)
    ticket = price_ticket([a, b], iterations=40_000, seed=9)
    assert not ticket.is_sgp
    assert ticket.joint_probability == pytest.approx(0.58 * 0.58, abs=0.01)
    assert describe_ticket_type([a, b]) == "2-Leg Cross-Game MULTI Parlay"


# ------------------------------------------------------- correlation gating
def test_negatively_correlated_same_game_pair_is_rejected():
    # the plan's canonical case: Under 38.5 with Over 3.5 passing TDs
    under = make_leg(market="totals", selection="Under", line=38.5,
                     player_name=None, team=None, p_model=0.56)
    pass_tds = make_leg(market="player_pass_tds", selection="Over", line=3.5,
                        player_name="QB One", team="KC", american_odds=120,
                        p_model=0.46)
    assert not same_game_pairs_are_correlated([under, pass_tds])

    tickets = enumerate_tickets([under, pass_tds], iterations=2_000)
    assert tickets == []


def test_weakly_correlated_same_game_pair_is_rejected():
    rb = make_leg(market="player_rush_yds", line=48.5, player_name="RB One",
                  team="KC", american_odds=100, p_model=0.60)
    qb = make_leg(player_name="QB One", team="KC", p_model=0.60)
    # RB rush yards vs own QB pass yards is slightly negative -> below the floor
    assert not same_game_pairs_are_correlated([rb, qb])


def test_cross_game_pairs_bypass_the_correlation_floor():
    a = make_leg(player_name="QB One", team="KC", p_model=0.60)
    b = make_leg(game_id="g2", sport="nba", market="player_points",
                 player_name="Guard A", team="BOS", line=25.5,
                 american_odds=-110, p_model=0.60)
    assert same_game_pairs_are_correlated([a, b])


# --------------------------------------------------------------- assembly
@pytest.fixture
def pool() -> list[Leg]:
    return [
        make_leg(player_name="QB One", team="KC", p_model=0.60),
        make_leg(player_name="WR One", team="KC", market="player_reception_yds",
                 line=64.5, american_odds=-105, p_model=0.60),
        make_leg(game_id="g2", sport="nba", market="player_points",
                 player_name="Guard A", team="BOS", line=25.5,
                 american_odds=-115, p_model=0.60),
        make_leg(game_id="g2", sport="nba", market="player_rebounds",
                 player_name="Big B", team="LAL", line=8.5,
                 american_odds=105, p_model=0.58),
    ]


def test_tickets_respect_leg_count_and_price_bands(pool):
    tickets = enumerate_tickets(pool, iterations=ITERATIONS, seed=5)
    assert tickets
    for ticket in tickets:
        assert settings.min_legs <= ticket.leg_count <= settings.max_legs
        assert settings.parlay_odds_min <= ticket.american_odds <= settings.parlay_odds_max
        assert ticket.ev > 0


def test_five_leg_tickets_are_never_built(pool):
    extra = make_leg(game_id="g3", market="player_receptions", line=4.5,
                     player_name="TE C", team="SF", american_odds=-105, p_model=0.60)
    tickets = enumerate_tickets(pool + [extra], iterations=1_000, seed=5)
    assert max(t.leg_count for t in tickets) <= 4


def test_legs_outside_the_price_band_never_reach_a_ticket(pool):
    junk = make_leg(game_id="g4", player_name="Longshot L", team="NYJ",
                    american_odds=400, p_model=0.40)
    tickets = enumerate_tickets(pool + [junk], iterations=1_000, seed=5)
    assert all("Longshot L" not in ticket.subjects for ticket in tickets)


# ------------------------------------------------------------------ one week
def test_a_ticket_never_spans_two_weeks():
    """The reported bug: the feed returns two weeks, the ILP combined them."""
    this_week = [
        make_leg(player_name="QB One", team="KC", p_model=0.60, slate_week="2026-09-15"),
        make_leg(player_name="WR One", team="KC", market="player_reception_yds",
                 line=64.5, american_odds=-105, p_model=0.60, slate_week="2026-09-15"),
    ]
    next_week = [
        make_leg(game_id="g9", player_name="QB Two", team="SF", p_model=0.60,
                 slate_week="2026-09-22"),
        make_leg(game_id="g9", player_name="WR Two", team="SF",
                 market="player_reception_yds", line=64.5, american_odds=-105,
                 p_model=0.60, slate_week="2026-09-22"),
    ]
    report = BuildReport()
    tickets = enumerate_tickets(
        this_week + next_week, iterations=1_000, seed=5, report=report
    )
    assert tickets, "each week on its own should still produce tickets"
    for ticket in tickets:
        assert len({leg.slate_week for leg in ticket.legs}) == 1
    assert report.rejected["legs from different weeks"] > 0


def test_a_ticket_reports_the_week_it_settles_on():
    legs = [
        make_leg(player_name="QB One", team="KC", p_model=0.60, slate_week="2026-09-15"),
        make_leg(player_name="WR One", team="KC", market="player_reception_yds",
                 line=64.5, american_odds=-105, p_model=0.60, slate_week="2026-09-15"),
    ]
    ticket = enumerate_tickets(legs, iterations=1_000, seed=5)[0]
    assert ticket.slate_week == "2026-09-15"
    assert ticket.week_display == "Sep 15 - Sep 21"
    assert ticket.as_dict()["week"] == "Sep 15 - Sep 21"


def test_a_slate_with_no_kickoff_times_still_builds(pool):
    """Unknown weeks are unknown, not a mismatch -- fixtures must still price."""
    assert all(leg.slate_week is None for leg in pool)
    assert enumerate_tickets(pool, iterations=1_000, seed=5)


def test_portfolio_never_repeats_a_subject(pool):
    tickets, report = build_parlays(pool, iterations=ITERATIONS, seed=5, bankroll=1_000)
    assert tickets
    assert report.solver_status == "Optimal"
    seen: set[str] = set()
    for ticket in tickets:
        assert not (ticket.subjects & seen)
        seen |= ticket.subjects


def test_portfolio_honours_the_ticket_cap(pool):
    tickets, _ = build_parlays(pool, max_tickets=1, iterations=ITERATIONS, seed=5)
    assert len(tickets) == 1


def test_portfolio_maximises_expected_value(pool):
    candidates = enumerate_tickets(pool, iterations=ITERATIONS, seed=5)
    selected = select_portfolio(candidates, max_tickets=1)
    assert selected[0].ev == max(c.ev for c in candidates)


def test_greedy_fallback_matches_the_constraints(pool):
    from src.optimizer import parlay_builder

    candidates = enumerate_tickets(pool, iterations=ITERATIONS, seed=5)
    greedy = parlay_builder._greedy_portfolio(candidates, 3)
    seen: set[str] = set()
    for ticket in greedy:
        assert not (ticket.subjects & seen)
        seen |= ticket.subjects


def test_empty_inputs_produce_an_empty_card():
    tickets, report = build_parlays([], iterations=500)
    assert tickets == []
    assert report.solver_status == "empty"
    assert summarise([]) == "no qualifying tickets"


def test_build_report_explains_rejections():
    under = make_leg(market="totals", selection="Under", line=38.5,
                     player_name=None, team=None, p_model=0.56)
    pass_tds = make_leg(market="player_pass_tds", selection="Over", line=3.5,
                        player_name="QB One", team="KC", american_odds=120, p_model=0.46)
    _, report = build_parlays([under, pass_tds], iterations=1_000)
    assert report.considered == 1
    assert report.rejected["same-game pair below correlation floor"] == 1


def test_summary_line_reads_like_a_bet_slip(pool):
    tickets, _ = build_parlays(pool, max_tickets=1, iterations=ITERATIONS, seed=5,
                               bankroll=1_000)
    text = summarise(tickets)
    assert "model" in text and "EV" in text and "stake" in text
    assert tickets[0].as_dict()["ticket_type"] == tickets[0].ticket_type
