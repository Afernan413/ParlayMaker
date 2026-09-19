"""Which week's slate a game belongs to."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.models.schedule import (
    one_week,
    parse_kickoff,
    week_by_game,
    week_key,
    week_label,
    week_start,
    weeks_on,
)


# ----------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------
def test_a_z_suffix_is_utc():
    assert parse_kickoff("2026-09-20T17:00:00Z") == datetime(
        2026, 9, 20, 17, 0, tzinfo=timezone.utc
    )


def test_an_offset_is_respected():
    assert parse_kickoff("2026-09-20T13:00:00-04:00").astimezone(timezone.utc).hour == 17


def test_a_naive_timestamp_is_read_as_utc():
    assert parse_kickoff("2026-09-20T17:00:00").tzinfo == timezone.utc


def test_a_datetime_passes_through():
    stamp = datetime(2026, 9, 20, 17, tzinfo=timezone.utc)
    assert parse_kickoff(stamp) is stamp


def test_junk_is_none_rather_than_an_exception():
    assert parse_kickoff("not a date") is None
    assert parse_kickoff("") is None
    assert parse_kickoff(None) is None


# ----------------------------------------------------------------------
# bucketing
# ----------------------------------------------------------------------
def test_a_sunday_afternoon_game_belongs_to_the_week_that_opened_on_tuesday():
    assert week_key("2026-09-20T17:00:00Z") == "2026-09-15"


def test_thursday_night_opens_the_week():
    """Thursday 8:15pm Eastern is Friday in UTC, and still that week's opener."""
    assert week_key("2026-09-18T00:15:00Z") == "2026-09-15"


def test_monday_night_closes_the_same_week_it_belongs_to():
    """8:15pm Eastern Monday is 00:15Z Tuesday -- the bug this guards against."""
    assert week_key("2026-09-22T00:15:00Z") == "2026-09-15"


def test_the_next_thursday_is_a_different_week():
    """The two the odds feed returns together, which must never share a ticket."""
    assert week_key("2026-09-20T17:00:00Z") != week_key("2026-09-25T00:15:00Z")


def test_a_college_saturday_shares_its_weeks_tuesday_maction():
    assert week_key("2026-09-16T23:00:00Z") == week_key("2026-09-19T19:30:00Z")


def test_tuesday_itself_opens_a_week():
    assert week_key("2026-09-22T23:00:00Z") == "2026-09-22"


def test_every_day_of_one_week_lands_in_the_same_bucket():
    kickoffs = [
        "2026-09-15T23:00:00Z",  # Tue evening
        "2026-09-17T00:00:00Z",  # Wed evening ET
        "2026-09-18T00:15:00Z",  # Thu night ET
        "2026-09-19T23:00:00Z",  # Sat
        "2026-09-20T17:00:00Z",  # Sun early
        "2026-09-21T00:20:00Z",  # Sun night ET
        "2026-09-22T00:15:00Z",  # Mon night ET
    ]
    assert {week_key(stamp) for stamp in kickoffs} == {"2026-09-15"}


def test_week_start_is_always_a_tuesday():
    for day in range(1, 29):
        start = week_start(f"2026-09-{day:02d}T20:00:00Z")
        assert start.weekday() == 1, f"{day} landed on {start}"


def test_an_unparseable_kickoff_has_no_week():
    assert week_key("whenever") is None
    assert week_start(None) is None


# ----------------------------------------------------------------------
# presentation and grouping
# ----------------------------------------------------------------------
def test_the_label_spans_tuesday_to_monday():
    assert week_label("2026-09-15") == "Sep 15 - Sep 21"


def test_a_label_for_nothing_says_so():
    assert week_label(None) == "Unscheduled"
    assert week_label("garbage") == "garbage"


def test_weeks_on_a_slate_come_back_in_order():
    games = [
        {"commence_time": "2026-09-25T00:15:00Z"},
        {"commence_time": "2026-09-20T17:00:00Z"},
        {"commence_time": "2026-09-20T20:05:00Z"},
        {"commence_time": "nonsense"},
    ]
    assert weeks_on(games) == ["2026-09-15", "2026-09-22"]


def test_week_by_game_skips_the_ones_it_cannot_place():
    games = [
        {"game_id": "a", "commence_time": "2026-09-20T17:00:00Z"},
        {"game_id": "b", "commence_time": None},
    ]
    assert week_by_game(games) == {"a": "2026-09-15"}


# ----------------------------------------------------------------------
# the check the optimizer runs
# ----------------------------------------------------------------------
def test_one_week_accepts_a_single_week():
    assert one_week(["2026-09-15", "2026-09-15"])


def test_one_week_rejects_two():
    assert not one_week(["2026-09-15", "2026-09-22"])


def test_an_unknown_week_is_not_treated_as_a_mismatch():
    """A slate with no kickoff times should still price; it just cannot check."""
    assert one_week([None, None])
    assert one_week(["2026-09-15", None])


def test_one_week_accepts_nothing_at_all():
    assert one_week([])
