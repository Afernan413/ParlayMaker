"""Which week's slate a game belongs to.

A parlay has to settle together. The Odds API hands back every upcoming event,
which on a Friday means this Sunday's games *and* next Thursday's -- so with no
notion of a week the optimizer will happily build a ticket that cannot resolve
for nine days, half of it priced off projections made a week too early. Worse,
the two halves are graded against different injury reports and different
weather. A slip has to be one week's slip.

A football week runs Tuesday to Monday: Thursday night through Monday night in
the NFL, Tuesday's MACtion through Saturday in college. Each week is named for
the Tuesday it starts on, so the keys sort chronologically as plain strings.

Kickoff times arrive in UTC, and a Monday night game kicks off on *Tuesday* in
UTC -- 20:15 Eastern is 00:15Z. Bucketing UTC directly would push every Monday
night game into the following week, so the timestamp is shifted back five hours
first, US Eastern's winter offset. Nothing in these leagues kicks off before
05:00 Eastern, so a fixed shift cannot misplace a game, and it needs no
timezone database to be right.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

#: US Eastern's winter offset. See the module docstring for why it is a fixed
#: number rather than a real timezone.
EASTERN_SHIFT = timedelta(hours=5)

#: Tuesday, as ``date.weekday()`` numbers it (Monday is 0).
WEEK_START_WEEKDAY = 1

MONTHS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)


def parse_kickoff(value: Any) -> datetime | None:
    """A kickoff timestamp as an aware UTC datetime, or ``None`` if unusable."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return None
    text = str(value).strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def week_start(commence_time: Any) -> date | None:
    """The Tuesday that opens this kickoff's week."""
    kickoff = parse_kickoff(commence_time)
    if kickoff is None:
        return None
    local = kickoff.astimezone(timezone.utc) - EASTERN_SHIFT
    offset = (local.weekday() - WEEK_START_WEEKDAY) % 7
    return local.date() - timedelta(days=offset)


def week_key(commence_time: Any) -> str | None:
    """Sortable identifier for a kickoff's week, e.g. ``"2026-09-22"``."""
    start = week_start(commence_time)
    return start.isoformat() if start else None


def week_label(key: str | None) -> str:
    """How the week reads to a person: ``"Sep 22 - Sep 28"``."""
    if not key:
        return "Unscheduled"
    try:
        start = date.fromisoformat(key)
    except ValueError:
        return str(key)
    end = start + timedelta(days=6)
    return f"{_short(start)} - {_short(end)}"


def _short(day: date) -> str:
    return f"{MONTHS[day.month - 1]} {day.day}"


def weeks_on(games: Iterable[Mapping[str, Any]]) -> list[str]:
    """Every week the slate spans, earliest first."""
    keys = {week_key(game.get("commence_time")) for game in games}
    return sorted(key for key in keys if key)


def week_by_game(games: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """``game_id -> week key`` for the games that have a usable kickoff."""
    index: dict[str, str] = {}
    for game in games:
        key = week_key(game.get("commence_time"))
        if key:
            index[str(game.get("game_id"))] = key
    return index


def one_week(weeks: Sequence[str | None]) -> bool:
    """True when every leg named here belongs to the same week.

    A missing week counts as unknown rather than as a mismatch: a slate with no
    kickoff times at all should still be priceable, it just cannot be checked.
    """
    named = {week for week in weeks if week}
    return len(named) <= 1
