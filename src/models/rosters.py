"""Who is on which team this week, and who can actually play.

The volume model learns from box scores, and box scores remember everyone who
ever played: the receiver traded in the spring, the veteran who retired, the
back released in camp. Measured on the 2026 week-3 slate, 40% of the player rows
the model projected from were one of those -- a player on a team he no longer
plays for (22%), on injured reserve, retired or on the practice squad (8%), or
on no roster at all (10%).

The weekly injury report cannot catch these. Players on injured reserve are not
listed on it, and retired or released players never are. The league's weekly
roster can, so it is the authority on two questions:

* **which team** a player is on -- a traded player's projection belongs to his
  new team, and a ghost projection for his old one must not exist (when a
  player faces his old team both would claim the same game and market, and
  whichever came last would win);
* **whether he can play** -- only the active roster takes the field. Reserve
  lists, the practice squad, the retired and the released do not.

NFL only: nflverse publishes weekly rosters for the NFL. Elsewhere the roster is
empty and nothing is filtered, which the run reports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

from src.models.roles import normalise_name

logger = logging.getLogger(__name__)

#: The one roster status that takes the field.
ACTIVE = "ACT"

#: What the other statuses mean, for the report.
STATUS_MEANING = {
    "ACT": "active",
    "RES": "reserve list (injured reserve and similar)",
    "DEV": "practice squad",
    "RET": "retired",
    "CUT": "released",
    "EXE": "exempt",
    "SUS": "suspended",
    "PUP": "physically unable to perform",
    "INA": "inactive",
}


@dataclass(frozen=True)
class RosterEntry:
    player: str
    team: str
    position: str
    status: str

    @property
    def active(self) -> bool:
        return self.status == ACTIVE


@dataclass(frozen=True)
class Roster:
    """The league's roster for one week, keyed by normalised player name."""

    entries: dict[str, RosterEntry] = field(default_factory=dict)
    week: int | None = None
    season: int | None = None

    @property
    def empty(self) -> bool:
        return not self.entries

    def get(self, player: Any) -> RosterEntry | None:
        return self.entries.get(normalise_name(player))

    def team_of(self, player: Any) -> str | None:
        entry = self.get(player)
        return entry.team if entry else None

    def can_play(self, player: Any) -> bool:
        """Active on a roster. With no roster loaded, everyone can."""
        if self.empty:
            return True
        entry = self.get(player)
        return entry is not None and entry.active

    def why_not(self, player: Any) -> str:
        """Why a player cannot play, in words."""
        entry = self.get(player)
        if entry is None:
            return "not on any roster"
        return STATUS_MEANING.get(entry.status, entry.status.lower())

    def coverage(self) -> dict[str, Any]:
        statuses: dict[str, int] = {}
        for entry in self.entries.values():
            statuses[entry.status] = statuses.get(entry.status, 0) + 1
        return {
            "players": len(self.entries),
            "active": statuses.get(ACTIVE, 0),
            "statuses": statuses,
            "week": self.week,
        }


def build_roster(frame: pd.DataFrame | None, *, week: int | None = None) -> Roster:
    """A :class:`Roster` from nflverse's weekly roster frame.

    Uses ``week`` if given, otherwise the latest week published. A player listed
    twice in a week -- moved mid-week -- keeps the active listing if there is one.
    """
    if frame is None or frame.empty:
        return Roster()
    rows = frame
    if "week" in rows.columns and len(rows):
        target = int(week) if week is not None else int(rows["week"].max())
        chosen = rows[rows["week"].astype(int) == target]
        if chosen.empty:
            chosen = rows[rows["week"].astype(int) == int(rows["week"].max())]
            target = int(rows["week"].max())
        rows = chosen
    else:
        target = week

    entries: dict[str, RosterEntry] = {}
    for record in rows.to_dict("records"):
        name = record.get("full_name") or " ".join(
            part for part in (record.get("first_name"), record.get("last_name")) if part
        )
        if not name:
            continue
        entry = RosterEntry(
            player=str(name),
            team=str(record.get("team") or "").upper(),
            position=str(record.get("position") or ""),
            status=str(record.get("status") or "").upper(),
        )
        key = normalise_name(name)
        existing = entries.get(key)
        if existing is None or (entry.active and not existing.active):
            entries[key] = entry
    season = int(rows["season"].max()) if "season" in rows.columns and len(rows) else None
    return Roster(entries=entries, week=target, season=season)


def load_nfl_roster(season: int, *, week: int | None = None) -> Roster:
    """The NFL's weekly roster for ``season``, latest week unless told otherwise."""
    try:
        import nflreadpy
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("nflreadpy is not installed") from exc
    frame = nflreadpy.load_rosters_weekly(seasons=[int(season)]).to_pandas()
    roster = build_roster(frame, week=week)
    logger.info("roster: %s", roster.coverage())
    return roster
