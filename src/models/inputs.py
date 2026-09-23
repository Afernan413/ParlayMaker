"""What the model actually knew when it priced a slate.

Every contextual input is optional in practice. Weather needs a key that may
not be set; the injury report may lag the week being priced, or not exist for
the sport at all; snap shares need a couple of games of the current season
before they mean anything. Each of those was previously a log line nobody
reads, or nothing at all -- so a page could imply the model had considered the
weather when the weather stage had never run.

This is the record. One :class:`InputStatus` per input per sport, carried
through the run summary and into the static bundle, so the answer to "did it
know about the injuries?" is on the page rather than in a build log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: The inputs worth reporting on, in the order a reader cares about them.
INPUT_ORDER = ("props", "rosters", "starters", "injuries", "weather")


@dataclass(frozen=True)
class InputStatus:
    """Whether one contextual input reached the projections, and why not.

    ``covered`` is how much of the slate it reached -- games, players,
    designations -- so "injuries: 3 of 246" is distinguishable from
    "injuries: 0".
    """

    name: str
    available: bool
    detail: str = ""
    covered: int = 0

    def as_row(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "detail": self.detail,
            "covered": self.covered,
        }


@dataclass
class ModelInputs:
    """Every input's status for one sport's slate."""

    sport: str
    statuses: dict[str, InputStatus] = field(default_factory=dict)

    def record(self, name: str, available: bool, detail: str = "", covered: int = 0) -> None:
        self.statuses[name] = InputStatus(
            name=name, available=available, detail=detail, covered=covered
        )

    @property
    def missing(self) -> list[str]:
        return [name for name in INPUT_ORDER if not self.statuses.get(name, _ABSENT).available]

    def as_rows(self) -> list[dict[str, Any]]:
        return [
            self.statuses[name].as_row() for name in INPUT_ORDER if name in self.statuses
        ]

    def summary(self) -> str:
        """One line for the console, naming what was and was not known."""
        parts = []
        for name in INPUT_ORDER:
            status = self.statuses.get(name)
            if status is None:
                continue
            parts.append(f"{name}={status.covered}" if status.available else f"{name}=none")
        return " ".join(parts)


_ABSENT = InputStatus(name="", available=False)


def weather_status(sport: str, *, snapshots: int, has_key: bool, venues: bool) -> InputStatus:
    """Why weather did or did not reach the projections.

    Basketball is indoors, so "not applicable" is the right answer rather than a
    gap. College football is a real gap: the venue coordinates the forecast
    lookup needs only exist for the 32 NFL stadiums.
    """
    if sport == "nba":
        return InputStatus("weather", True, "indoor sport, no weather to apply", 0)
    if not venues:
        return InputStatus(
            "weather", False,
            "no venue coordinates for this sport, so no forecast can be looked up", 0,
        )
    if not has_key:
        return InputStatus(
            "weather", False, "no OPENWEATHER_API_KEY set, so the forecast was never fetched", 0,
        )
    if snapshots == 0:
        return InputStatus("weather", False, "forecast fetch returned nothing", 0)
    return InputStatus("weather", True, f"{snapshots} venue forecast(s)", snapshots)


def injury_status(sport: str, *, designations: int, detail: str = "") -> InputStatus:
    if designations == 0:
        return InputStatus(
            "injuries", False,
            detail or f"no injury report reached the model for {sport}", 0,
        )
    return InputStatus("injuries", True, detail, designations)


def starter_status(*, players: int, detail: str = "") -> InputStatus:
    if players == 0:
        return InputStatus("starters", False, detail, 0)
    return InputStatus("starters", True, detail, players)
