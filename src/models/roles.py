"""Who is actually playing, and how much of the game they play.

The volume model averages a player's last four games. That is right for a
settled role and wrong in the one case that moves a line furthest: the player
ahead of them is out, and they are about to play three times the snaps their
average is built from. In week 2 of 2026 Atlanta had its listed starting
quarterback out; a four-game average of the man behind him says he throws for
fifteen yards.

So volume is projected *per snap* rather than per game:

    rate      = own production / own snaps, shrunk toward the same position
                group's rate on that team by how little the player has played
    projected = rate x the snaps we expect them to take

A third-string quarterback inherits almost all of his team's quarterback
production rate, because he has no rate of his own worth trusting. A settled
starter's own rate dominates, so nothing changes for him.

Expected snaps come from redistributing the snap share of everyone in the
position group who is out, in proportion to what the remaining players already
play. That conserves the snaps a team actually has to fill -- a team fields one
quarterback whoever is available -- which is what makes a promotion show up as
a promotion rather than as a five per cent nudge.

Snap counts exist for the NFL. College and basketball fall back to the
per-game recipe, and say so rather than pretending: see
:func:`RoleModel.coverage`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from config.settings import settings
from src.ingestion.injuries import STATUS_ORDER, normalize_status, status_multiplier

logger = logging.getLogger(__name__)

#: How much of a player's own rate to trust, in snaps *per appearance* -- the
#: same units as ``Role.snaps``. A full-time starter plays 55 to 60, so at 20 he
#: keeps about three quarters of his own rate; a backup on 8 snaps a game keeps
#: under a third and is priced mostly at his position group's rate, which is the
#: honest answer for someone nobody has seen carry a game.
#:
#: Set in per-appearance units on purpose. An earlier value of 60 shrank even a
#: settled starter more than halfway toward the group, because it was written as
#: if it were a total across the window rather than a per-game figure.
SNAP_SHRINKAGE = 20.0

#: A player cannot take more than every snap.
MAX_SHARE = 1.0

#: How far a snap share is allowed to move a projection. Early in a season the
#: share is measured over one or two games, and a thin sample should not be
#: trusted to quarter or triple a number. ``OUT`` is handled separately and is
#: absolute, so this bounds only the ambiguous cases.
SNAP_FACTOR_BOUNDS = (0.5, 2.5)

#: How much a workload has to move before the projection is rebuilt per snap.
#: Below it the per-game average stands, which keeps the model in step with the
#: calibration fitted on that average.
ROLE_CHANGE_THRESHOLD = 1.15

#: Games of the current season needed before the snap share says anything. One
#: game's share is noise, and the window deliberately does not reach back into
#: last season: rosters change, and a share that mixes two rosters is worse
#: than no share at all.
MIN_WINDOW_GAMES = 2

#: Positions whose snaps are worth redistributing. Offensive skill positions
#: only: a cornerback going out does not change anyone's receiving yards, which
#: is what the previous blanket "somebody on this team is out" rule implied.
SKILL_POSITIONS = frozenset({"QB", "RB", "FB", "WR", "TE"})


#: Generational suffixes, which the league's injury report carries and the snap
#: data does not -- "Michael Penix Jr." against "Michael Penix". Left unjoined,
#: a ruled-out starter reads as available.
NAME_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})


def normalise_name(name: Any) -> str:
    """Punctuation-free key for joining a feed's spelling to a box score's."""
    tokens = [
        "".join(ch for ch in part.lower() if ch.isalnum())
        for part in str(name or "").replace("-", " ").split()
    ]
    tokens = [token for token in tokens if token]
    # Only a trailing suffix, and only when a first and last name remain.
    while len(tokens) > 2 and tokens[-1] in NAME_SUFFIXES:
        tokens.pop()
    return "".join(tokens)


@dataclass(frozen=True)
class Role:
    """What one player's part in their team's offence looks like."""

    player: str
    team: str
    position: str
    #: Weighted offensive snaps in the games the player actually appeared in.
    #: This is the denominator for a production rate, because the production it
    #: divides is averaged the same way -- over appearances, not over the team's
    #: schedule.
    snaps: float
    #: Weighted share of the team's offensive snaps over the team's last games,
    #: counting a game the player sat out as zero. This is the quantity
    #: :func:`redistribute` moves around, because it is the one that sums to the
    #: slots a team fields.
    share: float
    status: str = "ACTIVE"
    #: Share once the group's ruled-out players are redistributed.
    expected_share: float = 0.0

    @property
    def out(self) -> bool:
        return self.status == "OUT"

    @property
    def availability(self) -> float:
        return status_multiplier(self.status)

    @property
    def starter(self) -> bool:
        """Playing over half a team's offensive snaps."""
        return self.share >= 0.5

    @property
    def snap_factor(self) -> float:
        """How much more (or less) they will play than their average says."""
        if self.share <= 0 or self.expected_share <= 0:
            return 1.0
        return float(np.clip(self.expected_share / self.share, *SNAP_FACTOR_BOUNDS))

    def expected_snaps(self, team_snaps: float) -> float:
        """Snaps to expect this week, given the team's snaps per game.

        Taken from the share rather than from the player's own snap count, so a
        player who has been in and out of the lineup is projected onto the
        snaps the team has to fill rather than onto his own patchy average.
        """
        if team_snaps <= 0:
            return self.snaps
        return float(np.clip(self.expected_share, 0.0, MAX_SHARE)) * team_snaps

    @property
    def promoted(self) -> bool:
        return self.snap_factor > ROLE_CHANGE_THRESHOLD

    @property
    def role_changed(self) -> bool:
        """Is this week's workload different enough to reprice the average?

        Only a real change earns the per-snap treatment. A settled role's
        per-game average is both the better estimate and the one the learned
        corrections were fitted on.
        """
        return (
            self.snap_factor > ROLE_CHANGE_THRESHOLD
            or self.snap_factor < 1.0 / ROLE_CHANGE_THRESHOLD
        )


@dataclass(frozen=True)
class RoleModel:
    """Every player's role on one slate.

    ``keyed`` is by ``(team, normalised player name)``. A team that is missing
    from the snap data has no roles, and the projection layer falls back to the
    per-game recipe for it.
    """

    roles: dict[tuple[str, str], Role] = field(default_factory=dict)
    #: Teams the snap data covered, so an absent role can be told apart from a
    #: player the model has simply never seen.
    covered_teams: frozenset[str] = frozenset()
    #: Weighted offensive snaps per game, per team. Turns a share into a count.
    team_snaps: dict[str, float] = field(default_factory=dict)
    #: ``{normalised name: status}`` from the injury report. Kept separately from
    #: ``roles`` because availability does not depend on the snap window: a
    #: ruled-out starter is ruled out in week one, when no share is yet worth
    #: measuring.
    statuses: dict[str, str] = field(default_factory=dict)
    #: The same keyed by ``(team, name)``. Two players can share a name across
    #: rosters -- there are two Brian Robinsons -- and applying one's
    #: designation to the other would rule out a healthy player.
    team_statuses: dict[tuple[str, str], str] = field(default_factory=dict)
    #: Which week's injury report the statuses came from, and how stale it is.
    report_week: int | None = None
    report_lag: int = 0
    #: When the depth chart used was published, if one was.
    depth_as_of: str | None = None
    #: The week whose ruled-out players were carried forward because this
    #: week's designations were not yet published; ``None`` if none were.
    carried_from: int | None = None
    #: ``(team, name) -> expected share`` for every quarterback on a depth
    #: chart, including those with no snaps this season.
    depth_shares: dict[tuple[str, str], float] = field(default_factory=dict)

    def quarterback_share(self, team: Any, player: Any) -> float | None:
        """A charted quarterback's expected share of the slot, else ``None``."""
        return self.depth_shares.get((str(team or "").upper(), normalise_name(player)))

    def snaps_for(self, team: Any) -> float:
        return self.team_snaps.get(str(team or "").upper(), 0.0)

    def status_for(self, player: Any, team: Any = None) -> str | None:
        """This player's designation, or ``None`` if the report does not name them.

        Prefers the team-specific entry: a name on its own is ambiguous when two
        rosters carry it.
        """
        name = normalise_name(player)
        if team:
            found = self.team_statuses.get((str(team).upper(), name))
            if found is not None:
                return found
            # The report names the player on another team, so this is not him.
            if any(key[1] == name for key in self.team_statuses):
                return None
        return self.statuses.get(name)

    def get(self, team: Any, player: Any) -> Role | None:
        return self.roles.get((str(team or "").upper(), normalise_name(player)))

    def covers(self, team: Any) -> bool:
        return str(team or "").upper() in self.covered_teams

    @property
    def empty(self) -> bool:
        return not self.roles

    @property
    def reason(self) -> str:
        """Why the role model is doing nothing, when it is doing nothing."""
        if self.roles:
            return ""
        return (
            f"fewer than {MIN_WINDOW_GAMES} games played this season, so a snap "
            "share would be noise -- availability still applies, role changes do not"
        )

    def coverage(self) -> dict[str, Any]:
        """What the model actually knew, for the run summary and the page."""
        promoted = [role for role in self.roles.values() if role.promoted]
        return {
            "players": len(self.roles),
            "teams": len(self.covered_teams),
            "designations": len(self.statuses),
            "out": sum(1 for status in self.statuses.values() if status == "OUT"),
            "promoted": len(promoted),
            "starters": sum(1 for role in self.roles.values() if role.starter),
            "report_week": self.report_week,
            "report_lag_weeks": self.report_lag,
            "depth_chart": self.depth_as_of,
            "carried_from": self.carried_from,
            "out_last_week": sum(
                1 for status in self.statuses.values() if status == "OUT_LAST_WEEK"
            ),
        }

    def promotions(self, limit: int = 12) -> list[Role]:
        """The biggest role changes, most promoted first. For the rationale."""
        promoted = [role for role in self.roles.values() if role.promoted]
        return sorted(promoted, key=lambda role: role.snap_factor, reverse=True)[:limit]


# ----------------------------------------------------------------------
# building it
# ----------------------------------------------------------------------
def snap_window(
    snap_counts: pd.DataFrame,
    *,
    season: int,
    week: int,
    weeks: int | None = None,
) -> pd.DataFrame:
    """Recency-weighted offensive snaps per player, from before ``week`` only.

    The window is the *team's* last few games, not the player's. That
    distinction is the whole point: measured over a player's own last four
    appearances, every quarterback who has ever started reads as a full-time
    starter, and a position group's shares sum to well over one. Measured over
    the team's last four games, a player who sat has a share of zero for those
    weeks, the group sums to roughly the slots a team fields, and
    :func:`redistribute` has a denominator that means something.

    Weighted with the same weights as the volume model, so snaps and production
    are summarised over the same window.
    """
    empty = pd.DataFrame(
        columns=["team", "player", "position", "snaps", "share", "team_snaps"]
    )
    if snap_counts is None or snap_counts.empty:
        return empty

    frame = snap_counts.dropna(subset=["player", "team", "season", "week"]).copy()
    frame["season"] = frame["season"].astype(int)
    frame["week"] = frame["week"].astype(int)
    # This season only. A window reaching back into last season mixes two
    # rosters, and a snap share built from two different squads is worse than
    # none: half of it describes players who have moved on.
    earlier = frame[(frame["season"] == int(season)) & (frame["week"] < int(week))]
    if earlier.empty:
        return empty

    window = weeks or settings.rolling_weeks
    weights = list(settings.rolling_weight_list)[:window] or [1.0]

    rows: list[dict[str, Any]] = []
    for team, played in earlier.groupby("team", sort=False):
        # The team's own last few games, newest first.
        games = (
            played[["season", "week"]].drop_duplicates()
            .sort_values(["season", "week"], ascending=False)
            .head(window)
        )
        if games.empty:
            continue
        if len(games) < MIN_WINDOW_GAMES:
            continue
        stamps = list(zip(games["season"], games["week"]))
        used = weights[: len(stamps)]
        total = sum(used)
        if total <= 0:
            continue
        weight_by_game = {stamp: weight for stamp, weight in zip(stamps, used)}

        in_window = played[
            played.apply(lambda r: (r["season"], r["week"]) in weight_by_game, axis=1)
        ]
        # How many offensive snaps this team runs in a game -- what a share is a
        # share of. Taken as the largest snap count in each game, since whoever
        # played every snap defines the total.
        per_game = in_window.groupby(["season", "week"])["offense_snaps"].max()
        team_snaps = sum(
            weight_by_game[stamp] * float(value or 0.0) for stamp, value in per_game.items()
        ) / total

        for player, appearances in in_window.groupby("player", sort=False):
            # Two denominators, deliberately. `share` divides by the whole
            # window so a game missed counts as zero; `snaps` divides by only
            # the games played, matching how the production it will be divided
            # into is averaged.
            share_weight = 0.0
            own_weight = 0.0
            snaps = 0.0
            for row in appearances.to_dict("records"):
                weight = weight_by_game[(row["season"], row["week"])]
                share_weight += weight * float(row.get("offense_pct") or 0.0)
                snaps += weight * float(row.get("offense_snaps") or 0.0)
                own_weight += weight
            rows.append(
                {
                    "team": str(team).upper(),
                    "player": str(player),
                    "position": str(appearances.iloc[0].get("position") or ""),
                    "snaps": snaps / own_weight if own_weight > 0 else 0.0,
                    "share": min(share_weight / total, MAX_SHARE),
                    "team_snaps": team_snaps,
                }
            )
    return pd.DataFrame(rows) if rows else empty


def redistribute(group: Sequence[Role], team_snaps: float) -> dict[str, float]:
    """Expected share per player once the group's ruled-out snaps move.

    Two quantities do different jobs here, and conflating them was wrong in
    both directions:

    * ``share`` -- a player's share of the team's last few games, counting one
      he sat out as zero. Summed over a position group it estimates the
      **slots** a team fields there: one quarterback, about two and a half
      receivers. That total is what a team has to cover whoever is available,
      so it is the conserved quantity.
    * ``snaps / team_snaps`` -- what a player plays **when he plays**. That is
      his role, and for a player who is available this week it is the better
      predictor than his attendance record. A receiver who plays four snaps in
      five when healthy but missed two games has a four-week share of thirty
      per cent and a role of eighty.

    So the slots are divided among the available players in proportion to their
    role, and nobody is given more than they take when healthy.
    """
    available = [role for role in group if not role.out]
    slots = sum(role.share for role in group)
    if not available or slots <= 0 or team_snaps <= 0:
        return {normalise_name(role.player): role.share for role in available}

    def when_playing(role: Role) -> float:
        return min(role.snaps / team_snaps, MAX_SHARE)

    roles_when_playing = {normalise_name(role.player): when_playing(role) for role in available}
    weight = sum(roles_when_playing.values())
    if weight <= 0:
        return {normalise_name(role.player): role.share for role in available}

    # Normalising to the slots is enough on its own: when the only available
    # quarterback has a role of 0.6, he takes the whole slot and comes out at
    # 1.0, which is what a starting quarterback plays. An earlier version capped
    # each player at the group's demonstrated maximum, but that maximum is
    # contaminated by the very injury that created the vacancy -- a starter who
    # left at halftime shows a 0.69 workload -- and it left promoted starters
    # priced as reserves.
    return {
        normalise_name(role.player): min(
            slots * roles_when_playing[normalise_name(role.player)] / weight, MAX_SHARE
        )
        for role in available
    }


#: What a demoted quarterback is expected to play: mop-up snaps, not a slot.
BACKUP_QB_SHARE = 0.05


@dataclass(frozen=True)
class DepthChart:
    """Each team's latest published depth chart, ``(team, name) -> rank``.

    The snap history can only say who *has* been playing. After a benching, a
    starter's return from injury or a trade, the depth chart is the first place
    the change shows up -- and for a quarterback it settles what snap shares
    cannot: two quarterbacks who each started a game do not split the next one.
    """

    ranks: dict[tuple[str, str], tuple[str, int]] = field(default_factory=dict)
    as_of: str | None = None

    @property
    def empty(self) -> bool:
        return not self.ranks

    def rank(self, team: Any, player: Any) -> int | None:
        found = self.ranks.get((str(team or "").upper(), normalise_name(player)))
        return found[1] if found else None

    def quarterbacks(self, team: Any) -> list[tuple[str, int]]:
        """``(normalised name, rank)`` for a team's quarterbacks, starter first."""
        key = str(team or "").upper()
        found = [
            (name, rank) for (club, name), (position, rank) in self.ranks.items()
            if club == key and position == "QB"
        ]
        return sorted(found, key=lambda item: item[1])


def build_depth_chart(frame: pd.DataFrame | None) -> DepthChart:
    """The latest snapshot per team from nflverse's dated depth charts."""
    if frame is None or frame.empty:
        return DepthChart()
    rows = frame.dropna(subset=["team", "player_name", "pos_rank"]).copy()
    if rows.empty:
        return DepthChart()
    rows["dt"] = pd.to_datetime(rows["dt"], utc=True, errors="coerce")
    rows = rows[rows["dt"] == rows.groupby("team")["dt"].transform("max")]

    ranks: dict[tuple[str, str], tuple[str, int]] = {}
    for record in rows.to_dict("records"):
        key = (str(record["team"]).upper(), normalise_name(record["player_name"]))
        position = str(record.get("pos_abb") or "")
        rank = int(record["pos_rank"])
        # A player listed at several spots keeps his best rank at each position;
        # only the quarterback ranks are acted on.
        existing = ranks.get(key)
        if existing is None or (position == "QB" and (existing[0] != "QB" or rank < existing[1])):
            ranks[key] = (position, rank)
    as_of = rows["dt"].max()
    return DepthChart(ranks=ranks, as_of=as_of.isoformat() if pd.notna(as_of) else None)


def starting_quarterback(team: str, depth: DepthChart, statuses: Mapping[str, str]) -> str | None:
    """The highest-ranked quarterback on the chart who is not ruled out.

    The chart lists an injured starter at the top -- it describes the roster,
    not this week's availability -- so the report decides who is skipped.
    """
    for name, _rank in depth.quarterbacks(team):
        if statuses.get((team, name), statuses.get(name, "ACTIVE")) != "OUT":
            return name
    return None


def build_role_model(
    snap_counts: pd.DataFrame | None,
    injuries: Mapping[str, str] | Iterable[Any] | None = None,
    *,
    season: int,
    week: int,
    weeks: int | None = None,
    report_week: int | None = None,
    report_lag: int = 0,
    depth: DepthChart | None = None,
    carried_from: int | None = None,
) -> RoleModel:
    """Roles for a slate: how much each player plays, and how much they will.

    ``injuries`` is either a ``{player_name: status}`` index or an iterable of
    :class:`~src.ingestion.injuries.InjuryRecord`. Only skill-position players
    are redistributed -- a cornerback going out does not change anyone's
    receiving yards.
    """
    status_by_name, status_by_team = _status_index(injuries)
    window = snap_window(snap_counts, season=season, week=week, weeks=weeks)
    if window.empty:
        # No usable snap history, but the injury report still stands: who is out
        # is known from week one, and it is the correction that matters most.
        return RoleModel(
            statuses=status_by_name, team_statuses=status_by_team,
            report_week=report_week, report_lag=report_lag, carried_from=carried_from,
        )

    team_snap_counts = (
        window.groupby("team")["team_snaps"].first().to_dict()
        if "team_snaps" in window.columns else {}
    )
    roles: dict[tuple[str, str], Role] = {}
    groups: dict[tuple[str, str], list[Role]] = {}

    for row in window.to_dict("records"):
        key = normalise_name(row["player"])
        role = Role(
            player=row["player"],
            team=row["team"],
            position=row["position"],
            snaps=row["snaps"],
            share=row["share"],
            status=status_by_team.get((row["team"], key), status_by_name.get(key, "ACTIVE")),
        )
        roles[(role.team, key)] = role
        if role.position in SKILL_POSITIONS:
            groups.setdefault((role.team, role.position), []).append(role)

    # Redistribute inside each position group, then write the expectation back.
    for (team, _position), group in groups.items():
        team_total = team_snap_counts.get(team, 0.0)
        for name, share in redistribute(group, team_total).items():
            existing = roles.get((team, name))
            if existing is not None:
                roles[(team, name)] = Role(
                    player=existing.player,
                    team=existing.team,
                    position=existing.position,
                    snaps=existing.snaps,
                    share=existing.share,
                    status=existing.status,
                    expected_share=share,
                )

    # Quarterbacks: the chart names the starter, and the slot is his. Snap
    # shares alone split it between everyone who has started lately.
    depth_shares: dict[tuple[str, str], float] = {}
    if depth is not None and not depth.empty:
        combined = dict(status_by_name)
        combined.update({key: value for key, value in status_by_team.items()})
        # Driven by the chart rather than by the snap groups, so a team whose
        # quarterbacks have no snaps this season is still covered.
        charted = {club for (club, _name), (pos, _rank) in depth.ranks.items() if pos == "QB"}
        for team in sorted(charted):
            group = groups.get((team, "QB"), [])
            starter = starting_quarterback(team, depth, combined)
            if starter is None:
                continue
            slot = min(max(sum(role.share for role in group), 1.0), MAX_SHARE)
            # How likely the starter is to play. His own projection is already
            # scaled by this through his availability; the next man up carries
            # the rest, so a doubtful starter's backup is not priced as if he
            # will certainly sit.
            plays = status_multiplier(combined.get((team, starter), combined.get(starter)))
            next_up = next(
                (
                    name for name, _rank in depth.quarterbacks(team)
                    if name != starter
                    and combined.get((team, name), combined.get(name, "ACTIVE")) != "OUT"
                ),
                None,
            )
            def expected(key: str, current: float) -> float:
                if key == starter:
                    return slot
                if key == next_up:
                    return plays * BACKUP_QB_SHARE + (1.0 - plays) * slot
                return min(current, BACKUP_QB_SHARE)

            # Every quarterback on the chart gets a share, including those with
            # no snaps this season: a starter back from injury, or one who
            # arrived by trade, is otherwise invisible here and would be priced
            # off his old average as a second full-time starter.
            for name, _rank in depth.quarterbacks(team):
                if combined.get((team, name), combined.get(name)) == "OUT":
                    continue
                depth_shares[(team, name)] = expected(name, BACKUP_QB_SHARE)

            for role in group:
                key = normalise_name(role.player)
                if role.out:
                    continue
                share = expected(key, role.share)
                depth_shares[(team, key)] = share
                roles[(team, key)] = Role(
                    player=role.player, team=role.team, position=role.position,
                    snaps=role.snaps, share=role.share, status=role.status,
                    expected_share=share,
                )

    # Everyone still at zero either sits outside a redistributed group -- a
    # lineman, say -- or is out. A player who is out expects no snaps at all;
    # anyone else expects what they already play.
    for key, role in list(roles.items()):
        if role.expected_share > 0:
            continue
        roles[key] = Role(
            player=role.player, team=role.team, position=role.position,
            snaps=role.snaps, share=role.share, status=role.status,
            expected_share=0.0 if role.out else role.share,
        )

    return RoleModel(
        roles=roles,
        covered_teams=frozenset(window["team"].unique()),
        team_snaps=team_snap_counts,
        statuses=status_by_name,
        team_statuses=status_by_team,
        report_week=report_week,
        report_lag=report_lag,
        depth_as_of=depth.as_of if depth is not None else None,
        carried_from=carried_from,
        depth_shares=depth_shares,
    )


def _status_index(injuries: Any) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    """Designations by name, and by ``(team, name)`` where the team is known.

    Feeds can disagree, so the most severe designation wins.
    """
    if not injuries:
        return {}, {}
    if isinstance(injuries, Mapping):
        triples = [(None, name, status) for name, status in injuries.items()]
    else:
        triples = [
            (record.team, record.player_name, record.status) for record in injuries
        ]

    by_name: dict[str, list[str]] = {}
    by_team: dict[tuple[str, str], list[str]] = {}
    for team, name, status in triples:
        key = normalise_name(name)
        normalised = normalize_status(status)
        by_name.setdefault(key, []).append(normalised)
        if team:
            by_team.setdefault((str(team).upper(), key), []).append(normalised)

    def worst(values: list[str]) -> str:
        return min(values, key=lambda value: STATUS_ORDER.index(value))

    return (
        {name: worst(values) for name, values in by_name.items()},
        {key: worst(values) for key, values in by_team.items()},
    )


# ----------------------------------------------------------------------
# what it does to a projection
# ----------------------------------------------------------------------
def per_snap_projection(
    *,
    own_total: float,
    own_snaps: float,
    group_total: float,
    group_snaps: float,
    expected_snaps: float,
    shrinkage: float = SNAP_SHRINKAGE,
) -> float:
    """Production per snap, shrunk toward the position group, times the snaps.

    ``own_total`` and ``own_snaps`` are the player's weighted production and
    snaps over the window; ``group_total`` and ``group_snaps`` the same for
    their position group on that team. With no snaps of their own a player is
    priced entirely at the group's rate, which is the honest answer for a
    backup nobody has seen play.
    """
    group_rate = group_total / group_snaps if group_snaps > 0 else 0.0
    if own_snaps <= 0:
        return max(group_rate * expected_snaps, 0.0)

    own_rate = own_total / own_snaps
    weight = own_snaps / (own_snaps + shrinkage)
    rate = weight * own_rate + (1.0 - weight) * group_rate
    return max(rate * expected_snaps, 0.0)


#: A week is treated as played once this many teams have snaps recorded. Below
#: it the week is in progress: the Thursday game is in the release and the
#: Sunday slate is what we are pricing.
WEEK_COMPLETE_TEAMS = 28


def next_week(snap_counts: pd.DataFrame | None, season: int) -> int:
    """The week being priced, from how much of each week has been played.

    Snaps only exist for games already played, so the naive answer is the
    highest week present plus one. That is wrong mid-week: the Thursday night
    game lands in the release days before the Sunday slate, and taking its week
    as finished points the model at the week after the one it is pricing --
    which in turn finds no injury report and concludes nobody is hurt. So a
    week only counts as played once most of the league appears in it.
    """
    if snap_counts is None or snap_counts.empty or "season" not in snap_counts.columns:
        return 1
    played = snap_counts[snap_counts["season"].astype(int) == int(season)]
    if played.empty:
        return 1
    by_week = played.groupby(played["week"].astype(int))["team"].nunique()
    complete = by_week[by_week >= WEEK_COMPLETE_TEAMS]
    return int(complete.index.max()) + 1 if len(complete) else int(by_week.index.min())


def load_nfl_roles(
    seasons: Iterable[int], *, season: int, week: int | None = None
) -> RoleModel:
    """Snap shares plus the league injury report, for an NFL slate.

    Both come from nflverse, which means they are the same source the learning
    loop replays history from -- the model is trained on the inputs it runs on.

    The report for the week being priced may not exist yet: it is filed
    Wednesday to Friday, and the nflverse release lags that. Rather than
    concluding that nobody is injured, the most recent published week is used
    and how stale it is is recorded, so a run can say what it actually knew. A
    week-old report still has everyone on injured reserve right; what it misses
    is this week's new injuries, which is what the live ESPN feed is for.
    """
    try:
        import nflreadpy
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "nflreadpy is not installed; `uv pip install -e '.[stats]'`"
        ) from exc

    years = sorted(set(seasons))
    snaps = nflreadpy.load_snap_counts(seasons=years).to_pandas()
    target_week = week if week is not None else next_week(snaps, season)
    report_week, lag, injuries, carried_from = latest_report(years, season, target_week)

    logger.info(
        "roles: %d snap rows for %s week %s; injury report from week %s (%d week(s) stale), "
        "%d players out",
        len(snaps), season, target_week, report_week, lag,
        sum(1 for record in injuries if record.status == "OUT"),
    )
    try:
        depth = build_depth_chart(nflreadpy.load_depth_charts(seasons=[int(season)]).to_pandas())
    except Exception as exc:  # the chart is an improvement, not a requirement
        logger.warning("depth chart unavailable: %s", exc)
        depth = DepthChart()
    return build_role_model(
        snaps, injuries, season=season, week=target_week,
        report_week=report_week, report_lag=lag, depth=depth,
        carried_from=carried_from,
    )


#: A week's report is final once game designations exist for this many teams.
#: The Wednesday and Thursday reports carry practice participation only; the
#: Out / Doubtful / Questionable designations are published on Friday.
FINAL_REPORT_TEAMS = 16


def report_is_final(frame: pd.DataFrame, season: int, week: int) -> bool:
    """Has this week's report reached the game designations yet?"""
    rows = frame[(frame["season"].astype(int) == int(season)) & (frame["week"].astype(int) == int(week))]
    designated = rows[rows["report_status"].notna()]
    return designated["team"].nunique() >= FINAL_REPORT_TEAMS


def latest_report(seasons: Sequence[int], season: int, week: int):
    """The injury information for ``week``, and how it was arrived at.

    Returns ``(report_week, weeks_stale, records, carried_from)``.

    When the week's report is final, it is used as it stands. When it is not --
    any day before Friday -- it has no game designations at all, and taking it
    at face value concludes that nobody is hurt. That was the behaviour, and on
    the Wednesday of week 3 it had the model treating a quarterback ruled out
    for both of the first two weeks as available.

    So mid-week, the last final report is carried forward: everyone it ruled out
    becomes ``OUT_LAST_WEEK``, priced at the measured 63% availability rather
    than as certainly out or certainly fine. Doubtful and questionable tags are
    not carried -- they are about one game. ``carried_from`` names the week the
    designations came from, or is ``None`` when the current report was used.
    """
    from src.ingestion.injuries import InjuryRecord, parse_nflverse_injuries

    import nflreadpy

    frame = nflreadpy.load_injuries(seasons=sorted(set(seasons))).to_pandas()
    if frame.empty:
        return None, 0, [], None
    in_season = frame[
        (frame["season"].astype(int) == int(season)) & (frame["week"].astype(int) <= week)
    ]
    if in_season.empty:
        return None, 0, [], None
    this_season = frame[frame["season"].astype(int) == int(season)]

    report_week = int(in_season["week"].astype(int).max())
    current = parse_nflverse_injuries(this_season, week=report_week)
    if report_is_final(frame, season, report_week):
        return report_week, max(week - report_week, 0), current, None

    final_weeks = [
        wk for wk in sorted(in_season["week"].astype(int).unique(), reverse=True)
        if wk < report_week and report_is_final(frame, season, wk)
    ]
    if not final_weeks:
        return report_week, max(week - report_week, 0), current, None
    previous = final_weeks[0]
    carried = [
        InjuryRecord(
            sport=record.sport, team=record.team, player_name=record.player_name,
            position=record.position, status="OUT_LAST_WEEK", practice=record.practice,
            detail=f"out in week {previous}; this week's designation not yet published",
            source=record.source, report_date=record.report_date,
        )
        for record in parse_nflverse_injuries(this_season, week=previous)
        if record.status == "OUT"
    ]
    # Anything the partial report already designates wins over the carry-over.
    designated = [record for record in current if record.status != "ACTIVE"]
    return report_week, max(week - report_week, 0), carried + designated, previous
