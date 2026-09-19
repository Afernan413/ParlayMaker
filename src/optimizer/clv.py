"""Closing line value benchmarking.

The de-vig plus projection stack claims to know a market's fair probability
better than the posted price does. The honest test of that claim is closing line
value: did the price we recommended beat the market's own final price?

Every recommended leg is written to ``bet_log`` with the price taken. Later
captures of the same market give the closing price; both are de-vigged and the
difference in fair probability is the CLV. Positive average CLV means the
model's edges are real signal rather than vig-shaped noise.
"""

from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from src.ingestion import db
from src.optimizer.ev_calculator import breakeven_probability, devig_selection


def new_run_id() -> str:
    """Identifier tying one card's legs together in ``bet_log``."""
    return uuid.uuid4().hex[:12]


def log_recommendations(
    tickets: Sequence[Any],
    *,
    sport: str,
    run_id: str | None = None,
    captured_at: str | None = None,
    db_path=None,
) -> str:
    """Persist a card's legs with the prices they were recommended at.

    ``captured_at`` defaults to now; pass it to backfill a historical card.
    """
    run_id = run_id or new_run_id()
    rows: list[dict[str, Any]] = []
    for index, ticket in enumerate(tickets, start=1):
        for leg in ticket.legs:
            rows.append(
                {
                    "run_id": run_id,
                    "sport": sport,
                    "ticket_index": index,
                    "ticket_type": ticket.ticket_type,
                    "ticket_odds": ticket.american_odds,
                    "game_id": leg.game_id,
                    "market": leg.market,
                    "player_name": leg.player_name,
                    "selection": leg.selection,
                    "line": leg.line,
                    "taken_odds": leg.american_odds,
                    "p_model": leg.p_model,
                    "p_implied": leg.p_implied,
                    "ev": leg.ev,
                    "stake": ticket.stake,
                    "captured_at": captured_at or db.utcnow(),
                }
            )
    db.insert_bets(rows, db_path=db_path)
    return run_id


@dataclass(frozen=True)
class ClvEntry:
    """One logged bet compared with its closing price."""

    run_id: str
    market: str
    subject: str
    selection: str
    line: float | None
    taken_odds: int
    closing_odds: int
    taken_fair: float
    closing_fair: float
    p_model: float | None

    @property
    def clv(self) -> float:
        """Fair-probability gain from the price we took."""
        return self.closing_fair - self.taken_fair

    @property
    def beat_close(self) -> bool:
        return self.clv > 0


@dataclass(frozen=True)
class ClvSummary:
    """Aggregate view over a set of :class:`ClvEntry`."""

    logged: int
    priced: int
    mean_clv: float
    median_clv: float
    beat_rate: float
    mean_model_edge: float

    @property
    def non_negative_drift(self) -> bool:
        """The verification criterion: average CLV is not negative."""
        return self.mean_clv >= 0.0


def _closing_rows(
    entry_row: dict[str, Any], db_path=None
) -> list[dict[str, Any]]:
    """Latest capture of a logged bet's market, taken strictly after the bet.

    Comparing against the same capture the bet was made from would measure
    nothing, so a market with no later capture yields no closing price.
    """
    is_prop = bool(entry_row.get("player_name"))
    table = "fanduel_props" if is_prop else "fanduel_lines"
    conditions = [
        "game_id = ?",
        "market = ?",
        "IFNULL(line, -999) = IFNULL(?, -999)",
        "captured_at > ?",
    ]
    params: list[Any] = [
        entry_row["game_id"],
        entry_row["market"],
        entry_row["line"],
        entry_row["captured_at"],
    ]
    if is_prop:
        conditions.append("player_name = ?")
        params.append(entry_row["player_name"])

    where = " AND ".join(conditions)
    latest = db.fetch_all(
        f"SELECT MAX(captured_at) AS ts FROM {table} WHERE {where}",
        params,
        db_path=db_path,
    )
    if not latest or not latest[0]["ts"]:
        return []
    return db.fetch_all(
        f"SELECT * FROM {table} WHERE {where} AND captured_at = ?",
        [*params, latest[0]["ts"]],
        db_path=db_path,
    )


def build_entries(
    *, sport: str | None = None, run_id: str | None = None, db_path=None
) -> list[ClvEntry]:
    """Compare every logged bet against the closing price of its market.

    Bets whose market never got a later capture are skipped -- there is no close
    to compare with yet.
    """
    conditions, params = [], []
    if sport:
        conditions.append("sport = ?")
        params.append(sport)
    if run_id:
        conditions.append("run_id = ?")
        params.append(run_id)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    logged = db.fetch_all(
        f"SELECT * FROM bet_log {where} ORDER BY id", params, db_path=db_path
    )

    entries: list[ClvEntry] = []
    for row in logged:
        closing = _closing_rows(row, db_path=db_path)
        if not closing:
            continue
        closing_price = next(
            (
                r["american_odds"]
                for r in closing
                if str(r["selection"]).lower() == str(row["selection"]).lower()
            ),
            None,
        )
        if closing_price is None:
            continue
        closing_fair = devig_selection(closing, row["selection"])
        if closing_fair is None:
            closing_fair = breakeven_probability(closing_price)
        taken_fair = row["p_implied"]
        if taken_fair is None:
            taken_fair = breakeven_probability(row["taken_odds"])

        entries.append(
            ClvEntry(
                run_id=row["run_id"],
                market=row["market"],
                subject=row["player_name"] or row["selection"],
                selection=row["selection"],
                line=row["line"],
                taken_odds=int(row["taken_odds"]),
                closing_odds=int(closing_price),
                taken_fair=float(taken_fair),
                closing_fair=float(closing_fair),
                p_model=row["p_model"],
            )
        )
    return entries


def summarise(entries: Sequence[ClvEntry], logged: int | None = None) -> ClvSummary:
    """Aggregate CLV statistics, including the non-negative-drift check."""
    if not entries:
        return ClvSummary(
            logged=logged or 0, priced=0, mean_clv=0.0, median_clv=0.0,
            beat_rate=0.0, mean_model_edge=0.0,
        )
    clvs = [entry.clv for entry in entries]
    edges = [
        entry.p_model - entry.taken_fair
        for entry in entries
        if entry.p_model is not None
    ]
    return ClvSummary(
        logged=logged if logged is not None else len(entries),
        priced=len(entries),
        mean_clv=float(statistics.fmean(clvs)),
        median_clv=float(statistics.median(clvs)),
        beat_rate=sum(entry.beat_close for entry in entries) / len(entries),
        mean_model_edge=float(statistics.fmean(edges)) if edges else 0.0,
    )


def report(
    *, sport: str | None = None, run_id: str | None = None, db_path=None
) -> tuple[list[ClvEntry], ClvSummary]:
    """Entries plus summary for the requested slice of ``bet_log``."""
    entries = build_entries(sport=sport, run_id=run_id, db_path=db_path)
    logged = db.fetch_all(
        "SELECT COUNT(*) c FROM bet_log"
        + (" WHERE sport = ?" if sport else ""),
        [sport] if sport else [],
        db_path=db_path,
    )[0]["c"]
    return entries, summarise(entries, logged=logged)


def render(entries: Iterable[ClvEntry], summary: ClvSummary) -> str:
    """Text CLV report for the CLI."""
    lines = ["Closing line value report", "=" * 72]
    for entry in entries:
        line = "" if entry.line is None else f" {entry.line:g}"
        lines.append(
            f"  {entry.subject} {entry.selection}{line:<8} "
            f"took {entry.taken_odds:>5} ({entry.taken_fair:.1%}) -> "
            f"close {entry.closing_odds:>5} ({entry.closing_fair:.1%})  "
            f"CLV {entry.clv:+.2%}"
        )
    if not summary.priced:
        lines.append(
            "  no logged bet has a later capture yet -- re-run ingestion closer "
            "to kickoff to build a closing sample"
        )
    lines += [
        "-" * 72,
        f"  logged bets:      {summary.logged}",
        f"  priced vs close:  {summary.priced}",
        f"  mean CLV:         {summary.mean_clv:+.2%}",
        f"  median CLV:       {summary.median_clv:+.2%}",
        f"  beat the close:   {summary.beat_rate:.0%}",
        f"  mean model edge:  {summary.mean_model_edge:+.2%}",
        f"  non-negative drift: {'PASS' if summary.non_negative_drift else 'FAIL'}",
    ]
    return "\n".join(lines)
