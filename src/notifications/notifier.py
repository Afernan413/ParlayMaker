"""Rendering and dispatch of parlay cards.

Rendering is pure (dict/str in, dict/str out) so the card layout is testable;
dispatch is the only part that touches the network, and in dry-run mode it
never does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import httpx

from config.settings import settings
from src.models.legs import format_odds
from src.optimizer.parlay_builder import ParlayTicket

logger = logging.getLogger(__name__)

DISCORD_COLOR_POSITIVE = 0x2ECC71
DISCORD_COLOR_NEUTRAL = 0x95A5A6
MAX_DISCORD_EMBEDS = 10


@dataclass
class DispatchResult:
    """Where a card actually went."""

    channels: list[str]
    delivered: bool
    detail: str = ""


def _edge_line(ticket: ParlayTicket) -> str:
    return (
        f"Model {ticket.joint_probability:.1%} vs implied "
        f"{ticket.implied_probability:.1%} -> edge {ticket.edge:+.1%} "
        f"(EV {ticket.ev:+.1%} per unit)"
    )


def _legs_block(ticket: ParlayTicket) -> str:
    return "\n".join(f"- {leg.describe()}" for leg in ticket.legs)


def _correlation_line(ticket: ParlayTicket) -> str:
    if not ticket.is_sgp:
        return "Cross-game legs: priced as independent."
    return (
        f"Same-game correlation: avg r={ticket.average_correlation:.2f}, "
        f"weakest r={ticket.weakest_correlation:.2f}, "
        f"joint x{ticket.correlation_lift:.2f} vs independence"
    )


def ticket_rationale(ticket: ParlayTicket) -> list[str]:
    """Bullet points, including Claude's adjustments when present."""
    bullets = list(ticket.rationale())
    bullets.append(_correlation_line(ticket))
    return bullets


def render_discord_embed(ticket: ParlayTicket) -> dict[str, Any]:
    """One Discord embed per ticket."""
    return {
        "title": f"{ticket.ticket_type} {format_odds(ticket.american_odds)}",
        "color": DISCORD_COLOR_POSITIVE if ticket.ev > 0 else DISCORD_COLOR_NEUTRAL,
        "fields": [
            {"name": "Legs", "value": _legs_block(ticket), "inline": False},
            {"name": "Edge", "value": _edge_line(ticket), "inline": False},
            {
                "name": "Stake (quarter Kelly)",
                "value": (
                    f"{ticket.stake:.2f} to win {ticket.to_win:.2f} "
                    f"({ticket.kelly_share:.2%} of bankroll)"
                ),
                "inline": False,
            },
            {
                "name": "Why",
                "value": "\n".join(f"- {note}" for note in ticket_rationale(ticket)),
                "inline": False,
            },
        ],
    }


def render_discord_payload(
    tickets: Sequence[ParlayTicket], *, header: str | None = None
) -> dict[str, Any]:
    """Full Discord webhook body."""
    content = header or f"**{len(tickets)} qualifying ticket(s)**"
    return {
        "content": content,
        "embeds": [render_discord_embed(t) for t in tickets[:MAX_DISCORD_EMBEDS]],
    }


def render_telegram_message(
    tickets: Sequence[ParlayTicket], *, header: str | None = None
) -> str:
    """Telegram-flavoured markdown body."""
    if not tickets:
        return "*No qualifying tickets* -- every candidate failed the EV, price or correlation filters."
    blocks = [header or f"*{len(tickets)} qualifying ticket(s)*"]
    for ticket in tickets:
        bullets = "\n".join(f"  - {note}" for note in ticket_rationale(ticket))
        blocks.append(
            f"*{ticket.ticket_type}* `{format_odds(ticket.american_odds)}`\n"
            f"{_legs_block(ticket)}\n"
            f"{_edge_line(ticket)}\n"
            f"Stake {ticket.stake:.2f} to win {ticket.to_win:.2f}\n"
            f"{bullets}"
        )
    return "\n\n".join(blocks)


def render_console(
    tickets: Sequence[ParlayTicket], *, header: str | None = None
) -> str:
    """Plain-text card for the CLI / dry runs."""
    if not tickets:
        return "No qualifying tickets: every candidate failed the EV, price or correlation filters."
    lines = [header or f"{len(tickets)} qualifying ticket(s)", "=" * 72]
    for index, ticket in enumerate(tickets, start=1):
        lines.append(
            f"[{index}] {ticket.ticket_type}  {format_odds(ticket.american_odds)}"
        )
        for leg in ticket.legs:
            lines.append(
                f"      {leg.describe():<46} model {leg.p_model:6.1%} "
                f"implied {leg.p_implied:6.1%} EV {leg.ev:+6.1%}"
            )
        lines.append(f"      {_edge_line(ticket)}")
        lines.append(
            f"      Stake {ticket.stake:.2f} to win {ticket.to_win:.2f} "
            f"({ticket.kelly_share:.2%} of bankroll, quarter Kelly)"
        )
        for note in ticket_rationale(ticket):
            lines.append(f"      * {note}")
        lines.append("-" * 72)
    return "\n".join(lines)


class Notifier:
    """Dispatches a rendered card to Discord and/or Telegram."""

    def __init__(
        self,
        *,
        discord_webhook_url: str | None = None,
        telegram_bot_token: str | None = None,
        telegram_chat_id: str | None = None,
        client: httpx.AsyncClient | None = None,
        dry_run: bool = False,
    ) -> None:
        self.discord_webhook_url = (
            settings.discord_webhook_url
            if discord_webhook_url is None
            else discord_webhook_url
        )
        self.telegram_bot_token = (
            settings.telegram_bot_token if telegram_bot_token is None else telegram_bot_token
        )
        self.telegram_chat_id = (
            settings.telegram_chat_id if telegram_chat_id is None else telegram_chat_id
        )
        self.dry_run = dry_run
        self._owns_client = client is None
        self._client = client

    async def __aenter__(self) -> "Notifier":
        if self._client is None and not self.dry_run:
            self._client = httpx.AsyncClient(timeout=settings.http_timeout_seconds)
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=settings.http_timeout_seconds)
        return self._client

    @property
    def configured_channels(self) -> list[str]:
        channels: list[str] = []
        if self.discord_webhook_url:
            channels.append("discord")
        if self.telegram_bot_token and self.telegram_chat_id:
            channels.append("telegram")
        return channels

    async def send(
        self, tickets: Sequence[ParlayTicket], *, header: str | None = None
    ) -> DispatchResult:
        """Deliver a card. Dry runs and unconfigured channels print instead."""
        if self.dry_run:
            print(render_console(tickets, header=header))
            return DispatchResult(channels=["console"], delivered=True, detail="dry-run")

        channels = self.configured_channels
        if not channels:
            print(render_console(tickets, header=header))
            return DispatchResult(
                channels=["console"], delivered=True, detail="no webhook configured"
            )

        delivered: list[str] = []
        problems: list[str] = []
        if "discord" in channels:
            ok, detail = await self._post_discord(tickets, header)
            (delivered if ok else problems).append(detail)
        if "telegram" in channels:
            ok, detail = await self._post_telegram(tickets, header)
            (delivered if ok else problems).append(detail)

        return DispatchResult(
            channels=channels,
            delivered=bool(delivered),
            detail="; ".join(delivered + problems),
        )

    async def _post_discord(
        self, tickets: Sequence[ParlayTicket], header: str | None
    ) -> tuple[bool, str]:
        payload = render_discord_payload(tickets, header=header)
        try:
            response = await self.client.post(self.discord_webhook_url, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("discord dispatch failed: %s", exc)
            return False, f"discord failed: {exc}"
        return True, "discord ok"

    async def _post_telegram(
        self, tickets: Sequence[ParlayTicket], header: str | None
    ) -> tuple[bool, str]:
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": render_telegram_message(tickets, header=header),
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("telegram dispatch failed: %s", exc)
            return False, f"telegram failed: {exc}"
        return True, "telegram ok"


def card_as_dicts(tickets: Iterable[ParlayTicket]) -> list[dict[str, Any]]:
    """JSON-serialisable card (for logging or a future webhook consumer)."""
    return [ticket.as_dict() for ticket in tickets]
