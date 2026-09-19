"""Card rendering and dispatch tests."""

from __future__ import annotations

import httpx
import pytest
import respx

from src.models.legs import Leg
from src.notifications.notifier import (
    Notifier,
    card_as_dicts,
    render_console,
    render_discord_payload,
    render_telegram_message,
    ticket_rationale,
)
from src.optimizer.parlay_builder import price_ticket


@pytest.fixture
def ticket():
    qb = Leg(game_id="g1", sport="nfl", market="player_pass_yds", selection="Over",
             american_odds=-110, line=274.5, player_name="QB One", team="KC",
             p_model=0.60, p_implied=0.52,
             rationale=["Wind sustained at 22mph; volume downgraded (x0.93)"])
    wr = Leg(game_id="g1", sport="nfl", market="player_reception_yds", selection="Over",
             american_odds=-105, line=64.5, player_name="WR One", team="KC",
             p_model=0.60, p_implied=0.52)
    return price_ticket([qb, wr], iterations=4_000, seed=3, bankroll=1_000)


def test_console_card_shows_price_edge_and_stake(ticket):
    text = render_console([ticket])
    assert "Correlated NFL SGP" in text
    assert "QB One Over 274.5 (-110)" in text
    assert "Stake" in text and "quarter Kelly" in text
    assert "Wind sustained at 22mph" in text
    assert "Same-game correlation" in text


def test_empty_card_is_explicit():
    assert "No qualifying tickets" in render_console([])
    assert "No qualifying tickets" in render_telegram_message([])


def test_discord_payload_shape(ticket):
    payload = render_discord_payload([ticket], header="NFL card")
    assert payload["content"] == "NFL card"
    embed = payload["embeds"][0]
    assert embed["title"].startswith("2-Leg Correlated NFL SGP +")
    names = [field["name"] for field in embed["fields"]]
    assert names == ["Legs", "Edge", "Stake (quarter Kelly)", "Why"]
    assert "Model" in embed["fields"][1]["value"]


def test_discord_payload_caps_embeds(ticket):
    payload = render_discord_payload([ticket] * 15)
    assert len(payload["embeds"]) == 10


def test_telegram_message_is_markdown(ticket):
    text = render_telegram_message([ticket])
    assert text.startswith("*1 qualifying ticket(s)*")
    assert "`+" in text  # odds in a code span
    assert "- QB One Over 274.5 (-110)" in text


def test_rationale_includes_the_correlation_note(ticket):
    bullets = ticket_rationale(ticket)
    assert any("Wind sustained" in bullet for bullet in bullets)
    assert any("correlation" in bullet for bullet in bullets)


def test_card_as_dicts_is_serialisable(ticket):
    import json

    payload = card_as_dicts([ticket])
    assert json.loads(json.dumps(payload))[0]["ticket_type"] == ticket.ticket_type


async def test_dry_run_prints_and_sends_nothing(ticket, capsys):
    async with Notifier(dry_run=True) as notifier:
        result = await notifier.send([ticket])
    assert result.channels == ["console"] and result.delivered
    assert "Correlated NFL SGP" in capsys.readouterr().out


async def test_unconfigured_channels_fall_back_to_console(ticket, capsys):
    async with Notifier(discord_webhook_url="", telegram_bot_token="",
                        telegram_chat_id="") as notifier:
        result = await notifier.send([ticket])
    assert result.channels == ["console"]
    assert "no webhook configured" in result.detail
    assert capsys.readouterr().out


@respx.mock
async def test_discord_dispatch_posts_the_payload(ticket):
    route = respx.post("https://discord.test/hook").mock(
        return_value=httpx.Response(204)
    )
    async with Notifier(
        discord_webhook_url="https://discord.test/hook",
        telegram_bot_token="", telegram_chat_id="",
    ) as notifier:
        result = await notifier.send([ticket], header="NFL card")

    assert route.called and result.delivered
    body = respx.calls[0].request.content.decode()
    assert "NFL card" in body and "embeds" in body


@respx.mock
async def test_telegram_dispatch_posts_markdown(ticket):
    route = respx.post("https://api.telegram.org/bot42:abc/sendMessage").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    async with Notifier(
        discord_webhook_url="", telegram_bot_token="42:abc", telegram_chat_id="99",
    ) as notifier:
        result = await notifier.send([ticket])

    assert route.called and result.delivered
    assert "telegram ok" in result.detail


@respx.mock
async def test_dispatch_failure_is_reported_not_raised(ticket):
    respx.post("https://discord.test/hook").mock(return_value=httpx.Response(500))
    async with Notifier(
        discord_webhook_url="https://discord.test/hook",
        telegram_bot_token="", telegram_chat_id="",
    ) as notifier:
        result = await notifier.send([ticket])
    assert result.delivered is False
    assert "discord failed" in result.detail


def test_configured_channels_reflect_credentials():
    both = Notifier(discord_webhook_url="x", telegram_bot_token="t", telegram_chat_id="c")
    assert both.configured_channels == ["discord", "telegram"]
    partial = Notifier(discord_webhook_url="", telegram_bot_token="t", telegram_chat_id="")
    assert partial.configured_channels == []
