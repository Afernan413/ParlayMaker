"""Pre-flight check tests. Every probe is mocked."""

from __future__ import annotations

import httpx
import pytest
import respx

from scripts import check_live_access
from scripts.check_live_access import FAIL, OK, SKIP, Check, render, run_checks

ODDS = "https://api.the-odds-api.com/v4/sports"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
WEATHER = "https://dataservice.accuweather.com/locations/v1/cities/geoposition/search"


def by_name(checks) -> dict:
    return {check.name: check for check in checks}


def mock_odds_ok(remaining: str = "412"):
    return respx.get(ODDS).mock(
        return_value=httpx.Response(
            200,
            json=[{"key": "americanfootball_nfl"}, {"key": "basketball_nba"}],
            headers={"x-requests-remaining": remaining, "x-requests-used": "88"},
        )
    )


@respx.mock
async def test_everything_reachable_reports_ready(monkeypatch):
    # The optional `stats` extra is not installed everywhere (CI runs without
    # it), and this test is about reachability, not packaging.
    monkeypatch.setattr(
        check_live_access, "check_stats_libraries",
        lambda: Check("stats libraries", OK, "stubbed for this test"),
    )
    mock_odds_ok()
    respx.get(ESPN).mock(return_value=httpx.Response(200, json={"injuries": []}))
    respx.get(WEATHER).mock(return_value=httpx.Response(200, json={"list": []}))

    checks = await run_checks(
        odds_key="secret-key-value", weather_key="weather-key", anthropic_key="sk-ant-xyz"
    )
    results = by_name(checks)

    assert results["ODDS_API_KEY"].status == OK
    assert "secret-key-value" not in results["ODDS_API_KEY"].detail  # masked
    assert results["api.the-odds-api.com"].status == OK
    assert "412 credits remaining" in results["api.the-odds-api.com"].detail
    assert "free" in results["api.the-odds-api.com"].detail
    assert results["site.api.espn.com"].status == OK
    assert results["dataservice.accuweather.com"].status == OK
    assert not any(check.blocking for check in checks)
    assert "READY" in render(checks)


@respx.mock
async def test_missing_key_blocks_without_touching_the_network():
    route = respx.get(ODDS)
    respx.get(ESPN).mock(return_value=httpx.Response(200, json={}))

    checks = await run_checks(odds_key="", weather_key="", anthropic_key="")
    results = by_name(checks)

    assert not route.called
    assert results["ODDS_API_KEY"].status == FAIL
    assert "the-odds-api.com" in results["ODDS_API_KEY"].detail
    assert results["ODDS_API_KEY"].blocking


@respx.mock
async def test_rejected_key_is_named_as_such():
    respx.get(ODDS).mock(return_value=httpx.Response(401, text="unauthorized"))
    respx.get(ESPN).mock(return_value=httpx.Response(200, json={}))

    checks = await run_checks(odds_key="bad-key", weather_key="", anthropic_key="")
    assert by_name(checks)["api.the-odds-api.com"].detail == "key rejected (401)"


@respx.mock
async def test_blocked_host_is_reported_as_unreachable():
    mock_odds_ok()
    respx.get(ESPN).mock(side_effect=httpx.ProxyError("403 from egress proxy"))

    checks = await run_checks(odds_key="key", weather_key="", anthropic_key="")
    espn = by_name(checks)["site.api.espn.com"]
    assert espn.status == FAIL and "unreachable" in espn.detail
    report = render(checks)
    assert "NOT READY" in report
    assert "egress/network policy" in report


@respx.mock
async def test_optional_services_only_skip():
    mock_odds_ok()
    respx.get(ESPN).mock(return_value=httpx.Response(200, json={}))

    checks = await run_checks(odds_key="key", weather_key="", anthropic_key="")
    results = by_name(checks)
    assert results["OPENWEATHER_API_KEY"].status == SKIP
    assert results["ANTHROPIC_API_KEY"].status == SKIP
    assert not results["OPENWEATHER_API_KEY"].blocking
    assert not results["ANTHROPIC_API_KEY"].blocking
    assert "READY" in render(checks)


@respx.mock
async def test_off_season_sport_is_flagged_without_blocking():
    respx.get(ODDS).mock(
        return_value=httpx.Response(
            200, json=[{"key": "americanfootball_nfl"}],
            headers={"x-requests-remaining": "500"},
        )
    )
    respx.get(ESPN).mock(return_value=httpx.Response(200, json={}))

    results = by_name(await run_checks(odds_key="key", weather_key="", anthropic_key=""))
    assert results["NFL slate available"].status == OK
    assert results["NBA slate available"].status == SKIP
    assert not results["NBA slate available"].blocking


@respx.mock
async def test_stats_libraries_are_reported_either_way():
    mock_odds_ok()
    respx.get(ESPN).mock(return_value=httpx.Response(200, json={}))
    check = by_name(await run_checks(odds_key="key", weather_key="", anthropic_key=""))[
        "stats libraries"
    ]
    assert check.status in {OK, FAIL}
    if check.status == FAIL:
        assert "[stats]" in check.detail
