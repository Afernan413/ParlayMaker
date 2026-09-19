#!/usr/bin/env python3
"""Pre-flight check for a live run.

Verifies that the keys are present and that every host the live path needs is
actually reachable, *before* a run spends API credits on requests that cannot
leave the machine.

    python scripts/check_live_access.py

The Odds API's ``/v4/sports`` endpoint is free -- it does not count against the
monthly allowance -- so this check costs nothing while still proving the key
works and reporting how many credits are left.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402

OK = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


@dataclass
class Check:
    """One pre-flight result."""

    name: str
    status: str
    detail: str
    required: bool = True

    @property
    def blocking(self) -> bool:
        return self.required and self.status == FAIL


def mask(secret: str) -> str:
    if not secret:
        return "(unset)"
    return f"{secret[:4]}…{secret[-4:]}" if len(secret) > 10 else "(set)"


async def check_odds_api(client: httpx.AsyncClient, api_key: str) -> list[Check]:
    """Key presence, host reachability and remaining credits (costs nothing)."""
    if not api_key:
        return [
            Check(
                "ODDS_API_KEY",
                FAIL,
                "not set -- get one at https://the-odds-api.com and put it in .env",
            )
        ]

    checks = [Check("ODDS_API_KEY", OK, mask(api_key))]
    try:
        response = await client.get(
            f"{settings.odds_api_base_url}/v4/sports",
            params={"apiKey": api_key},
            timeout=20.0,
        )
    except httpx.HTTPError as exc:
        checks.append(
            Check(
                "api.the-odds-api.com",
                FAIL,
                f"unreachable: {type(exc).__name__} -- {exc}",
            )
        )
        return checks

    if response.status_code == 401:
        checks.append(Check("api.the-odds-api.com", FAIL, "key rejected (401)"))
        return checks
    if response.status_code != 200:
        checks.append(
            Check("api.the-odds-api.com", FAIL, f"HTTP {response.status_code}")
        )
        return checks

    remaining = response.headers.get("x-requests-remaining", "?")
    used = response.headers.get("x-requests-used", "?")
    sports = [row.get("key") for row in response.json()]
    checks.append(
        Check(
            "api.the-odds-api.com",
            OK,
            f"{remaining} credits remaining ({used} used); "
            f"{len(sports)} sports listed; this check was free",
        )
    )
    for sport, key in (("NFL", "americanfootball_nfl"), ("NBA", "basketball_nba")):
        in_season = key in sports
        checks.append(
            Check(
                f"{sport} slate available",
                OK if in_season else SKIP,
                "in season" if in_season else "not currently listed (off-season)",
                required=False,
            )
        )
    return checks


async def check_injury_feed(client: httpx.AsyncClient) -> Check:
    try:
        response = await client.get(settings.nfl_injury_feed_url, timeout=20.0)
    except httpx.HTTPError as exc:
        return Check("site.api.espn.com", FAIL, f"unreachable: {type(exc).__name__}")
    return Check(
        "site.api.espn.com",
        OK if response.status_code == 200 else FAIL,
        f"HTTP {response.status_code} (injury/inactive feed, no key needed)",
    )


async def check_weather(client: httpx.AsyncClient, api_key: str) -> Check:
    if not api_key:
        return Check(
            "OPENWEATHER_API_KEY",
            SKIP,
            "not set -- NFL runs proceed without a forecast",
            required=False,
        )
    try:
        response = await client.get(
            f"{settings.openweather_base_url}/data/2.5/forecast",
            params={"lat": 42.7738, "lon": -78.7870, "units": "imperial", "appid": api_key},
            timeout=20.0,
        )
    except httpx.HTTPError as exc:
        return Check(
            "api.openweathermap.org", FAIL, f"unreachable: {type(exc).__name__}",
            required=False,
        )
    return Check(
        "api.openweathermap.org",
        OK if response.status_code == 200 else FAIL,
        f"HTTP {response.status_code}",
        required=False,
    )


def check_anthropic(api_key: str) -> Check:
    """Presence only -- calling the model here would cost tokens for nothing."""
    return Check(
        "ANTHROPIC_API_KEY",
        OK if api_key else SKIP,
        mask(api_key) if api_key else "not set -- runs fall back to baseline projections",
        required=False,
    )


def check_stats_libraries() -> Check:
    missing = []
    for module in ("nfl_data_py", "nba_api"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        return Check(
            "stats libraries",
            FAIL,
            f"missing {', '.join(missing)} -- run: uv pip install -e '.[stats]'",
        )
    return Check("stats libraries", OK, "nfl_data_py and nba_api importable")


async def run_checks(
    *,
    odds_key: str | None = None,
    weather_key: str | None = None,
    anthropic_key: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> list[Check]:
    """Every pre-flight check. Pass ``client`` to test without a network."""
    odds_key = settings.odds_api_key if odds_key is None else odds_key
    weather_key = settings.openweather_api_key if weather_key is None else weather_key
    anthropic_key = settings.anthropic_api_key if anthropic_key is None else anthropic_key

    owns_client = client is None
    client = client or httpx.AsyncClient(follow_redirects=True)
    try:
        checks = await check_odds_api(client, odds_key)
        checks.append(await check_injury_feed(client))
        checks.append(await check_weather(client, weather_key))
    finally:
        if owns_client:
            await client.aclose()

    checks.append(check_anthropic(anthropic_key))
    checks.append(check_stats_libraries())
    return checks


def render(checks: list[Check]) -> str:
    width = max(len(check.name) for check in checks) + 2
    lines = ["Live-run pre-flight", "=" * 72]
    for check in checks:
        lines.append(f"  [{check.status}] {check.name:<{width}} {check.detail}")
    lines.append("-" * 72)

    blocking = [check for check in checks if check.blocking]
    if blocking:
        lines.append("  NOT READY. Fix these first:")
        for check in blocking:
            lines.append(f"    - {check.name}: {check.detail}")
        if any("unreachable" in check.detail for check in blocking):
            lines.append(
                "    A host that is unreachable while others work is usually an "
                "egress/network policy, not your key."
            )
    else:
        lines.append(
            "  READY. Start small:  python run_pipeline.py --sport nfl --max-events 2"
        )
    return "\n".join(lines)


def main() -> int:
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:
        print(f"(outbound HTTPS goes through {proxy})\n")
    checks = asyncio.run(run_checks())
    print(render(checks))
    return 1 if any(check.blocking for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
