"""Access-control tests for the deployed app."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.web import app as web_app
from src.web.app import create_app
from src.web.auth import (
    COOKIE_NAME,
    LoginThrottle,
    bearer_token,
    generate_token,
    issue_session,
    loopback_only,
    verify_session,
)

TOKEN = "test-access-token"


@pytest.fixture
def client(tmp_path) -> TestClient:
    app = create_app(db_path=str(tmp_path / "auth.db"), use_mock=True, access_token=TOKEN)
    with TestClient(app, follow_redirects=False) as test_client:
        yield test_client


@pytest.fixture
def open_client(tmp_path) -> TestClient:
    app = create_app(db_path=str(tmp_path / "open.db"), use_mock=True, access_token="")
    with TestClient(app) as test_client:
        yield test_client


# ------------------------------------------------------------------ crypto
def test_session_round_trip():
    cookie = issue_session(TOKEN, ttl_seconds=3600)
    assert verify_session(cookie, TOKEN)
    assert not verify_session(cookie, "a-different-token")


def test_expired_session_is_rejected():
    assert not verify_session(issue_session(TOKEN, ttl_seconds=-1), TOKEN)


@pytest.mark.parametrize(
    "cookie", ["", None, "garbage", "not-a-number.abc", "9999999999.deadbeef", "9999999999."]
)
def test_malformed_cookies_are_rejected(cookie):
    assert not verify_session(cookie, TOKEN)


def test_cookie_never_contains_the_token():
    assert TOKEN not in issue_session(TOKEN, ttl_seconds=3600)


def test_generated_tokens_are_unique_and_long():
    first, second = generate_token(), generate_token()
    assert first != second and len(first) >= 32


@pytest.mark.parametrize(
    ("header", "expected"),
    [("Bearer abc", "abc"), ("bearer abc", "abc"), ("Basic abc", None), (None, None), ("Bearer", None)],
)
def test_bearer_parsing(header, expected):
    assert bearer_token(header) == expected


def test_login_throttle_opens_and_resets():
    throttle = LoginThrottle(window_seconds=60, max_attempts=3)
    for _ in range(3):
        assert not throttle.blocked("1.2.3.4", now=1_000)
        throttle.record_failure("1.2.3.4", now=1_000)
    assert throttle.blocked("1.2.3.4", now=1_000)
    assert not throttle.blocked("5.6.7.8", now=1_000)   # per client
    assert not throttle.blocked("1.2.3.4", now=1_100)   # window rolls off
    throttle.record_failure("1.2.3.4", now=1_100)
    throttle.reset("1.2.3.4")
    assert not throttle.blocked("1.2.3.4", now=1_100)


# -------------------------------------------------------------------- gate
def test_api_requires_authentication(client: TestClient):
    for path in ("/api/config", "/api/slate/nfl"):
        response = client.get(path)
        assert response.status_code == 401
        assert response.json()["code"] == "unauthorized"
    assert client.post("/api/price", json={"sport": "nfl", "leg_ids": []}).status_code == 401
    assert client.post("/api/build", json={"sport": "nfl"}).status_code == 401


def test_browser_paths_redirect_to_login(client: TestClient):
    response = client.get("/")
    assert response.status_code == 307
    assert response.headers["location"] == "/login"


def test_login_page_and_its_stylesheet_stay_public(client: TestClient):
    assert client.get("/login").status_code == 200
    assert client.get("/styles.css").status_code == 200


def test_health_is_public_but_withholds_detail(client: TestClient):
    body = client.get("/api/health").json()
    assert body == {"status": "ok"}  # no slate or data-source detail


def test_login_sets_a_session_and_unlocks_the_app(client: TestClient):
    assert client.post("/api/login", json={"token": "wrong"}).status_code == 401
    response = client.post("/api/login", json={"token": TOKEN})
    assert response.status_code == 200
    assert COOKIE_NAME in response.cookies

    assert client.get("/api/config").status_code == 200
    assert client.get("/").status_code == 200
    assert "slates" in client.get("/api/health").json()

    client.post("/api/logout")
    assert client.get("/api/config").status_code == 401


def test_bearer_token_works_without_a_cookie(tmp_path):
    app = create_app(db_path=str(tmp_path / "b.db"), use_mock=True, access_token=TOKEN)
    with TestClient(app) as client:
        assert client.get("/api/config", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
        assert client.get("/api/config", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_repeated_bad_logins_are_throttled(client: TestClient):
    codes = [
        client.post("/api/login", json={"token": f"guess-{index}"}).status_code
        for index in range(10)
    ]
    assert codes[0] == 401
    assert 429 in codes, "brute-force attempts should start being refused"


def test_open_instance_needs_no_token(open_client: TestClient):
    assert open_client.get("/api/config").status_code == 200
    assert open_client.get("/").status_code == 200


# ------------------------------------------------------------------- serve
def test_public_bind_without_a_token_is_refused(monkeypatch, capsys):
    monkeypatch.setattr(web_app.settings, "web_access_token", "")
    code = web_app.main(["--host", "0.0.0.0", "--mock"])
    assert code == 2
    assert "Refusing to serve" in capsys.readouterr().err


def test_print_token_exits_without_serving(capsys):
    assert web_app.main(["--print-token"]) == 0
    assert len(capsys.readouterr().out.strip()) >= 32


def test_loopback_detection():
    assert loopback_only("127.0.0.1") and loopback_only("localhost")
    assert not loopback_only("0.0.0.0") and not loopback_only("10.0.0.5")


# ---------------------------------------------------------- refresh floor
async def test_refresh_is_rate_limited(tmp_path):
    from src.web.service import SlateStore

    store = SlateStore(db_path=str(tmp_path / "r.db"), use_mock=True, min_refresh_seconds=300)
    first = await store.get("nfl")
    assert store.refresh_allowed("nfl") is False        # just built
    second = await store.get("nfl", refresh=True)
    assert second.built_at == first.built_at            # served from cache, no rebuild
