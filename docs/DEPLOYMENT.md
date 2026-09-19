# Deploying the parlay crafter as a web app

Getting from `localhost:8000` to a URL you can open on your phone. The app is a
single FastAPI process with a SQLite file — small, but it holds paid odds data
and can spend API credits, so the order below puts the lock before the door.

**Tested, and not.** Everything in steps 1 and 6 was run and verified. The
Dockerfile, `fly.toml` and the hosting steps were written but not executed —
there is no Docker daemon in the environment they were authored in. Treat them
as a checked recipe, not a proven deploy.

---

## Step 0 — Decide two things

**Who can reach it?** This is a single-user tool. Every route is behind one
shared token; there are no accounts. If a second person needs their own access,
put an identity proxy (Cloudflare Access, Tailscale, oauth2-proxy) in front
rather than growing `src/web/auth.py`.

**Where does the data come from?** `--mock` serves cached fixtures and costs
nothing; `--live` pulls real odds and spends API credits on every refresh. Ship
with `--mock` first, confirm the deployment works, then switch.

---

## Step 1 — Generate an access token

```bash
python -m src.web.app --print-token
# k3Jq8s...  (32+ url-safe characters)
```

Put it in `.env` locally, and in your host's secret store when you deploy:

```bash
WEB_ACCESS_TOKEN=k3Jq8s...
```

What this buys you:

* `/api/*` returns 401 without a valid bearer token or session cookie;
* browser routes redirect to `/login`, which exchanges the token for a signed,
  HttpOnly cookie (the token itself is never stored in the browser);
* repeated bad logins from one address start returning 429;
* `/api/health` stays public so platform probes work, but returns only
  `{"status": "ok"}` until you authenticate.

The app **refuses to start** on a non-loopback interface without a token:

```
Refusing to serve on 0.0.0.0 without WEB_ACCESS_TOKEN.
```

That is deliberate. An open instance hands out a paid provider's odds and lets
any passer-by trigger `?refresh=true`, which spends your credits.

---

## Step 2 — Pick a hosting path

| | Path A — Tunnel to your own machine | Path B — Fly.io / Render | Path C — VPS + Docker |
|---|---|---|---|
| Cost | free | ~$0–5/mo | ~$5/mo |
| Setup | 10 min | 20 min | 45 min |
| Always on | only while your machine is | yes | yes |
| Best when | it is just you, on your own hardware | you want it up without your laptop | you already run a server |

**For one person, Path A is the honest recommendation.** The app is a personal
research tool; putting it on the public internet adds attack surface and hosting
cost to solve a problem a tunnel already solves.

### Path A — Tailscale (private network, nothing public)

```bash
# On the machine that will run it
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# Run bound to the Tailscale interface
WEB_ACCESS_TOKEN=... python -m src.web.app --host 0.0.0.0 --port 8000 --live --max-events 6
```

Install Tailscale on your phone, join the same tailnet, and open
`http://<machine-name>:8000`. Nothing is exposed to the internet; the token is a
second layer, not the only one.

`tailscale serve https / http://127.0.0.1:8000` adds TLS inside the tailnet if
you want the cookie marked `secure`.

**Cloudflare Tunnel** is the alternative if you want a real hostname:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

That prints a public `trycloudflare.com` URL — public, so the token is now the
only thing between a stranger and your odds data. For anything beyond a quick
test, use a named tunnel with Cloudflare Access in front.

### Path B — Fly.io

```bash
fly launch --no-deploy --name parlay-crafter        # uses the committed fly.toml
fly volumes create parlay_data --size 1 --region iad
fly secrets set WEB_ACCESS_TOKEN=... ODDS_API_KEY=... ANTHROPIC_API_KEY=...
fly deploy
fly open
```

`fly.toml` is committed with the volume mount, a `/api/health` check, forced
HTTPS, and `auto_stop_machines` so it sleeps when idle. Change the CMD to live
data by editing the `command` in the Dockerfile or overriding it in `fly.toml`.

Render or Railway work the same way: point them at the Dockerfile, attach a
persistent disk at `/data`, set the same environment variables.

### Path C — VPS with Docker Compose and Caddy

```bash
git clone <your-repo> && cd ParlayMaker
cp .env.example .env         # set WEB_ACCESS_TOKEN, ODDS_API_KEY
docker compose up -d
```

`docker-compose.yml` publishes to `127.0.0.1:8000` only. Terminate TLS in front
of it:

```caddyfile
parlay.example.com {
    reverse_proxy 127.0.0.1:8000
}
```

Caddy gets a certificate automatically. Do not publish the container port
directly — without TLS the token crosses the network in clear text.

---

## Step 3 — Set the secrets

| Variable | Needed for | If missing |
|---|---|---|
| `WEB_ACCESS_TOKEN` | all non-loopback serving | refuses to start |
| `ODDS_API_KEY` | live odds and props | falls back to mock mode |
| `ANTHROPIC_API_KEY` | the context reasoning layer | baseline projections only |
| `OPENWEATHER_API_KEY` | NFL stadium forecasts | weather treated as unknown |
| `DB_PATH` | where SQLite lives | `data/sports_data.db` |
| `MIN_REFRESH_SECONDS` | floor between live rebuilds | 300 |

Use the platform's secret store (`fly secrets set`, Render's environment tab),
not a committed `.env`. `.env` is already gitignored; keep it that way.

---

## Step 4 — Give SQLite a real disk

The database holds captured odds, the bet log and therefore your CLV history. On
a container platform the filesystem is ephemeral — without a mounted volume,
every restart silently starts from an empty database and the CLV benchmark can
never accumulate a sample.

* Fly: the `[[mounts]]` block in `fly.toml` (volume at `/data`).
* Compose: the `parlay-data` named volume.
* Render/Railway: attach a persistent disk mounted at `/data`.

Then set `DB_PATH=/data/sports_data.db`, as all three configs do.

Back it up — it is one file:

```bash
sqlite3 /data/sports_data.db ".backup '/data/backup-$(date +%F).db'"
```

---

## Step 5 — Refresh on a schedule

The app builds a slate on first request and caches it. To keep it current
without opening the app, have cron call it with the bearer token:

```bash
# Every 30 minutes on game days. --refresh spends credits; the server enforces
# MIN_REFRESH_SECONDS regardless of how often you call.
*/30 * * * * curl -fsS -H "Authorization: Bearer $WEB_ACCESS_TOKEN" \
  "https://parlay.example.com/api/slate/nfl?refresh=true" > /dev/null
```

Budget first. With the default markets a live refresh costs
`3 + games × 6` credits for NFL. Refreshing 13 games every 30 minutes for one
Sunday would be ~1,300 credits — more than double the free monthly tier. Either
cap the games (`--max-events 4`) or refresh a few times a day, not continuously.

Watch what it actually spent:

```sql
SELECT endpoint, last_cost, requests_remaining, captured_at
FROM api_quota_log ORDER BY id DESC LIMIT 20;
```

---

## Step 6 — Verify the deployment

```bash
# 1. Health is public and answers
curl -s https://parlay.example.com/api/health
# {"status":"ok"}

# 2. The API is actually locked
curl -s -o /dev/null -w '%{http_code}\n' https://parlay.example.com/api/slate/nfl
# 401

# 3. The token works
curl -s -H "Authorization: Bearer $WEB_ACCESS_TOKEN" \
  https://parlay.example.com/api/slate/nfl | head -c 120
# {"meta":{"sport":"nfl",...

# 4. The browser flow works
open https://parlay.example.com      # -> /login -> paste token -> the app
```

Before switching to `--live`, run the pre-flight on the host:

```bash
python scripts/check_live_access.py
```

It reports key validity, per-host reachability and remaining credits, and costs
nothing.

---

## Operating notes

**Run exactly one worker.** The slate cache lives in process memory and the leg
ids the browser holds resolve against it; SQLite also prefers a single writer.
Two uvicorn workers means a price request can land on a worker that never built
that slate and get a 409. If you outgrow one process, move the slate cache to
Redis and the database to Postgres — not before.

**Memory.** numpy, scipy, pandas and pulp in one process want ~400–600 MB
resident. 512 MB instances will OOM during a copula-heavy build; `fly.toml` asks
for 1 GB.

**Keep it private.** The Odds API's terms restrict redistributing their odds
data; a public page showing live prices is a different thing from a private tool
that reads them. This is also a betting tool, and what you may lawfully operate
or share varies by jurisdiction — a private single-user instance avoids the
question entirely.

**The mock path stays useful after deployment.** `--mock` exercises every code
path with zero credits, so it is the right way to test a deploy, a proxy config
or a cron entry before pointing any of it at live data.
