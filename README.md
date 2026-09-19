# FanDuel Parlay Engine

An automated research pipeline for NFL and NBA parlays priced at FanDuel. It
pulls lines and player props, builds its own statistical projections, lets Claude
adjust those projections for late-breaking context inside hard bounds, prices
correlated same-game combinations through a Gaussian copula, and recommends only
the tickets that clear an expected-value floor.

> This is a modelling and research tool. It produces recommendations from public
> odds and public stats; it does not place bets, and nothing here is a promise of
> profit. Stake sizing is fractional Kelly on a bankroll **you** configure.

## How it works

```
ingest -> project -> reason -> price -> optimise -> notify
```

| Stage | Module | What it does |
| --- | --- | --- |
| Ingest | `src/ingestion/` | FanDuel odds + props (The Odds API), stadium weather, injury/inactive reports, all into SQLite with API-quota accounting |
| Project | `src/models/baseline.py` | 4-week recency-weighted volume, EPA/success-rate efficiency (NFL), per-minute rates scaled by pace and matchup (NBA) |
| Fit | `src/models/distributions.py` | Poisson / Negative-Binomial / Log-Normal / Normal fits, converted to a fair probability against each posted line |
| Reason | `src/reasoning/` | One Claude call per game returns bounded projection shifts for weather, inactives and scheme notes |
| Correlate | `src/models/correlation.py` | Latent correlation priors + Gaussian copula for joint same-game probabilities |
| Price | `src/optimizer/ev_calculator.py` | De-vig FanDuel's two-way markets, compute EV, size stakes at quarter Kelly |
| Optimise | `src/optimizer/parlay_builder.py` | Integer program that maximises portfolio EV under leg-count, price, correlation and diversification constraints |
| Notify | `src/notifications/notifier.py` | Discord embeds / Telegram markdown / console cards |
| Benchmark | `src/optimizer/clv.py` | Logs every recommendation, then scores its price against the closing line |
| Craft | `src/web/` | Browser UI: pick legs, see the price, the edge and the payout update live |

### The guardrails that matter

* **Quota safety.** Every Odds API response's quota headers are logged, and the
  client refuses to spend a request once fewer than 50 remain — mid-slate it
  stops cleanly instead of failing, keeping whatever it already wrote.
* **Bounded reasoning.** Claude can only scale a projection it was given, by at
  most ±20%, and only above a confidence threshold. Adjustments naming a player
  or market that was not in the payload are discarded. No key, a failed call or
  an unparseable response all degrade to the baseline projections.
* **Correlation, not optimism.** Two legs from the same game are only combined
  when their prior correlation is ≥ 0.25. Negatively correlated pairs (Under the
  total with Over on passing TDs) can never form a ticket.
* **Compounding discipline.** 2–4 legs per ticket, single legs between -140 and
  +130, tickets between +200 and +650, minimum 4% EV per leg, and a player or
  team may appear on only one ticket in a card.

## The fastest way to use it

A standalone build that needs no server, no keys and no Python at run time:

```bash
python scripts/build_static.py     # writes site/
open site/index.html               # or: xdg-open / just double-click it
```

That page carries a whole slate — game predictions, every priced bet, and the
correlations between them — and does the parlay maths in the browser, so it
works offline and on a phone. `site/` is committed with a sample slate, so a
fresh clone opens and works immediately.

To put it on the web, enable **Settings → Pages → Source: GitHub Actions**. The
committed workflow rebuilds and publishes it on a schedule; add an `ODDS_API_KEY`
repository secret and it switches from the sample slate to real odds by itself.

Two things that only bite on a hosted runner:

* **NBA cannot be built there.** `stats.nba.com` refuses datacenter IPs, so
  `nba_api` times out on GitHub's runners. The workflow builds NFL only; run
  `python scripts/build_static.py` locally for an NBA slate.
* **NFL stats come from `nflreadpy`**, not `nfl_data_py`. The latter still
  installs but 404s on every season after 2024.

| | Static build (`site/`) | Server app (`src/web/`) |
| --- | --- | --- |
| Needs a running process | no | yes |
| Refresh odds | rebuild (CLI or Action) | `?refresh=true` in the app |
| Runs on GitHub Pages | yes | no (Pages is static-only) |
| Pricing runs | in your browser | in Python |

## Quick start

```bash
uv venv && uv pip install -e '.[dev,web]'      # add ',stats' for live stat feeds
cp .env.example .env                           # fill in your API keys

# The UI, on cached fixtures: no API keys, no credits spent
python -m src.web.app --mock                   # http://127.0.0.1:8000

# Same pipeline from the command line
python run_pipeline.py --sport nfl --mock

# Live run, dispatching to your configured webhook
python run_pipeline.py --sport nba --mode live --legs 3
```

## The web UI

```bash
python -m src.web.app --mock              # cached fixtures (default)
python -m src.web.app --live --port 8080  # live odds, weather, injuries, stats
```

Left pane lists every priced bet on the slate — model probability against the
de-vigged market probability, with the EV the engine computes for each side.
Click to add a leg to the slip; the slip prices the whole combination through
the copula and shows:

* **what you collect** for the stake you type, with profit and the payout at
  other stakes;
* **expected value in dollars** next to the price, so a fat payout with a
  negative edge is obvious;
* **the suggested stake** at quarter Kelly against the bankroll you set;
* **model vs market probability** side by side, with the correlation lift
  spelled out — a same-game slip is not the legs multiplied, and the panel says
  by how much;
* **advisories** whenever a slip is outside a house rule: five legs, a leg
  priced past +130, a same-game pair below the 0.25 correlation floor (with a
  note when the two legs actively work against each other);
* **why the model disagrees**, quoting the context adjustments Claude applied.

"Build a card" hands the same slate to the ILP optimizer and loads any ticket it
returns straight into the slip.

Every number in the browser comes from the Python engine over
`/api/price` — the front end only scales a priced slip by the stake, because
payout, profit and EV are linear in it. The API is documented at `/docs`.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/config` | thresholds the UI displays |
| `GET /api/slate/{sport}?refresh=` | games and every priced leg |
| `POST /api/price` | price a slip: odds, probability, edge, payout, staking |
| `POST /api/build` | the optimizer's own card |
| `GET /api/health` | cached slates and data source |

### CLI

```
--sport nfl|nba          slate to build
--mode dry-run|live      dry-run prints the card; live dispatches to webhooks
--legs 2|3|4             pin every ticket to this many legs (default: 2-4)
--mock                   use cached fixtures instead of live APIs
--bankroll FLOAT         override the staking bankroll
--max-tickets INT        cap the card size
--iterations INT         copula iterations per ticket (default 10,000)
--no-props               game lines only (saves Odds API credits)
--no-game-markets        player props only
--json                   print the run summary as JSON
--clv-report             score logged recommendations against closing lines
--no-bet-log             do not record recommended legs
--db PATH                use a different SQLite file
```

A mock NFL dry run completes in about a second and prints a card like:

```
[2] 2-Leg Correlated NFL SGP  +282
      Ellis Ward Under 229.5 (-133)      model  60.3% implied  54.9% EV  +5.7%
      Owen Falk Under 34.5 (+118)        model  54.7% implied  43.5% EV +19.2%
      Model 41.5% vs implied 26.2% -> edge +15.3% (EV +58.4% per unit)
      Stake 51.80 to win 146.03 (5.18% of bankroll, quarter Kelly)
      * Ellis Ward: sustained wind in forecast (...) (x0.93)
      * Same-game correlation: avg r=0.55, weakest r=0.55, joint x1.26
```

## Deploying it

`docs/DEPLOYMENT.md` is the step-by-step guide: lock it with a token, pick a
hosting path (Tailscale tunnel, Fly.io, or a VPS with Docker Compose), mount a
volume for SQLite, schedule refreshes inside your credit budget, and verify.

The short version:

```bash
python -m src.web.app --print-token     # -> WEB_ACCESS_TOKEN
docker compose up -d                    # or: fly deploy
```

Without `WEB_ACCESS_TOKEN` the app refuses to bind to anything but loopback —
an open instance would hand out a paid provider's odds and let strangers spend
your API credits with `?refresh=true`.

## Configuration

All thresholds live in `config/settings.py` and can be overridden through the
environment or `.env` (see `.env.example`): the quota floor, EV floor,
correlation floor, odds bands, leg counts, Kelly fraction, bankroll, the ±20%
reasoning ceiling and the copula iteration count.

`config/bookmaker_keys.json` maps FanDuel market ids to canonical stats, their
distribution family and a dispersion prior — add a market there and the whole
pipeline picks it up.

## Data

SQLite (WAL mode) at `data/sports_data.db`, created on demand:

| Table | Contents |
| --- | --- |
| `games` | schedule, home/away, kickoff |
| `fanduel_lines` | moneyline, spread, total prices per capture |
| `fanduel_props` | player prop prices per capture |
| `weather_snapshots` | temperature, wind, precipitation, dome/wind/freeze flags |
| `injury_reports` | normalised availability (OUT → ACTIVE) with practice notes |
| `api_quota_log` | requests used/remaining per endpoint call |
| `bet_log` | every recommended leg with the price it was taken at |

Every DDL statement is `CREATE TABLE IF NOT EXISTS`; schema changes belong in
`MIGRATIONS` in `src/ingestion/db.py` as further idempotent statements.

### Mock fixtures

`data/mock/<sport>/` holds cached payloads in the same shapes the live providers
return, so `--mock` exercises the real parsers. Prices there are derived from the
model's own probabilities plus a deliberate few-point bias and a 4.5% overround,
which is what gives a dry run realistic edges to find. Regenerate with:

```bash
python scripts/generate_mock_data.py
```

Player and team names in the fixtures are fictional.

## Closing line value

Each card's legs are written to `bet_log` with the price they were recommended
at. Re-run ingestion closer to kickoff and the same markets get a later capture;
`--clv-report` then de-vigs both ends and reports the fair-probability gain:

```bash
python run_pipeline.py --sport nfl                 # logs the card
python run_pipeline.py --sport nfl --clv-report    # after a later ingestion
```

```
  WR One Over 64.5     took  +120 (44.5%) -> close  -115 (51.8%)  CLV +7.30%
  mean CLV:         +7.30%
  beat the close:   100%
  non-negative drift: PASS
```

A bet with no capture later than its own is not scored -- comparing a price with
itself would measure nothing. Mean CLV is the honest test of whether the
de-vigged probabilities carry signal; if it sits below zero, the projections are
losing to the market regardless of what the model's EV column claims.

## Tests

```bash
python -m pytest            # 182 tests, no network access
```

No test makes an unmocked HTTP call: The Odds API, OpenWeather and the injury
feeds are mocked with `respx`, and the Anthropic client is always a stub. The
suite covers the quota floor, the ±20% clamp (including an adversarial agent
that asks for 5×), the correlation rejection rules, the de-vig maths, the ILP
constraints and two end-to-end mock runs.

| Verification criterion | Proven by |
| --- | --- |
| Ingestion is fully mocked | `tests/test_ingestion.py` (21 tests, `respx`) |
| Quota use is tracked and halts a slate | `test_quota_floor_blocks_further_requests`, `test_ingest_slate_halts_props_when_quota_drops` |
| TD props reconcile to team totals; no negative yardage | `test_td_rates_are_reconciled_with_the_team_total`, `test_continuous_samples_are_never_negative` |
| Claude cannot move a projection by more than ±20% | `test_extreme_agent_output_cannot_move_a_projection_more_than_20_percent` |
| Negatively correlated legs are rejected | `test_negatively_correlated_same_game_pair_is_rejected` |
| De-vigged probabilities are benchmarked against CLV | `tests/test_clv.py` plus `--clv-report` |
| Dry run completes well under 45s and renders a card | `test_dry_run_produces_a_valid_card` (~1-2s per sport) |
| The UI's numbers match the engine's | `tests/test_web.py` (payout, EV, Kelly and re-priced optimizer tickets) |

## Layout

```
config/settings.py            thresholds, credentials, sport constants
config/bookmaker_keys.json    FanDuel market id -> stat/family/dispersion
src/ingestion/                db, odds_api, weather, injuries, mock
src/models/                   legs, baseline, distributions, correlation
src/reasoning/                prompts, context_agent
src/optimizer/                ev_calculator, leg_builder, parlay_builder, clv
src/notifications/notifier.py Discord / Telegram / console cards
src/web/                      FastAPI app, service layer and the browser UI
scripts/generate_mock_data.py fixture generator
run_pipeline.py               orchestration CLI
```

Two modules sit slightly outside the original design sketch: `src/models/legs.py`
holds the `Projection`/`Leg` vocabulary shared by every layer (which keeps the
imports acyclic), and `src/optimizer/leg_builder.py` joins stored prices to
projections, so `run_pipeline.py` stays orchestration only.

## Going live

### 1. Get an Odds API key

Sign up at **[the-odds-api.com](https://the-odds-api.com)** — the "Get API Key"
form takes an email address and sends the key back. The free tier is 500 credits
a month, which is enough to work with if you spend them deliberately.

### 2. Understand what a run costs

The Odds API bills **one credit per market per region**, so with the default
config:

| Call | Markets | Credits |
| --- | --- | --- |
| Game lines for the whole slate | h2h, spreads, totals | 3 |
| Player props, **per game** (NFL) | 6 markets | 6 |
| Player props, **per game** (NBA) | 4 markets | 4 |

A 13-game NFL Sunday with props is therefore about `3 + 13 × 6 = 81` credits —
roughly six full slates a month on the free tier. Controls:

```bash
python run_pipeline.py --sport nfl --max-events 4   # props for 4 games only
python run_pipeline.py --sport nfl --no-props       # game lines only: 3 credits
```

The engine logs the estimate before it spends anything, refuses to start a
request once fewer than 50 credits remain, and records what each call actually
cost (`x-requests-last`) in `api_quota_log`:

```sql
SELECT endpoint, last_cost, requests_remaining, captured_at
FROM api_quota_log ORDER BY id DESC LIMIT 10;
```

Trust that table over the estimate above — it is what the API charged.

### 3. Where the key goes

| How you run it | Where the key goes |
| --- | --- |
| Anything local (CLI, static build, local server) | `.env` in the repo root |
| GitHub Actions / Pages | repo **Settings → Secrets and variables → Actions → New repository secret**, named `ODDS_API_KEY` |
| Fly.io | `fly secrets set ODDS_API_KEY=...` |
| Render / Railway | the service's Environment tab |
| Docker Compose | `.env` in the repo root (`env_file: .env` picks it up) |

Locally:

```bash
cp .env.example .env
# then edit the one line:
ODDS_API_KEY=your_key_here
```

`.env` is gitignored, is read from the project root no matter which directory
you run from, and is overridden by a real environment variable if you set one.
Settings are read once at import, so restart anything already running after you
edit it.

### 4. Check the path before spending anything

```bash
python scripts/check_live_access.py
```

It verifies the keys and that every host the live path needs is reachable, and
reports the credits left on your account. It costs nothing: The Odds API's
`/v4/sports` endpoint is free. A host that comes back `unreachable` while others
pass is an egress/network policy, not a bad key — a sandboxed or corporate
network will block the data providers no matter what the key says.

### 5. Run it

```bash
uv pip install -e '.[stats]'                        # live stat feeds
python run_pipeline.py --sport nfl --max-events 4   # CLI
python -m src.web.app --live --max-events 4         # UI on live data
```

`--live` falls back to mock mode if `ODDS_API_KEY` is missing, so it never fails
halfway through a run for a missing key.

## Live runs need

* `ODDS_API_KEY` — FanDuel lines and props
* `ANTHROPIC_API_KEY` — the reasoning layer (optional; skipped without it)
* `OPENWEATHER_API_KEY` — NFL stadium forecasts (optional)
* `DISCORD_WEBHOOK_URL` or `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` — delivery
* `uv pip install -e '.[stats]'` — `nfl_data_py` / `nba_api` for live stat feeds
* `uv pip install -e '.[web]'` — FastAPI + uvicorn for the browser UI
