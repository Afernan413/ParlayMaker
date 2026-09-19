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

## Quick start

```bash
uv venv && uv pip install -e '.[dev]'          # add ',stats' for live stat feeds
cp .env.example .env                           # fill in your API keys

# End-to-end run on cached fixtures: no API keys, no credits spent
python run_pipeline.py --sport nfl --mock

# Live run, dispatching to your configured webhook
python run_pipeline.py --sport nba --mode live --legs 3
```

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

## Tests

```bash
python -m pytest            # 155 tests, no network access
```

No test makes an unmocked HTTP call: The Odds API, OpenWeather and the injury
feeds are mocked with `respx`, and the Anthropic client is always a stub. The
suite covers the quota floor, the ±20% clamp (including an adversarial agent
that asks for 5×), the correlation rejection rules, the de-vig maths, the ILP
constraints and two end-to-end mock runs.

## Layout

```
config/settings.py            thresholds, credentials, sport constants
config/bookmaker_keys.json    FanDuel market id -> stat/family/dispersion
src/ingestion/                db, odds_api, weather, injuries, mock
src/models/                   legs, baseline, distributions, correlation
src/reasoning/                prompts, context_agent
src/optimizer/                ev_calculator, leg_builder, parlay_builder
src/notifications/notifier.py Discord / Telegram / console cards
scripts/generate_mock_data.py fixture generator
run_pipeline.py               orchestration CLI
```

Two modules sit slightly outside the original design sketch: `src/models/legs.py`
holds the `Projection`/`Leg` vocabulary shared by every layer (which keeps the
imports acyclic), and `src/optimizer/leg_builder.py` joins stored prices to
projections, so `run_pipeline.py` stays orchestration only.

## Live runs need

* `ODDS_API_KEY` — FanDuel lines and props
* `ANTHROPIC_API_KEY` — the reasoning layer (optional; skipped without it)
* `OPENWEATHER_API_KEY` — NFL stadium forecasts (optional)
* `DISCORD_WEBHOOK_URL` or `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` — delivery
* `uv pip install -e '.[stats]'` — `nfl_data_py` / `nba_api` for live stat feeds
