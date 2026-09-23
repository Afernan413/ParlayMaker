# What the model takes into account

Three contextual inputs beyond the box score: who is starting, who is hurt, and
what the weather is doing. Every one of them is optional in practice, and the
honest position differs by sport. This is that position, written down, because
the failure mode is a page that *implies* the model weighed something it never
saw.

The run summary and the page both carry a live version of this — `starters`,
`injuries`, `weather` with what each one actually covered — from
`src/models/inputs.py`. If it says `starters=none`, the model did not know.

## Who is on the team at all

`src/models/rosters.py`. NFL only.

The volume model learns from box scores, and box scores remember everyone who
ever played. Measured on the week-3 slate of 2026, **40% of the player rows the
model projected from were wrong**: a player on a team he no longer plays for
(22%), on injured reserve, retired or on the practice squad (8%), or on no
roster at all (10%). A.J. Brown was still being projected for Philadelphia; he
is on New England, on injured reserve.

The weekly injury report cannot catch any of that — players on injured reserve
are not listed on it, and retired or released players never are. The league's
weekly roster can, so it decides two things:

* **which team** a player is on. A traded player's projection moves with him,
  and he no longer has a second, ghost row for the team he left. That row was
  not harmless: when a player faced his old team, both rows claimed the same
  game and market and whichever came last won.
* **whether he can play.** Only the active roster takes the field.

Players are grouped by `player_id` rather than by name, because names collide
— 2025 had two Byron Youngs and two Jonah Williamses. College carries no ids, so
it keeps grouping by name and team.

## How much of a player's history to trust

`src/learning/turnover.py`, which anyone can re-run.

The obvious response to a league that turns over is to trust old numbers less:
discount last season, discount games for a team a player has left. Both were
tested, fitted on 2023-24 and scored on 2025-26, and **both made the model
worse** — last season at half weight raised the Brier score, at a quarter it
raised it more, and discounting a player's old team hurt most for exactly the
players who had changed teams. A four-game window is short enough that stale
games age out on their own, and early in a season last year's games are the
best evidence there is.

What did help was the opposite: **more history, not less**. Averaging eight
games instead of four cut the error on the same projections by 0.23 yards (11
standard errors) and the Brier score from 0.2075 to 0.2005. Ten games was no
better than eight within the noise, so eight — the shorter window goes stale
more slowly. Role changes, which a longer average would be slow to notice, are
the role model's job below, on its own four-game snap window.

The decays are still in the code (`HISTORY_SEASON_DECAY`, `HISTORY_TEAM_DECAY`)
at 1.0, so the question can be re-asked as seasons accumulate rather than
settled by opinion.

## Who is starting

`src/models/roles.py`. NFL only.

The volume model averages a player's last four games. That is right for a
settled role and wrong in the case that moves a line furthest: the player ahead
of them is out, and they are about to play three times the snaps their average
is built from.

So where the role has changed, volume is projected **per snap** instead:

    rate      = own production / own snaps, shrunk toward the same position
                group's rate on that team by how little the player has played
    projected = rate x the snaps we expect them to take

A third-string quarterback inherits nearly all of his team's quarterback
production rate, because he has no rate of his own worth trusting. Expected
snaps come from redistributing the snap share of everyone in the position group
who is out, in proportion to what the remaining players play when they play —
so a team still fields one quarterback, and the vacancy shows up as a promotion
rather than a five per cent nudge.

Two things this deliberately does **not** do:

* **A settled role keeps its per-game average.** Only a workload moving more
  than 15% earns the per-snap treatment. The average is both the better
  estimate for a starter and the quantity `data/calibration.json` was fitted
  on; re-deriving every projection per snap would pull them all a quarter of
  the way toward their position group and quietly put the model out of step
  with its own corrections. For the minority of players whose role *has*
  changed, the calibration is an approximation — the dispersion was fitted on
  the per-game recipe. Closing that means joining snap counts into the
  walk-forward replay, which is not done yet.
* **It does not trust snaps to name a quarterback.** The depth chart does
  that — see below.

It needs two games of the current season before it says anything. The window
does not reach back into last season — rosters change, and a share built from
two different squads is worse than none. So early in a season the role model is
off, and says so.

**College football and basketball have no snap source**, so they keep the
per-game average throughout.

### Quarterbacks: the depth chart

Snap history can only say who *has* been playing. After a benching, a starter's
return from injury or a trade, the depth chart is where the change shows first.
For quarterbacks it decides the slot:

* the **starter** is the highest-ranked quarterback on the latest chart who is
  not ruled out — the chart lists an injured starter at the top, so the report
  decides who is skipped;
* the **next man up** carries the starts the starter might miss: if the starter
  is 63% to play, the backup is priced for the other 37%;
* everyone else is a backup.

Every charted quarterback gets a share, including those with no snaps this
season. That mattered on the first real slate: Seattle's starter missed week 2
and his backup took 96% of the snaps, so snap history alone kept the backup as
the starter and the returning starter as a 4% reserve. In Atlanta, the next man
up had arrived from another team with no Atlanta snaps, and was being priced off
his old average as a second full-time starter alongside the first.

## Who is hurt

For the NFL this is the league's own weekly report, via
`nflreadpy.load_injuries()` — practice participation plus the Friday game
designation. It carries the team and the position, which matters twice over:

* the position group is what tells you a vacancy exists. An earlier rule bumped
  every receiver by 5% whenever *anyone* on the team was out, cornerbacks
  included; with an injury on every team every week that is a constant, not a
  signal.
* a name on its own is ambiguous. Two rosters can carry the same name, and
  applying one player's designation to the other rules out a healthy man.

Designations become multipliers: `OUT` removes the player, `DOUBTFUL` 0.25,
`QUESTIONABLE` 0.88, `PROBABLE` 0.97.

**The report fills in over the week.** Wednesday's and Thursday's reports
carry practice participation only; the Out / Doubtful / Questionable
designations are published on Friday. On the Wednesday of week 3 the report had
22 rows from one team and no designations at all, and taking it at face value
concluded nobody in the league was hurt.

So a report only counts as final once designations exist for half the league.
Until then, last week's final report is carried forward — and not as a verdict.
A player ruled out last week is `OUT_LAST_WEEK`, priced at **63% availability**,
because that is what the history says: of 3,413 players ruled out in a week of
the 2023-25 regular seasons, 33% were out again the next week, 2% doubtful, 17%
questionable and 48% off the report. Weighted by the usual multipliers that is
0.63 — and the same 0.63 whether a player had been out one week or several.
Treating him as out would be wrong two times in three; treating him as fine,
one time in three.

Availability does not depend on the snap window, so it applies from week one of
a season even when role changes do not.

**College football**: the NCAA mandates no injury report. The Big Ten and SEC
publish availability reports; many programmes publish nothing. The ESPN
college-football feed is wired up and attempted, and whatever it returns is
used — but coverage is patchy by conference, and the run says how many
designations it actually got rather than pretending.

**Basketball**: the ESPN NBA feed, which works from a normal connection.

## Weather

`src/ingestion/weather.py`. NFL only, and only outdoors. The provider is
**AccuWeather**.

The key is read from `OPENWEATHER_API_KEY`. That name is historical — the
deployed secret is called that, and renaming it would mean re-adding it
everywhere — but the value is an AccuWeather key. Without it the forecast is
never fetched, which used to happen silently, so a build with no key looked
identical to one that had considered the weather and found it mild. It is now
reported.

Three things about AccuWeather shape the client:

* **A venue is addressed by an opaque location key**, not by coordinates, and
  looking one up costs a call. The keys are stable, so they are cached in the
  `weather_locations` table: 30-odd rows resolved once turn a
  two-call-per-venue forecast into one.
* **The hourly forecast only reaches twelve hours out.** A Saturday build for a
  Sunday afternoon kickoff is outside it, so the daily forecast covers anything
  further away and the hourly one is used when kickoff is close, being sharper.
  A daily reading says `(daily outlook)` in its conditions string, so nothing
  mistakes the day's high for the temperature at kickoff.
* **The free tier allows 50 calls a day** and answers `503` once spent. That is
  a clean stop rather than an outage: the slate stops asking, keeps what it has,
  and the rest goes unforecast. A plain `503` with no quota message is treated
  as a real outage and still tried per game.

Two further bounds, because the first live build showed why they were needed:

* **The location-key cache does not survive a hosted build.** It lives in
  SQLite, and the Pages runner starts from an empty database every time, so each
  run pays two calls per venue rather than one. A 29-game slate is 58 calls —
  past the day's whole allowance in a single build, and at a 20-second timeout it
  stalled the build for a quarter of an hour. Seeding the keys as a committed
  file would fix it properly; until then the budget below is what holds.
* **Each run has a call budget** (`WEATHER_CALL_BUDGET`, 24 by default) and its
  own shorter timeout (8 seconds, not the 20 used elsewhere — a forecast is not
  worth waiting that long for). Running out of budget is the same clean stop as
  running out of allowance: keep what was fetched, leave the rest unforecast.

Weather is also only fetched for the games props were fetched for. Weather
reaches a price through the reasoning layer's player-market adjustments, so a
game with no props has nothing for a forecast to move — with `--max-events 4`
that is 4 venues rather than 29.

Kickoff times are compared in local terms, not UTC: an 8:20pm Eastern game is
01:20 the next day in UTC, and matching UTC dates would ask for tomorrow's
forecast for tonight's game.

Retractable roofs and domes are marked in the stadium table and skipped: there
is no weather story indoors, and it saves quota.

**College football is a real gap.** The venue coordinates the forecast needs
exist for the 32 NFL stadiums only. There are 130-odd FBS venues and no
reachable table of their coordinates, and inventing them from memory would be
worse than admitting the gap. College weather therefore reports as
unavailable.

**Basketball is indoors**, so "not applicable" is the right answer there rather
than a gap.

## What the weather and injuries then do

The reasoning layer (`src/reasoning/`) turns the context into bounded shifts —
at most ±20% on a projection, above a confidence threshold, and only for a
player and market that were in the payload. The shifts are deliberately
**residual-sized**: the book has already moved its own line for the wind and
the inactives, so what is left to capture is the remainder, not the whole
effect. Oversized factors here would manufacture edges that are not there.

With no `ANTHROPIC_API_KEY` this degrades to `RuleBasedContextAgent`, which
applies the same shape of adjustment deterministically.
