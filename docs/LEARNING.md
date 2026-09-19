# Training the model on what actually happened

The projection recipe started as a set of reasonable guesses: a four-week
weighted average for the mean, and a dispersion per market picked by eye. This
is how those guesses get replaced with measurements, and how you check that the
replacement is actually better.

Everything lives in `src/learning/` and lands in one file, `data/calibration.json`,
which is committed. Loading that file corrects the whole pipeline at once --
including the page, because the static bundle exports the already-corrected
probability for every bet.

## The short version

```bash
# see what the model would learn, without changing anything
uv run python -m src.learning.train --sport nfl

# keep it, if the weeks it was not shown score better
uv run python -m src.learning.train --sport nfl --write
```

Nothing is written unless the holdout improves. `--force` overrides that; it is
there for when you know the data is fine and the guard is being pedantic.

## What is being measured

For every past player-week, the model is rebuilt using only the weeks before
it, asked for a projection, and compared with the box score. That walk-forward
replay is `src/learning/backtest.py`. The week being graded never reaches its
own projection, which is the only thing that makes the numbers mean anything.

Each projection is then probed at seven lines, from 55% to 170% of it. The wide
range is deliberate: a book's line sits near the player's *true* mean, and our
projection is a noisy estimate of that mean, so a real line routinely lands at
half or double what we projected. Those are exactly the legs a big-payout
parlay is built from, so they have to be measured rather than extrapolated
into.

## The three corrections

Fitted per market, in the order they compose, because each one changes what the
next has left to explain (`src/learning/calibrate.py`):

1. **Mean bias** -- `sum(actual) / sum(projected)`. Summing rather than
   averaging ratios weights by volume, so a 3-yard week for a backup cannot
   outvote a 300-yard week for a starter.
2. **Dispersion** -- matched to the residual spread, in whatever units the
   family's variance is defined in: a coefficient of variation for the
   log-normals, a variance multiple for the counts, a standard deviation in
   points for the normals.

   This includes the model's own uncertainty about the mean, not just the
   player's game-to-game variance -- which is correct. What a price needs is
   the spread of outcomes *given our projection*, and our projection is wrong
   by some amount too.

   A count market whose measured variance rejects the Poisson assumption
   (variance equals mean) is promoted to a Negative Binomial.
3. **Platt scaling** -- a logistic fit on the probability scale,
   `p' = sigmoid(a * logit(p) + b)`, mopping up what the first two leave: a
   wrong family shape, the zero mass in a receiving-yards line, the fact that a
   four-week average is a shrunk estimate rather than a true mean.

Game totals and spreads are fitted separately and need no projection model at
all: the closing line already is the market's mean, so the residual around it
is the spread the normal family should be given. `NFL_SCORE_SD` in
`src/models/baseline.py` is the prior this replaces.

Every fitted parameter is clipped to a guard rail (`MEAN_FACTOR_BOUNDS` and
friends in `src/models/calibration.py`). A fit outside those says something is
wrong with the data rather than with the prior.

## Why this matters for longshots

The first real run found the model systematically wrong in one direction: it
under-claimed low probabilities and over-claimed high ones. Claimed 7%, hit
21%. Claimed 84%, hit 61%. The distributions had tails that were too thin,
which means **longshot legs were underpriced by the model and favourites
overpriced** -- the precise failure mode that matters when the whole point is
turning a few dollars into a lot.

After fitting, the gap between claimed and observed collapsed from +0.14 / -0.23
to inside ±0.05, and the out-of-sample Brier score improved by about 5%. The
fitted dispersions were much wider than the priors across the board: receiving
yards 0.52 -> 0.75, rushing yards 0.45 -> 0.68, passing yards 0.27 -> 0.40.

## The forward loop

The replay can only measure the projection recipe. It cannot replay injuries,
weather or the reasoning layer's nudges, so it cannot measure the pipeline you
actually run.

So every leg the pipeline prices is written to the `projection_log` table when
it is priced -- every leg, not only the ones that made a card, so the training
set is not biased towards what the optimizer happened to like. Once the games
are played:

```bash
uv run python -m src.learning.journal --sport nfl    # grade what is due
```

Over a season that becomes a second training set, one that measures the whole
pipeline. `journal.graded_frame()` returns it in the shape the fits consume.

## Automation

`.github/workflows/train.yml` refits every Tuesday at 09:00 UTC, after Monday
night's box scores land. It commits `data/calibration.json` only when the
holdout says the fit is better, and the Pages workflow republishes on that file
changing, so the site starts pricing with it. The training report is written to
the workflow run summary.

## College football

College borrows the NFL fit for player markets: the market keys are the same
and there is far less college history to learn from. It never borrows the NFL
*score spread* -- a college final score swings far wider, so 13 points would be
worse than the prior. To fit college on its own data:

```bash
uv run python -m src.learning.train --sport ncaaf --seasons 2024 2025 --write
```

NBA cannot be trained from a hosted runner: `stats.nba.com` refuses datacenter
IPs. Run it from a machine on a residential connection.

## Reading the output

```
  before  brier 0.2251  log loss 0.6564  claimed 0.406 vs actual 0.380
  after   brier 0.2140  log loss 0.6179  claimed 0.404 vs actual 0.380
  brier improved by 4.91%
```

*Brier* is mean squared error on the probability -- lower is better, 0.25 is
what you get by guessing the base rate. *Log loss* punishes confident mistakes
much harder, so it is the one to watch if a fit makes the model bolder.
*Claimed vs actual* is the headline bias: the average probability the model
stated against how often those bets actually cashed.

The per-market table below it shows which markets improved, and the
calibration table shows claimed against observed by probability bucket, before
and after. A market whose `brier_after` is worse than its `brier_before` is
worth looking at by hand before trusting the fit.
