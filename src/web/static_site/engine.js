/* Parlay pricing in the browser.
 *
 * The Python engine already did the hard parts: projections, distribution fits,
 * de-vigging, the reasoning layer, and every pairwise correlation. What is left
 * here is arithmetic plus one Monte-Carlo draw, so this file has no model
 * opinions of its own -- change a correlation prior in Python and this picks it
 * up on the next build.
 *
 * Every function is pure and exported for the parity test that checks these
 * numbers against the Python implementation.
 */

(function (global) {
  "use strict";

  const DEFAULT_ITERATIONS = 10000;

  // ---------------------------------------------------------------- odds
  function americanToDecimal(american) {
    const value = Number(american);
    if (!value) throw new Error("american odds cannot be 0");
    return 1 + (value > 0 ? value / 100 : 100 / Math.abs(value));
  }

  function decimalToAmerican(decimal) {
    if (decimal <= 1) throw new Error("decimal odds must exceed 1.0");
    return decimal >= 2
      ? Math.round((decimal - 1) * 100)
      : Math.round(-100 / (decimal - 1));
  }

  function formatOdds(american) {
    const value = Math.round(american);
    return value > 0 ? `+${value}` : String(value);
  }

  function parlayDecimal(legs) {
    return legs.reduce((product, leg) => product * americanToDecimal(leg.odds), 1);
  }

  // ------------------------------------------------------- EV and staking
  function expectedValue(pTrue, decimal) {
    return pTrue * (decimal - 1) - (1 - pTrue);
  }

  function kellyShare(pTrue, decimal, fraction) {
    if (decimal <= 1) return 0;
    const full = (pTrue * decimal - 1) / (decimal - 1);
    return Math.max(full, 0) * fraction;
  }

  // -------------------------------------------------------------- random
  /** Deterministic PRNG, so the same slip always prices to the same number. */
  function mulberry32(seed) {
    let state = seed >>> 0;
    return function () {
      state = (state + 0x6d2b79f5) >>> 0;
      let t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /** Box-Muller: two independent standard normals per call. */
  function normalPair(random) {
    let u1 = random();
    const u2 = random();
    if (u1 < 1e-12) u1 = 1e-12;
    const radius = Math.sqrt(-2 * Math.log(u1));
    const angle = 2 * Math.PI * u2;
    return [radius * Math.cos(angle), radius * Math.sin(angle)];
  }

  /** Acklam's inverse normal CDF; max abs error ~1.15e-9. */
  function normalQuantile(p) {
    if (p <= 0) return -Infinity;
    if (p >= 1) return Infinity;
    const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
               1.383577518672690e2, -3.066479806614716e1, 2.506628277459239];
    const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
               6.680131188771972e1, -1.328068155288572e1];
    const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
               -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
               3.754408661907416];
    const low = 0.02425;
    let q, r;

    if (p < low) {
      q = Math.sqrt(-2 * Math.log(p));
      return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
             ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    if (p > 1 - low) {
      q = Math.sqrt(-2 * Math.log(1 - p));
      return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
              ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    q = p - 0.5;
    r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }

  // --------------------------------------------------------- correlation
  /**
   * Build the correlation matrix for these legs.
   *
   * ``lookup(legA, legB)`` is supplied by the caller and returns the
   * correlation the Python engine exported for that pair, so this file holds
   * no priors of its own.
   */
  function correlationMatrix(legs, lookup) {
    const size = legs.length;
    const matrix = [];
    for (let i = 0; i < size; i += 1) {
      matrix.push(new Array(size).fill(0));
      matrix[i][i] = 1;
    }
    for (let i = 0; i < size; i += 1) {
      for (let j = i + 1; j < size; j += 1) {
        const rho = lookup(legs[i], legs[j]);
        matrix[i][j] = rho;
        matrix[j][i] = rho;
      }
    }
    return matrix;
  }

  /** Cholesky factor, or null when the matrix is not positive definite. */
  function cholesky(matrix) {
    const size = matrix.length;
    const lower = matrix.map(() => new Array(size).fill(0));
    for (let i = 0; i < size; i += 1) {
      for (let j = 0; j <= i; j += 1) {
        let sum = matrix[i][j];
        for (let k = 0; k < j; k += 1) sum -= lower[i][k] * lower[j][k];
        if (i === j) {
          if (sum <= 1e-12) return null;
          lower[i][j] = Math.sqrt(sum);
        } else {
          lower[i][j] = sum / lower[j][j];
        }
      }
    }
    return lower;
  }

  /**
   * Shrink the off-diagonals toward zero until the matrix factorises.
   *
   * Python repairs an inconsistent correlation matrix by clipping eigenvalues;
   * shrinkage is the simpler equivalent here and lands in the same place for
   * the small matrices a 2-4 leg ticket produces. The shrink factor is
   * reported so the page can say the ticket needed one.
   */
  function repairedCholesky(matrix) {
    let factor = 1;
    for (let attempt = 0; attempt < 24; attempt += 1) {
      const scaled = matrix.map((row, i) =>
        row.map((value, j) => (i === j ? 1 : value * factor))
      );
      const lower = cholesky(scaled);
      if (lower) return { lower, shrink: factor };
      factor *= 0.9;
    }
    return { lower: matrix.map((_, i) => matrix.map((__, j) => (i === j ? 1 : 0))), shrink: 0 };
  }

  // -------------------------------------------------------------- copula
  /**
   * Joint probability that every leg cashes, via a Gaussian copula.
   *
   * A leg hits when its correlated latent draw falls below the quantile of its
   * own model probability, so the marginals come out exactly as the engine
   * priced them and only the dependency structure is added. Draws are
   * antithetic (each sample mirrored about the origin), which removes most of
   * the sampling error in those marginals.
   */
  function jointProbability(legs, lookup, options) {
    const settings = options || {};
    const iterations = settings.iterations || DEFAULT_ITERATIONS;
    const size = legs.length;
    if (size === 0) return { probability: 0, independent: 0, marginals: [], shrink: 1 };
    if (size === 1) {
      return {
        probability: legs[0].p_model,
        independent: legs[0].p_model,
        marginals: [legs[0].p_model],
        shrink: 1,
      };
    }

    const matrix = correlationMatrix(legs, lookup);
    const { lower, shrink } = repairedCholesky(matrix);
    const thresholds = legs.map((leg) =>
      normalQuantile(Math.min(Math.max(leg.p_model, 1e-9), 1 - 1e-9))
    );

    const random = mulberry32(settings.seed || 1337);
    const half = Math.ceil(iterations / 2);
    const raw = new Array(size);
    const latent = new Array(size);
    const hits = new Array(size).fill(0);
    let together = 0;
    let drawn = 0;

    for (let sample = 0; sample < half; sample += 1) {
      for (let i = 0; i < size; i += 2) {
        const pair = normalPair(random);
        raw[i] = pair[0];
        if (i + 1 < size) raw[i + 1] = pair[1];
      }
      // The sample and its mirror image.
      for (const sign of [1, -1]) {
        if (drawn >= iterations) break;
        let all = true;
        for (let i = 0; i < size; i += 1) {
          let value = 0;
          for (let k = 0; k <= i; k += 1) value += lower[i][k] * sign * raw[k];
          latent[i] = value;
        }
        for (let i = 0; i < size; i += 1) {
          if (latent[i] <= thresholds[i]) hits[i] += 1;
          else all = false;
        }
        if (all) together += 1;
        drawn += 1;
      }
    }

    const independent = legs.reduce((product, leg) => product * leg.p_model, 1);
    return {
      probability: together / drawn,
      independent,
      marginals: hits.map((count) => count / drawn),
      shrink,
    };
  }

  // ------------------------------------------------------------ weeks
  /**
   * Which weeks a set of legs settles on.
   *
   * A parlay has to resolve together. The odds feed returns every upcoming
   * event, so one slate routinely holds this Sunday's games and next
   * Thursday's; a ticket built across both would sit unresolved for nine days
   * with half of it priced off a week-old projection. The week key is computed
   * in Python (src/models/schedule.py) and travels on every leg, so the
   * browser never has to re-derive it.
   */
  function weeksOf(legs) {
    return [...new Set(legs.map((leg) => leg.week).filter(Boolean))].sort();
  }

  /** Legs belonging to one week; a blank week means "no filter". */
  function inWeek(legs, week) {
    if (!week) return legs.slice();
    return legs.filter((leg) => !leg.week || leg.week === week);
  }

  // ---------------------------------------------------------- slip price
  /** Price a slip: odds, probability, edge, payout and staking advice. */
  function priceSlip(legs, lookup, options) {
    const settings = options || {};
    const stake = Math.max(Number(settings.stake) || 0, 0);
    const bankroll = Number(settings.bankroll) || 0;
    const kellyFraction = settings.kellyFraction ?? 0.25;

    if (legs.length === 0) return { priceable: false, legs: [] };

    const decimal = parlayDecimal(legs);
    const american = decimalToAmerican(decimal);
    const simulation = jointProbability(legs, lookup, settings);
    const probability = simulation.probability;
    const implied = 1 / decimal;
    const evPerUnit = expectedValue(probability, decimal);
    const share = kellyShare(probability, decimal, kellyFraction);

    return {
      priceable: true,
      legs: legs.map((leg) => leg.id),
      legCount: legs.length,
      decimal,
      american,
      oddsDisplay: formatOdds(american),
      probability,
      implied,
      edge: probability - implied,
      independent: simulation.independent,
      correlationLift: simulation.independent > 0 ? probability / simulation.independent : 0,
      fairOdds: probability > 0 ? decimalToAmerican(1 / probability) : null,
      shrink: simulation.shrink,
      stake,
      profit: stake * (decimal - 1),
      payout: stake * decimal,
      evPerUnit,
      evDollars: evPerUnit * stake,
      kellyShare: share,
      recommendedStake: share * bankroll,
      isSameGame: new Set(legs.map((leg) => leg.game_id)).size === 1,
      weeks: weeksOf(legs),
      singleWeek: weeksOf(legs).length <= 1,
    };
  }

  // ------------------------------------------------------- house rules
  /** Report which engine guardrails a slip is outside. Advice, not a block. */
  function reviewSlip(legs, lookup, rules) {
    const advisories = [];
    const seen = new Set();

    if (legs.length < rules.min_legs) {
      advisories.push({
        level: "info",
        code: "too_few_legs",
        message: `Add at least ${rules.min_legs} bets to price a parlay.`,
      });
    }
    if (legs.length > rules.max_legs) {
      advisories.push({
        level: "warn",
        code: "too_many_legs",
        message: `${legs.length} legs is past the ${rules.max_legs}-leg ceiling; ` +
          "the house edge compounds faster than the payout.",
      });
    }
    if (weeksOf(legs).length > 1) {
      advisories.push({
        level: "block",
        code: "mixed_weeks",
        message: "These bets are from different weeks, so they cannot settle " +
          "together. Keep one slip to one week.",
      });
    }
    for (const leg of legs) {
      if (leg.odds < rules.leg_odds_min || leg.odds > rules.leg_odds_max) {
        advisories.push({
          level: "warn",
          code: "leg_price_band",
          message: `${leg.description} is outside the ` +
            `${formatOdds(rules.leg_odds_min)}/${formatOdds(rules.leg_odds_max)} single-leg band.`,
        });
      }
      if (leg.ev < rules.min_leg_ev) {
        advisories.push({
          level: "warn",
          code: "leg_ev_floor",
          message: `${leg.description} has ${(leg.ev * 100).toFixed(1)}% EV, under the ` +
            `${(rules.min_leg_ev * 100).toFixed(0)}% floor.`,
        });
      }
      const key = `${leg.subject}|${leg.market}`;
      if (seen.has(key)) {
        advisories.push({
          level: "block",
          code: "duplicate_selection",
          message: `${leg.subject} ${leg.market_label} is on the slip twice.`,
        });
      }
      seen.add(key);
    }
    for (let i = 0; i < legs.length; i += 1) {
      for (let j = i + 1; j < legs.length; j += 1) {
        if (legs[i].game_id !== legs[j].game_id) continue;
        const rho = lookup(legs[i], legs[j]);
        if (rho < rules.min_sgp_correlation) {
          advisories.push({
            level: "warn",
            code: "sgp_correlation",
            message: `${legs[i].description} and ${legs[j].description} are same-game at ` +
              `r=${rho.toFixed(2)}, under the ${rules.min_sgp_correlation.toFixed(2)} floor` +
              (rho < 0 ? " (they work against each other)" : ""),
          });
        }
      }
    }
    return advisories;
  }

  // --------------------------------------------------------- auto build
  /**
   * Longshot mode: the most likely route to a big payout.
   *
   * The house rules are off here by design -- no EV floor, no price band, no
   * correlation floor, up to eight legs. The only thing kept is coherence: the
   * same subject and market cannot appear twice, because both sides of one
   * market cannot win together.
   *
   * Tickets are ranked by model probability among those that clear the payout
   * target, which answers "what is the likeliest way to turn this into that".
   * Ranking that way also naturally favours correlated same-game tickets,
   * since correlation is what lifts a parlay's true joint probability above
   * the product of its legs -- and that gap is the one real edge a longshot
   * bettor has, because books price same-game legs closer to independent than
   * they actually are.
   */
  function buildLongshots(legs, lookup, settings) {
    const minMultiple = settings.minMultiple || 20;
    const minLegs = settings.minLegs || 2;
    const maxLegs = settings.maxLegs || 8;
    const poolSize = settings.maxPool || 18;
    const comboBudget = settings.budget || 250000;
    const shortlistSize = settings.shortlist || 250;
    const maxTickets = settings.maxTickets || 3;

    // Rank the pool by value, not probability. The most likely legs are all
    // short-priced favourites that cannot multiply up to a big payout;
    // ranking by EV keeps the legs the model thinks are underpriced, which is
    // what you want in every leg of a longshot. There is no EV *floor* -- a
    // negative-EV leg is allowed in, it just queues behind better ones.
    const usable = inWeek(legs, settings.week)
      .filter((leg) => leg.p_model > 0 && leg.p_model < 1);
    const byValue = usable.slice().sort((a, b) => b.ev - a.ev || b.p_model - a.p_model);
    // A big target is unreachable from value legs alone -- they are mostly
    // short prices that cannot multiply far enough. Seed part of the pool with
    // the longest prices on the board so the target is actually achievable,
    // then let the ranking decide which of them survive.
    const byPrice = usable.slice().sort((a, b) =>
      americanToDecimal(b.odds) - americanToDecimal(a.odds));

    const pool = [];
    const taken = new Set();
    const add = (leg) => {
      if (taken.has(leg.id)) return;
      taken.add(leg.id);
      pool.push(leg);
    };
    byValue.slice(0, Math.ceil(poolSize * 0.7)).forEach(add);
    byPrice.slice(0, Math.ceil(poolSize * 0.6)).forEach(add);

    // Phase 1 -- enumerate cheaply. Running the copula on every combination is
    // what froze the page: tens of thousands of Monte-Carlo runs on the main
    // thread. Independent probability and a correlation nudge cost nothing and
    // are only used to shortlist; the real pricing happens in phase 2.
    const shortlist = [];
    let checked = 0;
    let bestMultiple = 0;

    const screen = (chosen) => {
      const seen = new Set();
      for (const leg of chosen) {
        const key = `${leg.subject}|${leg.market}`;
        if (seen.has(key)) return;            // cannot win both sides
        seen.add(key);
      }
      const decimal = parlayDecimal(chosen);
      if (decimal > bestMultiple) bestMultiple = decimal;
      if (decimal < minMultiple) return;

      let independent = 1;
      for (const leg of chosen) independent *= leg.p_model;

      let correlation = 0;
      let pairs = 0;
      for (let i = 0; i < chosen.length; i += 1) {
        for (let j = i + 1; j < chosen.length; j += 1) {
          correlation += lookup(chosen[i], chosen[j]);
          pairs += 1;
        }
      }
      // Correlation lifts the true joint probability above the product, so
      // nudge correlated tickets up the shortlist rather than losing them.
      const score = independent * (1 + Math.max(pairs ? correlation / pairs : 0, 0));
      shortlist.push({ legs: chosen.slice(), score });
    };

    const combine = (start, chosen) => {
      if (checked > comboBudget) return;
      if (chosen.length >= minLegs) {
        checked += 1;
        screen(chosen);
      }
      if (chosen.length >= maxLegs) return;
      for (let index = start; index < pool.length; index += 1) {
        chosen.push(pool[index]);
        combine(index + 1, chosen);
        chosen.pop();
        if (checked > comboBudget) return;
      }
    };
    combine(0, []);

    // Phase 2 -- price only the best few, with the real copula.
    shortlist.sort((a, b) => b.score - a.score);
    const candidates = [];
    for (const entry of shortlist.slice(0, shortlistSize)) {
      const priced = priceSlip(entry.legs.slice(), lookup, {
        ...settings,
        iterations: settings.buildIterations || 4000,
      });
      priced.legs = entry.legs.slice();
      priced.subjects = entry.legs.map((leg) => leg.subject);
      candidates.push(priced);
    }

    // Likeliest first, bigger payout as the tie-break.
    candidates.sort((a, b) =>
      b.probability - a.probability || b.decimal - a.decimal);

    // Offer genuinely different tickets without demanding they be disjoint: a
    // longshot pool is small, and insisting on no shared players usually
    // leaves one option. Half-overlap keeps the choices distinguishable.
    const chosenTickets = [];
    for (const ticket of candidates) {
      if (chosenTickets.length >= maxTickets) break;
      const tooSimilar = chosenTickets.some((picked) => {
        const shared = ticket.legs.filter((leg) =>
          picked.legs.some((other) => other.id === leg.id)).length;
        return shared > Math.min(ticket.legCount, picked.legCount) / 2;
      });
      if (!tooSimilar) chosenTickets.push(ticket);
    }
    return {
      tickets: chosenTickets,
      considered: shortlist.length,
      checked,
      bestMultiple,
      mode: "longshot",
    };
  }

  /**
   * Enumerate feasible tickets and pick a diversified, EV-maximising set.
   *
   * Python solves the selection as an integer program; a greedy pick over the
   * same candidate set is used here, which the Python side also falls back to
   * when no solver is available.
   */
  function buildCard(legs, lookup, options) {
    const settings = options || {};
    if (settings.mode === "longshot") return buildLongshots(legs, lookup, settings);
    const rules = settings.rules;
    const minLegs = settings.legs || rules.min_legs;
    const maxLegs = settings.legs || rules.max_legs;
    const maxTickets = settings.maxTickets || 3;
    const maxPool = settings.maxPool || 26;

    // One week per ticket: a parlay that cannot settle together is not a
    // parlay, whatever its expected value says.
    const pool = inWeek(legs, settings.week)
      .filter((leg) =>
        leg.ev >= rules.min_leg_ev &&
        leg.odds >= rules.leg_odds_min &&
        leg.odds <= rules.leg_odds_max)
      .sort((a, b) => b.ev - a.ev)
      .slice(0, maxPool);

    const candidates = [];
    const combine = (start, chosen) => {
      if (chosen.length >= minLegs) {
        const ticket = considerTicket(chosen, lookup, settings, rules);
        if (ticket) candidates.push(ticket);
      }
      if (chosen.length >= maxLegs) return;
      for (let index = start; index < pool.length; index += 1) {
        chosen.push(pool[index]);
        combine(index + 1, chosen);
        chosen.pop();
      }
    };
    combine(0, []);

    candidates.sort((a, b) => b.evPerUnit - a.evPerUnit);

    const chosenTickets = [];
    const usedSubjects = new Set();
    for (const ticket of candidates) {
      if (chosenTickets.length >= maxTickets) break;
      if (ticket.subjects.some((subject) => usedSubjects.has(subject))) continue;
      chosenTickets.push(ticket);
      ticket.subjects.forEach((subject) => usedSubjects.add(subject));
    }
    return { tickets: chosenTickets, considered: candidates.length, mode: "value" };
  }

  function considerTicket(legs, lookup, settings, rules) {
    const subjects = legs.map((leg) => leg.subject);
    if (new Set(subjects).size !== subjects.length) return null;

    for (let i = 0; i < legs.length; i += 1) {
      for (let j = i + 1; j < legs.length; j += 1) {
        if (legs[i].game_id !== legs[j].game_id) continue;
        if (lookup(legs[i], legs[j]) < rules.min_sgp_correlation) return null;
      }
    }

    const decimal = parlayDecimal(legs);
    const american = decimalToAmerican(decimal);
    if (american < rules.parlay_odds_min || american > rules.parlay_odds_max) return null;

    const priced = priceSlip(legs.slice(), lookup, {
      ...settings,
      iterations: settings.buildIterations || 3000,
    });
    if (priced.evPerUnit <= 0) return null;
    return { ...priced, legs: legs.slice(), subjects };
  }

  const engine = {
    americanToDecimal,
    decimalToAmerican,
    formatOdds,
    parlayDecimal,
    expectedValue,
    kellyShare,
    normalQuantile,
    mulberry32,
    correlationMatrix,
    cholesky,
    jointProbability,
    weeksOf,
    inWeek,
    buildLongshots,
    priceSlip,
    reviewSlip,
    buildCard,
  };

  if (typeof module !== "undefined" && module.exports) module.exports = engine;
  global.ParlayEngine = engine;
})(typeof globalThis !== "undefined" ? globalThis : this);
