/* Parlay Model — standalone browser build.
 *
 * Reads the bundle the Python engine wrote into data.js and prices slips with
 * engine.js. No network, no server: it works from file://, a local server or
 * GitHub Pages, and keeps working offline once loaded. */

(function () {
  "use strict";

  const data = window.PARLAY_DATA;
  const engine = window.ParlayEngine;

  const state = {
    sport: "nfl",
    view: "games",
    slip: [],
    pricing: null,
    picksSeed: 1337,
    week: null,          // which week's slate we are building for
    noteTimer: null,
    mode: "longshot",   // what this tool is for; Value is a click away
    legsById: new Map(),
    correlations: new Map(),
  };

  const $ = (id) => document.getElementById(id);

  const money = (value) =>
    value.toLocaleString(undefined, {
      style: "currency", currency: "USD",
      minimumFractionDigits: Math.abs(value) < 100 && value % 1 !== 0 ? 2 : 0,
      maximumFractionDigits: 2,
    });
  const pct = (value, digits = 1) => `${(value * 100).toFixed(digits)}%`;
  const signedPct = (value, digits = 1) =>
    `${value >= 0 ? "+" : ""}${(value * 100).toFixed(digits)}%`;
  const signed = (value, digits = 1) =>
    `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;

  /** "in 3h", "Sun 5:00 PM", or "" when the kickoff is unreadable. */
  function kickoffLabel(iso) {
    const start = new Date(iso);
    if (Number.isNaN(start.getTime())) return "";
    const hours = (start.getTime() - Date.now()) / 36e5;
    if (hours < 0) return "started";
    if (hours < 1) return `in ${Math.max(Math.round(hours * 60), 1)} min`;
    if (hours < 24) return `in ${Math.round(hours)}h`;
    return start.toLocaleString(undefined, {
      weekday: "short", hour: "numeric", minute: "2-digit",
    });
  }

  const sportData = () => data.sports[state.sport];

  const SPORT_NAMES = { nfl: "NFL", ncaaf: "College football", nba: "NBA" };

  /**
   * Why a sport has no data, in words worth reading.
   *
   * The build records a reason for every sport it skipped or was not asked
   * for, so the page can say what happened and what to run instead of leaving
   * a button that does nothing when pressed.
   */
  function whySportIsMissing(sport) {
    const name = SPORT_NAMES[sport] || sport.toUpperCase();
    const reason = (data.skipped || {})[sport];
    return reason ? `${name}: ${reason}` : `${name} is not in this build.`;
  }
  const rules = () => data.settings;

  /** Correlation between two legs; 0 for anything the engine did not pair. */
  function lookup(legA, legB) {
    return state.correlations.get(legA.i * 100000 + legB.i) ?? 0;
  }

  function slipLegs() {
    return state.slip.map((id) => state.legsById.get(id)).filter(Boolean);
  }

  function hasEdge(leg) {
    const limits = rules();
    return (
      leg.ev >= limits.min_leg_ev &&
      leg.odds >= limits.leg_odds_min &&
      leg.odds <= limits.leg_odds_max
    );
  }

  // ------------------------------------------------------------ freshness
  function describeAge(iso) {
    const built = new Date(iso);
    if (Number.isNaN(built.getTime())) return { text: "unknown age", hours: 0 };
    const hours = (Date.now() - built.getTime()) / 36e5;
    if (hours < 1) return { text: `${Math.max(Math.round(hours * 60), 1)} min ago`, hours };
    if (hours < 48) return { text: `${Math.round(hours)} h ago`, hours };
    return { text: `${Math.round(hours / 24)} days ago`, hours };
  }

  function renderFreshness() {
    const sport = sportData();
    const age = describeAge(data.generated_at);
    const note = $("stale-note");

    if (sport.mock) {
      note.hidden = false;
      note.textContent =
        "Showing the bundled sample slate — fictional players and prices. " +
        "Add an Odds API key and rebuild to see real games.";
    } else if (age.hours > 12) {
      note.hidden = false;
      note.textContent =
        `These odds were captured ${age.text}. Lines move — re-run the build ` +
        "before betting anything off this page.";
    } else {
      note.hidden = true;
    }

    const games = weekGames();
    const legs = weekLegs();
    const span = weeksAvailable().length > 1 ? `${weekLabelOf(state.week)} · ` : "";
    $("status").textContent = span +
      `${games.length} games · ${legs.length} priced bets · ` +
      `${legs.filter(hasEdge).length} rated +EV · ` +
      `${sport.adjustments} context adjustments · built ${age.text}` +
      describeTraining();
    renderInputs();
  }

  /**
   * What the model could actually see: starters, injuries, weather.
   *
   * Every one of these is optional in practice -- a forecast needs a key, an
   * injury report needs the league to publish one, a snap share needs a couple
   * of games played. Listing them stops the page implying the model weighed
   * something it never saw.
   */
  function renderInputs() {
    const list = $("model-inputs");
    const rows = sportData().inputs || [];
    list.innerHTML = "";
    list.hidden = rows.length === 0;
    for (const row of rows) {
      const item = document.createElement("li");
      item.className = `model-input ${row.available ? "known" : "unknown"}`;
      const label = document.createElement("span");
      label.className = "model-input-name";
      label.textContent = row.name;
      const detail = document.createElement("span");
      detail.className = "model-input-detail";
      detail.textContent = row.detail || (row.available ? "in use" : "not available");
      item.append(label, detail);
      list.append(item);
    }
  }

  /**
   * What the model learned from past results, if anything.
   *
   * The corrections are already inside every price on this page; this only
   * says where they came from, so a stale model is visible rather than
   * implied.
   */
  function describeTraining() {
    const trained = (data.training || {})[state.sport];
    if (!trained || !trained.observations) return "";
    const seasons = (trained.seasons || []).join("/");
    const gain = trained.brier_gain
      ? ` (${(trained.brier_gain * 100).toFixed(1)}% sharper out of sample)`
      : "";
    return ` · trained on ${trained.observations.toLocaleString()} graded results` +
      `${seasons ? ` from ${seasons}` : ""}${gain}`;
  }

  // ---------------------------------------------------------------- weeks
  /**
   * A parlay has to settle together, so everything is scoped to one week.
   *
   * The feed hands back every upcoming event, so a slate routinely holds this
   * Sunday's games and next Thursday's. Mixing them was producing tickets that
   * could not resolve for nine days, half of them priced off a week-old
   * projection. The picker only appears when there is more than one week to
   * choose between.
   */
  function weeksAvailable() {
    return sportData().weeks || [];
  }

  function renderWeekPicker() {
    const weeks = weeksAvailable();
    const picker = $("week-picker");
    const select = $("week-select");

    if (!weeks.some((week) => week.key === state.week)) {
      state.week = weeks.length ? weeks[0].key : null;
    }

    picker.hidden = weeks.length < 2;
    select.innerHTML = "";
    for (const week of weeks) {
      const option = document.createElement("option");
      option.value = week.key;
      option.textContent = `${week.label} (${week.games} game${week.games === 1 ? "" : "s"})`;
      option.selected = week.key === state.week;
      select.append(option);
    }
  }

  /** Legs on the selected week. A leg with no week is never hidden. */
  function weekLegs() {
    return engine.inWeek(sportData().legs, state.week);
  }

  function weekGames() {
    const games = sportData().games;
    if (!state.week) return games;
    return games.filter((game) => !game.week || game.week === state.week);
  }

  function weekLabelOf(key) {
    const found = weeksAvailable().find((week) => week.key === key);
    return found ? found.label : key;
  }

  /** Say something once, briefly. Used when an action is refused. */
  function flashNote(message, milliseconds = 6000) {
    const note = $("slip-note");
    note.textContent = message;
    note.hidden = false;
    clearTimeout(state.noteTimer);
    state.noteTimer = setTimeout(() => {
      note.hidden = true;
    }, milliseconds);
  }

  // ---------------------------------------------------------------- load
  function loadSport() {
    const sport = sportData();
    state.legsById = new Map(sport.legs.map((leg) => [leg.id, leg]));
    state.correlations = new Map();
    for (const [a, b, r] of sport.correlations) {
      state.correlations.set(a * 100000 + b, r);
      state.correlations.set(b * 100000 + a, r);
    }
    state.slip = [];
    state.pricing = null;
    $("build-results").innerHTML = "";

    renderWeekPicker();
    renderFreshness();
    renderPicks();
    renderGames();
    renderFilters();
    renderLegs();
    renderSlip();
  }

  // --------------------------------------------------------------- picks
  /**
   * The landing view: what to bet, without having to go looking for it.
   *
   * Runs the same optimizer the Build panel uses, but on load and with
   * defaults, so the first thing on screen is a playable card rather than a
   * table to interpret.
   */
  function renderPicks() {
    const sport = sportData();
    const bankroll = Number($("build-bankroll").value) || rules().bankroll;
    const gamesById = new Map(sport.games.map((game) => [game.game_id, game]));

    // --- ready-made parlays ---
    const longshot = state.mode === "longshot";
    const lsStake = Math.max(Number($("ls-stake").value) || 5, 1);
    const list = $("picks-tickets");
    list.innerHTML = "";

    $("picks-parlays-title").textContent =
      longshot ? "Longshot tickets" : "Ready-made parlays";

    let built;
    try {
      built = engine.buildCard(sport.legs, lookup, {
        rules: rules(),
        mode: state.mode,
        week: state.week,
        maxTickets: 3,
        minMultiple: longshot ? Number($("ls-target").value) || 100 : undefined,
        maxLegs: longshot ? 8 : undefined,
        stake: longshot ? lsStake : undefined,
        bankroll,
        kellyFraction: rules().kelly_fraction,
        seed: state.picksSeed,
      });
    } catch (error) {
      // Never fail silently: a thrown solver is a bug worth seeing.
      const failed = document.createElement("li");
      failed.className = "picks-empty";
      failed.textContent = `Could not build tickets: ${error.message}`;
      list.append(failed);
      built = { tickets: [], considered: 0 };
    }

    if (built.tickets.length === 0) {
      const empty = document.createElement("li");
      empty.className = "picks-empty";
      empty.textContent = longshot
        ? `Nothing on this slate pays ${money(lsStake * (Number($("ls-target").value) || 100))} ` +
          `from ${money(lsStake)}. The biggest payout available is about ` +
          `${money(lsStake * (built.bestMultiple || 1))} — pick a smaller target.`
        : `Nothing here clears the model's value bar (${built.considered} candidates). ` +
          "That is a normal result — prices are fair more often than not. " +
          "Switch to Longshot for big-payout tickets instead.";
      list.append(empty);
    }

    for (const ticket of built.tickets) {
      const stake = longshot
        ? lsStake
        : Math.max(Math.round(ticket.kellyShare * bankroll), 1);
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "pick";
      button.dataset.ticket = ticket.legs.map((leg) => leg.id).join("~");
      button.dataset.stake = String(stake);
      button.innerHTML = `
        <span class="pick-headline"></span>
        <ul class="pick-legs"></ul>
        <span class="pick-chips"></span>
        <span class="pick-right">
          <span class="pick-odds">${ticket.oddsDisplay}</span>
          <span class="pick-return"></span>
        </span>`;
      button.querySelector(".pick-headline").textContent = longshot
        ? `${money(stake)} → ${money(stake * ticket.decimal)}`
        : `${ticket.legCount}-leg ${ticket.isSameGame ? "same-game parlay" : "parlay"}`;
      const ret = button.querySelector(".pick-return");
      ret.textContent = longshot
        ? (ticket.isSameGame ? "same game" : `${ticket.legCount} legs`)
        : `${money(stake)} → ${money(stake * ticket.decimal)}`;
      if (longshot) button.querySelector(".pick-headline").classList.add("pick-payout-headline");

      const legList = button.querySelector(".pick-legs");
      for (const leg of ticket.legs) {
        const legItem = document.createElement("li");
        legItem.textContent = leg.description;
        legList.append(legItem);
      }

      const chips = button.querySelector(".pick-chips");
      const odds = ticket.probability > 0 ? Math.round(1 / ticket.probability) : null;
      chips.append(chip(odds ? `hits about 1 in ${odds}` : `model ${pct(ticket.probability, 1)}`));
      chips.append(chip(`${ticket.legCount} legs`));
      // The long-run number stays visible even when it is bad, especially then.
      chips.append(
        ticket.evPerUnit >= 0
          ? chip(`${signedPct(ticket.evPerUnit, 0)} long-run`, "pick-chip-edge")
          : chip(`${signedPct(ticket.evPerUnit, 0)} long-run`, "pick-chip-cost"),
      );
      item.append(button);
      list.append(item);
    }

    // --- best single bets ---
    const singles = $("picks-singles");
    singles.innerHTML = "";
    const top = weekLegs()
      .filter(hasEdge)
      .sort((a, b) => b.ev - a.ev)
      .slice(0, 6);

    if (top.length === 0) {
      const empty = document.createElement("li");
      empty.className = "picks-empty";
      empty.textContent = "No single bet clears the model's minimum edge today.";
      singles.append(empty);
    }

    for (const leg of top) {
      const game = gamesById.get(leg.game_id);
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "pick";
      button.dataset.legId = leg.id;
      button.setAttribute("aria-pressed", String(state.slip.includes(leg.id)));
      const line = leg.line === null || leg.line === undefined ? "" : ` ${leg.line}`;
      button.innerHTML = `
        <span class="pick-headline"></span>
        <span class="pick-sub"></span>
        <span class="pick-chips"></span>
        <span class="pick-right">
          <span class="pick-odds">${engine.formatOdds(leg.odds)}</span>
          <span class="pick-return"></span>
        </span>`;
      button.querySelector(".pick-headline").textContent =
        `${leg.label} ${leg.selection}${line}`;
      button.querySelector(".pick-sub").textContent =
        `${leg.market_label} · ${leg.game}`;
      button.querySelector(".pick-return").textContent =
        `$10 → ${money(10 * leg.decimal)}`;

      const chips = button.querySelector(".pick-chips");
      chips.append(
        chip(`model ${pct(leg.p_model, 0)}`),
        chip(`book ${pct(leg.p_implied, 0)}`),
        chip(`${signedPct(leg.edge, 0)} edge`, "pick-chip-edge"),
      );
      const when = kickoffLabel(game?.kickoff);
      if (when) chips.append(chip(when, "pick-chip-time"));

      item.append(button);
      singles.append(item);
    }
  }

  /**
   * Repaint a "searching" line before a solve that blocks the main thread.
   *
   * A longshot search takes a second or two. Without this the page simply
   * freezes and the control looks dead -- which is exactly how a working
   * button gets reported as broken.
   */
  function renderPicksBusy(message) {
    const list = $("picks-tickets");
    list.innerHTML = "";
    const busy = document.createElement("li");
    busy.className = "picks-empty";
    busy.textContent = message;
    list.append(busy);
    setTimeout(() => { renderPicks(); syncLegButtons(); }, 20);
  }

  function chip(text, extra) {
    const node = document.createElement("span");
    node.className = `pick-chip${extra ? " " + extra : ""}`;
    node.textContent = text;
    return node;
  }

  // --------------------------------------------------------------- games
  function renderGames() {
    const grid = $("game-grid");
    const games = weekGames();
    grid.innerHTML = "";
    $("games-count").textContent = `${games.length} games`;

    for (const game of games) {
      const card = document.createElement("article");
      card.className = "game-card";

      const kickoff = new Date(game.kickoff);
      const when = Number.isNaN(kickoff.getTime())
        ? ""
        : kickoff.toLocaleString(undefined, {
            weekday: "short", hour: "numeric", minute: "2-digit",
          });

      const hasProjection = game.model_total !== undefined;
      const favourite = hasProjection
        ? (game.model_margin >= 0 ? game.home_team : game.away_team)
        : null;

      card.innerHTML = `
        <div class="game-card-head">
          <span class="game-teams"></span>
          <span class="game-time"></span>
        </div>
        <p class="score-line"></p>
        <p class="score-note"></p>
        <div class="compare">
          <span class="compare-head">Line</span>
          <span class="compare-head compare-value">Model</span>
          <span class="compare-head compare-value">Market</span>
        </div>
        <p class="game-script"></p>
        <div class="game-actions">
          <button type="button" class="btn btn-sm" data-game="${game.game_id}">
            See this game's bets
          </button>
        </div>`;

      card.querySelector(".game-teams").textContent = game.label;
      card.querySelector(".game-time").textContent = when;

      if (hasProjection) {
        const score = card.querySelector(".score-line");
        score.innerHTML =
          `<span>${game.model_away_points.toFixed(0)}</span>` +
          `<span class="versus">–</span>` +
          `<span>${game.model_home_points.toFixed(0)}</span>`;
        card.querySelector(".score-note").textContent =
          `Projected score · ${favourite} by ${Math.abs(game.model_margin).toFixed(1)}`;

        const compare = card.querySelector(".compare");
        const rows = [
          ["Total", game.model_total.toFixed(1),
           game.market_total !== undefined && game.market_total !== null
             ? game.market_total.toFixed(1) : "—"],
          ["Spread (home)", signed(-game.model_margin),
           game.market_spread !== undefined && game.market_spread !== null
             ? signed(game.market_spread) : "—"],
        ];
        for (const [label, model, market] of rows) {
          const name = document.createElement("span");
          name.textContent = label;
          const modelCell = document.createElement("span");
          modelCell.className = "compare-value compare-model";
          modelCell.textContent = model;
          const marketCell = document.createElement("span");
          marketCell.className = "compare-value";
          marketCell.textContent = market;
          compare.append(name, modelCell, marketCell);
        }
      } else {
        card.querySelector(".score-line").remove();
        card.querySelector(".score-note").textContent =
          "No projection for this game — not enough stat history.";
      }

      const script = card.querySelector(".game-script");
      if (game.game_script) script.textContent = game.game_script;
      else script.remove();

      grid.append(card);
    }
  }

  // ------------------------------------------------------------ bet list
  function renderFilters() {
    const sport = sportData();
    const gameSelect = $("filter-game");
    gameSelect.innerHTML = '<option value="">All games</option>';
    for (const game of weekGames()) {
      const option = document.createElement("option");
      option.value = game.game_id;
      option.textContent = game.label;
      gameSelect.append(option);
    }

    const markets = [...new Set(weekLegs().map((leg) => leg.market_label))].sort();
    const marketSelect = $("filter-market");
    marketSelect.innerHTML = '<option value="">All markets</option>';
    for (const market of markets) {
      const option = document.createElement("option");
      option.value = market;
      option.textContent = market;
      marketSelect.append(option);
    }
  }

  function visibleLegs() {
    const search = $("filter-search").value.trim().toLowerCase();
    const game = $("filter-game").value;
    const market = $("filter-market").value;
    const edgesOnly = $("filter-edges").checked;
    const sort = $("filter-sort").value;

    const legs = weekLegs().filter((leg) => {
      if (edgesOnly && !hasEdge(leg)) return false;
      if (game && leg.game_id !== game) return false;
      if (market && leg.market_label !== market) return false;
      if (search) {
        const haystack =
          `${leg.subject} ${leg.team ?? ""} ${leg.market_label} ${leg.selection} ${leg.game}`.toLowerCase();
        if (!haystack.includes(search)) return false;
      }
      return true;
    });

    const comparators = {
      ev: (a, b) => b.ev - a.ev,
      edge: (a, b) => b.edge - a.edge,
      prob: (a, b) => b.p_model - a.p_model,
      odds: (a, b) => b.decimal - a.decimal,
    };
    return legs.sort(comparators[sort] ?? comparators.ev);
  }

  function renderLegs() {
    const list = $("leg-list");
    const legs = visibleLegs();
    list.innerHTML = "";
    $("legs-count").textContent = `${legs.length} of ${weekLegs().length}`;
    $("legs-empty").hidden = legs.length > 0;

    const fragment = document.createDocumentFragment();
    for (const leg of legs) {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "leg";
      button.dataset.legId = leg.id;
      button.setAttribute("aria-pressed", String(state.slip.includes(leg.id)));

      const line = leg.line === null || leg.line === undefined ? "" : ` ${leg.line}`;
      button.innerHTML = `
        <span class="leg-main">
          <span class="leg-name"></span>
          <span class="leg-bet"></span>
        </span>
        <span class="leg-meta">
          <span class="leg-game"></span>
          <span>model ${pct(leg.p_model)} · market ${pct(leg.p_implied)}</span>
        </span>
        <span class="leg-right">
          <span class="leg-odds">${engine.formatOdds(leg.odds)}</span>
          <span class="leg-ev ${hasEdge(leg) ? "ev-good" : "ev-bad"}">${signedPct(leg.ev)} EV</span>
        </span>`;
      button.querySelector(".leg-name").textContent = leg.label;
      button.querySelector(".leg-bet").textContent =
        `${leg.market_label} ${leg.selection}${line}`;
      button.querySelector(".leg-game").textContent = leg.game;
      item.append(button);
      fragment.append(item);
    }
    list.append(fragment);
  }

  function syncLegButtons() {
    for (const button of document.querySelectorAll(".leg, .pick[data-leg-id]")) {
      button.setAttribute("aria-pressed", String(state.slip.includes(button.dataset.legId)));
    }
  }

  // --------------------------------------------------------------- slip
  function renderSlip() {
    const list = $("slip-legs");
    list.innerHTML = "";
    const legs = slipLegs();
    $("slip-empty").hidden = legs.length > 0;
    $("slip-body").hidden = legs.length === 0;
    $("clear-slip").hidden = legs.length === 0;

    for (const leg of legs) {
      const item = document.createElement("li");
      item.className = "slip-leg";
      item.innerHTML = `
        <span class="slip-leg-body">
          <span class="slip-leg-name"></span>
          <span class="slip-leg-sub"></span>
        </span>
        <span class="slip-leg-odds">${engine.formatOdds(leg.odds)}</span>
        <button type="button" class="remove" data-remove="${leg.id}"
                aria-label="Remove this leg">×</button>`;
      const line = leg.line === null || leg.line === undefined ? "" : ` ${leg.line}`;
      item.querySelector(".slip-leg-name").textContent =
        `${leg.label} ${leg.selection}${line}`;
      item.querySelector(".slip-leg-sub").textContent =
        `${leg.market_label} · ${leg.game} · model ${pct(leg.p_model)}`;
      list.append(item);
    }
  }

  /** Running total that follows you down a phone screen. */
  function renderMiniSlip() {
    const bar = $("mini-slip");
    const pricing = state.pricing;
    if (!pricing || !pricing.priceable) {
      bar.hidden = true;
      return;
    }
    bar.hidden = false;
    $("mini-count").textContent = String(pricing.legCount);
    $("mini-odds").textContent = pricing.oddsDisplay;
    $("mini-payout").textContent =
      `${money(stakeValue())} → ${money(stakeValue() * pricing.decimal)}`;
  }

  function stakeValue() {
    const raw = Number($("stake").value);
    return Number.isFinite(raw) && raw > 0 ? raw : 0;
  }

  /** Re-run the copula. Only needed when the legs change, not the stake. */
  function priceSlip() {
    const legs = slipLegs();
    renderSlip();
    if (legs.length === 0) {
      state.pricing = null;
      renderMiniSlip();
      return;
    }
    state.pricing = engine.priceSlip(legs, lookup, {
      stake: stakeValue(),
      bankroll: Number($("build-bankroll").value) || rules().bankroll,
      kellyFraction: rules().kelly_fraction,
      iterations: 10000,
      seed: 1337,
    });
    state.pricing.advisories = engine.reviewSlip(legs, lookup, rules());
    renderPricing();
  }

  function renderPricing() {
    const pricing = state.pricing;
    if (!pricing || !pricing.priceable) return;

    const legs = slipLegs();
    const stake = stakeValue();
    const profit = stake * (pricing.decimal - 1);
    const payout = stake + profit;
    const evDollars = pricing.evPerUnit * stake;

    $("hero-payout").textContent = money(payout);
    $("hero-detail").textContent =
      `${money(stake)} on a ${pricing.legCount}-leg ` +
      `${pricing.isSameGame ? "same-game parlay" : "parlay"} at ${pricing.oddsDisplay} — ` +
      `${money(profit)} profit if all ${pricing.legCount} land.`;

    $("tile-odds").textContent = pricing.oddsDisplay;
    $("tile-odds-sub").textContent =
      `${pricing.decimal.toFixed(2)}× · fair price ${engine.formatOdds(pricing.fairOdds)}`;

    $("tile-profit").textContent = money(profit);
    $("tile-profit-sub").textContent = `${(pricing.decimal - 1).toFixed(2)}× your stake`;

    const evTile = $("tile-ev");
    evTile.textContent = `${evDollars >= 0 ? "+" : ""}${money(evDollars)}`;
    evTile.className = `tile-value ${pricing.evPerUnit >= 0 ? "ev-good" : "ev-bad"}`;
    $("tile-ev-sub").textContent =
      `${signedPct(pricing.evPerUnit)} per $1 · long-run average, not this ticket`;

    const bankroll = Number($("build-bankroll").value) || rules().bankroll;
    $("tile-kelly").textContent = money(pricing.kellyShare * bankroll);
    $("tile-kelly-sub").textContent = pricing.kellyShare > 0
      ? `¼ Kelly · ${pct(pricing.kellyShare, 2)} of ${money(bankroll)}`
      : "no edge — the model says pass";

    const scale = Math.max(pricing.probability, pricing.implied, 0.0001);
    $("bar-model").style.width = `${(pricing.probability / scale) * 100}%`;
    $("bar-market").style.width = `${(pricing.implied / scale) * 100}%`;
    $("prob-model").textContent = pct(pricing.probability);
    $("prob-market").textContent = pct(pricing.implied);

    const edgeBadge = $("prob-edge");
    edgeBadge.textContent = `${signedPct(pricing.edge)} edge`;
    edgeBadge.className = `badge ${pricing.edge > 0 ? "badge-good" : "badge-bad"}`;

    $("prob-note").textContent = pricing.legCount < 2
      ? "Single leg — add another to build a parlay."
      : pricing.isSameGame
        ? `Same-game legs move together: the model puts this at ` +
          `${pct(pricing.probability)}, ${pricing.correlationLift.toFixed(2)}× what ` +
          `multiplying the legs would give (${pct(pricing.independent)}).`
        : `Legs from different games are priced as independent ` +
          `(${pct(pricing.independent)} multiplied out).`;

    renderAdvisories(pricing.advisories);
    renderPayoutTable(pricing, stake);
    renderCorrelations(legs);
    renderWhy(legs);
    renderMiniSlip();
  }

  function renderAdvisories(advisories) {
    const list = $("advisories");
    list.innerHTML = "";
    const icons = { block: "✕", warn: "!", info: "i" };
    for (const advisory of advisories ?? []) {
      const item = document.createElement("li");
      item.className = `advisory advisory-${advisory.level}`;
      const icon = document.createElement("span");
      icon.className = "advisory-icon";
      icon.textContent = icons[advisory.level] ?? "i";
      const text = document.createElement("span");
      text.textContent = advisory.message;
      item.append(icon, text);
      list.append(item);
    }
  }

  function renderPayoutTable(pricing, stake) {
    const rows = $("payout-rows");
    rows.innerHTML = "";
    const amounts = [...new Set([5, 10, 25, 50, 100, stake])]
      .filter((amount) => amount > 0)
      .sort((a, b) => a - b);

    for (const amount of amounts) {
      const profit = amount * (pricing.decimal - 1);
      const row = document.createElement("tr");
      if (Math.abs(amount - stake) < 1e-9) row.className = "is-current";
      for (const value of [
        money(amount), money(profit), money(amount + profit),
        `${pricing.evPerUnit >= 0 ? "+" : ""}${money(pricing.evPerUnit * amount)}`,
      ]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      rows.append(row);
    }
  }

  function renderCorrelations(legs) {
    const list = $("corr-list");
    list.innerHTML = "";
    $("corr-block").hidden = legs.length < 2;
    for (let i = 0; i < legs.length; i += 1) {
      for (let j = i + 1; j < legs.length; j += 1) {
        const rho = lookup(legs[i], legs[j]);
        const sameGame = legs[i].game_id === legs[j].game_id;
        const item = document.createElement("li");
        const cls = rho > 0.01 ? "corr-pos" : rho < -0.01 ? "corr-neg" : "";
        item.innerHTML = `<span class="corr-value ${cls}">r=${rho.toFixed(2)}</span> `;
        item.append(document.createTextNode(
          `(${sameGame ? "same game" : "different games"}) ` +
          `${legs[i].description} ↔ ${legs[j].description}`
        ));
        list.append(item);
      }
    }
  }

  function renderWhy(legs) {
    const list = $("why-list");
    list.innerHTML = "";
    const notes = [...new Set(legs.flatMap((leg) => leg.why ?? []))];
    $("why-block").hidden = notes.length === 0;
    for (const note of notes) {
      const item = document.createElement("li");
      item.textContent = note;
      list.append(item);
    }
  }

  // ---------------------------------------------------------- auto build
  function runBuild() {
    const button = $("build-run");
    button.disabled = true;
    button.textContent = "Solving…";

    // Let the button repaint before the synchronous solve blocks the thread.
    setTimeout(() => {
      try {
        const legsValue = $("build-legs").value;
        const result = engine.buildCard(sportData().legs, lookup, {
          rules: rules(),
          mode: state.mode,
          week: state.week,
          legs: state.mode === "longshot" ? null : (legsValue ? Number(legsValue) : null),
          minMultiple: state.mode === "longshot"
            ? Number($("ls-target").value) || 100 : undefined,
          maxLegs: state.mode === "longshot" ? 8 : undefined,
          maxTickets: Number($("build-tickets").value) || 3,
          bankroll: Number($("build-bankroll").value) || rules().bankroll,
          kellyFraction: rules().kelly_fraction,
          seed: 1337,
        });
        renderBuild(result);
      } catch (error) {
        // A silent no-op button is worse than an ugly message.
        const list = $("build-results");
        list.innerHTML = "";
        const item = document.createElement("li");
        item.className = "empty";
        item.textContent = `Could not build a card: ${error.message}`;
        list.append(item);
      } finally {
        button.disabled = false;
        button.textContent = "Build a card";
      }
    }, 10);
  }

  function renderBuild(result) {
    const list = $("build-results");
    list.innerHTML = "";
    if (result.tickets.length === 0) {
      const item = document.createElement("li");
      item.className = "empty";
      item.textContent =
        `No ticket cleared every rule (${result.considered} candidates). ` +
        "Try a different leg count, or switch to Longshot at the top for " +
        "big-payout tickets with the filters off.";
      list.append(item);
      return;
    }

    const bankroll = Number($("build-bankroll").value) || rules().bankroll;
    for (const ticket of result.tickets) {
      const item = document.createElement("li");
      item.className = "build-ticket";
      item.innerHTML = `
        <div class="build-ticket-head">
          <span class="build-ticket-type"></span>
          <span class="build-ticket-odds">${ticket.oddsDisplay}</span>
        </div>
        <ul class="build-ticket-legs"></ul>
        <div class="build-ticket-foot">
          <span>model ${pct(ticket.probability)} vs market ${pct(ticket.implied)} · ${signedPct(ticket.evPerUnit)} EV</span>
          <button type="button" class="btn btn-sm">Load into slip</button>
        </div>`;
      item.querySelector(".build-ticket-type").textContent =
        `${ticket.legCount}-Leg ${ticket.isSameGame ? "Same-Game" : "Cross-Game"} Parlay`;

      const legList = item.querySelector(".build-ticket-legs");
      for (const leg of ticket.legs) {
        const legItem = document.createElement("li");
        legItem.textContent = leg.description;
        legList.append(legItem);
      }

      item.querySelector("button").addEventListener("click", () => {
        state.slip = ticket.legs.map((leg) => leg.id);
        const suggested = Math.round(ticket.kellyShare * bankroll);
        if (suggested > 0) $("stake").value = suggested;
        syncLegButtons();
        priceSlip();
        $("slip-heading").scrollIntoView({ behavior: "smooth", block: "center" });
      });
      list.append(item);
    }
  }

  // -------------------------------------------------------------- events
  function toggleLeg(legId) {
    const index = state.slip.indexOf(legId);
    if (index >= 0) {
      state.slip.splice(index, 1);
    } else {
      const leg = state.legsById.get(legId);
      const clash = slipWeek();
      if (leg && leg.week && clash && leg.week !== clash) {
        // Refused rather than warned: a slip spanning two weeks cannot settle
        // together, so there is no version of it worth pricing.
        flashNote(
          `That bet is from ${weekLabelOf(leg.week)}; your slip is ` +
          `${weekLabelOf(clash)}. Clear the slip to switch weeks.`
        );
        return;
      }
      state.slip.push(legId);
    }
    syncLegButtons();
    priceSlip();
  }

  /** The week the slip is already committed to, if any. */
  function slipWeek() {
    for (const id of state.slip) {
      const leg = state.legsById.get(id);
      if (leg && leg.week) return leg.week;
    }
    return null;
  }

  function showView(view) {
    state.view = view;
    $("view-picks").hidden = view !== "picks";
    $("view-games").hidden = view !== "games";
    $("view-bets").hidden = view !== "bets";
    for (const tab of document.querySelectorAll(".tab[data-view]")) {
      tab.setAttribute("aria-selected", String(tab.dataset.view === view));
    }
  }

  function debounce(fn, wait) {
    let timer;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), wait);
    };
  }

  function wireEvents() {
    $("leg-list").addEventListener("click", (event) => {
      const button = event.target.closest(".leg");
      if (button) toggleLeg(button.dataset.legId);
    });

    $("slip-legs").addEventListener("click", (event) => {
      const button = event.target.closest("[data-remove]");
      if (button) toggleLeg(button.dataset.remove);
    });

    $("clear-slip").addEventListener("click", () => {
      state.slip = [];
      state.pricing = null;
      syncLegButtons();
      renderSlip();
    });

    // Picks: a ticket loads the whole slip, a single bet toggles like any leg.
    $("view-picks").addEventListener("click", (event) => {
      const ticket = event.target.closest("[data-ticket]");
      if (ticket) {
        state.slip = ticket.dataset.ticket.split("~").filter((id) => state.legsById.has(id));
        $("stake").value = ticket.dataset.stake;
        syncLegButtons();
        priceSlip();
        $("slip-heading").scrollIntoView({ behavior: "smooth", block: "center" });
        return;
      }
      const single = event.target.closest(".pick[data-leg-id]");
      if (single) toggleLeg(single.dataset.legId);
    });

    for (const button of document.querySelectorAll(".mode[data-mode]")) {
      button.addEventListener("click", () => {
        if (state.mode === button.dataset.mode) return;
        state.mode = button.dataset.mode;
        for (const other of document.querySelectorAll(".mode[data-mode]")) {
          other.setAttribute("aria-pressed", String(other === button));
        }
        const longshot = state.mode === "longshot";
        $("longshot-controls").hidden = !longshot;
        $("longshot-note").hidden = !longshot;
        if (longshot) renderPicksBusy("Searching for the likeliest big payouts…");
        else { renderPicks(); syncLegButtons(); }
      });
    }

    const research = () => renderPicksBusy("Searching for the likeliest big payouts…");
    $("ls-run").addEventListener("click", research);
    $("ls-target").addEventListener("change", research);
    $("ls-stake").addEventListener("change", research);

    $("mini-slip").addEventListener("click", () => {
      $("slip-heading").scrollIntoView({ behavior: "smooth", block: "start" });
    });

    $("game-grid").addEventListener("click", (event) => {
      const button = event.target.closest("[data-game]");
      if (!button) return;
      $("filter-game").value = button.dataset.game;
      renderLegs();
      showView("bets");
    });

    for (const id of ["filter-game", "filter-market", "filter-sort", "filter-edges"]) {
      $(id).addEventListener("change", renderLegs);
    }
    $("filter-search").addEventListener("input", debounce(renderLegs, 120));

    // Payout, profit and EV are linear in the stake, so rescale locally.
    $("stake").addEventListener("input", () => {
      if (state.pricing?.priceable) renderPricing();
    });
    for (const chip of document.querySelectorAll(".chip[data-stake]")) {
      chip.addEventListener("click", () => {
        $("stake").value = chip.dataset.stake;
        if (state.pricing?.priceable) renderPricing();
      });
    }
    $("build-bankroll").addEventListener("change", () => {
      if (state.pricing?.priceable) renderPricing();
    });

    for (const button of document.querySelectorAll(".seg[data-sport]")) {
      button.addEventListener("click", () => {
        const sport = button.dataset.sport;
        if (!data.sports[sport]) {
          flashNote(whySportIsMissing(sport), 12000);
          return;
        }
        if (state.sport === sport) return;
        state.sport = sport;
        for (const other of document.querySelectorAll(".seg[data-sport]")) {
          other.setAttribute("aria-pressed", String(other === button));
        }
        loadSport();
      });
    }

    for (const tab of document.querySelectorAll(".tab[data-view]")) {
      tab.addEventListener("click", () => showView(tab.dataset.view));
    }

    $("week-select").addEventListener("change", (event) => {
      state.week = event.target.value || null;
      loadSport();   // clears the slip: it belonged to the other week
    });

    $("build-run").addEventListener("click", runBuild);

    $("theme-toggle").addEventListener("click", () => {
      const root = document.documentElement;
      const dark = root.dataset.theme === "dark"
        || (!root.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
      root.dataset.theme = dark ? "light" : "dark";
      try { localStorage.setItem("parlay-theme", root.dataset.theme); } catch (_) { /* private mode */ }
    });
  }

  /** Keep the explainer open until it has been read once. */
  function restoreExplainer() {
    const explainer = $("explainer");
    let seen = false;
    try { seen = localStorage.getItem("parlay-explainer-seen") === "1"; } catch (_) { /* private mode */ }
    explainer.open = !seen;
    explainer.addEventListener("toggle", () => {
      if (explainer.open) return;
      try { localStorage.setItem("parlay-explainer-seen", "1"); } catch (_) { /* private mode */ }
    });
  }

  function restoreTheme() {
    try {
      const stored = localStorage.getItem("parlay-theme");
      if (stored) document.documentElement.dataset.theme = stored;
    } catch (_) { /* private mode: fall back to the OS setting */ }
  }

  function start() {
    restoreTheme();
    if (!data || !engine) {
      $("status").textContent =
        "Could not load the model data. Rebuild with: python scripts/build_static.py";
      return;
    }
    const available = Object.keys(data.sports);
    if (available.length === 0) {
      $("status").textContent = "No slate in this build. Rebuild with: python scripts/build_static.py";
      return;
    }
    if (!available.includes(state.sport)) state.sport = available[0];

    for (const button of document.querySelectorAll(".seg[data-sport]")) {
      const sport = button.dataset.sport;
      const present = available.includes(sport);
      // Marked rather than `disabled`: a disabled button fires no click, so
      // tapping it would say nothing at all. The reason is worth more than the
      // dead press, and a title tooltip does not exist on a phone.
      button.classList.toggle("seg-empty", !present);
      button.setAttribute("aria-disabled", String(!present));
      button.setAttribute("aria-pressed", String(sport === state.sport));
      button.title = present ? "" : whySportIsMissing(sport);
    }
    wireEvents();
    restoreExplainer();
    showView("picks");
    loadSport();
  }

  start();
})();
