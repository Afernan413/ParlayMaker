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

  const sportData = () => data.sports[state.sport];
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

    $("status").textContent =
      `${sport.games.length} games · ${sport.legs.length} priced bets · ` +
      `${sport.legs.filter(hasEdge).length} rated +EV · ` +
      `${sport.adjustments} context adjustments · built ${age.text}`;
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

    renderFreshness();
    renderGames();
    renderFilters();
    renderLegs();
    renderSlip();
  }

  // --------------------------------------------------------------- games
  function renderGames() {
    const sport = sportData();
    const grid = $("game-grid");
    grid.innerHTML = "";
    $("games-count").textContent = `${sport.games.length} games`;

    for (const game of sport.games) {
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
    for (const game of sport.games) {
      const option = document.createElement("option");
      option.value = game.game_id;
      option.textContent = game.label;
      gameSelect.append(option);
    }

    const markets = [...new Set(sport.legs.map((leg) => leg.market_label))].sort();
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

    const legs = sportData().legs.filter((leg) => {
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
    $("legs-count").textContent = `${legs.length} of ${sportData().legs.length}`;
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
    for (const button of document.querySelectorAll(".leg")) {
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
          legs: legsValue ? Number(legsValue) : null,
          maxTickets: Number($("build-tickets").value) || 3,
          bankroll: Number($("build-bankroll").value) || rules().bankroll,
          kellyFraction: rules().kelly_fraction,
          seed: 1337,
        });
        renderBuild(result);
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
        "Try a different leg count.";
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
    if (index >= 0) state.slip.splice(index, 1);
    else state.slip.push(legId);
    syncLegButtons();
    priceSlip();
  }

  function showView(view) {
    state.view = view;
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
        if (state.sport === button.dataset.sport) return;
        state.sport = button.dataset.sport;
        for (const other of document.querySelectorAll(".seg[data-sport]")) {
          other.setAttribute("aria-pressed", String(other === button));
        }
        loadSport();
      });
    }

    for (const tab of document.querySelectorAll(".tab[data-view]")) {
      tab.addEventListener("click", () => showView(tab.dataset.view));
    }

    $("build-run").addEventListener("click", runBuild);

    $("theme-toggle").addEventListener("click", () => {
      const root = document.documentElement;
      const dark = root.dataset.theme === "dark"
        || (!root.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
      root.dataset.theme = dark ? "light" : "dark";
      try { localStorage.setItem("parlay-theme", root.dataset.theme); } catch (_) { /* private mode */ }
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

    const skipped = data.skipped || {};
    for (const button of document.querySelectorAll(".seg[data-sport]")) {
      const sport = button.dataset.sport;
      const present = available.includes(sport);
      button.disabled = !present;
      button.setAttribute("aria-pressed", String(sport === state.sport));
      if (!present) {
        // Say why it is missing rather than leaving a dead button.
        button.title = skipped[sport]
          ? `${sport.toUpperCase()} was skipped in this build: ${skipped[sport]}`
          : `${sport.toUpperCase()} is not in this build`;
      }
    }
    wireEvents();
    showView("games");
    loadSport();
  }

  start();
})();
