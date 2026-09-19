/* Parlay Crafter front end.
 *
 * All pricing comes from the Python engine: the browser never re-implements the
 * copula, the de-vig or Kelly. The one thing it does locally is scale a priced
 * slip by the stake -- payout, profit and EV are linear in stake, so the stake
 * box can respond instantly without another request. */

const state = {
  sport: "nfl",
  config: null,
  slate: null,
  legsById: new Map(),
  slip: [],          // leg_id order
  pricing: null,     // last /api/price response
  requestToken: 0,
};

const $ = (id) => document.getElementById(id);

const money = (value) =>
  value.toLocaleString(undefined, {
    style: "currency", currency: "USD",
    minimumFractionDigits: Math.abs(value) < 100 && value % 1 !== 0 ? 2 : 0,
    maximumFractionDigits: 2,
  });
const pct = (value, digits = 1) => `${(value * 100).toFixed(digits)}%`;
const signedPct = (value, digits = 1) => `${value >= 0 ? "+" : ""}${(value * 100).toFixed(digits)}%`;

function setStatus(text, tone = "") {
  const node = $("status");
  node.textContent = text;
  node.dataset.tone = tone;
}

/* ------------------------------------------------------------- fetching */
async function api(path, options) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body.detail) detail = body.detail;
    } catch (_) { /* keep the status text */ }
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

async function loadConfig() {
  state.config = await api("/api/config");
  const badge = $("source-badge");
  badge.textContent = state.config.mock ? "cached fixtures" : "live data";
  badge.className = `badge ${state.config.mock ? "badge-muted" : "badge-good"}`;
  badge.title = state.config.mock
    ? "Reading data/mock fixtures - no API credits spent"
    : "Live odds, weather, injuries and stats";
}

async function loadSlate({ refresh = false } = {}) {
  setStatus(refresh ? "Rebuilding the slate…" : "Building the slate…");
  $("refresh").disabled = true;
  try {
    const payload = await api(`/api/slate/${state.sport}?refresh=${refresh}`);
    state.slate = payload;
    state.legsById = new Map(payload.legs.map((leg) => [leg.leg_id, leg]));
    state.slip = state.slip.filter((id) => state.legsById.has(id));
    renderFilters();
    renderLegs();
    renderSlip();
    const meta = payload.meta;
    setStatus(
      `${meta.games} games · ${meta.legs} priced bets · ${meta.edges} clear the ` +
      `${pct(state.config.engine.min_leg_ev, 0)} EV floor · ` +
      `${meta.adjustments} context adjustments · built in ${meta.seconds}s`
    );
    $("build-results").innerHTML = "";
  } catch (error) {
    setStatus(`Could not build the slate: ${error.message}`, "error");
  } finally {
    $("refresh").disabled = false;
  }
}

/* ------------------------------------------------------------ rendering */
function renderFilters() {
  const gameSelect = $("filter-game");
  gameSelect.innerHTML = '<option value="">All games</option>';
  for (const game of state.slate.games) {
    const option = document.createElement("option");
    option.value = game.game_id;
    option.textContent = game.label;
    gameSelect.append(option);
  }

  const markets = [...new Set(state.slate.legs.map((leg) => leg.market_label))].sort();
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

  const legs = state.slate.legs.filter((leg) => {
    if (edgesOnly && !leg.has_edge) return false;
    if (game && leg.game_id !== game) return false;
    if (market && leg.market_label !== market) return false;
    if (search) {
      const haystack = `${leg.subject} ${leg.team ?? ""} ${leg.market_label} ${leg.selection} ${leg.game}`.toLowerCase();
      if (!haystack.includes(search)) return false;
    }
    return true;
  });

  const comparators = {
    ev: (a, b) => b.ev - a.ev,
    edge: (a, b) => b.edge - a.edge,
    prob: (a, b) => b.p_model - a.p_model,
    odds: (a, b) => b.decimal_odds - a.decimal_odds,
  };
  return legs.sort(comparators[sort] ?? comparators.ev);
}

function evClass(leg) {
  if (!leg.has_edge) return "ev-bad";
  return leg.ev >= state.config.engine.min_leg_ev ? "ev-good" : "ev-flat";
}

function renderLegs() {
  const list = $("leg-list");
  const legs = visibleLegs();
  list.innerHTML = "";
  $("legs-count").textContent = `${legs.length} of ${state.slate.legs.length}`;
  $("legs-empty").hidden = legs.length > 0;

  const fragment = document.createDocumentFragment();
  for (const leg of legs) {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "leg";
    button.setAttribute("aria-pressed", String(state.slip.includes(leg.leg_id)));
    button.dataset.legId = leg.leg_id;

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
        <span class="leg-odds">${leg.odds_display}</span>
        <span class="leg-ev ${evClass(leg)}">${signedPct(leg.ev)} EV</span>
      </span>`;
    button.querySelector(".leg-name").textContent = leg.label;
    button.querySelector(".leg-bet").textContent = `${leg.market_label} ${leg.selection}${line}`;
    button.querySelector(".leg-game").textContent = leg.game;
    item.append(button);
    fragment.append(item);
  }
  list.append(fragment);
}

function renderSlip() {
  const list = $("slip-legs");
  list.innerHTML = "";
  const hasLegs = state.slip.length > 0;
  $("slip-empty").hidden = hasLegs;
  $("slip-body").hidden = !hasLegs;
  $("clear-slip").hidden = !hasLegs;

  for (const legId of state.slip) {
    const leg = state.legsById.get(legId);
    if (!leg) continue;
    const item = document.createElement("li");
    item.className = "slip-leg";
    item.innerHTML = `
      <span class="slip-leg-body">
        <span class="slip-leg-name"></span>
        <span class="slip-leg-sub"></span>
      </span>
      <span class="slip-leg-odds">${leg.odds_display}</span>
      <button type="button" class="remove" data-remove="${leg.leg_id}"
              aria-label="Remove this leg">×</button>`;
    const line = leg.line === null || leg.line === undefined ? "" : ` ${leg.line}`;
    item.querySelector(".slip-leg-name").textContent =
      `${leg.label} ${leg.selection}${line}`;
    item.querySelector(".slip-leg-sub").textContent =
      `${leg.market_label} · ${leg.game} · model ${pct(leg.p_model)}`;
    list.append(item);
  }
}

/* ------------------------------------------------------------- pricing */
async function priceSlip() {
  if (state.slip.length === 0) {
    state.pricing = null;
    renderSlip();
    return;
  }
  const token = ++state.requestToken;
  try {
    const pricing = await api("/api/price", {
      method: "POST",
      body: JSON.stringify({
        sport: state.sport,
        leg_ids: state.slip,
        stake: stakeValue(),
        bankroll: Number($("build-bankroll").value) || undefined,
      }),
    });
    if (token !== state.requestToken) return;  // a newer slip is in flight
    state.pricing = pricing;
    renderPricing();
  } catch (error) {
    if (error.status === 409) {
      setStatus("That bet is no longer on the slate — reloading it.", "error");
      await loadSlate({ refresh: false });
      return;
    }
    setStatus(`Pricing failed: ${error.message}`, "error");
  }
}

function stakeValue() {
  const raw = Number($("stake").value);
  return Number.isFinite(raw) && raw > 0 ? raw : 0;
}

function renderPricing() {
  renderSlip();
  const pricing = state.pricing;
  if (!pricing || !pricing.priceable) {
    renderAdvisories(pricing ? pricing.advisories : []);
    return;
  }

  const stake = stakeValue();
  const profit = stake * (pricing.decimal_odds - 1);
  const payout = stake + profit;
  const evDollars = pricing.ev_per_unit * stake;

  $("hero-payout").textContent = money(payout);
  $("hero-detail").textContent =
    `${money(stake)} on a ${pricing.leg_count}-leg ${pricing.is_sgp ? "same-game parlay" : "parlay"} ` +
    `at ${pricing.odds_display} — ${money(profit)} profit if all ${pricing.leg_count} land.`;

  $("tile-odds").textContent = pricing.odds_display;
  $("tile-odds-sub").textContent =
    `${pricing.decimal_odds.toFixed(2)}× · fair price ${pricing.fair_odds > 0 ? "+" : ""}${pricing.fair_odds}`;

  $("tile-profit").textContent = money(profit);
  $("tile-profit-sub").textContent = `${(pricing.decimal_odds - 1).toFixed(2)}× your stake`;

  const evTile = $("tile-ev");
  evTile.textContent = `${evDollars >= 0 ? "+" : ""}${money(evDollars)}`;
  evTile.className = `tile-value ${pricing.ev_per_unit >= 0 ? "ev-good" : "ev-bad"}`;
  $("tile-ev-sub").textContent =
    `${signedPct(pricing.ev_per_unit)} per $1 · long-run average, not this ticket`;

  $("tile-kelly").textContent = money(pricing.recommended_stake);
  $("tile-kelly-sub").textContent =
    pricing.recommended_stake > 0
      ? `¼ Kelly · ${pct(pricing.kelly_share, 2)} of ${money(pricing.bankroll)}`
      : "no edge — the model says pass";

  const model = pricing.p_model;
  const market = pricing.p_implied;
  const scale = Math.max(model, market, 0.0001);
  $("bar-model").style.width = `${(model / scale) * 100}%`;
  $("bar-market").style.width = `${(market / scale) * 100}%`;
  $("prob-model").textContent = pct(model);
  $("prob-market").textContent = pct(market);

  const edgeBadge = $("prob-edge");
  edgeBadge.textContent = `${signedPct(pricing.edge)} edge`;
  edgeBadge.className = `badge ${pricing.edge > 0 ? "badge-good" : "badge-bad"}`;

  const lift = pricing.correlation_lift;
  $("prob-note").textContent = pricing.leg_count < 2
    ? "Single leg — add another to build a parlay."
    : pricing.is_sgp
      ? `Same-game legs move together: the copula puts this at ${pct(model)}, ` +
        `${lift.toFixed(2)}× what multiplying the legs would give ` +
        `(${pct(pricing.independent_probability)}).`
      : `Legs from different games are priced as independent ` +
        `(${pct(pricing.independent_probability)} multiplied out).`;

  renderAdvisories(pricing.advisories);
  renderPayoutTable(pricing, stake);
  renderCorrelations(pricing);
  renderWhy(pricing);
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
  const amounts = [...new Set([...pricing.payout_table.map((row) => row.stake), stake])]
    .filter((amount) => amount > 0)
    .sort((a, b) => a - b);

  for (const amount of amounts) {
    const profit = amount * (pricing.decimal_odds - 1);
    const row = document.createElement("tr");
    if (Math.abs(amount - stake) < 1e-9) row.className = "is-current";
    for (const value of [
      money(amount),
      money(profit),
      money(amount + profit),
      `${pricing.ev_per_unit >= 0 ? "+" : ""}${money(pricing.ev_per_unit * amount)}`,
    ]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    rows.append(row);
  }
}

function renderCorrelations(pricing) {
  const list = $("corr-list");
  const block = $("corr-block");
  list.innerHTML = "";
  const pairs = pricing.correlation_pairs ?? [];
  block.hidden = pairs.length === 0;
  for (const pair of pairs) {
    const item = document.createElement("li");
    const relation = pair.same_game ? "same game" : "different games";
    const cls = pair.correlation > 0.01 ? "corr-pos" : pair.correlation < -0.01 ? "corr-neg" : "";
    item.innerHTML = `<span class="corr-value ${cls}">r=${pair.correlation.toFixed(2)}</span> `;
    item.append(
      document.createTextNode(`(${relation}) ${pair.a_label} ↔ ${pair.b_label}`)
    );
    list.append(item);
  }
}

function renderWhy(pricing) {
  const list = $("why-list");
  const block = $("why-block");
  list.innerHTML = "";
  const notes = pricing.rationale ?? [];
  block.hidden = notes.length === 0;
  for (const note of notes) {
    const item = document.createElement("li");
    item.textContent = note;
    list.append(item);
  }
}

/* --------------------------------------------------------- auto build */
async function runBuild() {
  const button = $("build-run");
  button.disabled = true;
  button.textContent = "Solving…";
  try {
    const legs = $("build-legs").value;
    const payload = await api("/api/build", {
      method: "POST",
      body: JSON.stringify({
        sport: state.sport,
        legs: legs ? Number(legs) : null,
        max_tickets: Number($("build-tickets").value) || 3,
        bankroll: Number($("build-bankroll").value) || undefined,
        seed: 11,
      }),
    });
    renderBuild(payload);
  } catch (error) {
    setStatus(`Optimizer failed: ${error.message}`, "error");
  } finally {
    button.disabled = false;
    button.textContent = "Build a card";
  }
}

function renderBuild(payload) {
  const list = $("build-results");
  list.innerHTML = "";
  if (payload.tickets.length === 0) {
    const item = document.createElement("li");
    item.className = "empty";
    item.textContent =
      `No ticket cleared every constraint (${payload.report.candidates} candidates ` +
      `from ${payload.report.considered} combinations). Try a different leg count.`;
    list.append(item);
    return;
  }

  for (const ticket of payload.tickets) {
    const item = document.createElement("li");
    item.className = "build-ticket";
    item.innerHTML = `
      <div class="build-ticket-head">
        <span class="build-ticket-type"></span>
        <span class="build-ticket-odds">${ticket.odds_display}</span>
      </div>
      <ul class="build-ticket-legs"></ul>
      <div class="build-ticket-foot">
        <span>model ${pct(ticket.p_model)} vs market ${pct(ticket.p_implied)} · ${signedPct(ticket.ev)} EV</span>
        <button type="button" class="btn btn-sm">Load into slip</button>
      </div>`;
    item.querySelector(".build-ticket-type").textContent = ticket.ticket_type;
    const legList = item.querySelector(".build-ticket-legs");
    for (const leg of ticket.legs) {
      const legItem = document.createElement("li");
      legItem.textContent = leg.description;
      legList.append(legItem);
    }
    item.querySelector("button").addEventListener("click", () => {
      state.slip = ticket.leg_ids.filter((id) => state.legsById.has(id));
      $("stake").value = ticket.stake > 0 ? Math.round(ticket.stake) : stakeValue();
      syncLegButtons();
      priceSlip();
      $("slip-heading").scrollIntoView({ behavior: "smooth", block: "center" });
    });
    list.append(item);
  }
}

/* ------------------------------------------------------------- events */
/* Reflect slip membership without rebuilding the list, so the click target
 * keeps focus and the scroll position survives. */
function syncLegButtons() {
  for (const button of document.querySelectorAll(".leg")) {
    button.setAttribute("aria-pressed", String(state.slip.includes(button.dataset.legId)));
  }
}

function toggleLeg(legId) {
  const index = state.slip.indexOf(legId);
  if (index >= 0) state.slip.splice(index, 1);
  else state.slip.push(legId);
  syncLegButtons();
  priceSlip();
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

  for (const id of ["filter-game", "filter-market", "filter-sort", "filter-edges"]) {
    $(id).addEventListener("change", renderLegs);
  }
  $("filter-search").addEventListener("input", debounce(renderLegs, 120));

  // Payout, profit and EV are linear in stake, so scale the last price locally.
  $("stake").addEventListener("input", () => {
    if (state.pricing?.priceable) renderPricing();
  });
  for (const chip of document.querySelectorAll(".chip[data-stake]")) {
    chip.addEventListener("click", () => {
      $("stake").value = chip.dataset.stake;
      if (state.pricing?.priceable) renderPricing();
    });
  }
  $("build-bankroll").addEventListener("change", () => priceSlip());

  for (const button of document.querySelectorAll(".seg[data-sport]")) {
    button.addEventListener("click", async () => {
      if (state.sport === button.dataset.sport) return;
      state.sport = button.dataset.sport;
      state.slip = [];
      state.pricing = null;
      for (const other of document.querySelectorAll(".seg[data-sport]")) {
        other.setAttribute("aria-pressed", String(other === button));
      }
      await loadSlate();
    });
  }

  $("refresh").addEventListener("click", () => loadSlate({ refresh: true }));
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

async function start() {
  restoreTheme();
  wireEvents();
  try {
    await loadConfig();
  } catch (error) {
    setStatus(`Could not reach the engine: ${error.message}`, "error");
    return;
  }
  await loadSlate();
}

start();
