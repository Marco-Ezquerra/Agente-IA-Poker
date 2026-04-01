/**
 * poker.js — Lógica frontend de la mesa de póker HUNL
 *
 * Flujo:
 *   1. POST /api/nueva_partida  → render estado inicial
 *   2. Humano elige acción      → POST /api/accion
 *   3. Servidor responde con nuevo estado (incluyendo acción del bot)
 *   4. Render nuevo estado
 *   5. Si HAND_OVER → mostrar overlay de resultado
 *   6. Botón "Siguiente mano"   → POST /api/nueva_mano → volver al paso 2
 */

'use strict';

/* ── Constantes / utilidades de palos ──────────────────────── */
const SUIT_SYMBOL = { s: '♠', h: '♥', d: '♦', c: '♣' };
const SUIT_COLOR  = { s: 'black', h: 'red', d: 'red', c: 'black' };
const RANK_DISPLAY = {
  '2':'2','3':'3','4':'4','5':'5','6':'6',
  '7':'7','8':'8','9':'9','T':'10','J':'J','Q':'Q','K':'K','A':'A',
};

/* ── Estado de la sesión (cliente) ─────────────────────────── */
let _state        = null;
let _raiseMin     = 1.0;
let _raiseMax     = 200.0;
let _wins         = 0;
let _losses       = 0;
let _totalHands   = 0;
let _balanceBB    = 0;
let _prevStack    = 200;  // seguimiento del stack inicial para calcular balance

/* ── DOM ───────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);

/* ════════════════════════════════════════════════════════════
   API helpers
   ════════════════════════════════════════════════════════════ */

async function apiCall(endpoint, method = 'GET', body = null) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(endpoint, opts);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function fetchStats() {
  try {
    const s = await apiCall('/api/stats');
    renderStats(s);
  } catch (_) {}
}

/* ════════════════════════════════════════════════════════════
   Acciones del usuario
   ════════════════════════════════════════════════════════════ */

async function startNewGame() {
  setLoading(true);
  try {
    _wins = _losses = _totalHands = _balanceBB = 0;
    _prevStack = 200;
    updateResultsSummary();
    const estado = await apiCall('/api/nueva_partida', 'POST');
    render(estado);
  } catch (e) {
    showStatus('Error al conectar con el servidor: ' + e.message);
  } finally {
    setLoading(false);
  }
}

async function sendAction(tipo, amount = null) {
  if (!_state || _state.game_state !== 'HUMAN_TURN') return;
  setLoading(true);
  hideRaiseSlider();
  try {
    const body = { tipo };
    if (amount !== null) body.amount = amount;
    const estado = await apiCall('/api/accion', 'POST', body);
    render(estado);
    fetchStats();
  } catch (e) {
    showStatus('Error: ' + e.message);
  } finally {
    setLoading(false);
  }
}

async function nextHand() {
  hideResultOverlay();
  setLoading(true);
  try {
    const estado = await apiCall('/api/nueva_mano', 'POST');
    render(estado);
    fetchStats();
  } catch (e) {
    showStatus('Error: ' + e.message);
  } finally {
    setLoading(false);
  }
}

/* ════════════════════════════════════════════════════════════
   Render principal
   ════════════════════════════════════════════════════════════ */

function render(state) {
  _state = state;

  // Número de mano
  $('hand-number').textContent = `Mano #${state.mano_numero}`;

  // Blueprint badge
  // (solo se actualiza al cargar; no cambia durante la partida)

  // Posiciones
  const humanPos = state.tu_pos || '?';
  const botPos   = humanPos === 'SB' ? 'BB' : 'SB';
  $('human-pos-badge').textContent = humanPos;
  $('bot-pos-badge').textContent   = botPos;

  // Stacks
  $('human-stack').textContent = `${state.tu_stack} BB`;
  $('bot-stack').textContent   = `${state.bot_stack} BB`;

  // Pot
  $('pot-value').textContent = `${state.pot} BB`;

  // Calle
  $('street-badge').textContent = streetDisplay(state.street);

  // Cartas humano
  renderHumanCards(state.tu_mano || []);

  // Cartas del bot
  renderBotCards(state.bot_mano || null, state.game_state === 'HAND_OVER');

  // Cartas comunitarias
  renderCommunity(state.community || []);

  // Chips de apuesta (apuesta actual)
  renderBetChip('human-bet', state.to_call, state.to_call > 0);

  // Panel de acciones
  if (state.game_state === 'HUMAN_TURN') {
    renderActions(state.valid_actions || [], state.to_call);
    showStatus(actionPrompt(state));
  } else if (state.game_state === 'HAND_OVER') {
    clearActions();
    showStatus('');
    showResultOverlay(state.resultado, state.tu_mano, state.bot_mano, state.community);
    recordResult(state);
  } else {
    clearActions();
    showStatus('Esperando...');
  }
}

/* ── Cartas humano ──────────────────────────────────────────── */
function renderHumanCards(hand) {
  const container = $('human-cards');
  container.innerHTML = '';
  if (!hand || hand.length === 0) {
    container.append(makeCardEmpty(), makeCardEmpty());
    return;
  }
  hand.forEach((c, i) => {
    const el = makeCard(c);
    el.classList.add('card-flip');
    container.appendChild(el);
  });
}

/* ── Cartas bot ─────────────────────────────────────────────── */
function renderBotCards(hand, reveal) {
  const container = $('bot-cards');
  container.innerHTML = '';
  if (reveal && hand && hand.length === 2) {
    hand.forEach(c => {
      const el = makeCard(c);
      el.classList.add('card-flip');
      container.appendChild(el);
    });
  } else {
    container.append(makeCardBack(), makeCardBack());
  }
}

/* ── Cartas comunitarias ────────────────────────────────────── */
function renderCommunity(community) {
  const container = $('community-cards');
  container.innerHTML = '';
  const slots = 5;
  for (let i = 0; i < slots; i++) {
    if (i < community.length) {
      const el = makeCard(community[i]);
      if (i >= (community.length - (community.length === 3 ? 3
               : community.length === 4 ? 1 : 1))) {
        el.classList.add('card-flip');
      }
      container.appendChild(el);
    } else {
      container.appendChild(makeCardEmpty());
    }
  }
}

/* ── Chip de apuesta ────────────────────────────────────────── */
function renderBetChip(elId, amount, show) {
  const el = $(elId);
  if (show && amount > 0) {
    el.textContent = amount + ' BB';
    el.style.display = 'flex';
    el.classList.remove('chip-pop');
    void el.offsetWidth;
    el.classList.add('chip-pop');
  } else {
    el.style.display = 'none';
  }
}

/* ════════════════════════════════════════════════════════════
   Construcción de botones de acción
   ════════════════════════════════════════════════════════════ */

function renderActions(validActions, toCall) {
  const container = $('action-buttons');
  container.innerHTML = '';
  hideRaiseSlider();

  let hasRaiseOptions = false;

  validActions.forEach(action => {
    if (action.tipo === 'fold') {
      container.appendChild(makeActionBtn('Fold', 'btn-fold', () => sendAction('fold')));

    } else if (action.tipo === 'call') {
      const label = action.amount > 0
        ? `Call ${action.amount} BB`
        : 'Call';
      container.appendChild(makeActionBtn(label, 'btn-call', () => sendAction('call')));

    } else if (action.tipo === 'check') {
      container.appendChild(makeActionBtn('Check', 'btn-check', () => sendAction('check')));

    } else if (action.tipo === 'all_in') {
      const label = `All-in ${action.amount} BB`;
      container.appendChild(makeActionBtn(label, 'btn-allin', () => sendAction('all_in')));

    } else if (action.tipo === 'raise_options') {
      hasRaiseOptions = true;
      _raiseMin = action.min;
      _raiseMax = action.max;

      // Botones de preset
      action.presets.forEach(p => {
        const btn = makeActionBtn(`${p.label} (${p.amount} BB)`, 'btn-raise',
          () => sendAction('raise', p.amount));
        container.appendChild(btn);
      });

      // Botón "Raise custom" que abre el slider
      const btnCustom = makeActionBtn('Raise custom…', 'btn-raise btn-sm',
        () => openRaiseSlider(action.min, action.max));
      container.appendChild(btnCustom);
    }
  });
}

function makeActionBtn(label, cssClasses, onClick) {
  const btn = document.createElement('button');
  btn.className = 'btn ' + cssClasses;
  btn.textContent = label;
  btn.addEventListener('click', onClick);
  return btn;
}

function clearActions() {
  $('action-buttons').innerHTML = '';
  hideRaiseSlider();
}

/* ── Slider de raise ────────────────────────────────────────── */
function openRaiseSlider(min, max) {
  const row    = $('raise-slider-row');
  const slider = $('raise-slider');
  slider.min   = min;
  slider.max   = max;
  slider.step  = 0.5;
  slider.value = Math.round(min * 2) / 2;
  updateRaiseLabel(slider.value);
  row.style.display = 'flex';
}

function hideRaiseSlider() {
  $('raise-slider-row').style.display = 'none';
}

function updateRaiseLabel(val) {
  $('raise-label').textContent = `${parseFloat(val).toFixed(1)} BB`;
}

/* ════════════════════════════════════════════════════════════
   Result overlay
   ════════════════════════════════════════════════════════════ */

function showResultOverlay(resultado, humanoHand, botHand, community) {
  if (!resultado) return;

  const overlay = $('result-overlay');
  const card    = $('result-card');
  const icon    = $('result-icon');
  const title   = $('result-title');
  const sub     = $('result-subtitle');
  const hands   = $('revealed-hands');

  // Clases de resultado
  card.className = 'result-card';
  const r = resultado.resultado_humano || resultado.tipo || '';
  if (r === 'win' || r === 'fold_win') {
    card.classList.add('result-win');
    icon.textContent  = '🏆';
    title.textContent = '¡Ganaste!';
  } else if (r === 'lose') {
    card.classList.add('result-lose');
    icon.textContent  = '😞';
    title.textContent = 'Perdiste';
  } else if (r === 'split') {
    card.classList.add('result-split');
    icon.textContent  = '🤝';
    title.textContent = 'Split';
  } else {
    icon.textContent  = '🃏';
    title.textContent = resultado.ganador + ' gana';
  }

  sub.textContent = `Pot: ${resultado.pot} BB`;

  // Manos reveladas (solo en showdown)
  hands.innerHTML = '';
  if (resultado.tipo === 'showdown' && botHand && botHand.length) {
    hands.appendChild(makeRevealedHand('Tú', humanoHand || []));
    hands.appendChild(makeRevealedHand('Bot', botHand));
  }

  overlay.style.display = 'flex';
}

function hideResultOverlay() {
  $('result-overlay').style.display = 'none';
}

function makeRevealedHand(label, hand) {
  const div = document.createElement('div');
  div.className = 'revealed-hand';
  const lbl = document.createElement('div');
  lbl.className = 'revealed-hand-label';
  lbl.textContent = label;
  const row = document.createElement('div');
  row.className = 'cards-row';
  hand.forEach(c => row.appendChild(makeCard(c)));
  div.appendChild(lbl);
  div.appendChild(row);
  return div;
}

/* ════════════════════════════════════════════════════════════
   Estadísticas
   ════════════════════════════════════════════════════════════ */

function renderStats(s) {
  $('s-hands').textContent = s.hands_seen;
  $('s-vpip').textContent  = pct(s.vpip);
  $('s-pfr').textContent   = pct(s.pfr);
  $('s-af').textContent    = s.af_total.toFixed(2);
  $('s-ftb').textContent   = pct(s.ftb_total);
  $('s-wtsd').textContent  = pct(s.wtsd);
  $('s-tipo').textContent  = s.clasificacion || '–';
}

function pct(v) { return (v * 100).toFixed(0) + '%'; }

/* ── Resultados sesión ──────────────────────────────────────── */
function recordResult(state) {
  if (!state.resultado) return;
  const r = state.resultado.resultado_humano || '';
  _totalHands++;
  if (r === 'win' || r === 'fold_win') _wins++;
  else if (r === 'lose') _losses++;
  // Balance: stack actual - stack inicial acumulado
  _balanceBB = Math.round((state.tu_stack - 200) * 10) / 10;
  updateResultsSummary();
  addHistoryItem(state.mano_numero, r, state.resultado.pot);
}

function updateResultsSummary() {
  $('r-total').textContent   = _totalHands;
  $('r-wins').textContent    = _wins;
  $('r-losses').textContent  = _losses;
  $('r-balance').textContent = (_balanceBB >= 0 ? '+' : '') + _balanceBB;
}

function addHistoryItem(mano, resultado, pot) {
  const list = $('history-list');
  const div  = document.createElement('div');
  div.className = `history-item ${resultado === 'win' || resultado === 'fold_win'
    ? 'win' : resultado === 'lose' ? 'lose' : 'split'}`;
  div.innerHTML = `<span>#${mano}</span>
    <span>${resultado === 'win' || resultado === 'fold_win' ? '▲' : resultado === 'lose' ? '▼' : '='} ${pot}BB</span>`;
  list.insertBefore(div, list.firstChild);
  // Máx 30 entradas
  while (list.children.length > 30) list.removeChild(list.lastChild);
}

/* ════════════════════════════════════════════════════════════
   Construcción de elementos carta
   ════════════════════════════════════════════════════════════ */

/**
 * Crea un elemento .card con su rango y palo a partir de un compact card string.
 * @param {string} compact - e.g. 'Ah', 'Ts', 'Kd'
 */
function makeCard(compact) {
  const rank = compact.slice(0, -1);
  const suit = compact.slice(-1);
  const el   = document.createElement('div');
  el.className = `card ${SUIT_COLOR[suit] || 'black'}`;

  const rankDisplay = RANK_DISPLAY[rank] || rank;
  const suitSymbol  = SUIT_SYMBOL[suit]  || suit;

  el.innerHTML = `
    <div class="card-rank-top">${rankDisplay}<br>${suitSymbol}</div>
    <div class="card-suit-center">${suitSymbol}</div>
    <div class="card-rank-bot">${rankDisplay}<br>${suitSymbol}</div>
  `;
  return el;
}

function makeCardBack() {
  const el = document.createElement('div');
  el.className = 'card card-back';
  return el;
}

function makeCardEmpty() {
  const el = document.createElement('div');
  el.className = 'card card-empty';
  return el;
}

/* ════════════════════════════════════════════════════════════
   Helpers de UI
   ════════════════════════════════════════════════════════════ */

function showStatus(msg) {
  $('status-msg').textContent = msg;
}

function actionPrompt(state) {
  const toCall = state.to_call;
  const pot    = state.pot;
  if (toCall > 0) {
    return `Tu turno — ${toCall} BB para igualar. Pot: ${pot} BB`;
  }
  return `Tu turno — puedes apostar o hacer check. Pot: ${pot} BB`;
}

function streetDisplay(s) {
  const m = { preflop: 'Preflop', flop: 'Flop', turn: 'Turn', river: 'River' };
  return m[s] || (s || '');
}

function setLoading(on) {
  $('loading-overlay').style.display = on ? 'flex' : 'none';
}

/* ════════════════════════════════════════════════════════════
   Event listeners
   ════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
  // Nueva partida
  $('btn-nueva-partida').addEventListener('click', startNewGame);

  // Siguiente mano
  $('btn-next-hand').addEventListener('click', nextHand);

  // Slider de raise
  $('raise-slider').addEventListener('input', e => updateRaiseLabel(e.target.value));

  // Confirmar raise custom
  $('btn-confirm-raise').addEventListener('click', () => {
    const amount = parseFloat($('raise-slider').value);
    sendAction('raise', amount);
  });

  // Verificar si hay blueprint disponible (ping rápido)
  fetch('/api/stats').then(r => r.json()).then(s => {
    // Si respondió, blueprint puede estar disponible
    fetch('/api/recargar_blueprint', { method: 'POST' })
      .then(r => r.json())
      .then(d => {
        const badge = $('blueprint-badge');
        if (d.ok) {
          badge.className = 'badge badge-on';
          badge.textContent = `Blueprint (${(d.infosets/1000).toFixed(0)}k InfoSets)`;
        }
      }).catch(() => {});
  }).catch(() => {});

  // Auto-arrancar
  startNewGame();
});
