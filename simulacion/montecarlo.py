# montecarlo.py
"""
Módulo de cálculo de equity y callbacks de decisión para el agente de póker.

Provee:
- montecarlo_equity          : simulación Monte Carlo para estimar probabilidad de ganar.
- get_equity_cached          : versión cacheada con límite de tamaño (LRU aproximado).
- advanced_ia_action_callback: IA basada en equity + pot odds por fase.
- human_bot_action_callback  : bot rival que simula un jugador mediocre.
- blueprint_action_callback  : IA GTO híbrida:  blueprint MCCFR + subgame search
                               en tiempo real + ajuste por modelo del oponente.
"""

import random
from itertools import combinations
from template import eval_hand_from_strings
from poker_engine import compact_card

# === CACHE GLOBAL DE EQUITY ===
_equity_cache: dict = {}
_EQUITY_CACHE_MAX_SIZE = 4096  # evita crecimiento ilimitado en partidas largas

# Funciones de equity Montecarlo

def generar_baraja_compacta():
    palos = ['h', 'd', 'c', 's']
    valores = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    return [v + p for v in valores for p in palos]


def montecarlo_equity(hole_cards, community_cards, num_players=2, num_simulations=500):
    wins = ties = losses = 0
    deck_base = generar_baraja_compacta()
    # eliminar cartas visibles
    for card in hole_cards + community_cards:
        if card in deck_base:
            deck_base.remove(card)
    for _ in range(num_simulations):
        deck = deck_base.copy()
        # repartir a oponentes
        opponents = []
        for _ in range(num_players - 1):
            opp = random.sample(deck, 2)
            for c in opp:
                deck.remove(c)
            opponents.append(opp)
        # completar board
        board = community_cards.copy()
        board += random.sample(deck, 5 - len(board))
        # evaluar
        my_score = eval_hand_from_strings(hole_cards, board)
        best = my_score
        tie = False
        for opp in opponents:
            score = eval_hand_from_strings(opp, board)
            if score > best:
                best = score; tie = False
            elif score == best:
                tie = True
        if my_score == best and not tie:
            wins += 1
        elif tie:
            ties += 1
        else:
            losses += 1
    total = wins + ties + losses
    return (wins + 0.5 * ties) / total if total else 0


def get_equity_cached(hole_cards, community_cards, num_players=2, num_simulations=500):
    key = (tuple(sorted(hole_cards)), tuple(sorted(community_cards)), num_players, num_simulations)
    if key in _equity_cache:
        return _equity_cache[key]
    eq = montecarlo_equity(hole_cards, community_cards, num_players, num_simulations)
    if len(_equity_cache) >= _EQUITY_CACHE_MAX_SIZE:
        # Elimina la mitad de entradas más antiguas (FIFO aproximado)
        for old_key in list(_equity_cache.keys())[:_EQUITY_CACHE_MAX_SIZE // 2]:
            del _equity_cache[old_key]
    _equity_cache[key] = eq
    return eq

# === IA avanzada ===

def advanced_ia_action_callback(jugador, state, valid_actions):
    """
    IA basada en equity Montecarlo ajustada por fase (preflop, flop, turn, river).
    Usa presupuestos de simulación diferentes por fase.
    """
    my_cards = [compact_card(c) for c in jugador.mano]
    fase = state.get('fase', 'preflop')
    community = state.get('community', [])
    pot = state.get('pot', 0)
    current = state.get('current_bet', 0)
    contrib = state.get('contributions', {})
    to_call = current - contrib.get(jugador.id, 0)

    # Presupuesto de simulaciones por fase
    sims_map = {
        'preflop': 500,
        'flop': 1000,
        'turn': 2000,
        'river': 5000
    }
    num_sims = sims_map.get(fase, 500)

    # Calcular equity según fase y board actual
    equity = get_equity_cached(my_cards, community, num_players=2, num_simulations=num_sims)

    # Acciones de raise disponibles
    raises = [a for a in valid_actions if isinstance(a, tuple) and a[0] == 'raise']
    allowed = sorted([amt for _, amt in raises])
    pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0

    def pick_raise(ratio):
        if not allowed:
            return None
        target = ratio * pot
        return min(allowed, key=lambda x: abs(x - target))

    # Lógica de decisión según fase
    if to_call > 0:
        if fase == 'preflop':
            if equity > 0.65:
                action = ('raise', pick_raise(1.0)) if allowed else 'call'
            elif equity > 0.45:
                action = 'call'
            else:
                action = 'fold'
        elif fase == 'flop':
            if equity > 0.75:
                action = ('raise', pick_raise(1.2)) if allowed else 'call'
            elif equity < pot_odds:
                action = ('raise', pick_raise(0.4)) if allowed and random.random() < 0.35 else 'fold'
            elif equity > 0.5:
                action = 'call'
            else:
                action = 'fold'
        elif fase == 'turn':
            if equity > 0.80:
                action = ('raise', pick_raise(1.3)) if allowed else 'call'
            elif equity < pot_odds:
                action = ('raise', pick_raise(0.5)) if allowed and random.random() < 0.20 else 'fold'
            else:
                action = 'call'
        else:  # river
            if equity > 0.85:
                action = ('raise', pick_raise(1.5)) if allowed else 'call'
            elif equity < pot_odds:
                action = ('raise', pick_raise(0.5)) if allowed and random.random() < 0.10 else 'fold'
            else:
                action = 'call'
    else:
        # Sin coste de llamada (to_call == 0)
        if equity > 0.5:
            action = ('raise', pick_raise(0.5)) if allowed and random.random() < 0.60 else 'check'
        elif equity < 0.10:
            # bluff ocasionales
            prob_bluff = {'flop': 0.25, 'turn': 0.15, 'river': 0.10}.get(fase, 0.10)
            if random.random() < prob_bluff:
                action = ('raise', pick_raise(1.0)) if allowed else 'check'
            else:
                action = 'check'
        else:
            action = 'check'

    # Etiquetado de tipo de jugada
    tipo = (
        'valor' if isinstance(action, tuple) and action[0] == 'raise' and equity > 0.65 else
        'bluff' if isinstance(action, tuple) and action[0] == 'raise' else
        'neutral' if action == 'call' and 0.4 < equity < 0.7 else
        'defensivo' if action == 'call' and equity < 0.4 else
        'pasivo' if action in ['check', 'fold'] else 'otro'
    )

    log_entry = ['ia', jugador.id, fase, round(equity, 4), my_cards, community,
                 action if isinstance(action, str) else action[0], tipo]
    state.setdefault('history', []).append(log_entry)
    state.setdefault('ia_logs', []).append(log_entry)
    return action

# BOT humano simple

def human_bot_action_callback(jugador, state, valid_actions):
    """
    Bot rival que simula un jugador humano mediocre.
    Toma decisiones basadas en equity y pot odds con algo de aleatoriedad.
    Corregido: fold cuando equity < pot_odds (antes hacía call incorrectamente).
    """
    my_cards = [compact_card(c) for c in jugador.mano]
    community = state.get('community', [])
    pot = state.get('pot', 0)
    current = state.get('current_bet', 0)
    contrib = state.get('contributions', {})
    to_call = current - contrib.get(jugador.id, 0)
    equity = get_equity_cached(my_cards, community, 2, 1000)
    pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0
    raises = [a for a in valid_actions if isinstance(a, tuple) and a[0] == 'raise']
    allowed = sorted([amt for _, amt in raises])

    if to_call > 0:
        if equity < pot_odds:
            # Sin valor esperado positivo: fold, con bluff esporádico
            if random.random() < 0.15 and allowed:
                decision = ('raise', random.choice(allowed[:2]))
            else:
                decision = 'fold'
        elif equity < pot_odds + 0.05:
            decision = 'call'
        else:
            if allowed:
                extra = (equity - pot_odds) * pot
                raise_amt = max(allowed[0], min(extra, allowed[-1]))
                decision = ('raise', raise_amt)
            else:
                decision = 'call'
    else:
        if equity > 0.5 and allowed and random.random() < 0.40:
            decision = ('raise', random.choice(allowed[:2]))
        else:
            decision = 'check'
    return decision


# ── Blueprint action callback (GTO híbrido) ────────────────────────────────────

# Estado global: blueprint y motor de búsqueda cargados en la primera llamada.
_blueprint       = None
_search_engine   = None
_blueprint_tried = False   # evita reintentar cargas fallidas


def _try_load_blueprint():
    """Carga el blueprint y el motor de búsqueda de forma lazy y segura."""
    global _blueprint, _search_engine, _blueprint_tried
    if _blueprint_tried:
        return _blueprint
    _blueprint_tried = True
    try:
        from cfr.mccfr_trainer  import MCCFRTrainer
        from cfr.realtime_search import RealtimeSearch
        if MCCFRTrainer.exists():
            _blueprint     = MCCFRTrainer.load()
            _search_engine = RealtimeSearch(
                blueprint=_blueprint, depth=1, iterations=120)
    except Exception:
        pass   # Sin blueprint: el callback fallará a advanced_ia_action_callback
    return _blueprint


def _abstract_to_real(abstract_act, pot, to_call, stack, valid_actions):
    """
    Convierte una acción abstracta del MCCFR en una acción concreta del motor.

    Busca en valid_actions la opción más próxima al tamaño abstracto.
    Si la acción no es posible (no hay valid_raise), retrocede a call/check.

    Parámetros
    ----------
    abstract_act  : str   – constante de abstracciones.infoset_encoder
    pot           : float
    to_call       : float
    stack         : float
    valid_actions : list  – acciones del motor en el estado actual

    Retorna
    -------
    str o tuple('raise', float) compatible con el motor de juego
    """
    from abstracciones.infoset_encoder import (
        FOLD, CALL, ALLIN,
        RAISE_RATIOS,
    )

    if abstract_act == FOLD:
        return 'fold' if 'fold' in valid_actions else 'check'

    if abstract_act == CALL:
        if to_call == 0:
            return 'check'
        return 'call' if 'call' in valid_actions else 'check'

    if abstract_act == ALLIN:
        return 'all in' if 'all in' in valid_actions else 'call'

    # Raises: RAISE_THIRD, RAISE_HALF, RAISE_POT, RAISE_2POT
    ratio        = RAISE_RATIOS.get(abstract_act, 1.0)
    target_total = to_call + ratio * pot            # fichas totales objetivo
    target_total = min(target_total, stack)

    raises = [(a, amt) for a in valid_actions
              if isinstance(a, tuple) and a[0] == 'raise'
              for amt in [a[1]]]
    if not raises:
        # No hay raises disponibles → call o check
        return 'call' if to_call > 0 else 'check'

    # Mejor raise disponible (el más cercano al target)
    best = min(raises, key=lambda x: abs(x[1] - target_total))
    return best[0]


def blueprint_action_callback(jugador, state, valid_actions,
                               opp_model=None, opp_samples=8):
    """
    Callback GTO híbrido que combina:
      1. Blueprint MCCFR (política base unexploitable)
      2. Real-time subgame search  (refina para el estado concreto)
      3. Opponent model adjustments (se desvía del GTO para explotar fallos)

    Si el blueprint no está disponible (aún no entrenado), delega a
    advanced_ia_action_callback para mantener la compatibilidad.

    Parámetros
    ----------
    jugador       : Jugador      – jugador activo
    state         : dict         – estado del motor (fase, pot, community, ...)
    valid_actions : list         – acciones válidas en el estado actual
    opp_model     : OpponentModel | None  – modelo del rival (opcional)
    opp_samples   : int          – manos del oponente a muestrear en search

    Retorna
    -------
    str o tuple('raise', float) – acción para el motor de juego
    """
    # Cargar blueprint en primera llamada
    _try_load_blueprint()

    my_cards  = [compact_card(c) for c in jugador.mano]
    fase      = state.get('fase', 'preflop')
    community = state.get('community', [])
    pot       = float(state.get('pot', 1.5))
    current   = float(state.get('current_bet', 0))
    contrib   = state.get('contributions', {})
    to_call   = max(0.0, current - float(contrib.get(jugador.id, 0)))
    stack     = float(jugador.fichas)

    # Determinar position (0=SB, 1=BB) a partir del id del jugador
    position  = jugador.id % 2    # SB=0, BB=1 en HUNL

    # Stacks  y contributions en formato [SB, BB]
    stacks_arr   = [0.0, 0.0]
    contribs_arr = [0.0, 0.0]
    for pid, amt in contrib.items():
        idx = int(pid) % 2
        contribs_arr[idx] = float(amt)
    stacks_arr[0] = float(state.get('stack_small_blind', 100))
    stacks_arr[1] = float(state.get('stack_big_blind',  100))

    # Historial abstracto de la calle actual (no disponible en el state simple)
    bet_hist = []
    n_raises = 0

    # ── Si no hay blueprint → fallback a IA basada en equity ─────────────────
    if _search_engine is None:
        return advanced_ia_action_callback(jugador, state, valid_actions)

    # ── Real-time subgame search ──────────────────────────────────────────────
    try:
        abstract_act = _search_engine.get_action(
            traverser   = position,
            my_hand     = my_cards,
            board       = community,
            street_str  = fase,
            pot         = pot,
            stacks      = stacks_arr,
            contribs    = contribs_arr,
            bet_hist    = bet_hist,
            n_raises    = n_raises,
            to_call     = to_call,
            opp_samples = opp_samples,
            bucket_sims = 30,
        )
    except Exception:
        return advanced_ia_action_callback(jugador, state, valid_actions)

    # ── Ajuste por opponent model ─────────────────────────────────────────────
    if opp_model is not None:
        adj = opp_model.get_counter_adjustments()
        from abstracciones.infoset_encoder import FOLD, CALL

        # Si el modelo dice que el rival es muy foldy y tenemos para bluffear,
        # potencialmente ajustamos de CALL a un raise pequeño como bluff
        if (abstract_act == CALL
                and adj['bluff_freq_mult'] > 1.5
                and to_call == 0
                and random.random() < 0.25 * adj['bluff_freq_mult']):
            from abstracciones.infoset_encoder import RAISE_THIRD
            abstract_act = RAISE_THIRD

        # Si el modelo dice que el rival llama mucho (loose-passive),
        # convertir algunos bluffs en check/fold
        if (adj['bluff_freq_mult'] < 0.5
                and abstract_act not in (FOLD, CALL)):
            equity = get_equity_cached(my_cards, community, 2, 300)
            if equity < 0.45:
                abstract_act = FOLD if to_call > 0 else CALL

    # ── Convertir acción abstracta en acción concreta ─────────────────────────
    return _abstract_to_real(abstract_act, pot, to_call, stack, valid_actions)
