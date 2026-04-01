"""
Aplicación web Flask para jugar contra el agente de póker IA.

Correcciones implementadas respecto a la versión anterior:
  1. abstract_position corregida: 0 si bot == SB, 1 si bot == BB
     (antes usaba self.bot_idx=1 fijo → 50 % de manos con estrategia invertida).
  2. _to_abstract_bet_hist reconstruye pot progresivamente con running_pot
     y raises reales → clasificación de raises correcta en vez de lookup fallido.
  3. _translate_abstract_action con SPR caps:
       ALLIN → RAISE_2POT si SPR > 2.5
       ALLIN → RAISE_POT  si SPR > 5
     (evita raises de 40-100 BB en pots de 5 BB).
  4. Override de equity-pura solo en true all-in (≥95 % del stack),
     no al ≥50 % como antes (que sobreescribía el blueprint en casi toda
     situación postflop).
  5. _online_learn(): tras cada mano completa ejecuta 80 traversals MCCFR
     sobre el deal real (Experience Replay CFR, estilo Pluribus).
  6. Blueprint se guarda automáticamente cada 20 manos aprendidas.

Uso
---
    cd simulacion
    python web/app.py            # modo debug
    python web/app.py --port 8080 --host 0.0.0.0

Endpoints
---------
  GET  /                           → página principal HTML
  POST /api/nueva_mano             → inicia una nueva mano
  POST /api/accion                 → el jugador humano realiza una acción
  GET  /api/estado                 → estado JSON de la partida actual
  POST /api/recargar_blueprint     → recarga el blueprint desde disco
  GET  /api/stats                  → estadísticas de convergencia y aprendizaje
"""

import os
import sys
import random
import threading
import time

# Asegurar que simulacion/ está en el path
_SIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)

from flask import Flask, request, jsonify

from abstracciones.card_abstractor import (
    preflop_bucket, postflop_bucket,
    PREFLOP_BUCKETS, POSTFLOP_BUCKETS,
)
from abstracciones.infoset_encoder import (
    FOLD, CALL, RAISE_THIRD, RAISE_HALF, RAISE_POT, RAISE_2POT, ALLIN,
    ABSTRACT_ACTIONS, ACTION_IDX, RAISE_RATIOS, encode_infoset,
)
from cfr.mccfr_trainer import MCCFRTrainer, _deal as _cfr_deal

app = Flask(__name__)

# ── Estado global del blueprint ────────────────────────────────────────────────

_blueprint_lock   = threading.Lock()
_blueprint        = None      # MCCFRTrainer o None
_blueprint_tried  = False
_hands_learned    = 0         # manos procesadas por _online_learn
_SAVE_EVERY       = 20        # guardar blueprint cada N manos aprendidas

BLUEPRINT_PATH = os.path.join(_SIM_DIR, 'cfr', 'blueprint.pkl')


def _load_blueprint():
    """Carga el blueprint una sola vez de forma lazy y thread-safe."""
    global _blueprint, _blueprint_tried
    with _blueprint_lock:
        if _blueprint_tried:
            return _blueprint
        _blueprint_tried = True
        try:
            if MCCFRTrainer.exists(BLUEPRINT_PATH):
                _blueprint = MCCFRTrainer.load(BLUEPRINT_PATH)
        except Exception as exc:
            print(f"[web/app] No se pudo cargar blueprint: {exc}")
        return _blueprint


def _get_blueprint() -> MCCFRTrainer | None:
    if not _blueprint_tried:
        _load_blueprint()
    return _blueprint


# ── Estado de la partida ───────────────────────────────────────────────────────

_game_state: dict = {}


def _new_game_state(stack_hu: float = 100.0) -> dict:
    """
    Crea un estado de partida fresco para Heads-Up No-Limit Hold'em.

    Posiciones:
      SB (small blind) = posición 0  →  apuesta 0.5 BB
      BB (big blind)   = posición 1  →  apuesta 1.0 BB

    En cada mano el bot alterna entre SB y BB para que la simulación
    sea simétrica (como en partidas reales de heads-up).
    """
    # Primera mano: posición aleatoria; manos sucesivas: alternancia.
    if not _game_state:
        bot_is_sb = random.choice([True, False])
    else:
        prev_bot_sb = _game_state.get('bot_is_sb', True)
        bot_is_sb   = not prev_bot_sb          # alternar posición en cada mano

    sb_idx = 0 if bot_is_sb else 1        # índice SB en arrays [bot, human]
    bb_idx = 1 - sb_idx

    stacks = [stack_hu, stack_hu]

    # Repartir cartas
    deck = _make_deck()
    random.shuffle(deck)
    bot_hand   = [deck[0], deck[1]]
    human_hand = [deck[2], deck[3]]
    board      = deck[4:9]                # flop(3) + turn(1) + river(1) pre-generados

    return {
        'bot_is_sb':   bot_is_sb,
        'sb_idx':      sb_idx,
        'bb_idx':      bb_idx,
        'stacks':      stacks,            # [bot_stack, human_stack]
        'contribs':    [0.5 if bot_is_sb else 1.0,
                        1.0 if bot_is_sb else 0.5],
        'pot':         1.5,
        'street':      'preflop',
        'street_idx':  0,
        'board':       [],
        'full_board':  board,             # cartas comunitarias no reveladas aún
        'bot_hand':    bot_hand,
        'human_hand':  human_hand,
        'deck':        deck,
        'bet_hist':    [],                # acciones abstractas de la calle actual
        'n_raises':    0,
        'to_call':     0.5,               # SB debe completar 0.5 a 1 BB
        'active':      sb_idx,            # SB actúa primero preflop
        'street_actions': [],             # acciones concretas de la calle actual
        'hand_history':  [],              # historial completo de la mano
        'finished':    False,
        'result':      None,
        'winner':      None,
    }


def _make_deck() -> list:
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['s', 'h', 'd', 'c']
    deck  = [r + s for r in ranks for s in suits]
    random.shuffle(deck)
    return deck


# ── Abstracción de historial de apuestas ──────────────────────────────────────

def _to_abstract_bet_hist(street_actions: list, initial_pot: float) -> tuple:
    """
    Convierte la lista de acciones concretas de una calle en un historial
    abstracto compatible con las claves de InfoSet del blueprint MCCFR.

    La reconstrucción usa running_pot para calcular los ratios de raise
    CORRECTAMENTE, incluso cuando hay varias acciones en la misma calle.

    Parámetros
    ----------
    street_actions : list[dict]  – acciones concretas con claves
                                   'type' ('fold'|'call'|'check'|'raise'|'allin')
                                   'amount' (en BBs, solo para raises)
    initial_pot    : float       – tamaño del pot al inicio de la calle

    Retorna
    -------
    tuple[str] – secuencia de acciones abstractas (FOLD, CALL, RAISE_*, ALLIN)
    """
    abstract_hist = []
    running_pot   = initial_pot
    last_raise    = 0.0

    for act in street_actions:
        atype  = act.get('type', '')
        amount = float(act.get('amount', 0.0))

        if atype == 'fold':
            abstract_hist.append(FOLD)
            break

        elif atype in ('call', 'check'):
            abstract_hist.append(CALL)
            if last_raise > 0:
                running_pot += last_raise
                last_raise   = 0.0

        elif atype == 'allin':
            abstract_hist.append(ALLIN)
            running_pot += amount
            last_raise   = amount

        elif atype == 'raise':
            # Clasificar el raise según su ratio respecto al pot corriente.
            # Los umbrales son los puntos medios entre las razones abstractas:
            #   RAISE_THIRD ≈ 1/3 (0.33), RAISE_HALF ≈ 1/2 (0.50),
            #   RAISE_POT ≈ 1× (1.00), RAISE_2POT ≈ 2× (2.00)
            # Puntos de corte: 0.42 = (0.33+0.50)/2, 0.62 = (0.50+1.00)/2,
            #                  1.25 = (1.00+2.00)/2
            ratio = amount / running_pot if running_pot > 0 else 1.0
            if   ratio <= 0.42:  abstract_hist.append(RAISE_THIRD)
            elif ratio <= 0.62:  abstract_hist.append(RAISE_HALF)
            elif ratio <= 1.25:  abstract_hist.append(RAISE_POT)
            else:                abstract_hist.append(RAISE_2POT)
            running_pot += amount
            last_raise   = amount

    return tuple(abstract_hist)


# ── Traducción de acción abstracta → acción concreta con SPR caps ─────────────

def _translate_abstract_action(abstract_act: str,
                                pot: float,
                                to_call: float,
                                bot_stack: float,
                                valid_raise_amounts: list) -> dict:
    """
    Convierte una acción abstracta del blueprint en una acción concreta,
    aplicando SPR caps para evitar raises desproporcionados cuando el blueprint
    no ha convergido bien (ALLIN frecuente con pocas iteraciones).

    SPR caps (Stack-to-Pot Ratio = bot_stack / pot):
      - Si SPR > 5   y acción == ALLIN → degradar a RAISE_POT
      - Si SPR > 2.5 y acción == ALLIN → degradar a RAISE_2POT
      (Con SPR ≤ 2.5 el all-in es razonable y se deja pasar.)

    Parámetros
    ----------
    abstract_act        : str   – acción abstracta del blueprint
    pot                 : float – tamaño del pot en BBs
    to_call             : float – fichas necesarias para igualar
    bot_stack           : float – stack actual del bot
    valid_raise_amounts : list  – montos de raise válidos en el motor

    Retorna
    -------
    dict con claves 'type' y (opcional) 'amount'
    """
    spr = bot_stack / pot if pot > 0 else 999.0

    # ── SPR caps: limitar all-in cuando el pot es pequeño relativo al stack ──
    if abstract_act == ALLIN:
        if spr > 5.0:
            abstract_act = RAISE_POT
        elif spr > 2.5:
            abstract_act = RAISE_2POT

    # ── Traducir acción abstracta → concreta ─────────────────────────────────
    if abstract_act == FOLD:
        return {'type': 'fold'}

    if abstract_act == CALL:
        if to_call <= 0:
            return {'type': 'check'}
        return {'type': 'call', 'amount': min(to_call, bot_stack)}

    if abstract_act == ALLIN:
        return {'type': 'allin', 'amount': bot_stack}

    # Raises proporcionales
    ratio  = RAISE_RATIOS.get(abstract_act, 1.0)
    target = to_call + ratio * pot
    target = min(target, bot_stack)

    if not valid_raise_amounts:
        # Sin raises disponibles → call o check
        if to_call > 0:
            return {'type': 'call', 'amount': min(to_call, bot_stack)}
        return {'type': 'check'}

    # Elegir el monto de raise más cercano al objetivo
    best = min(valid_raise_amounts, key=lambda x: abs(x - target))
    return {'type': 'raise', 'amount': best}


# ── Consulta al blueprint con posición correcta ───────────────────────────────

def _blueprint_query(state: dict, bucket_sims: int = 80) -> str:
    """
    Consulta el blueprint MCCFR para obtener la acción abstracta del bot.

    Corrección clave: la posición abstracta (0=SB, 1=BB) se determina
    comparando bot_is_sb con la posición SB en el árbol CFR,
    NO usando un índice fijo (el error original que causaba decisiones
    completamente invertidas en el 50 % de las manos).

    Parámetros
    ----------
    state       : dict – estado actual de la partida (_game_state)
    bucket_sims : int  – simulaciones Monte Carlo para calcular buckets

    Retorna
    -------
    str – una de las constantes ABSTRACT_ACTIONS
    """
    bp = _get_blueprint()
    if bp is None:
        return CALL   # fallback sin blueprint

    bot_is_sb  = state['bot_is_sb']
    # Posición abstracta: 0=SB, 1=BB  (corrección del bug de posición fija)
    abstract_position = 0 if bot_is_sb else 1

    street     = state['street']
    bot_hand   = state['bot_hand']
    board      = state['board']

    # Reconstruir historial abstracto de la calle actual
    initial_pot = _street_initial_pot(state)
    bet_hist    = list(_to_abstract_bet_hist(state['street_actions'], initial_pot))

    key = encode_infoset(
        position    = abstract_position,
        street_str  = street,
        hand_cards  = bot_hand,
        board_cards = board,
        bet_history = bet_hist,
        hand_sims   = bucket_sims,
        board_sims  = bucket_sims,
    )
    return bp.get_action(key)


def _street_initial_pot(state: dict) -> float:
    """Estima el pot al inicio de la calle actual para la reconstrucción de bet history."""
    street_idx = state['street_idx']
    if street_idx == 0:
        return 1.5   # preflop: SB(0.5) + BB(1.0)
    # Postflop: el pot al inicio de la calle es el pot actual menos las
    # apuestas de la calle actual
    street_invested = sum(
        float(a.get('amount', 0))
        for a in state['street_actions']
        if a.get('type') in ('call', 'raise', 'allin')
    )
    return max(state['pot'] - street_invested, 1.5)


# ── Override de equity pura (solo true all-in) ────────────────────────────────

def _equity_override(state: dict, abstract_act: str) -> str:
    """
    Sobreescribe la acción del blueprint con equity pura SOLO en situaciones
    de true all-in (bot tiene ≥95 % del stack en el pot), para evitar fold
    en spots obligatorios o call cuando hay equity suficiente.

    Corrección: el umbral anterior era ≥50 % del stack, lo que sobreescribía
    el blueprint en casi toda situación postflop. Ahora solo se activa con
    ≥95 % del stack comprometido (true all-in).

    Parámetros
    ----------
    state       : dict – estado actual de la partida
    abstract_act: str  – acción sugerida por el blueprint

    Retorna
    -------
    str – acción (posiblemente modificada)
    """
    bot_idx   = 0
    bot_stack = state['stacks'][bot_idx]
    pot       = state['pot']
    to_call   = state['to_call']

    # True all-in: el call comprometería ≥95 % del stack total inicial
    initial_stack = 100.0   # stack de inicio
    committed     = initial_stack - bot_stack
    if_call       = committed + to_call

    true_allin = (if_call / initial_stack) >= 0.95 if to_call > 0 else False

    if not true_allin:
        return abstract_act   # no sobreescribir en spots normales

    # En true all-in: decidir por equity pura
    try:
        from montecarlo import get_equity_cached
        board    = state['board']
        bot_hand = state['bot_hand']
        equity   = get_equity_cached(bot_hand, board, 2, 400)

        # Pot odds mínimas para call
        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0

        if equity >= pot_odds:
            return CALL
        else:
            return FOLD if to_call > 0 else CALL
    except Exception:
        return abstract_act


# ── Online learning (Experience Replay CFR) ───────────────────────────────────

def _online_learn(deal: dict, n_traversals: int = 80):
    """
    Tras cada mano completa, ejecuta n_traversals iteraciones MCCFR sobre
    el deal real para mejorar el blueprint (Experience Replay CFR, Pluribus).

    Esta técnica permite que el agente mejore su estrategia en spots
    que realmente ocurren en partidas, convergiendo más rápido en situaciones
    relevantes que el entrenamiento offline uniforme.

    Parámetros
    ----------
    deal        : dict – debe contener 'bot_hand', 'human_hand', 'full_board'
    n_traversals: int  – traversals MCCFR a ejecutar (default: 80)
    """
    global _blueprint, _hands_learned

    bp = _get_blueprint()
    if bp is None:
        return   # sin blueprint no hay nada que actualizar

    try:
        # Reconstruir el deal real para el trainer
        bot_hand   = deal['bot_hand']
        human_hand = deal['human_hand']
        full_board  = deal['full_board']   # 5 cartas pre-generadas

        bot_is_sb  = deal['bot_is_sb']
        hand0 = bot_hand   if bot_is_sb else human_hand
        hand1 = human_hand if bot_is_sb else bot_hand

        # El full_board tiene 5 cartas: flop(3), turn(1), river(1)
        flop  = full_board[:3]
        turn  = full_board[3:4]
        river = full_board[4:5]
        boards = [flop, turn, river]

        # Precomputar buckets una vez para todos los traversals
        bkts = MCCFRTrainer._precompute_buckets([hand0, hand1], boards, sims=60)

        init_kwargs = dict(
            bkts=bkts, boards=boards,
            street_idx=0, pot=1.5,
            stacks=[99.5, 99.0],
            contribs=[0.5, 1.0],
            bet_hist=[], n_raises=0,
            to_call=0.5, position=0,
        )

        with _blueprint_lock:
            for _ in range(n_traversals // 2):
                for traverser in [0, 1]:
                    bp._cfr(traverser, **init_kwargs)
            bp.iterations += n_traversals // 2

        _hands_learned += 1

        # Guardar blueprint cada _SAVE_EVERY manos aprendidas
        if _hands_learned % _SAVE_EVERY == 0:
            with _blueprint_lock:
                try:
                    bp.save(BLUEPRINT_PATH)
                    print(f"[online_learn] Blueprint guardado tras {_hands_learned} manos "
                          f"({bp.iterations:,} iters totales)")
                except Exception as save_exc:
                    print(f"[online_learn] Error guardando blueprint: {save_exc}")

    except Exception as exc:
        print(f"[online_learn] Error en traversal: {exc}")


# ── Lógica de juego ────────────────────────────────────────────────────────────

def _advance_street(state: dict):
    """Avanza a la siguiente calle revelando las cartas comunitarias."""
    street_map = {
        'preflop': ('flop',  3),
        'flop':    ('turn',  4),
        'turn':    ('river', 5),
    }
    current = state['street']
    if current not in street_map:
        state['finished'] = True
        return

    next_street, n_cards = street_map[current]
    state['street']       = next_street
    state['street_idx']  += 1
    state['board']        = state['full_board'][:n_cards]
    state['bet_hist']     = []
    state['n_raises']     = 0
    state['to_call']      = 0.0
    state['street_actions'] = []
    # En postflop el BB actúa primero
    state['active']       = state['bb_idx']


def _resolve_showdown(state: dict) -> dict:
    """Determina el ganador en showdown y calcula las ganancias."""
    try:
        from template import eval_hand_from_strings
        board     = state['full_board'][:5]
        bot_score = eval_hand_from_strings(state['bot_hand'],   board)
        hum_score = eval_hand_from_strings(state['human_hand'], board)
        pot       = state['pot']
        if bot_score > hum_score:
            winner, gain_bot, gain_human = 'bot',   pot, 0.0
        elif hum_score > bot_score:
            winner, gain_bot, gain_human = 'human', 0.0, pot
        else:
            winner, gain_bot, gain_human = 'tie',   pot / 2, pot / 2
    except Exception:
        # Fallback sin evaluador nativo
        winner, gain_bot, gain_human = 'tie', state['pot'] / 2, state['pot'] / 2

    return {
        'winner':      winner,
        'pot':         state['pot'],
        'gain_bot':    gain_bot,
        'gain_human':  gain_human,
        'bot_hand':    state['bot_hand'],
        'human_hand':  state['human_hand'],
        'board':       state['full_board'][:5],
    }


def _apply_action(state: dict, actor: str, action: dict) -> bool:
    """
    Aplica una acción al estado de la partida.

    Parámetros
    ----------
    state  : dict – estado de la partida
    actor  : str  – 'bot' o 'human'
    action : dict – {'type': ..., 'amount': ...}

    Retorna
    -------
    bool – True si la mano terminó
    """
    atype    = action.get('type', 'check')
    amount   = float(action.get('amount', 0.0))
    actor_idx = 0 if actor == 'bot' else 1
    opp_idx   = 1 - actor_idx

    state['hand_history'].append({'actor': actor, **action})
    state['street_actions'].append(action)

    if atype == 'fold':
        # El que hace fold pierde su contribución al pot
        state['finished'] = True
        state['winner']   = 'human' if actor == 'bot' else 'bot'
        gain = state['pot']
        if state['winner'] == 'bot':
            state['stacks'][0] += gain
        else:
            state['stacks'][1] += gain
        state['result'] = {
            'winner':    state['winner'],
            'reason':    'fold',
            'pot':       state['pot'],
            'gain_bot':  gain if state['winner'] == 'bot'   else 0.0,
            'gain_human': gain if state['winner'] == 'human' else 0.0,
        }
        return True

    elif atype in ('call', 'check'):
        call_amount = min(state['to_call'], state['stacks'][actor_idx])
        state['stacks'][actor_idx] -= call_amount
        state['pot']               += call_amount
        state['contribs'][actor_idx] += call_amount
        state['to_call'] = 0.0

        # Comprobar si la calle terminó (ambos igualaron)
        if state['contribs'][0] == state['contribs'][1] or state['stacks'][actor_idx] == 0:
            if state['street'] == 'river':
                result = _resolve_showdown(state)
                state['finished'] = True
                state['winner']   = result['winner']
                state['result']   = result
                # Actualizar stacks
                state['stacks'][0] += result['gain_bot']
                state['stacks'][1] += result['gain_human']
                return True
            else:
                _advance_street(state)
                return False

        # No terminó: el oponente actúa
        state['active'] = opp_idx
        return False

    elif atype in ('raise', 'allin'):
        state['stacks'][actor_idx]   -= amount
        state['pot']                 += amount
        state['contribs'][actor_idx] += amount
        state['to_call']              = max(0.0, state['contribs'][actor_idx]
                                            - state['contribs'][opp_idx])
        state['n_raises']            += 1
        state['active']               = opp_idx
        return False

    return False


def _bot_decision(state: dict) -> dict:
    """
    Calcula la acción del bot combinando blueprint + SPR caps + equity override.

    Retorna
    -------
    dict – {'type': ..., 'amount': ...}
    """
    pot       = state['pot']
    to_call   = state['to_call']
    bot_stack = state['stacks'][0]

    # 1. Consultar blueprint con posición correcta
    abstract_act = _blueprint_query(state)

    # 2. Override de equity pura (solo true all-in ≥95 % del stack)
    abstract_act = _equity_override(state, abstract_act)

    # 3. Traducir a acción concreta con SPR caps
    valid_raises = _valid_raise_amounts(state)
    return _translate_abstract_action(abstract_act, pot, to_call,
                                      bot_stack, valid_raises)


def _valid_raise_amounts(state: dict) -> list:
    """
    Genera una lista de montos de raise válidos para el bot basados en el
    pot y el stack actual (similar al motor de juego real).
    """
    pot       = state['pot']
    to_call   = state['to_call']
    bot_stack = state['stacks'][0]

    amounts = []
    for ratio in [1.0 / 3.0, 0.5, 1.0, 2.0]:
        amt = to_call + ratio * pot
        if amt < bot_stack:
            amounts.append(round(amt, 2))
    return amounts


# ── Endpoints Flask ────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Página principal con UI HTML básica."""
    return _html_page()


@app.route('/api/nueva_mano', methods=['POST'])
def nueva_mano():
    """
    Inicia una nueva mano de heads-up.

    Body JSON (opcional):
      { "stack": 100 }

    Respuesta:
      Estado inicial de la partida. Si el bot actúa primero (SB preflop),
      su acción ya va incluida en el estado.
    """
    global _game_state
    data       = request.get_json(silent=True) or {}
    stack      = float(data.get('stack', 100.0))
    _game_state = _new_game_state(stack)
    _load_blueprint()

    # Preflop: SB actúa primero. Si el bot es SB, que decida.
    _maybe_bot_acts()

    return jsonify(_public_state())


@app.route('/api/accion', methods=['POST'])
def accion():
    """
    El jugador humano realiza una acción.

    Body JSON:
      { "type": "fold" | "call" | "check" | "raise" | "allin",
        "amount": <float>  (solo para raise) }

    Respuesta:
      Estado actualizado de la partida, con la respuesta del bot ya incluida.
    """
    if not _game_state:
        return jsonify({'error': 'No hay partida en curso. Llama /api/nueva_mano primero.'}), 400
    if _game_state.get('finished'):
        return jsonify({'error': 'La mano ya terminó.', 'state': _public_state()}), 400

    # Verificar que es el turno del humano
    if _game_state.get('active') != 1:
        return jsonify({'error': 'No es tu turno.'}), 400

    data   = request.get_json(silent=True) or {}
    action = {'type': data.get('type', 'check'),
               'amount': float(data.get('amount', 0))}

    finished = _apply_action(_game_state, 'human', action)

    if not finished and not _game_state.get('finished'):
        _maybe_bot_acts()

    state_json = _public_state()

    # Aprendizaje online al terminar la mano
    if _game_state.get('finished'):
        threading.Thread(
            target=_online_learn,
            args=({
                'bot_hand':   _game_state['bot_hand'],
                'human_hand': _game_state['human_hand'],
                'full_board': _game_state['full_board'],
                'bot_is_sb':  _game_state['bot_is_sb'],
            },),
            daemon=True,
        ).start()

    return jsonify(state_json)


@app.route('/api/estado', methods=['GET'])
def estado():
    """Devuelve el estado público actual de la partida."""
    if not _game_state:
        return jsonify({'error': 'No hay partida en curso.'}), 404
    return jsonify(_public_state())


@app.route('/api/recargar_blueprint', methods=['POST'])
def recargar_blueprint():
    """
    Recarga el blueprint desde disco.
    Útil tras completar un entrenamiento nuevo con pre_entrenamiento.py.
    """
    global _blueprint, _blueprint_tried
    with _blueprint_lock:
        _blueprint_tried = False
        _blueprint       = None
    bp = _load_blueprint()
    if bp is None:
        return jsonify({
            'ok':      False,
            'message': f'Blueprint no encontrado en {BLUEPRINT_PATH}',
        }), 404
    return jsonify({
        'ok':        True,
        'message':   'Blueprint recargado correctamente.',
        'iters':     bp.iterations,
        'infosets':  len(bp.regret_sum),
    })


@app.route('/api/stats', methods=['GET'])
def stats():
    """Devuelve estadísticas de convergencia y aprendizaje online."""
    bp = _get_blueprint()
    data = {
        'blueprint_loaded':  bp is not None,
        'hands_learned':     _hands_learned,
        'save_every':        _SAVE_EVERY,
        'postflop_buckets':  POSTFLOP_BUCKETS,
        'preflop_buckets':   PREFLOP_BUCKETS,
    }
    if bp is not None:
        n_is = len(bp.regret_sum)
        iters = bp.iterations
        data.update({
            'blueprint_iters':    iters,
            'blueprint_infosets': n_is,
            'visits_per_infoset': round(iters * 2 / max(n_is, 1), 1),
        })
    return jsonify(data)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _maybe_bot_acts():
    """Si el turno es del bot y la mano no terminó, ejecuta su acción."""
    while (not _game_state.get('finished')
           and _game_state.get('active') == 0):
        action   = _bot_decision(_game_state)
        finished = _apply_action(_game_state, 'bot', action)
        if finished:
            break


def _public_state() -> dict:
    """Estado de la partida seguro para enviar al cliente (sin mostrar mano del bot)."""
    gs = _game_state
    return {
        'finished':      gs.get('finished', False),
        'winner':        gs.get('winner'),
        'result':        gs.get('result'),
        'street':        gs.get('street'),
        'board':         gs.get('board', []),
        'pot':           round(gs.get('pot', 0), 2),
        'to_call':       round(gs.get('to_call', 0), 2),
        'your_hand':     gs.get('human_hand', []),
        'your_stack':    round(gs.get('stacks', [100, 100])[1], 2),
        'bot_stack':     round(gs.get('stacks', [100, 100])[0], 2),
        'bot_is_sb':     gs.get('bot_is_sb', True),
        'active':        'bot' if gs.get('active') == 0 else 'human',
        'hand_history':  gs.get('hand_history', []),
        # Revelar mano del bot solo al terminar
        'bot_hand':      gs.get('bot_hand', []) if gs.get('finished') else [],
    }


def _html_page() -> str:
    """Página HTML mínima para interactuar con la partida vía fetch()."""
    return """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Agente IA Poker</title>
<style>
  body { font-family: monospace; background: #1a1a1a; color: #eee; padding: 20px; }
  h1   { color: #f0c040; }
  pre  { background: #222; padding: 10px; border-radius: 4px; white-space: pre-wrap; }
  button { margin: 4px; padding: 8px 16px; font-size: 14px; cursor: pointer;
           background: #333; color: #eee; border: 1px solid #555; border-radius: 4px; }
  button:hover { background: #555; }
  input[type=number] { width: 80px; padding: 6px; background: #333; color: #eee;
                        border: 1px solid #555; border-radius: 4px; }
  #log { max-height: 300px; overflow-y: auto; }
</style>
</head>
<body>
<h1>🂡 Agente IA Poker (HUNL)</h1>
<div>
  <button onclick="nuevaMano()">Nueva mano</button>
  <button onclick="cargarStats()">Stats</button>
  <button onclick="recargarBlueprint()">Recargar blueprint</button>
</div>
<hr>
<pre id="estado">Pulsa "Nueva mano" para empezar.</pre>
<div id="acciones"></div>
<hr>
<pre id="log"></pre>
<script>
async function api(method, path, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch(path, opts);
  return r.json();
}
function mostrar(data) {
  document.getElementById('estado').textContent = JSON.stringify(data, null, 2);
  mostrarAcciones(data);
}
function log(msg) {
  const el = document.getElementById('log');
  el.textContent += msg + '\\n';
  el.scrollTop = el.scrollHeight;
}
async function nuevaMano() {
  const data = await api('POST', '/api/nueva_mano', {});
  mostrar(data); log('→ Nueva mano iniciada.');
}
async function accion(tipo, amount) {
  const body = { type: tipo };
  if (amount !== undefined) body.amount = amount;
  const data = await api('POST', '/api/accion', body);
  mostrar(data);
  if (data.finished) log('✓ Mano terminada. Ganador: ' + data.winner);
}
async function cargarStats() {
  const data = await api('GET', '/api/stats');
  log('Stats: ' + JSON.stringify(data));
}
async function recargarBlueprint() {
  const data = await api('POST', '/api/recargar_blueprint');
  log('Reload: ' + JSON.stringify(data));
}
function mostrarAcciones(state) {
  const div = document.getElementById('acciones');
  if (state.finished || state.active !== 'human') { div.innerHTML = ''; return; }
  const pot   = state.pot   || 1;
  const small = Math.round(pot * 0.33 * 100) / 100;
  const big   = Math.round(pot * 1.0  * 100) / 100;
  div.innerHTML = `
    <button onclick="accion('fold')">Fold</button>
    <button onclick="accion('call')">Call / Check</button>
    <button onclick="accion('raise', ${small})">Raise 1/3 pot (${small} BB)</button>
    <button onclick="accion('raise', ${big})">Raise pot (${big} BB)</button>
    <span>Custom raise: <input type="number" id="custom_amt" value="${small}" min="0" step="0.5">
    <button onclick="accion('raise', +document.getElementById('custom_amt').value)">Raise</button></span>
    <button onclick="accion('allin', ${state.your_stack})">All-in (${state.your_stack} BB)</button>
  `;
}
</script>
</body>
</html>"""


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Servidor web del agente IA de póker.')
    parser.add_argument('--host',  default='127.0.0.1', help='Host (default: 127.0.0.1)')
    parser.add_argument('--port',  type=int, default=5000, help='Puerto (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Modo debug Flask')
    args = parser.parse_args()

    print(f"Iniciando servidor en http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
