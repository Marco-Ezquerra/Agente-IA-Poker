"""
Codificador de Information Sets (InfoSets) para MCCFR en HUNL.

Un InfoSet representa el conjunto de estados del juego que el jugador activo
NO puede distinguir entre sí (ve sus propias cartas y el historial público,
pero no las cartas del rival).

Clave canónica
--------------
    (position, street_idx, hand_bucket, board_bucket_tuple, bet_history_tuple)

  position         : int  – 0 = SB, 1 = BB
  street_idx       : int  – 0=preflop, 1=flop, 2=turn, 3=river
  hand_bucket      : int  – resultado de preflop_bucket o postflop_bucket
  board_bucket_tuple : tuple[int] – bucket por calle hasta la actual
                       vacía en preflop, (flop_b,) en flop, etc.
  bet_history_tuple  : tuple[str] – secuencia de acciones abstractas
                       de la calle actual

Acciones abstractas
-------------------
  FOLD, CALL (incluye check), RAISE_THIRD (1/3 pot), RAISE_HALF,
  RAISE_POT, RAISE_2POT, ALLIN

Referencia: Bowling et al. (2015) "Heads-up limit hold'em poker is solved"
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from .card_abstractor import (
    preflop_bucket, postflop_bucket,
    PREFLOP_BUCKETS, POSTFLOP_BUCKETS,
)

# ── Acciones abstractas ────────────────────────────────────────────────────────

FOLD        = 'f'
CALL        = 'c'       # incluye check (to_call == 0)
RAISE_THIRD = 'r1'      # raise ≈ 1/3 pot
RAISE_HALF  = 'r2'      # raise ≈ 1/2 pot
RAISE_POT   = 'r3'      # raise ≈ 1× pot
RAISE_2POT  = 'r4'      # raise ≈ 2× pot
ALLIN       = 'ai'

ABSTRACT_ACTIONS = [FOLD, CALL, RAISE_THIRD, RAISE_HALF, RAISE_POT, RAISE_2POT, ALLIN]
NUM_ACTIONS      = len(ABSTRACT_ACTIONS)   # 7
ACTION_IDX       = {a: i for i, a in enumerate(ABSTRACT_ACTIONS)}

# Ratios de tamaño de raise respecto al pot
RAISE_RATIOS = {
    RAISE_THIRD: 1.0 / 3.0,
    RAISE_HALF:  1.0 / 2.0,
    RAISE_POT:   1.0,
    RAISE_2POT:  2.0,
    ALLIN:       999.0,
}

STREETS     = ['preflop', 'flop', 'turn', 'river']
STREET_IDX  = {s: i for i, s in enumerate(STREETS)}


# ── Codificación del InfoSet ───────────────────────────────────────────────────

def encode_infoset(position, street_str, hand_cards, board_cards, bet_history,
                   hand_sims=200, board_sims=150):
    """
    Genera la clave hashable del InfoSet para la situación actual.

    Parámetros
    ----------
    position    : int       – 0 (SB) o 1 (BB)
    street_str  : str       – 'preflop' | 'flop' | 'turn' | 'river'
    hand_cards  : list[str] – 2 cartas del jugador en formato compacto
    board_cards : list[str] – cartas comunitarias visibles hasta ahora
    bet_history : list[str] – secuencia de acciones abstractas de esta calle
    hand_sims   : int       – simulaciones para calcular hand bucket
    board_sims  : int       – simulaciones para calcular board buckets

    Retorna
    -------
    tuple hashable que identifica unívocamente el InfoSet.
    """
    street = STREET_IDX.get(street_str, 0)

    if street == 0:
        # Preflop: solo bucket de la mano (no hay board)
        hb       = preflop_bucket(hand_cards, num_sims=hand_sims)
        bb_tuple = ()
    else:
        # Postflop: bucket de la mano con board actual
        hb = postflop_bucket(hand_cards, board_cards, num_sims=hand_sims)

        # Un bucket por cada calle revelada HASTA la actual
        # (captura textura del board en cada calle independientemente)
        flop  = board_cards[:3] if len(board_cards) >= 3 else []
        turn  = board_cards[:4] if len(board_cards) >= 4 else []
        river = board_cards[:5] if len(board_cards) >= 5 else []

        flop_b  = postflop_bucket(hand_cards, flop,  num_sims=board_sims) if flop  else 0
        turn_b  = postflop_bucket(hand_cards, turn,  num_sims=board_sims) if turn  else 0
        river_b = postflop_bucket(hand_cards, river, num_sims=board_sims) if river else 0

        # Solo incluir calles que ya han ocurrido
        all_buckets = (flop_b, turn_b, river_b)
        bb_tuple    = all_buckets[:street]

    return (position, street, hb, bb_tuple, tuple(bet_history))


# ── Conversión acción real → acción abstracta ─────────────────────────────────

def abstract_action(action, pot):
    """
    Convierte una acción del motor de juego en una acción abstracta.

    Parámetros
    ----------
    action : str o tuple – 'fold', 'call', 'check', 'all in', ('raise', amount)
    pot    : float       – tamaño actual del bote (en BBs)

    Retorna
    -------
    str – una de las constantes ABSTRACT_ACTIONS
    """
    if action in ('fold', 'f'):
        return FOLD
    if action in ('call', 'check', 'c'):
        return CALL
    if action in ('all in', 'allin'):
        return ALLIN

    if isinstance(action, tuple) and action[0] == 'raise':
        amount = action[1]
        ratio  = amount / pot if pot > 0 else 1.0
        if   ratio <= 0.42:  return RAISE_THIRD
        elif ratio <= 0.62:  return RAISE_HALF
        elif ratio <= 1.25:  return RAISE_POT
        else:                return RAISE_2POT

    # Fallback
    return CALL


def concrete_raise_amount(abstract_act, pot, to_call, stack):
    """
    Convierte una acción abstracta de raise en un monto concreto (en BBs).

    Parámetros
    ----------
    abstract_act : str   – RAISE_THIRD | RAISE_HALF | RAISE_POT | RAISE_2POT | ALLIN
    pot          : float – bote actual
    to_call      : float – fichas necesarias para igualar
    stack        : float – fichas disponibles del jugador activo

    Retorna
    -------
    float – monto total a añadir (to_call + raise extra), acotado por stack
    """
    if abstract_act == ALLIN:
        return stack

    ratio      = RAISE_RATIOS.get(abstract_act, 1.0)
    raise_extra = ratio * pot
    total       = to_call + raise_extra
    return min(total, stack)
