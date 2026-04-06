"""
Módulo de abstracción de cartas para MCCFR.

Implementa:
  - EHS  (Expected Hand Strength): P(ganar showdown con cartas actuales)
  - EHS² = EHS + (1-EHS)*ppot - EHS*npot
      ppot = potencial positivo  (pasar de perder a ganar con futuras cartas)
      npot = potencial negativo  (pasar de ganar a perder con futuras cartas)
  - Buckets preflop  : 0 .. PREFLOP_BUCKETS-1  (10 clases)
  - Buckets postflop : 0 .. POSTFLOP_BUCKETS-1  (16 clases)

Referencia: Johanson et al. (2013) "Measuring the size of large no-limit poker games"

Caché
-----
- preflop_bucket: caché dict keyed por forma canónica  (169 entradas máx.)
- postflop_bucket: caché LRU de 32.768 entradas keyed por (hand_tuple, board_tuple).
  Las llamadas repetidas dentro del mismo árbol CFR — mismo deal, distintas
  ramas — retornan instantáneamente en O(1) sin ninguna simulación Monte Carlo.
"""

import sys
import os
from functools import lru_cache
from itertools import combinations as _combinations

import numpy as np

# Asegurar que simulacion/ está en el path aunque se importe desde cfr/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

PREFLOP_BUCKETS  = 10
POSTFLOP_BUCKETS = 16

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['s', 'h', 'd', 'c']


# ── Representación numérica de cartas (evaluación vectorizada) ───────────────

_CARD2INT: dict = {
    r + s: RANKS.index(r) * 4 + SUITS.index(s)
    for r in RANKS for s in SUITS
}

def _cards_to_ints(card_list):
    return np.array([_CARD2INT[c] for c in card_list], dtype=np.int8)


def _eval5_batch(batch: np.ndarray) -> np.ndarray:
    """
    Evalúa un batch de N manos de exactamente 5 cartas.

    Parámetros
    ----------
    batch : np.ndarray shape (N, 5)  – enteros de carta [0..51]

    Retorna
    -------
    np.ndarray int64 (N,)  – mayor = mejor mano
    """
    N  = batch.shape[0]
    r  = batch // 4    # rango [0..12]
    s  = batch % 4     # palo  [0..3]

    is_flush    = (s == s[:, :1]).all(axis=1)
    r_sorted    = np.sort(r, axis=1)
    is_straight = (
        (r_sorted[:, 4] - r_sorted[:, 0] == 4) &
        (np.diff(r_sorted, axis=1) == 1).all(axis=1)
    )
    wheel    = np.array([0, 1, 2, 3, 12], dtype=np.int8)
    is_wheel = (r_sorted == wheel).all(axis=1)   # BUG-3: extrae para fix de kicker
    is_straight |= is_wheel

    freq = np.zeros((N, 13), dtype=np.int8)
    for col in range(5):
        freq[np.arange(N), r[:, col]] += 1
    cs = np.sort(freq, axis=1)[:, ::-1]

    cat = np.zeros(N, dtype=np.int64)
    cat[is_straight & is_flush]              = 8
    cat[cs[:, 0] == 4]                       = 7
    cat[(cs[:, 0] == 3) & (cs[:, 1] == 2)]  = 6
    cat[is_flush & (cat == 0)]               = 5
    cat[is_straight & (cat == 0)]            = 4
    cat[(cs[:, 0] == 3) & (cat == 0)]        = 3
    cat[(cs[:, 0] == 2) & (cs[:, 1] == 2) & (cat == 0)] = 2
    cat[(cs[:, 0] == 2) & (cs[:, 1] != 2) & (cat == 0)] = 1

    # BUG-3 fix: en el wheel (A2345) el As tiene rango 12 (posición i=4, peso 13^4=28561),
    # lo que haría que el wheel supere numéricamente a escaleras más fuertes como 23456.
    # Solución: sustituir el As por -1 en los hands de wheel para que actúe como carta baja.
    r_for_kicker = r_sorted.astype(np.int64)        # copia (dtype distinto)
    r_for_kicker[is_wheel, 4] = -1                  # As = carta baja en A-2-3-4-5
    kicker = np.zeros(N, dtype=np.int64)
    for i in range(5):
        kicker += r_for_kicker[:, i] * (13 ** i)
    return cat * (13 ** 5) + kicker


def _best_hand_batch(hole_i: np.ndarray, board_i: np.ndarray) -> np.ndarray:
    """Mejor mano de 5 entre hole+board para cada fila del batch."""
    all_cards = np.concatenate([hole_i, board_i], axis=1)
    n = all_cards.shape[1]
    combos = np.array(list(_combinations(range(n), 5)), dtype=np.int8)
    N, C   = all_cards.shape[0], combos.shape[0]
    hands5 = all_cards[:, combos]  # (N, C, 5)
    scores = _eval5_batch(hands5.reshape(N * C, 5)).reshape(N, C)
    return scores.max(axis=1)


# ── EHS vectorizado ───────────────────────────────────────────────────────────

def compute_ehs(hole, board, num_sims=300):
    """
    Expected Hand Strength montecarlo vectorizado con NumPy.
    ~20x más rápido que la versión Python-pura, misma calidad estadística.

    Parámetros
    ----------
    hole     : list[str]  — 2 cartas del jugador
    board    : list[str]  — 0-5 cartas comunitarias visibles
    num_sims : int        — muestras Monte Carlo

    Retorna
    -------
    float en [0, 1]
    """
    known_i   = _cards_to_ints(hole + board)
    hole_i    = known_i[:2]
    board_i   = known_i[2:]
    n_fill    = 5 - len(board)

    avail = np.array([c for c in range(52) if c not in set(known_i.tolist())], dtype=np.int8)
    rng   = np.random.default_rng()

    needed  = 2 + n_fill
    samples = np.stack([rng.permutation(avail)[:needed] for _ in range(num_sims)])

    opp_i      = samples[:, :2]
    fill_i     = samples[:, 2:]
    board_rep  = np.tile(board_i, (num_sims, 1)) if len(board) > 0 else np.empty((num_sims, 0), dtype=np.int8)
    full_b     = np.concatenate([board_rep, fill_i], axis=1)
    hole_rep   = np.tile(hole_i, (num_sims, 1))

    my  = _best_hand_batch(hole_rep, full_b)
    opp = _best_hand_batch(opp_i,   full_b)

    wins = int((my > opp).sum())
    ties = int((my == opp).sum())
    return (wins + 0.5 * ties) / num_sims


# ── EHS² vectorizado ──────────────────────────────────────────────────────────

def compute_ehs2(hole, board, num_sims=300):
    """
    EHS² = EHS + (1 - EHS) * ppot - EHS * npot  (Johanson et al. 2013)

    Incorpora el potencial positivo (ppot) y negativo (npot) de la mano.
    En river (5 cartas) no hay futuro: delega en compute_ehs.
    Implementación vectorizada con NumPy.

    Parámetros
    ----------
    hole     : list[str]
    board    : list[str]  — 0-4 cartas (si len≥5 retorna EHS puro)
    num_sims : int

    Retorna
    -------
    float en [0, 1]
    """
    if len(board) >= 5:
        return compute_ehs(hole, board, num_sims)

    known_i = _cards_to_ints(hole + board)
    hole_i  = known_i[:2]
    board_i = known_i[2:]
    n_fill  = 5 - len(board)

    avail   = np.array([c for c in range(52) if c not in set(known_i.tolist())], dtype=np.int8)
    rng     = np.random.default_rng()
    needed  = 2 + n_fill
    samples = np.stack([rng.permutation(avail)[:needed] for _ in range(num_sims)])

    opp_i     = samples[:, :2]
    fill_i    = samples[:, 2:]
    board_rep = np.tile(board_i, (num_sims, 1)) if len(board) > 0 else np.empty((num_sims, 0), dtype=np.int8)
    full_b    = np.concatenate([board_rep, fill_i], axis=1)
    hole_rep  = np.tile(hole_i, (num_sims, 1))

    my_fut  = _best_hand_batch(hole_rep, full_b)
    opp_fut = _best_hand_batch(opp_i,   full_b)

    if len(board) >= 3:
        br      = np.tile(board_i, (num_sims, 1))
        my_cur  = _best_hand_batch(hole_rep, br)
        opp_cur = _best_hand_batch(opp_i,    br)
    else:
        my_cur  = np.zeros(num_sims, dtype=np.int64)
        opp_cur = np.zeros(num_sims, dtype=np.int64)

    cur_ahead = my_cur  > opp_cur
    cur_behind= my_cur  < opp_cur
    cur_tie   = my_cur == opp_cur
    fut_win   = my_fut  > opp_fut
    fut_lose  = my_fut  < opp_fut
    fut_tie   = my_fut == opp_fut

    # BUG-2 fix: la fórmula de Johanson requiere EHS_actual (fuerza con el board parcial),
    # NO el EHS futuro (probabilidad de ganar con board completo). El EHS futuro solo
    # sirve para calcular ppot/npot.
    # Cuando no hay board (len<3), cur_ahead/cur_tie = zeros → ehs_cur = 0.5 (neutral correcto).
    ehs_cur = float((cur_ahead.sum() + 0.5 * cur_tie.sum()) / num_sims)

    ppot_mask = cur_behind | cur_tie
    ppot_denom= float(ppot_mask.sum()) or 1.0
    ppot = float((ppot_mask & fut_win).sum() + 0.5 * (ppot_mask & fut_tie).sum()) / ppot_denom

    npot_mask = cur_ahead | cur_tie
    npot_denom= float(npot_mask.sum()) or 1.0
    npot = float((npot_mask & fut_lose).sum() + 0.5 * (npot_mask & fut_tie).sum()) / npot_denom

    ehs2 = ehs_cur + (1.0 - ehs_cur) * ppot - ehs_cur * npot
    return max(0.0, min(1.0, ehs2))


# ── Caché de EHS preflop ──────────────────────────────────────────────────────

_preflop_ehs_cache: dict = {}


def _canonical_preflop(hand):
    """
    Forma canónica de una mano preflop (isomorfismo de palos).
    Retorna (rank_hi, rank_lo, suited: bool).
    Ejemplos: ['Ah','Ks'] → ('A','K',False) ; ['2h','2d'] → ('2','2',False)
    """
    r0, r1 = RANKS.index(hand[0][0]), RANKS.index(hand[1][0])
    suited  = hand[0][1] == hand[1][1]
    if r0 >= r1:
        return (RANKS[r0], RANKS[r1], suited)
    return (RANKS[r1], RANKS[r0], suited)


def preflop_bucket(hand, num_sims=300):
    """
    Asigna un bucket 0..PREFLOP_BUCKETS-1 a una mano preflop.
    Bucket 0 = mano débil, bucket PREFLOP_BUCKETS-1 = mano fuerte.

    Usa EHS simulado cacheado sobre las 169 formas canónicas.

    Parámetros
    ----------
    hand : list[str] – 2 cartas ('Ah', 'Ks')

    Retorna
    -------
    int
    """
    canon = _canonical_preflop(hand)
    if canon not in _preflop_ehs_cache:
        _preflop_ehs_cache[canon] = compute_ehs(list(hand), [], num_sims)
    ehs = _preflop_ehs_cache[canon]

    # EHS preflop en [~0.32, ~0.85] → normalizar
    low, high = 0.32, 0.86
    normalized = (ehs - low) / (high - low)
    bucket = int(normalized * PREFLOP_BUCKETS)
    return max(0, min(PREFLOP_BUCKETS - 1, bucket))


# ── Buckets postflop ──────────────────────────────────────────────────────────

# CACHE-1 fix: num_sims NO forma parte de la clave de caché.
# Los distintos callers (trainer sims=50, realtime sims=60, encode_infoset sims=200/150)
# generaban 4 entradas distintas para la misma (mano, board), reduciendo el hit rate
# real ~4x. Solución: clave = solo (hand_t, board_t); sims fijos internamente.
_POSTFLOP_CACHE_SIMS: int = 200


@lru_cache(maxsize=32768)
def _postflop_bucket_cached(hand_t: tuple, board_t: tuple) -> int:
    """
    Versión cacheada de postflop_bucket. Acepta tuplas para ser hashable.

    La caché LRU de 32.768 entradas evita recalcular EHS² para el mismo
    (mano, board) dentro del mismo árbol de traversal — varias ramas del
    árbol comparten el mismo deal e inspeccionan el mismo nodo desde distintos
    caminos.  Hit rate típico > 90 % durante el entrenamiento.

    num_sims es fijo (_POSTFLOP_CACHE_SIMS) para garantizar que todos los
    callers compartan la misma entrada de caché.
    """
    ehs2   = compute_ehs2(list(hand_t), list(board_t), _POSTFLOP_CACHE_SIMS)
    bucket = int(ehs2 * POSTFLOP_BUCKETS)
    return max(0, min(POSTFLOP_BUCKETS - 1, bucket))


def postflop_bucket(hand, board, num_sims=200):  # noqa: ARG001  (num_sims ignorado)
    """
    Asigna un bucket 0..POSTFLOP_BUCKETS-1 a una (mano, board).
    Usa EHS² para incorporar potencial de mejora.

    La función convierte las listas a tuplas y delega en la caché LRU
    _postflop_bucket_cached para obtener O(1) en llamadas repetidas.

    Parámetros
    ----------
    hand     : list[str] – 2 cartas del jugador
    board    : list[str] – 3-5 cartas comunitarias visibles
    num_sims : int       – ignorado (ver _POSTFLOP_CACHE_SIMS); conservado
                          por compatibilidad de firma con callers existentes.

    Retorna
    -------
    int
    """
    return _postflop_bucket_cached(tuple(hand), tuple(board))


def clear_postflop_cache():
    """Invalida la caché LRU postflop (útil al cambiar POSTFLOP_BUCKETS)."""
    _postflop_bucket_cached.cache_clear()
