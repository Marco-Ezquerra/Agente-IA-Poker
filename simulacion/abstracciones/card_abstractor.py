"""
Módulo de abstracción de cartas para MCCFR.

Implementa:
  - EHS  (Expected Hand Strength): P(ganar showdown con cartas actuales)
  - EHS² = EHS + (1-EHS)*ppot - EHS*npot
      ppot = potencial positivo  (pasar de perder a ganar con futuras cartas)
      npot = potencial negativo  (pasar de ganar a perder con futuras cartas)
  - Buckets preflop  : 0 .. PREFLOP_BUCKETS-1  (10 clases)
  - Buckets postflop : 0 .. POSTFLOP_BUCKETS-1  (8 clases)

Con 8 buckets postflop el espacio de InfoSets pasa de ~795k a ~21k
(×38 mejor cobertura). A 200k iters se consiguen ~19 visitas/InfoSet
(convergencia real frente a estrategia esencialmente aleatoria con 50 buckets).

Referencia: Johanson et al. (2013) "Measuring the size of large no-limit poker games"
"""

import random
import sys
import os

# Asegurar que simulacion/ está en el path aunque se importe desde cfr/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

PREFLOP_BUCKETS  = 10
POSTFLOP_BUCKETS = 8

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['s', 'h', 'd', 'c']


# ── Utilidades de baraja ──────────────────────────────────────────────────────

def _full_deck():
    return [r + s for r in RANKS for s in SUITS]


def _remove_known(deck, known):
    known_set = set(known)
    return [c for c in deck if c not in known_set]


# ── Evaluación de mano ────────────────────────────────────────────────────────

try:
    from template import eval_hand_from_strings as _native_eval
    _NATIVE = True
except Exception:
    _NATIVE = False


def _rank_five(five):
    """
    Evalúa exactamente 5 cartas con heurística rápida.
    Retorna un entero (mayor = mejor mano).
    Categorías: 0=high card, 1=pair, 2=two pair, 3=trips,
                4=straight, 5=flush, 6=full house, 7=quads, 8=str.flush
    Dentro de cada categoría se añade un desempate de rangos.
    """
    from collections import Counter
    ranks = [RANKS.index(c[0]) for c in five]
    suits = [c[1] for c in five]
    cnt   = Counter(ranks)
    counts = sorted(cnt.values(), reverse=True)
    is_flush    = len(set(suits)) == 1
    sorted_r    = sorted(ranks)
    is_straight = (sorted_r[-1] - sorted_r[0] == 4 and len(set(sorted_r)) == 5)
    # Rueda (A-2-3-4-5)
    if sorted_r == [0, 1, 2, 3, 12]:
        is_straight = True
        sorted_r    = [0, 1, 2, 3, 4]

    if   is_straight and is_flush:  cat = 8
    elif counts[0] == 4:            cat = 7
    elif counts[:2] == [3, 2]:      cat = 6
    elif is_flush:                  cat = 5
    elif is_straight:               cat = 4
    elif counts[0] == 3:            cat = 3
    elif counts[:2] == [2, 2]:      cat = 2
    elif counts[0] == 2:            cat = 1
    else:                           cat = 0

    # Desempate: kickers en orden descendente de frecuencia
    kickers = sorted(ranks, key=lambda r: (cnt[r], r), reverse=True)
    tiebreak = sum(k * (13 ** i) for i, k in enumerate(reversed(kickers)))
    return cat * (13 ** 5) + tiebreak


def _eval_hand(hole, board):
    """Mejor mano de 5 usando las 7 cartas (2 hole + hasta 5 board)."""
    if _NATIVE:
        return _native_eval(hole, board)
    from itertools import combinations
    cards = hole + board
    return max(_rank_five(list(c)) for c in combinations(cards, 5))


# ── EHS ───────────────────────────────────────────────────────────────────────

def compute_ehs(hole, board, num_sims=300):
    """
    Expected Hand Strength: probabilidad de ganar contra una mano aleatoria
    del oponente, completando el board aleatoriamente.

    Parámetros
    ----------
    hole   : list[str]  – 2 cartas del jugador ('Ah', 'Ks', ...)
    board  : list[str]  – 0-5 cartas comunitarias visibles
    num_sims : int      – iteraciones Monte Carlo

    Retorna
    -------
    float en [0, 1]
    """
    wins = ties = 0
    known = set(hole + board)
    deck  = _remove_known(_full_deck(), known)
    n_fill = 5 - len(board)

    for _ in range(num_sims):
        random.shuffle(deck)
        opp  = deck[:2]
        fill = deck[2:2 + n_fill]
        full_board = board + fill

        my = _eval_hand(hole, full_board)
        op = _eval_hand(opp,  full_board)
        if   my > op: wins += 1
        elif my == op: ties += 1

    total = num_sims
    return (wins + 0.5 * ties) / total


# ── EHS² ──────────────────────────────────────────────────────────────────────

def compute_ehs2(hole, board, num_sims=300):
    """
    EHS² = EHS + (1 - EHS) * ppot - EHS * npot

    Incorpora el potencial positivo (ppot) y negativo (npot) de la mano.
    En river (5 cartas) no hay futuro: devuelve EHS directamente.

    Parámetros
    ----------
    hole   : list[str]
    board  : list[str]  – 0-4 cartas (si =5 se usa EHS puro)
    num_sims : int

    Retorna
    -------
    float en [0, 1]
    """
    if len(board) >= 5:
        return compute_ehs(hole, board, num_sims)

    # Matriz de potencial HP[estado_actual][estado_futuro]
    # estado: 0=ganando, 1=empate, 2=perdiendo
    HP = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    known  = set(hole + board)
    deck   = _remove_known(_full_deck(), known)
    n_fill = 5 - len(board)

    for _ in range(num_sims):
        random.shuffle(deck)
        opp  = deck[:2]
        fill = deck[2:2 + n_fill]
        full_board = board + fill

        # Estado actual (con board incompleto si lo hay)
        if board:
            from itertools import combinations
            cards_my = hole + board
            cards_op = opp  + board
            n = len(cards_my)
            my_cur = max(_rank_five(list(c)) for c in combinations(cards_my, min(5, n))) if n >= 5 else 0
            op_cur = max(_rank_five(list(c)) for c in combinations(cards_op, min(5, n))) if n >= 5 else 0
        else:
            my_cur = op_cur = 0  # preflop sin board: tratar como empate

        if   my_cur > op_cur: cur = 0
        elif my_cur < op_cur: cur = 2
        else:                  cur = 1

        # Estado futuro (board completo)
        my_fut = _eval_hand(hole, full_board)
        op_fut = _eval_hand(opp,  full_board)
        if   my_fut > op_fut: fut = 0
        elif my_fut < op_fut: fut = 2
        else:                  fut = 1

        HP[cur][fut] += 1

    total = num_sims or 1

    # EHS con potencial
    ehs = (sum(HP[0]) + 0.5 * sum(HP[1])) / total

    # ppot: estábamos perdiendo/empate y ganamos
    ppot_num = HP[2][0] + 0.5 * HP[2][1] + 0.5 * HP[1][0]
    ppot_den = sum(HP[2]) + sum(HP[1]) or 1
    ppot = ppot_num / ppot_den

    # npot: estábamos ganando/empate y perdemos
    npot_num = HP[0][2] + 0.5 * HP[0][1] + 0.5 * HP[1][2]
    npot_den = sum(HP[0]) + sum(HP[1]) or 1
    npot = npot_num / npot_den

    ehs2 = ehs + (1.0 - ehs) * ppot - ehs * npot
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

def postflop_bucket(hand, board, num_sims=200):
    """
    Asigna un bucket 0..POSTFLOP_BUCKETS-1 a una (mano, board).
    Usa EHS² para incorporar potencial de mejora.

    Parámetros
    ----------
    hand  : list[str] – 2 cartas del jugador
    board : list[str] – 3-5 cartas comunitarias visibles
    num_sims : int

    Retorna
    -------
    int
    """
    ehs2   = compute_ehs2(hand, board, num_sims)
    bucket = int(ehs2 * POSTFLOP_BUCKETS)
    return max(0, min(POSTFLOP_BUCKETS - 1, bucket))
