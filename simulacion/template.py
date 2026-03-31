import ctypes
import os
import random
from collections import Counter
from itertools import combinations

##  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/lib/.libs !!!!
## IMPORTANTE INCLUIR LA RUTA DE LA LIBRERIA EN LA TERMINAL
# === Configuración de ruta a la librería desde subcarpeta ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LIB_PATH = os.path.join(BASE_DIR, 'libhand_eval.so')
LIBS_DIR = os.path.join(BASE_DIR, 'lib/.libs')
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':' + LIBS_DIR

# === Mapeo de cartas ===
rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5,
            '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
suit_map = {'s': 0, 'h': 1, 'd': 2, 'c': 3}
suit_inv = ['s', 'h', 'd', 'c']
rank_inv = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

# === Evaluador nativo (poker-eval via ctypes) — opcional =====================
_lib = None
try:
    os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':' + LIBS_DIR
    _lib = ctypes.CDLL(LIB_PATH)
    _lib.eval_hand.argtypes = [ctypes.c_int] * 14
    _lib.eval_hand.restype  = ctypes.c_int
except OSError:
    _lib = None   # fallback: treys o evaluador puro Python

# === Evaluador treys (rápido, puro Python, sin dependencias externas) ========
_treys_eval = None
try:
    from treys import Card as _TreysCard, Evaluator as _TreysEvaluator
    _treys_eval = _TreysEvaluator()
except ImportError:
    _treys_eval = None

# === Evaluador puro Python (fallback cuando libhand_eval.so no está disponible) =

_RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']


def _rank_five(five):
    """
    Evalúa exactamente 5 cartas con heurística rápida.
    Retorna un entero (mayor = mejor mano).
    Categorías: 0=high card … 8=straight flush.
    """
    ranks = [_RANKS.index(c[0]) for c in five]
    suits = [c[1] for c in five]
    cnt   = Counter(ranks)
    counts = sorted(cnt.values(), reverse=True)
    is_flush    = len(set(suits)) == 1
    sorted_r    = sorted(ranks)
    is_straight = (sorted_r[-1] - sorted_r[0] == 4 and len(set(sorted_r)) == 5)
    if sorted_r == [0, 1, 2, 3, 12]:          # rueda A-2-3-4-5
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

    kickers  = sorted(ranks, key=lambda r: (cnt[r], r), reverse=True)
    tiebreak = sum(k * (13 ** i) for i, k in enumerate(reversed(kickers)))
    return cat * (13 ** 5) + tiebreak


def _py_eval_hand(hand, board):
    """Mejor mano de 5 usando las 7 cartas (2 hole + hasta 5 board). Puro Python."""
    cards = hand + board
    return max(_rank_five(list(combo)) for combo in combinations(cards, 5))


# === Funciones públicas =====================================================

def card_to_tuple(card_str):
    """Convierte una carta tipo 'As' en (rank_int, suit_int)."""
    return (rank_map[card_str[0]], suit_map[card_str[1]])


def tuple_to_card(card_tuple):
    """Convierte (rank_int, suit_int) en 'As'."""
    return rank_inv[card_tuple[0]] + suit_inv[card_tuple[1]]


def eval_hand_from_strings(hand, board):
    """
    Evalúa una mano de póker (2 hole + hasta 5 board) y devuelve su score.

    Prioridad: libhand_eval.so (C) → treys → Python puro.
    Convención: mayor = mejor mano.

    Parámetros
    ----------
    hand  : list[str]  – 2 cartas del jugador ('Ah', 'Ks', ...)
    board : list[str]  – 3-5 cartas comunitarias

    Retorna
    -------
    int – puntuación de la mejor mano de 5 (mayor = mejor)
    """
    if _lib is not None:
        cards = hand + board
        ints = [val for card in cards for val in card_to_tuple(card)]
        while len(ints) < 14:
            ints.append(0)
        return _lib.eval_hand(*ints[:14])
    if _treys_eval is not None and len(board) == 5:
        # treys devuelve 1 (mejor) … 7462 (peor) → negamos para mayor=mejor
        board_cards = [_TreysCard.new(c) for c in board]
        hole_cards  = [_TreysCard.new(c) for c in hand]
        return -_treys_eval.evaluate(board_cards, hole_cards)
    return _py_eval_hand(hand, board)


def create_deck():
    """Devuelve una baraja completa tipo ['2s', '2h', ..., 'As']."""
    return [r + s for r in rank_map for s in suit_map]


def draw_random_cards(n):
    """Extrae n cartas aleatorias sin repetición."""
    return random.sample(create_deck(), n)
