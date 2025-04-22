import ctypes
import os
import random

##  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/lib/.libs !!!!
## IMPORTANTE INCLUIR LA RUTA DE LA LIBRERIA EN LA TERMINAL
# === Configuración de ruta a la librería desde subcarpeta ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LIB_PATH = os.path.join(BASE_DIR, 'libhand_eval.so')
LIBS_DIR = os.path.join(BASE_DIR, 'lib/.libs')
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':' + LIBS_DIR

# === Cargar la librería ===
lib = ctypes.CDLL(LIB_PATH)
lib.eval_hand.argtypes = [ctypes.c_int] * 14
lib.eval_hand.restype = ctypes.c_int

# === Mapeo de cartas ===
rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5,
            '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
suit_map = {'s': 0, 'h': 1, 'd': 2, 'c': 3}
suit_inv = ['s', 'h', 'd', 'c']
rank_inv = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

# === Funciones útiles ===

def card_to_tuple(card_str):
    """Convierte una carta tipo 'As' en (12, 0)"""
    return (rank_map[card_str[0]], suit_map[card_str[1]])

def tuple_to_card(card_tuple):
    """Convierte (12, 0) en 'As'"""
    return rank_inv[card_tuple[0]] + suit_inv[card_tuple[1]]

def eval_hand_from_strings(hand, board):
    """Evalúa una mano y devuelve su score usando 2+5 cartas tipo 'As'"""
    cards = hand + board
    ints = [val for card in cards for val in card_to_tuple(card)]
    
    return lib.eval_hand(*ints)

def create_deck():
    """Devuelve una baraja completa tipo ['2s', '2h', ..., 'As']"""
    return [r + s for r in rank_map for s in suit_map]

def draw_random_cards(n):
    """Extrae n cartas aleatorias sin repetición"""
    return random.sample(create_deck(), n)
