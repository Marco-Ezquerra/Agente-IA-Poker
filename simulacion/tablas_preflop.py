# preflop_range_setup.py

import json
import random
from itertools import combinations
from template import rank_map, suit_map      # Importar mapas de rango y palo
from poker_engine import compact_card

# === preflop_ranges.json (skeleton) ===
# Guarda esto en la raíz de tu proyecto como 'preflop_ranges.json'
preflop_ranges_skeleton = {
    "SB": {
        # SB actúa primero: open (raise), limp (call), fold
        "open": ["AsAh", "KsKh"],  # ejemplo
        "limp": [],
        "fold": [],
        # Si SB abrió y BB 3-betea, SB puede fold, call o 4-bet
        "vs_3bet": {
            "fold": [],
            "call": [],
            "4bet": []
        }
    },
    "BB": {
        # BB responde: call, check, fold, 3-bet
        "call": [],
        "check": [],
        "fold": [],
        "3bet": [],
        # Si BB hace 3-bet y SB 4-betea, BB puede fold, call o 5-bet
        "vs_4bet": {
            "fold": [],
            "call": [],
            "5bet": []
        }
    }
}

def save_skeleton(path="preflop_ranges.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(preflop_ranges_skeleton, f, indent=2, ensure_ascii=False)

def load_preflop_ranges(path="preflop_ranges.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# cargar o crear skeleton si no existe
try:
    preflop_ranges = load_preflop_ranges()
except FileNotFoundError:
    save_skeleton()
    preflop_ranges = load_preflop_ranges()


def determine_action_key_from_history(history):
    """
    Devuelve clave principal:
      - 'open' si SB (id=0) hace raise
      - 'limp' si SB hace call sin raise previo
      - 'fold' si SB foldea
      - '3bet' si BB (id=1) hace raise tras open
    """
    phase_started = False
    sb_opened = False
    for event in history:
        if event[0] == "phase" and event[1] == "preflop":
            phase_started = True
        if not phase_started:
            continue
        if event[0] == "r" and event[1] == 0:
            sb_opened = True
            return "open"
        if event[0] == "c" and event[1] == 0:
            return "limp"
        if event[0] == "fold" and event[1] == 0:
            return "fold"
        if sb_opened and event[0] == "r" and event[1] == 1:
            return "3bet"
    return "open"


def determine_subkey_from_history(history):
    """
    Para vs_3bet o vs_4bet, devuelve:
      - 'fold','call','4bet' para vs_3bet
      - 'fold','call','5bet' para vs_4bet
    """
    for event in history:
        if event[0] == "fold" and event[1] == 0 and "vs_3bet" in preflop_ranges["SB"]:
            return "fold"
        if event[0] == "c" and event[1] == 0 and "vs_3bet" in preflop_ranges["SB"]:
            return "call"
        if event[0] == "r" and event[1] == 0 and "vs_3bet" in preflop_ranges["SB"]:
            return "4bet"
        if event[0] == "fold" and event[1] == 1 and "vs_4bet" in preflop_ranges["BB"]:
            return "fold"
        if event[0] == "c" and event[1] == 1 and "vs_4bet" in preflop_ranges["BB"]:
            return "call"
        if event[0] == "r" and event[1] == 1 and "vs_4bet" in preflop_ranges["BB"]:
            return "5bet"
    return None


def get_rival_hand(position, action_key, subkey=None):
    """
    Retorna una tupla de dos strings de cartas basadas en posición y acción.
    Fallback: mano aleatoria.
    """
    bucket = preflop_ranges.get(position, {})
    if subkey and isinstance(bucket.get(action_key), dict):
        hands = bucket[action_key].get(subkey, [])
    else:
        hands = bucket.get(action_key, [])
    if not hands:
        deck = [r + s for r in rank_map for s in suit_map]
        combos = list(combinations(deck, 2))
        return random.choice(combos)
    return random.choice(hands)
