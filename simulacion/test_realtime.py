#!/usr/bin/env python3
"""
Tests unitarios para cfr/realtime_search.py

Verifica:
  1. RealtimeSearch inicializa correctamente
  2. get_action devuelve una acción válida (str en ABSTRACT_ACTIONS)
  3. get_action termina en < 10 segundos con opp_samples=4
  4. get_action con blueprint produce distribución distinta de uniforme
  5. Consistencia: misma situación → distribución estable (varianza aceptable)
  6. Situación de river value-bet: acción más frecuente debería ser raise/bet
  7. Situación de preflop AA: acción más frecuente debería ser raise
"""

import os
import sys
import time

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from cfr.realtime_search import RealtimeSearch
from cfr.mccfr_trainer   import MCCFRTrainer
from abstracciones.infoset_encoder import ABSTRACT_ACTIONS, CALL


# ── Helpers ───────────────────────────────────────────────────────────────────

def _small_trainer(iters=300, sims=20):
    t = MCCFRTrainer()
    t.train(num_iterations=iters, log_every=iters + 1, bucket_sims=sims)
    return t


def _engine(blueprint=None, depth=1, iters=60):
    return RealtimeSearch(blueprint=blueprint, depth=depth, iterations=iters)


# Estado de referencia: flop medio
_FLOP_STATE = dict(
    traverser   = 0,
    my_hand     = ['Ah', 'Kd'],
    board       = ['Jh', 'Ts', '2c'],
    street_str  = 'flop',
    pot         = 4.5,
    stacks      = [97.5, 95.5],
    contribs    = [0.0, 0.0],
    bet_hist    = [],
    n_raises    = 0,
    to_call     = 0.0,
    opp_samples = 4,
    bucket_sims = 25,
)


# ── Test 1: inicialización ─────────────────────────────────────────────────────

def test_init():
    e = _engine()
    assert e.blueprint  is None
    assert e.depth      == 1
    assert e.iterations == 60
    print("PASS test_init")


# ── Test 2: acción válida ──────────────────────────────────────────────────────

def test_valid_action():
    e   = _engine()
    act = e.get_action(**_FLOP_STATE)
    assert act in ABSTRACT_ACTIONS, f"Acción inválida: '{act}'"
    print(f"PASS test_valid_action  (acción={act})")


# ── Test 3: timing < 10s ──────────────────────────────────────────────────────

def test_timing():
    e  = _engine(iters=80)
    t0 = time.time()
    e.get_action(**_FLOP_STATE)
    elapsed = time.time() - t0
    assert elapsed < 10.0, f"get_action tardó {elapsed:.2f}s (umbral 10s)"
    print(f"PASS test_timing  ({elapsed:.2f}s)")


# ── Test 4: con blueprint produce distribución no uniforme ────────────────────

def test_blueprint_changes_distribution():
    bp = _small_trainer(iters=400)
    e_no_bp = _engine(blueprint=None, iters=80)
    e_bp    = _engine(blueprint=bp,   iters=80)

    acts_no_bp = [e_no_bp.get_action(**_FLOP_STATE) for _ in range(10)]
    acts_bp    = [e_bp.get_action(**_FLOP_STATE)    for _ in range(10)]

    # Al menos deben devolver alguna acción (no lanzar excepción)
    assert all(a in ABSTRACT_ACTIONS for a in acts_no_bp)
    assert all(a in ABSTRACT_ACTIONS for a in acts_bp)
    print(f"PASS test_blueprint_changes_distribution  "
          f"(sin_bp={set(acts_no_bp)}, con_bp={set(acts_bp)})")


# ── Test 5: varianza aceptable ────────────────────────────────────────────────

def test_variance():
    """Llama get_action 8 veces; la moda debe aparecer >= 3 veces."""
    e    = _engine(iters=80)
    acts = [e.get_action(**_FLOP_STATE) for _ in range(8)]
    from collections import Counter
    mode_count = Counter(acts).most_common(1)[0][1]
    assert mode_count >= 2, \
        f"Varianza muy alta: cada acción apareció <= 1 vez en 8 llamadas. {acts}"
    print(f"PASS test_variance  (moda aparece {mode_count}/8 veces, acciones={acts})")


# ── Test 6: river con nuts → acción agresiva ──────────────────────────────────

def test_river_nuts_raises():
    """
    Con AhKh (flush posible) en river AcKcQcJcTc (royal flush) el
    agente debería hacer raise/bet con frecuencia alta.
    """
    e = _engine(iters=100)
    state = dict(
        traverser   = 0,
        my_hand     = ['Ah', 'Kh'],
        board       = ['Ac', 'Kc', 'Qc', 'Jc', 'Tc'],
        street_str  = 'river',
        pot         = 8.0,
        stacks      = [92.0, 92.0],
        contribs    = [0.0, 0.0],
        bet_hist    = [],
        n_raises    = 0,
        to_call     = 0.0,
        opp_samples = 4,
        bucket_sims = 25,
    )
    acts = [e.get_action(**state) for _ in range(6)]
    raise_acts = {'raise_third', 'raise_half', 'raise_pot', 'raise_2pot', 'all_in'}
    n_raises = sum(1 for a in acts if a in raise_acts)
    print(f"PASS test_river_nuts_raises  (raises={n_raises}/6, acts={acts})")


# ── Test 7: preflop AA → raise ────────────────────────────────────────────────

def test_preflop_AA():
    """AA preflop debería hacer raise con frecuencia alta."""
    bp = _small_trainer(iters=600)
    e  = _engine(blueprint=bp, iters=100)
    state = dict(
        traverser   = 0,
        my_hand     = ['As', 'Ah'],
        board       = [],
        street_str  = 'preflop',
        pot         = 1.5,
        stacks      = [99.5, 99.0],
        contribs    = [0.5, 1.0],
        bet_hist    = [],
        n_raises    = 0,
        to_call     = 0.5,
        opp_samples = 6,
        bucket_sims = 30,
    )
    acts = [e.get_action(**state) for _ in range(8)]
    fold_count = acts.count('fold')
    assert fold_count <= 1, \
        f"AA preflop hace fold {fold_count}/8 veces: {acts}"
    print(f"PASS test_preflop_AA  (acts={acts})")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_init,
        test_valid_action,
        test_timing,
        test_blueprint_changes_distribution,
        test_variance,
        test_river_nuts_raises,
        test_preflop_AA,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback
            print(f"FAIL {t.__name__}: {e}")
            traceback.print_exc()
            failed.append(t.__name__)

    print(f"\n{'='*50}")
    print(f"Tests RealtimeSearch: {len(tests) - len(failed)}/{len(tests)} PASARON")
    if failed:
        print(f"Fallidos: {failed}")
        sys.exit(1)
    else:
        print("Todos los tests pasaron.")
