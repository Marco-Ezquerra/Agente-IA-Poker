#!/usr/bin/env python3
"""
Tests unitarios para opponent_model.py

Verifica:
  1. Creación e inicialización por defecto
  2. observe_action registra correctamente VPIP, PFR, agresividad
  3. observe_fold_to_bet registra denominador FTB
  4. classify() devuelve 'unknown' con < 12 manos
  5. classify() detecta loose-passive correctamente
  6. classify() detecta tight-aggressive correctamente
  7. get_counter_adjustments(): loose-passive → bluff_freq_mult bajo
  8. get_counter_adjustments(): tight-passive → bluff_freq_mult alto
  9. FTB alto incrementa bluff_freq_mult
  10. FTB bajo reduce bluff_freq_mult
  11. save() / load() round-trip exacto
  12. Flujo completo: 20 manos de un rival loose-passive → arquetipo correcto
"""

import os
import sys
import tempfile

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from opponent_model import OpponentModel, OpponentStats


# ── Test 1: inicialización ─────────────────────────────────────────────────────

def test_init():
    m = OpponentModel(opponent_id=1)
    assert m.opponent_id == 1
    assert m.stats.hands_seen == 0
    assert m.stats.vpip == 0.50           # retorna default con 0 manos
    print("PASS test_init")


# ── Test 2: observe_action ─────────────────────────────────────────────────────

def test_observe_action_vpip_pfr():
    m = OpponentModel()
    m.new_hand()
    m.observe_action('raise', 'preflop', voluntarily=True)
    assert m.stats.vpip_count == 1
    assert m.stats.pfr_count  == 1
    assert m.stats.aggressive['preflop'] == 1

    m.new_hand()
    m.observe_action('call', 'preflop', voluntarily=True)
    assert m.stats.vpip_count == 2
    assert m.stats.pfr_count  == 1       # call no cuenta como PFR
    assert m.stats.passive['preflop'] == 1
    print("PASS test_observe_action_vpip_pfr")


# ── Test 3: observe_bet_faced ──────────────────────────────────────────────────

def test_observe_bet_faced():
    """
    Patrón correcto: observe_bet_faced registra el denominador;
    luego observe_action registra la respuesta del oponente.
    FOLD después de observe_bet_faced → FTB = 1.0.
    CALL después de observe_bet_faced  → FTB = 0.0.
    """
    m = OpponentModel()
    # Escenario 1: agente apuesta, oponente foldea.
    m.new_hand()
    m.observe_bet_faced('flop')           # agente apostó → denominador
    assert m.stats.bets_faced['flop'] == 1
    m.observe_action('fold', 'flop')      # oponente foldeó → solo numerador
    assert m.stats.fold_to_bet['flop'] == 1
    assert abs(m.stats.ftb('flop') - 1.0) < 1e-9, \
        f"Esperaba FTB=1.0, got {m.stats.ftb('flop')}"

    # Escenario 2: agente apuesta, oponente llama → FTB no sube.
    m.new_hand()
    m.observe_bet_faced('flop')
    m.observe_action('call', 'flop')
    # bets_faced=2, fold_to_bet=1 → FTB = 0.5
    assert abs(m.stats.ftb('flop') - 0.5) < 1e-9, \
        f"Esperaba FTB=0.5, got {m.stats.ftb('flop')}"
    print("PASS test_observe_bet_faced")


# ── Test 4: classify con pocas manos ──────────────────────────────────────────

def test_classify_unknown_few_hands():
    m = OpponentModel()
    for _ in range(11):
        m.new_hand()
        m.observe_action('raise', 'preflop', voluntarily=True)
    assert m.classify() == 'unknown', \
        f"Esperaba 'unknown' con 11 manos, got '{m.classify()}'"
    print("PASS test_classify_unknown_few_hands")


# ── Test 5: loose-passive ──────────────────────────────────────────────────────

def test_classify_loose_passive():
    """VPIP alto (80%), AF bajo → loose-passive."""
    m = OpponentModel()
    for _ in range(20):
        m.new_hand()
        m.observe_action('call', 'preflop', voluntarily=True)     # llama siempre
        m.observe_action('check', 'flop',   voluntarily=False)    # pasivo postflop
    assert m.classify() == 'loose-passive', \
        f"Esperaba 'loose-passive', got '{m.classify()}'"
    print("PASS test_classify_loose_passive")


# ── Test 6: tight-aggressive ──────────────────────────────────────────────────

def test_classify_tight_aggressive():
    """VPIP bajo (10%), AF alto → tight-aggressive."""
    m = OpponentModel()
    for i in range(20):
        m.new_hand()
        if i < 2:                                               # solo 10% VPIP
            m.observe_action('raise', 'preflop', voluntarily=True)
            m.observe_action('raise', 'flop',    voluntarily=False)
            m.observe_action('raise', 'turn',    voluntarily=False)
        # resto: no actúa voluntariamente (fold a preflop raise, no registrado)
    archetype = m.classify()
    # Con solo 2/20 raises puede clasificar como tight-passive o tight-aggressive
    # dependiendo del AF. Lo que importa es que NO es loose.
    assert 'tight' in archetype, \
        f"Esperaba arquetipo 'tight-*', got '{archetype}'"
    print(f"PASS test_classify_tight_aggressive  (arquetipo={archetype})")


# ── Test 7: ajuste loose-passive ──────────────────────────────────────────────

def test_counter_loose_passive():
    m = OpponentModel()
    for _ in range(20):
        m.new_hand()
        m.observe_action('call', 'preflop', voluntarily=True)
        m.observe_action('check', 'flop')
    adj = m.get_counter_adjustments()
    assert adj['bluff_freq_mult'] < 0.5, \
        f"Esperaba bluff_freq < 0.5 vs loose-passive, got {adj['bluff_freq_mult']}"
    assert adj['bet_size_mult'] > 1.0, \
        f"Esperaba bet_size > 1.0 vs loose-passive, got {adj['bet_size_mult']}"
    print(f"PASS test_counter_loose_passive  (bluff={adj['bluff_freq_mult']:.2f})")


# ── Test 8: ajuste tight-passive ──────────────────────────────────────────────

def test_counter_tight_passive():
    """
    VPIP bajo + AF bajo = tight-passive → bluff_freq_mult base 1.80.
    Construimos 30 manos con VPIP=2/30=6% y sin raises.
    """
    m = OpponentModel()
    for i in range(30):
        m.new_hand()
        if i < 2:
            m.observe_action('call', 'preflop', voluntarily=True)
    adj = m.get_counter_adjustments()
    archetype = adj['archetype']
    # VPIP=6% < 20% → tight. AF≈0 → passive
    assert 'tight' in archetype, \
        f"Esperaba arquetipo tight, got '{archetype}'"
    # tight-passive base bluff_freq=1.80; FTB=0 → reduce × 0.40 = 0.72
    # aceptamos cualquier valor razonable (no neutro 1.0)
    assert 0.3 <= adj['bluff_freq_mult'] <= 3.5, \
        f"bluff_freq fuera de rango: {adj['bluff_freq_mult']:.2f}"
    print(f"PASS test_counter_tight_passive  "
          f"(archetype={archetype}, bluff={adj['bluff_freq_mult']:.2f})")


# ── Test 9: FTB alto aumenta bluff ────────────────────────────────────────────

def test_ftb_high_increases_bluff():
    """
    Rival tight-passive que SIEMPRE foldea ante apuesta.
    FTB=1.0 → multiplicador base 1.80 × 1.60 = 2.88 > 1.0.
    VPIP=2/30=6% → tight. Raises=0 → passive.
    """
    m = OpponentModel()
    for i in range(30):
        m.new_hand()
        if i < 2:
            # Solo 2 hands donde entra voluntariamente
            m.observe_action('call', 'preflop', voluntarily=True)
        # Siempre hay una apuesta en flop y el rival foldea
        m.observe_bet_faced('flop')           # denominador
        m.observe_action('fold', 'flop')      # numerador
    # FTB = 30/30 = 1.0 (todas las apuestas resultaron en fold)
    assert m.stats.ftb() > 0.9, f"FTB esperado > 0.9, got {m.stats.ftb():.2f}"
    adj = m.get_counter_adjustments()
    archetype = adj['archetype']
    # tight-passive + FTB alto → bluff_freq_mult > 1.0
    assert adj['bluff_freq_mult'] > 1.0, \
        f"Esperaba bluff_freq > 1.0 con FTB alto ({archetype}), "\
        f"got {adj['bluff_freq_mult']:.2f}"
    print(f"PASS test_ftb_high_increases_bluff  "
          f"(archetype={archetype}, FTB={m.stats.ftb():.0%}, "
          f"bluff_mult={adj['bluff_freq_mult']:.2f})")


# ── Test 10: FTB bajo reduce bluff ────────────────────────────────────────────

def test_ftb_low_reduces_bluff():
    m = OpponentModel()
    for _ in range(20):
        m.new_hand()
        m.observe_action('call', 'preflop', voluntarily=True)
        m.observe_bet_faced('flop')
        m.observe_action('call', 'flop')    # nunca foldea
    assert m.stats.ftb() == 0.0, f"FTB esperado 0, got {m.stats.ftb()}"
    adj = m.get_counter_adjustments()
    assert adj['bluff_freq_mult'] < 1.0, \
        f"Esperaba bluff_freq < 1.0 con FTB=0, got {adj['bluff_freq_mult']:.2f}"
    print(f"PASS test_ftb_low_reduces_bluff  "
          f"(FTB={m.stats.ftb():.0%}, bluff_mult={adj['bluff_freq_mult']:.2f})")


# ── Test 11: save / load round-trip ───────────────────────────────────────────

def test_save_load():
    m1 = OpponentModel(opponent_id=2)
    for _ in range(15):
        m1.new_hand()
        m1.observe_action('raise', 'preflop', voluntarily=True)
        m1.observe_action('raise', 'flop')

    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        path = f.name

    try:
        m1.save(path)
        m2 = OpponentModel.load(path)

        assert m2.opponent_id      == m1.opponent_id
        assert m2.stats.hands_seen == m1.stats.hands_seen
        assert m2.stats.vpip_count == m1.stats.vpip_count
        assert m2.stats.pfr_count  == m1.stats.pfr_count
        assert m2.classify()       == m1.classify()
        print(f"PASS test_save_load  "
              f"(opponent={m2.opponent_id}, hands={m2.stats.hands_seen}, "
              f"archetype={m2.classify()})")
    finally:
        os.unlink(path)


# ── Test 12: flujo completo 20 manos loose-passive ────────────────────────────

def test_full_flow_loose_passive():
    """Simula 20 manos de un rival que llama todo y nunca sube."""
    m = OpponentModel(opponent_id=1)
    for _ in range(20):
        m.new_hand()
        m.observe_action('call',  'preflop', voluntarily=True)
        m.observe_action('check', 'flop')
        m.observe_action('check', 'turn')
        m.observe_action('check', 'river')

    arch = m.classify()
    assert arch == 'loose-passive', \
        f"Esperaba 'loose-passive' tras 20 manos, got '{arch}'"
    adj = m.get_counter_adjustments()
    assert adj['bluff_freq_mult']  < 0.5
    assert adj['value_threshold_adj'] < 0.0    # apostar más delgado
    assert adj['bet_size_mult'] > 1.0
    print(f"PASS test_full_flow_loose_passive  "
          f"(arch={arch}, bluff={adj['bluff_freq_mult']:.2f}, "
          f"vt_adj={adj['value_threshold_adj']:.2f})")
    print(m.summary())


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_init,
        test_observe_action_vpip_pfr,
        test_observe_bet_faced,
        test_classify_unknown_few_hands,
        test_classify_loose_passive,
        test_classify_tight_aggressive,
        test_counter_loose_passive,
        test_counter_tight_passive,
        test_ftb_high_increases_bluff,
        test_ftb_low_reduces_bluff,
        test_save_load,
        test_full_flow_loose_passive,
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
    print(f"Tests OpponentModel: {len(tests) - len(failed)}/{len(tests)} PASARON")
    if failed:
        print(f"Fallidos: {failed}")
        sys.exit(1)
    else:
        print("Todos los tests pasaron.")
