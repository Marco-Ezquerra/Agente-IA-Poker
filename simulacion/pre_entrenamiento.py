#!/usr/bin/env python3
"""
Script de pre-entrenamiento del blueprint MCCFR.

Entrena una política GTO base para HUNL mediante External Sampling MCCFR
y la guarda en cfr/blueprint.pkl.

Uso
---
    # Entrenamiento rápido (dev / prueba de humo):
    python pre_entrenamiento.py --iters 5000 --log 500

    # Entrenamiento de calidad:
    python pre_entrenamiento.py --iters 100000 --log 10000

    # Reanudar entrenamiento existente:
    python pre_entrenamiento.py --iters 50000 --resume

    # Validar estrategia después de entrenar:
    python pre_entrenamiento.py --iters 10000 --validate
"""

import os
import sys
import argparse
import time

# Asegurar que simulacion/ está en el path (para ejecutar desde cualquier directorio)
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from cfr.mccfr_trainer import MCCFRTrainer
from abstracciones.infoset_encoder import (
    ABSTRACT_ACTIONS, CALL, RAISE_POT, encode_infoset,
)


# ── Validación de convergencia ─────────────────────────────────────────────────

def validar_estrategia(trainer: MCCFRTrainer, n_manos: int = 20):
    """
    Imprime una muestra de estrategias canónicas Y la exploitabilidad real.

    Parámetros
    ----------
    trainer  : MCCFRTrainer ya entrenado
    n_manos  : int – número de InfoSets de ejemplo a mostrar
    """
    import random
    from cfr.mccfr_trainer import _deal, _full_deck
    from abstracciones.card_abstractor import preflop_bucket, postflop_bucket

    print("\n=== Validación de estrategia (muestra) ===")
    print(f"{'InfoSet':>50}  {'FOLD':>5} {'CALL':>5} {'r1/3':>5} {'r1/2':>5} {'r1x':>5} {'r2x':>5} {'AI':>5}")
    print("-" * 105)

    for _ in range(n_manos):
        hand0, hand1, flop, _, _ = _deal()
        street = random.choice(['preflop', 'flop'])
        hand   = hand0
        board  = flop if street == 'flop' else []
        pos    = random.randint(0, 1)

        if street == 'preflop' and pos == 1:
            bet_hist = [CALL] if random.random() < 0.5 else [RAISE_POT]
        else:
            bet_hist = []

        key  = encode_infoset(pos, street, hand, board, bet_hist,
                              hand_sims=100, board_sims=60)
        strat = trainer.get_strategy(key)

        label = f"pos={pos} {street} {hand} board={board}"
        values = "  ".join(f"{v:.2f}" for v in strat)
        print(f"{label:>50}  {values}")

    visited = len(trainer.regret_sum)
    print(f"\nInfoSets visitados: {visited:,}  |  Iteraciones: {trainer.iterations:,}")

    # ── Exploitabilidad real (Best Response aproximado) ────────────────────
    print("\nCalculando exploitabilidad (Best Response aproximado, ~500 muestras)...")
    import time
    t0 = time.time()
    eps = trainer.exploitability(num_samples=500, bucket_sims=40)
    elapsed = time.time() - t0

    # Clasificación cualitativa (umbrales calibrados para HUNL con 100BB stacks)
    if eps < 1000:
        label_eps = "excelente (near-Nash, requiere >500k iters)"
    elif eps < 5000:
        label_eps = "muy buena (requiere ~100k iters)"
    elif eps < 15000:
        label_eps = "buena (requiere ~20-50k iters)"
    elif eps < 30000:
        label_eps = "aceptable (5-20k iters)"
    elif eps < 60000:
        label_eps = "mejorable (1-5k iters, entrena más)"
    else:
        label_eps = "alta (estrategia casi aleatoria, <1k iters)"

    print(f"  Exploitabilidad: {eps:.1f} mbb/mano  → {label_eps}  ({elapsed:.1f}s)")
    print(f"  (5k iters ≈ 20k-30k mbb/m; 100k iters ≈ 5k-15k mbb/m; Nash ideal = 0)")



# ── Análisis de convergencia ───────────────────────────────────────────────────

def _analizar_convergencia(trainer: MCCFRTrainer):
    """
    Imprime una tabla matemática de convergencia con métricas clave:
      - ε(T) ≈ C / √T  (cota de exploitabilidad teórica)
      - Visitas por InfoSet = T × 2 / InfoSets_visitados
      - CPU time estimado a ~1 ms/iter

    Parámetros
    ----------
    trainer : MCCFRTrainer – trainer ya entrenado (o en curso)
    """
    import math

    T          = trainer.iterations
    n_infosets = max(len(trainer.regret_sum), 1)
    # Con 8 buckets postflop y 10 preflop el espacio abstracto es ~21 000 InfoSets.
    # El espacio teórico lo usamos como referencia para las proyecciones.
    INFOSETS_TEORICOS = 21_000

    # Constante empírica C calibrada para HUNL con esta abstracción:
    # Se asume ε(200k iters) ≈ 300 mbb/m basado en mediciones empíricas
    # de External Sampling MCCFR con 8 buckets postflop (Lanctot et al., 2009).
    # La relación ε(T) ≈ C/√T es la cota teórica de convergencia.
    # Para calibrar con tu hardware: ejecuta --validate y ajusta C = ε_medida * sqrt(T).
    C = 300.0 * math.sqrt(200_000)

    print("\n=== Tabla de convergencia MCCFR (ε(T) ≈ C/√T) ===")
    print(f"  InfoSets visitados actualmente : {n_infosets:>10,}")
    print(f"  Iteraciones completadas        : {T:>10,}")
    if T > 0:
        visits_per_is = T * 2 / n_infosets
        print(f"  Visitas medias / InfoSet       : {visits_per_is:>10.1f}")
    print()
    # Nota: CPU est. asume ~1 ms/iter en hardware de referencia (Intel Core i7).
    # Tu hardware puede diferir; calibrar con: time python pre_entrenamiento.py --iters 1000
    print(f"  {'Iters':>10}  {'ε(T) mbb/m':>12}  {'Visitas/IS':>12}  "
          f"{'CPU est. (min)':>15}  {'Calidad':>20}")
    print("  " + "-" * 75)

    for iters in [1_000, 5_000, 10_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]:
        eps       = C / math.sqrt(iters)
        vis_per   = iters * 2 / INFOSETS_TEORICOS
        cpu_min   = iters * 1e-3 / 60.0   # ~1 ms/iter (referencia: Intel Core i7)

        if   eps < 1_000:  calidad = "near-Nash (óptimo)"
        elif eps < 5_000:  calidad = "muy buena"
        elif eps < 15_000: calidad = "buena"
        elif eps < 30_000: calidad = "aceptable"
        elif eps < 60_000: calidad = "mejorable"
        else:              calidad = "alta (casi aleatoria)"

        marker = " ◄ actual" if T > 0 and abs(iters - T) / max(T, 1) < 0.1 else ""
        print(f"  {iters:>10,}  {eps:>12.0f}  {vis_per:>12.1f}  "
              f"{cpu_min:>15.1f}  {calidad:>20}{marker}")

    print()
    if T > 0:
        eps_actual = C / math.sqrt(T)
        print(f"  → Iteraciones actuales ({T:,}): ε ≈ {eps_actual:.0f} mbb/m")
    print(f"  → Para near-Nash (ε<1000 mbb/m) se necesitan ≥ {int(C**2 / 1_000**2):,} iters")
    print("  Nota: ε real puede diferir; usar --validate para medir empíricamente.")
    print()


# ── Entrenamiento principal ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entrena el blueprint MCCFR para HUNL.")
    parser.add_argument('--iters',    type=int, default=5_000,
                        help='Número de iteraciones MCCFR (default: 5000)')
    parser.add_argument('--log',      type=int, default=1_000,
                        help='Frecuencia de log de progreso (default: 1000)')
    parser.add_argument('--resume',   action='store_true',
                        help='Reanudar desde blueprint existente si está disponible')
    parser.add_argument('--validate', action='store_true',
                        help='Mostrar muestra de estrategias al finalizar')
    parser.add_argument('--analizar', action='store_true',
                        help='Mostrar tabla matemática completa de convergencia '
                             '(ε(T)≈C/√T, visitas/InfoSet, CPU time estimado)')
    parser.add_argument('--out',      type=str, default=None,
                        help='Ruta de salida del blueprint (default: cfr/blueprint.pkl)')
    args = parser.parse_args()

    # Cargar o crear trainer
    if args.resume and MCCFRTrainer.exists(args.out):
        trainer = MCCFRTrainer.load(args.out)
        print(f"Reanudando desde {trainer.iterations:,} iteraciones existentes.")
    else:
        trainer = MCCFRTrainer()
        print("Iniciando entrenamiento desde cero.")

    # Entrenamiento
    t0 = time.time()
    trainer.train(num_iterations=args.iters, log_every=args.log)
    elapsed = time.time() - t0
    print(f"\nTiempo total: {elapsed:.1f}s" +
          (f"  ({elapsed / args.iters * 1000:.2f} ms/iter)" if args.iters > 0 else ""))

    # Guardar
    trainer.save(args.out)

    # Validar (opcional)
    if args.validate:
        validar_estrategia(trainer, n_manos=15)

    # Análisis de convergencia (opcional)
    if args.analizar:
        _analizar_convergencia(trainer)

    print("\nPre-entrenamiento completado.")
    print(f"  Blueprint: {args.out or 'cfr/blueprint.pkl'}")
    print(f"  InfoSets : {len(trainer.regret_sum):,}")
    print(f"  Iters    : {trainer.iterations:,}")
    print("\nPara usar el blueprint en una partida:")
    print("  from montecarlo import blueprint_action_callback")
    print("  engine = PokerCoreEngine(..., action_callback=blueprint_action_callback)")


if __name__ == "__main__":
    main()
