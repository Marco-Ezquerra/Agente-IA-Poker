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

    print("\nPre-entrenamiento completado.")
    print(f"  Blueprint: {args.out or 'cfr/blueprint.pkl'}")
    print(f"  InfoSets : {len(trainer.regret_sum):,}")
    print(f"  Iters    : {trainer.iterations:,}")
    print("\nPara usar el blueprint en una partida:")
    print("  from montecarlo import blueprint_action_callback")
    print("  engine = PokerCoreEngine(..., action_callback=blueprint_action_callback)")


if __name__ == "__main__":
    main()
