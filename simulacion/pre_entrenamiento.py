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
    NUM_ACTIONS,
)
from abstracciones.card_abstractor import PREFLOP_BUCKETS, POSTFLOP_BUCKETS


# ── Análisis matemático de iteraciones necesarias ─────────────────────────────

def analizar_iteraciones():
    """
    Calcula las iteraciones MCCFR necesarias para distintos niveles de
    calidad, basándose en el tamaño del espacio de InfoSets y la tasa de
    convergencia teórica de External Sampling CFR.

    Fundamento matemático
    ---------------------
    La exploitabilidad del blueprint decrece como:

        ε(T) ≈ C / √T    (para External Sampling MCCFR, Lanctot 2009)

    donde T = número de iteraciones y C depende del tamaño del juego.
    Con nuestras abstracciones (PREFLOP_BUCKETS=10, POSTFLOP_BUCKETS=8,
    7 acciones, MAX_RAISES=4), el número de InfoSets reachables es:

        |I| ≈ 2 × [10 + 8×3] × promedio_apuestas_por_calle

    Para 'visitas_objetivo' visitas promedio por InfoSet:

        T = |I| × visitas_objetivo / 2   (2 traversers por iteración)

    Referencia: Brown & Sandholm (2017) "Libratus"; Zinkevich et al. (2007)
    """
    import math

    print("\n" + "═" * 68)
    print("  ANÁLISIS MATEMÁTICO DE CONVERGENCIA MCCFR")
    print("═" * 68)

    # Estimación del espacio de InfoSets con la abstracción actual.
    # Calibrado empíricamente: con 50 postflop buckets → 795k InfoSets @ 55k iters.
    # Reducción a 8 buckets: 795k × (8/50) ≈ 127k InfoSets (mismo espacio de
    # bet_histories, menos granularidad de cartas).
    # Fórmula: |I| ≈ 2pos × (10preflop + 8³postflop) × avg_bet_hist_depth
    avg_bet_hist_depth = 5.0   # profundidad media de bet history por InfoSet
    n_infosets_est = int(
        2                                                         # posiciones
        * (PREFLOP_BUCKETS + POSTFLOP_BUCKETS ** 3)               # manos×boards
        * avg_bet_hist_depth                                      # historiales
    )

    print(f"\n  Configuración de abstracción:")
    print(f"    PREFLOP_BUCKETS  = {PREFLOP_BUCKETS}")
    print(f"    POSTFLOP_BUCKETS = {POSTFLOP_BUCKETS}  (reducido de 50 → convergencia CPU)")
    print(f"    Acciones         = {NUM_ACTIONS}  (FOLD,CALL,r1/3,r1/2,r1x,r2x,AI)")
    print(f"    InfoSets estimados (reachables) ≈ {n_infosets_est:,}")

    print(f"\n  {'Nivel':20s}  {'Iters':>10}  {'Visitas/IS':>12}  {'ε est. (mbb/m)':>16}  {'CPU (min)':>10}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*12}  {'-'*16}  {'-'*10}")

    # Constante C empírica (ajustada a nuestro sistema: 55k iters → ~23k mbb/m)
    # ε(T) ≈ C/√T → C = ε × √T = 23000 × √55000 ≈ 5.39M
    C_empirical = 23_000 * math.sqrt(55_000)

    targets = [
        ("Humo (test)",       5_000,    "~estrategia aleatoria"),
        ("Básico (jugable)", 50_000,    "mucho ruido, vale para jugar"),
        ("Bueno",           200_000,    "objetivo mínimo para este proyecto"),
        ("Muy bueno",       500_000,    "comportamiento coherente"),
        ("Excelente",     1_000_000,    "near-GTO en el espacio abstracto"),
        ("Near-GTO",      5_000_000,    "exploitabilidad < 500 mbb/m"),
    ]

    iter_per_min = 55_000 / 5.0   # ~11k iter/min medido empíricamente

    for name, iters, comment in targets:
        visits    = (iters * 2) / max(n_infosets_est, 1)
        eps_est   = C_empirical / math.sqrt(iters)
        cpu_mins  = iters / iter_per_min
        print(f"  {name:20s}  {iters:>10,}  {visits:>12.1f}  {eps_est:>14.0f}    {cpu_mins:>8.1f}   # {comment}")

    print(f"\n  RECOMENDACIÓN: 200k iters (~18 min) para calidad 'Buena'")
    print(f"  Con online learning (80 iters/mano × N manos jugadas) se")
    print(f"  refuerzan los InfoSets más frecuentes → convergencia más rápida")
    print(f"  en los spots reales de la partida.\n")
    print("═" * 68)



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
    parser.add_argument('--analizar', action='store_true',
                        help='Mostrar análisis matemático de iteraciones necesarias')
    parser.add_argument('--out',      type=str, default=None,
                        help='Ruta de salida del blueprint (default: cfr/blueprint.pkl)')
    args = parser.parse_args()

    # Análisis de iteraciones (opción independiente)
    if args.analizar:
        analizar_iteraciones()

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


if __name__ == "__main__":
    main()
