"""
Script de entrenamiento del blueprint MCCFR.

Uso
---
    cd simulacion
    python cfr/train_blueprint.py                          # 50 000 iters (rápido)
    python cfr/train_blueprint.py --iters 200000           # entrenamiento medio
    python cfr/train_blueprint.py --iters 1000000          # producción

El blueprint resultante se guarda en cfr/blueprint.pkl y es cargado
automáticamente por blueprint_action_callback en montecarlo.py.

Estrategia de entrenamiento recomendada
----------------------------------------
  Ronda 1 – 50 000 iters   : calibración rápida, comprobar que converge
  Ronda 2 – 200 000 iters  : blueprint útil en juego  (≈ 5-15 min en CPU)
  Ronda 3 – 1 000 000 iters: producción  (≈ 30-90 min en CPU)

Se puede hacer fine-tuning reanudando desde un blueprint previo:
    trainer = MCCFRTrainer.load()
    trainer.train(num_iterations=200_000)
    trainer.save()
"""

import argparse
import os
import sys
import time

# Asegurar que simulacion/ está en el path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cfr.mccfr_trainer import MCCFRTrainer


def parse_args():
    p = argparse.ArgumentParser(
        description='Entrena el blueprint MCCFR para el agente de póker HUNL.')
    p.add_argument('--iters',     type=int, default=50_000,
                   help='Número de iteraciones de MCCFR (default: 50 000)')
    p.add_argument('--log-every', type=int, default=5_000,
                   help='Frecuencia de log (default: 5 000)')
    p.add_argument('--resume',    action='store_true',
                   help='Reanudar entrenamiento desde blueprint existente')
    p.add_argument('--output',    type=str, default=None,
                   help='Ruta de salida del blueprint (default: cfr/blueprint.pkl)')
    return p.parse_args()


def main():
    args = parse_args()

    if args.resume and MCCFRTrainer.exists(args.output):
        print("Reanudando entrenamiento desde blueprint existente…")
        trainer = MCCFRTrainer.load(args.output)
    else:
        print("Iniciando entrenamiento desde cero…")
        trainer = MCCFRTrainer()

    t0 = time.time()
    trainer.train(num_iterations=args.iters, log_every=args.log_every)
    elapsed = time.time() - t0

    trainer.save(args.output)
    print(f"\nTiempo total: {elapsed:.1f}s  |  "
          f"{args.iters / elapsed:.0f} iters/s")

    # Verificación básica: consultar algunos InfoSets
    print("\nVerificación: estrategia para InfoSet de ejemplo (AA preflop):")
    try:
        from abstracciones.infoset_encoder import encode_infoset, ABSTRACT_ACTIONS
        key = encode_infoset(0, 'preflop', ['Ah', 'As'], [], [])
        strat = trainer.get_strategy(key)
        for a, p in zip(ABSTRACT_ACTIONS, strat):
            print(f"  {a:6s}: {p:.3f}")
    except Exception as e:
        print(f"  (verificación fallida: {e})")


if __name__ == '__main__':
    main()
