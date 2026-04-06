#!/usr/bin/env python3
"""
Prepara el cache obligatorio de clustering EMD para entrenamiento blueprint.
"""

import os
import sys
import time

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from abstracciones.card_abstractor import build_equity_clusters


def main():
    t0 = time.time()
    print('Generando equity_clusters.pkl con n_samples=100000, bins=20...')
    build_equity_clusters(n_samples=100000, bins=20)
    elapsed = time.time() - t0
    print(f'OK: equity_clusters.pkl generado. Tiempo: {elapsed:.1f}s')


if __name__ == '__main__':
    main()
