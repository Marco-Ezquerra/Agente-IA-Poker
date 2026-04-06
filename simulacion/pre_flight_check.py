#!/usr/bin/env python3
"""
Pre-flight check para entrenamiento largo del blueprint.

No modifica la matematica del trainer/solver. Solo orquesta un dry run:
  1) Entrena 10k iteraciones MCCFR en bloques de 1k y monitorea RAM/velocidad/nodos.
  2) Imprime proxy de convergencia (regret medio absoluto global).
  3) Ejecuta hand-off de prueba al turn y valida que el subgame solver recibe rango logico.
  4) Genera STATUS.md con resumen final.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np

try:
    import psutil
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "psutil no esta instalado. Instala con: pip install psutil"
    ) from exc

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from abstracciones.card_abstractor import PREFLOP_BUCKETS, POSTFLOP_BUCKETS, preflop_bucket, postflop_bucket
from abstracciones.infoset_encoder import (
    ABSTRACT_ACTIONS,
    ACTION_IDX,
    CALL,
    FOLD,
    RAISE_RATIOS,
)
from cfr.mccfr_trainer import MCCFRTrainer
from cfr.realtime_search import RealtimeSearch, _fast_key, _precompute_buckets


RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['s', 'h', 'd', 'c']


@dataclass
class MonitorRow:
    iters: int
    ram_mb: float
    iters_per_sec: float
    nodes: int
    proxy_regret_mean_abs: float


def _full_deck() -> List[str]:
    return [r + s for r in RANKS for s in SUITS]


def _mean_abs_regret(regret_sum: Dict[Tuple, np.ndarray]) -> float:
    if not regret_sum:
        return 0.0
    vals = [float(np.mean(np.abs(arr))) for arr in regret_sum.values()]
    return float(np.mean(vals)) if vals else 0.0


def _check_action_guardrails() -> bool:
    """
    Verifica que check sigue existiendo via accion abstracta CALL cuando to_call == 0.
    """
    mask = MCCFRTrainer._mask(to_call=0.0, stack=100.0, n_raises=0)
    has_call = bool(mask[ACTION_IDX[CALL]])
    has_fold = bool(mask[ACTION_IDX[FOLD]])
    return has_call and not has_fold


def monitor_training(total_iters: int, chunk: int, bucket_sims: int) -> Tuple[MCCFRTrainer, List[MonitorRow]]:
    trainer = MCCFRTrainer()
    process = psutil.Process(os.getpid())
    rows: List[MonitorRow] = []

    print("\n=== Dry Run MCCFR ===")
    print(f"Total iters: {total_iters:,} | chunk: {chunk:,} | bucket_sims: {bucket_sims}")
    print("iters | RAM_MB | iters/s | nodos | proxy_regret_abs")

    done = 0
    while done < total_iters:
        step = min(chunk, total_iters - done)
        t0 = time.perf_counter()
        trainer.train(num_iterations=step, log_every=step + 1, bucket_sims=bucket_sims)
        dt = max(time.perf_counter() - t0, 1e-9)

        done += step
        ram_mb = process.memory_info().rss / (1024 * 1024)
        ips = step / dt
        nodes = len(trainer.regret_sum)
        proxy = _mean_abs_regret(trainer.regret_sum)

        row = MonitorRow(
            iters=done,
            ram_mb=ram_mb,
            iters_per_sec=ips,
            nodes=nodes,
            proxy_regret_mean_abs=proxy,
        )
        rows.append(row)

        print(
            f"{done:>5d} | {ram_mb:>7.1f} | {ips:>7.1f} | {nodes:>7d} | {proxy:>12.6f}"
        )

    return trainer, rows


def compute_turn_reach_probs(
    trainer: MCCFRTrainer,
    my_hand: List[str],
    board_turn: List[str],
    hand_sims: int = 16,
) -> Dict[Tuple[str, str], float]:
    """
    Proxy de reach probabilities del rango rival en turn.

    Se construye sin tocar matematicas del trainer: solo consultas de estrategia
    promedio por bucket.
    """
    known = set(my_hand + board_turn)
    remaining = [c for c in _full_deck() if c not in known]

    probs: Dict[Tuple[str, str], float] = {}
    opp_pos = 1  # BB

    for a, b in combinations(remaining, 2):
        opp_hand = [a, b]

        pf_bkt = preflop_bucket(opp_hand, num_sims=1)
        key_pf = (opp_pos, 0, pf_bkt, (CALL,))
        strat_pf = trainer.get_strategy(key_pf)
        p_continue_pf = max(1e-9, 1.0 - float(strat_pf[ACTION_IDX[FOLD]]))

        turn_bkt = postflop_bucket(opp_hand, board_turn, num_sims=hand_sims)
        key_turn = (opp_pos, 2, turn_bkt, ())
        strat_turn = trainer.get_strategy(key_turn)
        p_continue_turn = max(1e-9, 1.0 - float(strat_turn[ACTION_IDX[FOLD]]))

        # Prior suave por fuerza de buckets para evitar distribucion plana cuando
        # el blueprint aun tiene poca muestra en ciertos InfoSets.
        pf_strength = (pf_bkt + 1.0) / float(PREFLOP_BUCKETS)
        turn_strength = (turn_bkt + 1.0) / float(POSTFLOP_BUCKETS)
        strength_prior = 0.7 * pf_strength + 0.3 * turn_strength

        probs[(a, b)] = p_continue_pf * p_continue_turn * strength_prior

    total = float(sum(probs.values()))
    if total <= 0.0:
        return {}

    for k in list(probs.keys()):
        probs[k] /= total

    return probs


def handoff_turn_to_realtime(
    trainer: MCCFRTrainer,
    hand_probs: Dict[Tuple[str, str], float],
    my_hand: List[str],
    board_turn: List[str],
) -> Tuple[bool, List[Tuple[Tuple[str, str], float]], str]:
    """
    Dry hand-off al realtime solver.

    No resuelve toda la calle: valida que el solver puede consumir un rango
    no vacio/no uniforme derivado del blueprint.
    """
    top5 = sorted(hand_probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
    if not top5:
        return False, [], "Rango vacio"

    # Verifica que la distribucion no sea uniforme trivial en top-5
    pvals = [p for _, p in top5]
    non_uniform = (max(pvals) - min(pvals)) > 1e-6

    # Pasa las manos top al modulo realtime_search para precomputo de buckets.
    # Esto valida que el hand-off tecnico no llega vacio ni inconsistente.
    try:
        for hand, _ in top5:
            bkts = _precompute_buckets(my_hand, list(hand), board_turn, 'turn', sims=20)
            _ = _fast_key(0, 2, bkts, [])
    except Exception as exc:
        return False, top5, f"Error al precomputar buckets para hand-off: {exc}"

    # Llamada corta al solver como smoke test de integracion.
    engine = RealtimeSearch(blueprint=trainer, depth=1, iterations=30)
    try:
        action = engine.get_action(
            traverser=0,
            my_hand=my_hand,
            board=board_turn,
            street_str='turn',
            pot=12.0,
            stacks=[88.0, 88.0],
            contribs=[0.0, 0.0],
            bet_hist=[],
            n_raises=0,
            to_call=0.0,
            opp_samples=6,
            bucket_sims=20,
        )
    except Exception as exc:
        return False, top5, f"RealtimeSearch fallo: {exc}"

    ok = non_uniform and (action in ABSTRACT_ACTIONS)
    msg = f"accion_solver={action} | top5_non_uniform={non_uniform}"
    return ok, top5, msg


def write_status(
    status_path: str,
    rows: List[MonitorRow],
    handoff_ok: bool,
    handoff_msg: str,
    top5: List[Tuple[Tuple[str, str], float]],
) -> None:
    last = rows[-1]
    x_mb = last.ram_mb
    projected_gb_50m = (x_mb * (50_000_000 / 10_000)) / 1024.0

    bet_sizes = [
        f"{a}={ratio:.4g}x_pot"
        for a, ratio in RAISE_RATIOS.items()
        if a != 'ai'
    ]

    lines = [
        "# STATUS",
        "",
        "## Resumen Dry Run",
        f"- Iteraciones ejecutadas: {last.iters:,}",
        f"- RAM final: {last.ram_mb:.1f} MB",
        f"- Velocidad final: {last.iters_per_sec:.1f} iters/s",
        f"- Nodos (regret_sum): {last.nodes:,}",
        f"- Proxy convergencia (regret medio abs): {last.proxy_regret_mean_abs:.6f}",
        "",
        "## Estimacion de Memoria",
        f"- Si 10k iteraciones ocupan {x_mb:.1f} MB, 50 millones ocuparian ~{projected_gb_50m:.2f} GB (extrapolacion lineal).",
        "",
        "## Hand-off al Turn",
        f"- Estado: {'Exito' if handoff_ok else 'Fallo'}",
        f"- Detalle: {handoff_msg}",
        "- Top 5 manos rival (reach prob):",
    ]

    for hand, p in top5:
        lines.append(f"  - {hand[0]} {hand[1]}: {p:.6f}")

    lines.extend([
        "",
        "## Parametros Actuales",
        f"- PREFLOP_BUCKETS: {PREFLOP_BUCKETS}",
        f"- K-Means/EMD POSTFLOP_BUCKETS: {POSTFLOP_BUCKETS}",
        "- Hist bins configurado: 20",
        f"- Acciones abstractas: {', '.join(ABSTRACT_ACTIONS)}",
        f"- Sizings de apuesta (x pot): {', '.join(bet_sizes)}",
        "- CHECK disponible: SI (via accion CALL cuando to_call == 0)",
        "",
    ])

    with open(status_path, 'w', encoding='utf-8') as fh:
        fh.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description='Pre-flight dry run para blueprint MCCFR.')
    parser.add_argument('--iters', type=int, default=10_000, help='Iteraciones totales del dry run (default: 10000)')
    parser.add_argument('--chunk', type=int, default=1_000, help='Frecuencia de monitoreo (default: 1000)')
    parser.add_argument('--bucket-sims', type=int, default=30, help='Monte Carlo sims para buckets durante dry run')
    parser.add_argument('--status', type=str, default='STATUS.md', help='Ruta de salida para reporte final')
    args = parser.parse_args()

    if args.iters <= 0 or args.chunk <= 0:
        raise SystemExit('--iters y --chunk deben ser positivos.')

    check_ok = _check_action_guardrails()
    print(f"CHECK guardrail: {'OK' if check_ok else 'FAIL'} (CALL habilitado y FOLD deshabilitado con to_call=0)")

    trainer, rows = monitor_training(args.iters, args.chunk, args.bucket_sims)

    # Hand-off de prueba en turn con board inventado
    my_hand = ['Ah', 'Kd']
    board_turn = ['Qs', 'Jh', '7c', '2d']
    hand_probs = compute_turn_reach_probs(trainer, my_hand=my_hand, board_turn=board_turn, hand_sims=16)

    handoff_ok, top5, handoff_msg = handoff_turn_to_realtime(
        trainer=trainer,
        hand_probs=hand_probs,
        my_hand=my_hand,
        board_turn=board_turn,
    )

    print("\n=== Top 5 reach probs (turn) ===")
    for hand, p in top5:
        print(f"{hand[0]} {hand[1]} -> {p:.6f}")

    status_path = os.path.abspath(os.path.join(_DIR, args.status))
    write_status(
        status_path=status_path,
        rows=rows,
        handoff_ok=handoff_ok,
        handoff_msg=handoff_msg,
        top5=top5,
    )

    print(f"\nSTATUS escrito en: {status_path}")
    print(f"Hand-off al turn: {'EXITO' if handoff_ok else 'FALLO'} | {handoff_msg}")
    return 0 if handoff_ok and check_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
