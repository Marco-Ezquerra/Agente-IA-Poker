#!/usr/bin/env python3
"""
Benchmark: IA-GTO (blueprint MCCFR + subgame search) vs IA-MCTS (legacy).

Enfrenta dos políticas de decisión en N manos de HUNL y reporta:
  - winrate en BB/100  (big blinds cada 100 manos)
  - IC 95% por bootstrap
  - tabla de resultados por mano
  - resumen estadístico final

Uso
---
    python comparador.py --manos 200
    python comparador.py --manos 500 --fichas 200 --quiet
    python comparador.py --solo-equity --manos 100   # sin blueprint (fallback)
"""

import os
import sys
import argparse
import random
import math

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import logging
logging.disable(logging.CRITICAL)   # silenciar logs del motor durante el benchmark

from poker_engine import PokerCoreEngine


# ── Estadísticas ──────────────────────────────────────────────────────────────

def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _bootstrap_ci(xs, n_boot=2000, ci=0.95):
    """Intervalo de confianza por bootstrap."""
    if len(xs) < 2:
        return 0.0, 0.0
    means = []
    for _ in range(n_boot):
        sample = [random.choice(xs) for _ in range(len(xs))]
        means.append(_mean(sample))
    means.sort()
    alpha = 1 - ci
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return lo, hi


# ── Callbacks ─────────────────────────────────────────────────────────────────

def _make_gto_callback(opp_model=None, opp_samples=5):
    """Crea el callback GTO (blueprint + subgame search + opponent model)."""
    from montecarlo import blueprint_action_callback
    def cb(jugador, state, valid_actions):
        return blueprint_action_callback(
            jugador, state, valid_actions,
            opp_model=opp_model, opp_samples=opp_samples,
        )
    return cb


def _make_mcts_callback():
    """Crea el callback MCTS legacy envolviendo run_mcts."""
    from mcts_modulo import run_mcts

    def mcts_callback(jugador, state, valid_actions):
        mcts_state = [
            state.get('pot', 1.5),
            [[j['id'], j['fichas'], j.get('mano', [])]
             for j in state.get('jugadores', [{'id': jugador.id, 'fichas': jugador.fichas, 'mano': []}])],
            state.get('community', [])
        ]
        return run_mcts(mcts_state, jugador.id, valid_actions, num_simulaciones=200)

    return mcts_callback


def _make_equity_callback():
    """Callback basado en equity puro (sin blueprint, sin MCTS)."""
    from montecarlo import advanced_ia_action_callback
    return advanced_ia_action_callback


# ── Benchmark ────────────────────────────────────────────────────────────────

def run_benchmark(num_manos: int = 100,
                  fichas: int = 100,
                  gto_callback=None,
                  rival_callback=None,
                  verbose: bool = True,
                  gto_nombre: str = "GTO",
                  rival_nombre: str = "MCTS") -> dict:
    """
    Enfrenta gto_callback (jugador 0) contra rival_callback (jugador 1).

    Retorna dict con:
      resultados   : list[float] – ganancia de GTO en BBs por mano
      winrate_bb100: float       – BBs ganados cada 100 manos
      ci_lo, ci_hi : float       – IC 95% del winrate
      std          : float       – desviación estándar por mano
    """
    engine = PokerCoreEngine(
        nombres_jugadores=[gto_nombre, rival_nombre],
        action_callback=None,
    )
    engine.mesa.jugadores[0].fichas = fichas
    engine.mesa.jugadores[1].fichas = fichas

    def action_callback(jugador, state, valid_actions):
        if jugador.id == 0:
            return gto_callback(jugador, state, valid_actions)
        return rival_callback(jugador, state, valid_actions)

    engine.action_callback = action_callback

    resultados = []
    fichas_gto   = fichas
    fichas_rival = fichas

    for mano_idx in range(1, num_manos + 1):
        try:
            engine.reset()
            engine.mesa.jugadores[0].fichas = fichas_gto
            engine.mesa.jugadores[1].fichas = fichas_rival
            engine.action_callback = action_callback

            final_state = engine.play_round()
            _, players_finales, _ = final_state
            fichas_post = {p[0]: p[1] for p in players_finales}

            delta = fichas_post[0] - fichas_gto
            resultados.append(delta)

            fichas_gto   = fichas_post[0]
            fichas_rival = fichas_post[1]

            if verbose:
                print(f"  Mano {mano_idx:>4} | {gto_nombre}: {fichas_gto:7.1f}  "
                      f"{rival_nombre}: {fichas_rival:7.1f}  "
                      f"Δ {gto_nombre}: {delta:+.1f}")

            # Recompra si alguno se queda sin fichas
            if fichas_gto <= 0:
                fichas_gto = fichas
            if fichas_rival <= 0:
                fichas_rival = fichas

        except Exception as e:
            logging.warning("Error en mano %d: %s", mano_idx, e)
            continue

    if not resultados:
        return {'resultados': [], 'winrate_bb100': 0.0,
                'ci_lo': 0.0, 'ci_hi': 0.0, 'std': 0.0}

    wr     = _mean(resultados) * 100.0   # BB/100
    ci_lo, ci_hi = _bootstrap_ci(resultados)
    ci_lo *= 100.0;  ci_hi *= 100.0

    return {
        'resultados':    resultados,
        'winrate_bb100': wr,
        'ci_lo':         ci_lo,
        'ci_hi':         ci_hi,
        'std':           _std(resultados),
        'n_manos':       len(resultados),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GTO vs MCTS para el agente de póker HUNL"
    )
    parser.add_argument('--manos',       type=int, default=100,
                        help='Número de manos (default: 100)')
    parser.add_argument('--fichas',      type=int, default=100,
                        help='Fichas iniciales en BBs (default: 100)')
    parser.add_argument('--quiet',       action='store_true',
                        help='Suprimir output por mano')
    parser.add_argument('--solo-equity', action='store_true',
                        help='Usar equity puro en lugar del blueprint GTO')
    parser.add_argument('--opp-samples', type=int, default=5,
                        help='Manos del oponente a muestrear en subgame search (default: 5)')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Benchmark HUNL: {args.manos} manos")
    print(f"  Fichas iniciales: {args.fichas} BBs")
    print(f"{'='*60}\n")

    # ── Callbacks ──────────────────────────────────────────
    from opponent_model import OpponentModel
    opp_model = OpponentModel(opponent_id=1)

    if args.solo_equity:
        gto_cb      = _make_equity_callback()
        gto_nombre  = "IA-Equity"
    else:
        gto_cb      = _make_gto_callback(opp_model=opp_model,
                                          opp_samples=args.opp_samples)
        gto_nombre  = "IA-GTO"

    rival_cb    = _make_mcts_callback()
    rival_nombre = "IA-MCTS"

    # ── Ejecución ──────────────────────────────────────────
    res = run_benchmark(
        num_manos     = args.manos,
        fichas        = args.fichas,
        gto_callback  = gto_cb,
        rival_callback= rival_cb,
        verbose       = not args.quiet,
        gto_nombre    = gto_nombre,
        rival_nombre  = rival_nombre,
    )

    # ── Resultados ──────────────────────────────────────────
    wr    = res['winrate_bb100']
    ci_lo = res['ci_lo']
    ci_hi = res['ci_hi']
    std   = res['std']
    n     = res['n_manos']

    print(f"\n{'='*60}")
    print(f"  {gto_nombre} vs {rival_nombre}  –  {n} manos")
    print(f"{'='*60}")
    print(f"  Winrate  : {wr:+.1f} BB/100")
    print(f"  IC 95%   : [{ci_lo:+.1f}, {ci_hi:+.1f}] BB/100")
    print(f"  Std/mano : {std:.3f} BBs")
    ganador = gto_nombre if wr > 0 else rival_nombre
    print(f"  Resultado: {'WIN' if wr > 0 else 'LOSS'} para {gto_nombre}")
    print(f"\n{opp_model.summary()}")
    print(f"{'='*60}\n")

    # Significancia estadística (z-test simple)
    if std > 0 and n > 30:
        se = std / math.sqrt(n)
        z  = abs(_mean(res['resultados'])) / se
        p  = 2 * (1 - _normal_cdf(z))
        print(f"  z={z:.2f}  p-value≈{p:.4f}  "
              f"({'significativo' if p < 0.05 else 'no significativo'} al 5%)  \n")


def _normal_cdf(z):
    """CDF de la normal estándar (aproximación de Abramowitz y Stegun)."""
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    p = (0.319381530 * t - 0.356563782 * t**2 +
         1.781477937 * t**3 - 1.821255978 * t**4 +
         1.330274429 * t**5)
    prob = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z**2) * p
    return prob if z >= 0 else 1 - prob


if __name__ == "__main__":
    main()

