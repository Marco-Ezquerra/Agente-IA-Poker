"""
Real-time Subgame Search para HUNL.

Dado el estado actual de la partida en tiempo real (cartas del agente,
tablero, historial de apuestas), realiza una búsqueda de subgame con
horizonte limitado usando el blueprint como leaf evaluator.

Arquitectura
------------
Para cada muestra de mano del oponente:
  1. Se precomputan TODOS los buckets de ambos jugadores en todas las calles
     (preflop, flop, turn, river) — UNA sola vez, O(sims_pequeños).
  2. Dentro del árbol CFR, los InfoSet keys se generan con _fast_key → O(1).
  → Coste total por llamada a get_action: O(opp_samples × bucket_sims)
    en lugar de O(opp_samples × iterations × nodos × bucket_sims).

Técnica
-------
External Sampling CFR sobre el subgame a profundidad D (calles hacia adelante).
La imperfect information sobre las cartas del rival se resuelve muestreando
N manos del oponente de la baraja residual y promediando estrategias.

Referencia: Brown & Sandholm (2017) "Safe and Nested Subgame Solving for
Imperfect-Information Games", NeurIPS.

Uso
---
    from cfr.realtime_search import RealtimeSearch
    from cfr.mccfr_trainer  import MCCFRTrainer

    blueprint = MCCFRTrainer.load()
    engine    = RealtimeSearch(blueprint=blueprint, depth=2, iterations=200)

    abstract_action = engine.get_action(
        traverser=0, my_hand=['Ah','Kd'], board=['Jh','Ts','2c'],
        street_str='flop', pot=4.5, stacks=[97.5, 95.5],
        contribs=[0.0, 0.0], bet_hist=[], n_raises=0, to_call=0.0,
        opp_samples=8,
    )
"""

import os
import sys
import random

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from abstracciones.card_abstractor import (
    preflop_bucket, postflop_bucket, POSTFLOP_BUCKETS,
)
from abstracciones.infoset_encoder import (
    FOLD, CALL, RAISE_THIRD, RAISE_HALF, RAISE_POT, RAISE_2POT, ALLIN,
    ABSTRACT_ACTIONS, NUM_ACTIONS, ACTION_IDX, RAISE_RATIOS,
)

RANKS      = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS      = ['s', 'h', 'd', 'c']
STREETS    = ['preflop', 'flop', 'turn', 'river']
STREET_IDX = {s: i for i, s in enumerate(STREETS)}


# ── Utilidades ─────────────────────────────────────────────────────────────────

def _full_deck():
    return [r + s for r in RANKS for s in SUITS]


def _regret_match(regrets, mask):
    pos   = np.maximum(0.0, regrets) * mask
    total = pos.sum()
    if total > 0.0:
        return pos / total
    valid = mask.astype(np.float64)
    return valid / valid.sum()


def _fast_key(player, street_idx, bkts, bet_hist):
    """
    Genera la clave del InfoSet usando buckets precomputados → O(1).

    BUG-5 fix: clave de 4 elementos idéntica a la de mccfr_trainer._fast_key.
    La versión anterior usaba 5 elementos (añadía 'bb' con buckets de calles
    anteriores), lo que provocaba que blueprint.get_strategy(key) retornase
    siempre distribución uniforme (miss total en el blueprint entrenado).

    bkts : dict {(player_idx, street_idx) → bucket_int}
    """
    hb = bkts[(player, street_idx)]
    return (player, street_idx, hb, tuple(bet_hist))


def _precompute_buckets(hand0, hand1, board, street_str, sims=60):
    """
    Precomputa los buckets de ambos jugadores para todas las calles.
    Llamar UNA vez por muestra de mano del oponente.

    Parámetros
    ----------
    hand0, hand1 : list[str] – cartas de SB y BB
    board        : list[str] – cartas comunitarias visibles
    street_str   : str       – calle actual
    sims         : int       – simulaciones Monte Carlo para EHS²

    Retorna
    -------
    dict {(player_idx, street_idx) → bucket_int}
    """
    flop_board  = board[:3] if len(board) >= 3 else board
    turn_board  = board[:4] if len(board) >= 4 else board
    river_board = board[:5] if len(board) >= 5 else board

    bkts = {}
    for p, hand in [(0, hand0), (1, hand1)]:
        bkts[(p, 0)] = preflop_bucket(hand, num_sims=sims)
        bkts[(p, 1)] = postflop_bucket(hand, flop_board,  num_sims=sims) if flop_board  else 0
        bkts[(p, 2)] = postflop_bucket(hand, turn_board,  num_sims=sims) if turn_board  else 0
        bkts[(p, 3)] = postflop_bucket(hand, river_board, num_sims=sims) if river_board else 0
    return bkts


# ── RealtimeSearch ─────────────────────────────────────────────────────────────

class RealtimeSearch:
    """
    Búsqueda de subgame en tiempo real con blueprint como leaf evaluator.

    Parámetros
    ----------
    blueprint  : MCCFRTrainer | None  – política GTO preentrenada
    depth      : int  – calles hacia adelante a explorar (default 1)
    iterations : int  – iteraciones CFR por llamada (default 200)
    """

    def __init__(self, blueprint=None, depth: int = 1, iterations: int = 200):
        self.blueprint  = blueprint
        self.depth      = depth
        self.iterations = iterations
        self._regret:    dict = {}
        self._strat_sum: dict = {}

    # ── Acceso a tablas locales ───────────────────────────────────────────────

    def _get_regrets(self, key):
        if key not in self._regret:
            self._regret[key] = np.zeros(NUM_ACTIONS, dtype=np.float64)
        return self._regret[key]

    def _get_strat_sum(self, key):
        if key not in self._strat_sum:
            self._strat_sum[key] = np.zeros(NUM_ACTIONS, dtype=np.float64)
        return self._strat_sum[key]

    def _strategy(self, key, mask):
        return _regret_match(self._get_regrets(key), mask)

    # ── Máscara de acciones válidas ───────────────────────────────────────────

    @staticmethod
    def _mask(to_call, stack, n_raises, raise_max=2):
        m = np.ones(NUM_ACTIONS, dtype=bool)
        if to_call == 0.0:
            m[ACTION_IDX[FOLD]] = False
        if n_raises >= raise_max:
            for a in [RAISE_THIRD, RAISE_HALF, RAISE_POT, RAISE_2POT]:
                m[ACTION_IDX[a]] = False
        if stack <= to_call:
            for a in [RAISE_THIRD, RAISE_HALF, RAISE_POT, RAISE_2POT]:
                m[ACTION_IDX[a]] = False
        return m

    # ── Evaluación de hoja (blueprint leaf) ──────────────────────────────────

    def _leaf_value(self, traverser, bkts, street_idx, pot):
        """
        Valor estimado del subgame cuando se supera el horizonte de búsqueda.
        Usa el bucket del jugador en la calle actual como proxy de equity.
        Si hay blueprint disponible, pondera con la agresividad del blueprint.
        """
        bucket = bkts[(traverser, street_idx)]
        eq_raw = bucket / max(POSTFLOP_BUCKETS, 1)

        if self.blueprint is not None:
            key  = _fast_key(traverser, street_idx, bkts, [])
            strat = self.blueprint.get_strategy(key)
            aggr  = sum(strat[ACTION_IDX[a]]
                        for a in [RAISE_THIRD, RAISE_HALF, RAISE_POT, RAISE_2POT, ALLIN])
            eq_raw = eq_raw * 0.7 + min(aggr, 1.0) * 0.3

        return (2.0 * eq_raw - 1.0) * pot * 0.4

    # ── Showdown con buckets ──────────────────────────────────────────────────

    def _showdown(self, traverser, bkts, pot):
        """Estima el resultado del showdown usando los buckets de river."""
        b0  = bkts[(0, 3)]
        b1  = bkts[(1, 3)]
        eq0 = 0.5 + 0.5 * (b0 - b1) / max(POSTFLOP_BUCKETS, 1)
        eq0 = max(0.0, min(1.0, eq0))
        eq  = eq0 if traverser == 0 else (1.0 - eq0)
        return eq * pot - (1.0 - eq) * pot

    # ── Paso de acción ────────────────────────────────────────────────────────

    def _step(self, action, traverser, bkts,
              street_idx, pot, stacks, contribs,
              bet_hist, n_raises, to_call, position, depth_left):
        active   = position
        opponent = 1 - active

        if action == FOLD:
            return float(pot) if traverser == opponent else -float(contribs[active])

        elif action == CALL:
            amount = min(to_call, stacks[active])
            ns = list(stacks);  nc = list(contribs)
            ns[active] -= amount
            nc[active] += amount
            new_pot = pot + amount

            if nc[0] == nc[1] or ns[active] == 0.0:
                # BUG-9 fix: limp de SB preflop — el BB tiene opción de check/raise.
                # En mccfr_trainer este caso existe; sin este bloque el game tree
                # del realtime difiere del blueprint, invalidando la estrategia
                # aprendida para cualquier mano que empiece con limp.
                if street_idx == 0 and active == 0 and not bet_hist:
                    return self._search(
                        traverser, bkts, street_idx, new_pot, ns, nc,
                        [CALL], 0, 0.0, opponent, depth_left)
                if street_idx < 3:
                    next_si = street_idx + 1
                    if depth_left <= 0:
                        return self._leaf_value(traverser, bkts, next_si, new_pot)
                    return self._search(
                        traverser, bkts, next_si, new_pot, ns,
                        [0.0, 0.0], [], 0, 0.0, 1, depth_left - 1)
                else:
                    return self._showdown(traverser, bkts, new_pot)

            return self._search(
                traverser, bkts, street_idx, new_pot, ns, nc,
                bet_hist + [CALL], n_raises,
                nc[opponent] - nc[active], opponent, depth_left)

        elif action == ALLIN:
            amount = stacks[active]
            ns = list(stacks);  nc = list(contribs)
            ns[active] = 0.0
            nc[active] += amount
            new_pot     = pot + amount
            new_to_call = max(0.0, nc[active] - nc[opponent])
            return self._search(
                traverser, bkts, street_idx, new_pot, ns, nc,
                bet_hist + [ALLIN], n_raises + 1, new_to_call, opponent, depth_left)

        else:
            ratio       = RAISE_RATIOS.get(action, 1.0)
            raise_extra = min(ratio * pot, stacks[active] - to_call)
            total_add   = to_call + raise_extra
            if total_add >= stacks[active]:
                total_add = stacks[active]; action = ALLIN
            ns = list(stacks);  nc = list(contribs)
            ns[active]  -= total_add
            nc[active]  += total_add
            new_pot      = pot + total_add
            new_to_call  = nc[active] - nc[opponent]
            return self._search(
                traverser, bkts, street_idx, new_pot, ns, nc,
                bet_hist + [action], n_raises + 1, new_to_call, opponent, depth_left)

    # ── CFR local del subgame ─────────────────────────────────────────────────

    def _search(self, traverser, bkts,
                street_idx, pot, stacks, contribs,
                bet_hist, n_raises, to_call, position, depth_left):
        """
        External Sampling CFR sobre el subgame.
        Usa bkts precomputados para InfoSet keys → O(1) por nodo.
        """
        active = position

        key      = _fast_key(active, street_idx, bkts, bet_hist)
        mask     = self._mask(to_call, stacks[active], n_raises)
        strategy = self._strategy(key, mask)

        if active == traverser:
            valid_idxs  = np.where(mask)[0]
            action_vals = np.zeros(NUM_ACTIONS)
            for idx in valid_idxs:
                action_vals[idx] = self._step(
                    ABSTRACT_ACTIONS[idx], traverser, bkts,
                    street_idx, pot, stacks, contribs,
                    bet_hist, n_raises, to_call, position, depth_left)

            ev = float(np.dot(strategy, action_vals))
            self._get_regrets(key)[:] += (action_vals - ev) * mask
            self._get_strat_sum(key)[:] += strategy
            return ev
        else:
            idx = int(np.random.choice(NUM_ACTIONS, p=strategy))
            self._get_strat_sum(key)[:] += strategy
            return self._step(
                ABSTRACT_ACTIONS[idx], traverser, bkts,
                street_idx, pot, stacks, contribs,
                bet_hist, n_raises, to_call, position, depth_left)

    # ── API pública ───────────────────────────────────────────────────────────

    def get_action(self, traverser: int,
                   my_hand: list,
                   board: list,
                   street_str: str,
                   pot: float,
                   stacks: list,
                   contribs: list,
                   bet_hist: list,
                   n_raises: int,
                   to_call: float,
                   opp_samples: int = 8,
                   bucket_sims: int = 50) -> str:
        """
        Calcula la acción abstracta recomendada para el agente.

        Para cada muestra de mano del oponente:
          1. Precomputa todos los buckets UNA vez → O(bucket_sims)
          2. Ejecuta iters_per_sample iteraciones de CFR → O(1) por nodo

        Parámetros
        ----------
        traverser    : int         – id del agente (SB=0, BB=1)
        my_hand      : list[str]   – 2 cartas del agente
        board        : list[str]   – cartas comunitarias visibles
        street_str   : str         – 'preflop'|'flop'|'turn'|'river'
        pot          : float       – bote en BBs
        stacks       : list[float] – [stack_SB, stack_BB]
        contribs     : list[float] – contribuciones en BBs esta calle
        bet_hist     : list[str]   – historial abstracto de esta calle
        n_raises     : int         – raises ya realizados en esta calle
        to_call      : float       – fichas para igualar
        opp_samples  : int         – manos del oponente a muestrear
        bucket_sims  : int         – simulaciones para EHS² por muestra

        Retorna
        -------
        str – una de las constantes ABSTRACT_ACTIONS
        """
        self._regret.clear()
        self._strat_sum.clear()

        street_idx      = STREET_IDX.get(street_str, 0)
        known           = set(my_hand + board)
        deck_rem        = [c for c in _full_deck() if c not in known]
        iters_per_sample = max(1, self.iterations // max(opp_samples, 1))

        for _ in range(opp_samples):
            random.shuffle(deck_rem)
            opp_hand = deck_rem[:2]

            hand0 = my_hand  if traverser == 0 else opp_hand
            hand1 = opp_hand if traverser == 0 else my_hand

            # Precomputar buckets UNA vez para este par de manos → O(bucket_sims)
            bkts = _precompute_buckets(hand0, hand1, board, street_str,
                                       sims=bucket_sims)

            # CFR sobre el subgame → O(1) por nodo
            for _ in range(iters_per_sample):
                for t in [0, 1]:
                    self._search(
                        t, bkts, street_idx,
                        pot, stacks, contribs, bet_hist,
                        n_raises, to_call, traverser, self.depth)

        # Extraer estrategia promedio del InfoSet actual del agente
        # Reutilizar los bkts de la última muestra como proxy
        if opp_samples > 0:
            my_key = _fast_key(traverser, street_idx, bkts, bet_hist)
            ss = self._strat_sum.get(my_key, None)
            if ss is not None and ss.sum() > 0:
                avg = ss / ss.sum()
                return str(ABSTRACT_ACTIONS[int(np.argmax(avg))])

        # Fallback: acción uniforme
        return CALL
