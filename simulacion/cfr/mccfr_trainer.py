"""
MCCFR Trainer – External Sampling para Heads-Up No-Limit Hold'em (HUNL).

Implementa el algoritmo de External Sampling MCCFR descrito en:
  Lanctot et al. (2009) "Monte Carlo Sampling for Regret Minimization in
  Extensive Games", NeurIPS.

Juego abstracto
---------------
  Jugadores  : 2  (P0 = SB, P1 = BB)
  Calles     : preflop → flop → turn → river
  Acciones   : 7 acciones abstractas por nodo
                (FOLD, CALL, RAISE_THIRD, RAISE_HALF, RAISE_POT, RAISE_2POT, ALLIN)
  InfoSets   : (position, street, hand_bucket, board_bucket_tuple, bet_hist)

Eficiencia
----------
Los buckets (EHS / EHS²) se precomputan UNA sola vez por iteración antes de
iniciar el traversal del árbol. Dentro del árbol solo se hacen accesos O(1)
a la tabla de buckets → cada iteración tarda milisegundos, no minutos.

Convergencia
------------
La estrategia promedio (strategy_sum / sum) converge a Nash en juegos de
suma cero de dos jugadores. El blueprint resultante es la política GTO base
que se usa como prior en el real-time search.

Uso
---
    from cfr.mccfr_trainer import MCCFRTrainer

    trainer = MCCFRTrainer()
    trainer.train(num_iterations=100_000, log_every=10_000)
    trainer.save()   # → cfr/blueprint.pkl

    # Consultar estrategia en un InfoSet:
    key      = encode_infoset(position, street, hand, board, bet_hist)
    strategy = trainer.get_strategy(key)   # np.ndarray shape (7,)
"""

import os
import sys
import random
import pickle

import numpy as np

# Asegurar que simulacion/ está en el path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from abstracciones.card_abstractor import (
    preflop_bucket, postflop_bucket, POSTFLOP_BUCKETS,
)
from abstracciones.infoset_encoder import (
    FOLD, CALL, RAISE_THIRD, RAISE_HALF, RAISE_POT, RAISE_2POT, ALLIN,
    ABSTRACT_ACTIONS, NUM_ACTIONS, ACTION_IDX,
    RAISE_RATIOS, encode_infoset,
)

BLUEPRINT_PATH = os.path.join(os.path.dirname(__file__), 'blueprint.pkl')

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['s', 'h', 'd', 'c']
STREETS = ['preflop', 'flop', 'turn', 'river']


# ── Utilidades de baraja ───────────────────────────────────────────────────────

def _full_deck():
    return [r + s for r in RANKS for s in SUITS]


def _deal():
    """Reparte 2+2 hole cards + flop+turn+river aleatoriamente."""
    deck = _full_deck()
    random.shuffle(deck)
    return deck[0:2], deck[2:4], deck[4:7], [deck[7]], [deck[8]]


# ── Regret Matching ───────────────────────────────────────────────────────────

def _regret_match(regrets, mask):
    """
    Regret Matching: σ(a) = max(0, R(a)) / Σ max(0, R(·))
    Si todos los regrets son ≤ 0, distribuye uniformemente entre acciones válidas.
    """
    pos   = np.maximum(0.0, regrets) * mask
    total = pos.sum()
    if total > 0.0:
        return pos / total
    valid = mask.astype(np.float64)
    return valid / valid.sum()


# ── Clave rápida de InfoSet (sin Monte Carlo) ─────────────────────────────────

def _fast_key(player, street_idx, bkts, bet_hist):
    """
    Crea la clave del InfoSet usando buckets precomputados.

    Idéntica en estructura a encode_infoset() pero O(1) en tiempo:
    no llama nunca a Monte Carlo porque los buckets ya están en bkts.

    bkts : dict {(player, street_idx) → bucket_int}
    """
    hb    = bkts[(player, street_idx)]
    # Board texture: buckets de las calles anteriores (flop, turn, river parcial)
    all_b = (bkts[(player, 1)], bkts[(player, 2)], bkts[(player, 3)])
    bb    = all_b[:street_idx]     # solo calles ya reveladas
    return (player, street_idx, hb, bb, tuple(bet_hist))


# ── Trainer ───────────────────────────────────────────────────────────────────

class MCCFRTrainer:
    """
    Entrena una estrategia blueprint mediante External Sampling MCCFR.

    Atributos públicos
    ------------------
    regret_sum   : dict[tuple, np.ndarray]  – regrets acumulados
    strategy_sum : dict[tuple, np.ndarray]  – estrategia acumulada (para promedio)
    iterations   : int                      – iteraciones completadas
    """

    def __init__(self):
        self.regret_sum:   dict = {}
        self.strategy_sum: dict = {}
        self.iterations:   int  = 0

    # ── Acceso a tablas ──────────────────────────────────────────────────────

    def _regrets(self, key) -> np.ndarray:
        if key not in self.regret_sum:
            self.regret_sum[key] = np.zeros(NUM_ACTIONS, dtype=np.float64)
        return self.regret_sum[key]

    def _strat_sum(self, key) -> np.ndarray:
        if key not in self.strategy_sum:
            self.strategy_sum[key] = np.zeros(NUM_ACTIONS, dtype=np.float64)
        return self.strategy_sum[key]

    def _strategy(self, key, mask) -> np.ndarray:
        return _regret_match(self._regrets(key), mask)

    # ── Precompute de buckets ────────────────────────────────────────────────

    @staticmethod
    def _precompute_buckets(hands, boards, sims=100):
        """
        Calcula todos los buckets necesarios para UN traversal completo.
        Solo se llama UNA vez por iteración (fuera del bucle recursivo).

        Retorna dict {(player_idx, street_idx) → bucket_int}.
        """
        bkts = {}
        flop_board  = boards[0]
        turn_board  = boards[0] + boards[1]
        river_board = boards[0] + boards[1] + boards[2]

        for p in [0, 1]:
            hand = hands[p]
            bkts[(p, 0)] = preflop_bucket(hand,            num_sims=sims)
            bkts[(p, 1)] = postflop_bucket(hand, flop_board,  num_sims=sims)
            bkts[(p, 2)] = postflop_bucket(hand, turn_board,  num_sims=sims)
            bkts[(p, 3)] = postflop_bucket(hand, river_board, num_sims=sims)
        return bkts

    # ── Máscara de acciones válidas ──────────────────────────────────────────

    @staticmethod
    def _mask(to_call, stack, n_raises, raise_max=2):
        """Booleano por acción: True = acción válida en este estado."""
        m = np.ones(NUM_ACTIONS, dtype=bool)
        if to_call == 0.0:
            m[ACTION_IDX[FOLD]] = False          # no se puede fold sin apuesta
        if n_raises >= raise_max:
            for a in [RAISE_THIRD, RAISE_HALF, RAISE_POT, RAISE_2POT]:
                m[ACTION_IDX[a]] = False         # límite de raises alcanzado
        if stack <= to_call:
            for a in [RAISE_THIRD, RAISE_HALF, RAISE_POT, RAISE_2POT]:
                m[ACTION_IDX[a]] = False         # no hay fichas para raise
        return m

    # ── Lógica de transición ─────────────────────────────────────────────────

    def _apply_action(self, action, traverser,
                      bkts, boards, street_idx,
                      pot, stacks, contribs, bet_hist, n_raises, to_call,
                      position):
        """
        Aplica una acción abstracta y devuelve el valor (en BBs) para el traverser.
        """
        active   = position
        opponent = 1 - active

        if action == FOLD:
            if traverser == opponent:
                return float(pot)
            else:
                return -float(contribs[active])

        elif action == CALL:
            amount = min(to_call, stacks[active])
            ns = list(stacks);  nc = list(contribs)
            ns[active]  -= amount
            nc[active]  += amount
            new_pot = pot + amount

            if nc[0] == nc[1] or ns[active] == 0.0:
                # Caso especial: limp de SB preflop (street_idx=0, active=0,
                # bet_hist vacío). En HUNL el BB tiene opción de subir tras el limp.
                if street_idx == 0 and active == 0 and not bet_hist:
                    return self._cfr(
                        traverser, bkts, boards, street_idx, new_pot,
                        ns, nc, [CALL],
                        n_raises, 0.0, opponent)   # to_call=0: BB puede check o raise
                if street_idx < 3:
                    return self._next_street(
                        traverser, bkts, boards, street_idx + 1, new_pot, ns)
                else:
                    return self._showdown(traverser, bkts, new_pot, nc[traverser])
            else:
                return self._cfr(
                    traverser, bkts, boards, street_idx, new_pot,
                    ns, nc, bet_hist + [CALL],
                    n_raises, nc[opponent] - nc[active], opponent)

        elif action == ALLIN:
            amount = stacks[active]
            ns = list(stacks);  nc = list(contribs)
            ns[active] = 0.0
            nc[active] += amount
            new_pot    = pot + amount
            new_to_call = max(0.0, nc[active] - nc[opponent])
            return self._cfr(
                traverser, bkts, boards, street_idx, new_pot,
                ns, nc, bet_hist + [ALLIN],
                n_raises + 1, new_to_call, opponent)

        else:
            # Raise proporcional
            ratio       = RAISE_RATIOS.get(action, 1.0)
            raise_extra = min(ratio * pot, stacks[active] - to_call)
            total_add   = to_call + raise_extra
            if total_add >= stacks[active]:
                total_add = stacks[active]
                action    = ALLIN

            ns = list(stacks);  nc = list(contribs)
            ns[active]  -= total_add
            nc[active]  += total_add
            new_pot      = pot + total_add
            new_to_call  = nc[active] - nc[opponent]
            return self._cfr(
                traverser, bkts, boards, street_idx, new_pot,
                ns, nc, bet_hist + [action],
                n_raises + 1, new_to_call, opponent)

    def _next_street(self, traverser, bkts, boards, next_street_idx, pot, stacks):
        """Inicia la siguiente calle. BB actúa primero en postflop."""
        return self._cfr(
            traverser, bkts, boards, next_street_idx, pot,
            stacks, [0.0, 0.0], [], 0, 0.0,
            position=1)

    def _showdown(self, traverser, bkts, pot, my_contrib=None):
        """
        Resuelve el showdown usando buckets de EHS² como proxy de equity.
        Los buckets ya están precomputados en bkts [(player, 3)] = river bucket.

        EV = eq * pot - my_contrib
        (cuánto gana el traverser: equity del bote menos lo que ya puso)
        """
        b0 = bkts[(0, 3)]
        b1 = bkts[(1, 3)]

        if   b0 > b1: eq0 = 0.5 + 0.5 * (b0 - b1) / POSTFLOP_BUCKETS
        elif b1 > b0: eq0 = 0.5 - 0.5 * (b1 - b0) / POSTFLOP_BUCKETS
        else:         eq0 = 0.5

        eq    = eq0 if traverser == 0 else (1.0 - eq0)
        contrib = my_contrib if my_contrib is not None else pot / 2.0
        return eq * pot - contrib

    # ── CFR recursivo (External Sampling) ────────────────────────────────────

    def _cfr(self, traverser,
             bkts, boards, street_idx,
             pot, stacks, contribs,
             bet_hist, n_raises, to_call, position):
        """
        External Sampling MCCFR sobre el árbol abstracto.
        Usa bkts (precomputado) para InfoSet keys → O(1) por nodo.

        - traverser: explora TODAS las acciones  → actualiza regrets
        - oponente : muestrea UNA acción según estrategia actual
        """
        active  = position

        # Construir la clave del InfoSet sin Monte Carlo (O(1))
        key      = _fast_key(active, street_idx, bkts, bet_hist)
        mask     = self._mask(to_call, stacks[active], n_raises)
        strategy = self._strategy(key, mask)

        if active == traverser:
            valid_idxs  = np.where(mask)[0]
            action_vals = np.zeros(NUM_ACTIONS)
            for idx in valid_idxs:
                action_vals[idx] = self._apply_action(
                    ABSTRACT_ACTIONS[idx], traverser,
                    bkts, boards, street_idx,
                    pot, stacks, contribs, bet_hist, n_raises, to_call, position)

            ev = float(np.dot(strategy, action_vals))
            self._regrets(key)[:] += (action_vals - ev) * mask
            self._strat_sum(key)[:] += strategy
            return ev

        else:
            idx = int(np.random.choice(NUM_ACTIONS, p=strategy))
            self._strat_sum(key)[:] += strategy
            return self._apply_action(
                ABSTRACT_ACTIONS[idx], traverser,
                bkts, boards, street_idx,
                pot, stacks, contribs, bet_hist, n_raises, to_call, position)

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def train(self, num_iterations: int = 50_000, log_every: int = 5_000,
              bucket_sims: int = 100):
        """
        Ejecuta num_iterations iteraciones de External Sampling MCCFR.

        En cada iteración:
          1. Reparte una mano completa (chance node)
          2. Precomputa todos los buckets (1 vez por iteración)
          3. Traversal desde perspectiva P0
          4. Traversal desde perspectiva P1

        Parámetros
        ----------
        num_iterations : int  – iteraciones de entrenamiento
        log_every      : int  – frecuencia de log de progreso
        bucket_sims    : int  – simulaciones MoneCarlo para precomputar buckets
        """
        print(f"Iniciando MCCFR. Objetivo: {num_iterations:,} iteraciones.")
        for i in range(1, num_iterations + 1):
            hand0, hand1, flop, turn, river = _deal()
            hands  = [hand0, hand1]
            boards = [flop, turn, river]

            # Precomputar buckets UNA sola vez para este deal
            bkts = self._precompute_buckets(hands, boards, sims=bucket_sims)

            # Blinds: SB=0.5, BB=1.0 → pot=1.5, SB to_call=0.5
            init_kwargs = dict(
                bkts=bkts, boards=boards,
                street_idx=0, pot=1.5,
                stacks=[99.5, 99.0],
                contribs=[0.5, 1.0],
                bet_hist=[], n_raises=0,
                to_call=0.5, position=0,
            )
            for traverser in [0, 1]:
                self._cfr(traverser, **init_kwargs)

            self.iterations += 1
            if i % log_every == 0:
                print(f"  iter {i:>8,}  |  InfoSets: {len(self.regret_sum):>8,}")

        print(f"Entrenamiento completado. InfoSets: {len(self.regret_sum):,}")

    # ── Consulta de estrategia ────────────────────────────────────────────────

    def get_strategy(self, key) -> np.ndarray:
        """
        Devuelve la estrategia promedio (blueprint) para un InfoSet dado.

        La estrategia promedio —no la última— converge a equilibrio de Nash.
        Si el InfoSet no ha sido visitado, retorna distribución uniforme.

        Parámetros
        ----------
        key : tuple – clave generada por encode_infoset o _fast_key

        Retorna
        -------
        np.ndarray shape (NUM_ACTIONS,) normalizado a suma 1.
        """
        if key not in self.strategy_sum:
            return np.ones(NUM_ACTIONS, dtype=np.float64) / NUM_ACTIONS
        s     = self.strategy_sum[key].copy()
        total = s.sum()
        if total > 0.0:
            return s / total
        return np.ones(NUM_ACTIONS, dtype=np.float64) / NUM_ACTIONS

    def get_action(self, key) -> str:
        """Muestrea una acción de la estrategia promedio para el InfoSet dado."""
        strat = self.get_strategy(key)
        return str(np.random.choice(ABSTRACT_ACTIONS, p=strat))

    # ── Serialización ─────────────────────────────────────────────────────────

    def save(self, path=None):
        """Serializa el blueprint en disco (pickle)."""
        path = path or BLUEPRINT_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'regret_sum':   self.regret_sum,
                'strategy_sum': self.strategy_sum,
                'iterations':   self.iterations,
            }, f, protocol=4)
        print(f"Blueprint guardado en '{path}'  ({self.iterations:,} iters, "
              f"{len(self.regret_sum):,} InfoSets)")

    @classmethod
    def load(cls, path=None):
        """Carga un blueprint previamente guardado con save()."""
        path = path or BLUEPRINT_PATH
        trainer = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        trainer.regret_sum   = data['regret_sum']
        trainer.strategy_sum = data['strategy_sum']
        trainer.iterations   = data['iterations']
        print(f"Blueprint cargado desde '{path}'  ({trainer.iterations:,} iters, "
              f"{len(trainer.regret_sum):,} InfoSets)")
        return trainer

    @staticmethod
    def exists(path=None):
        """Comprueba si existe un blueprint guardado."""
        return os.path.exists(path or BLUEPRINT_PATH)

    # ── Exploitabilidad (Best Response aproximado) ────────────────────────────

    def exploitability(self, num_samples: int = 500,
                       bucket_sims: int = 50) -> float:
        """
        Calcula la exploitabilidad aproximada del blueprint en mbb/mano.

        Método: Best Response (BR) aproximado por muestreo de manos.
        Usa utilidad NETA: ganancia = eq × pot − inversión_total.

          ε = (BR_0(σ_1) + BR_1(σ_0)) / 2  [BB/mano] → × 1000 = mbb/mano

        En Nash exacto ε = 0. Disminuye monotónicamente con las iteraciones.
        Con ~5k iters y abstracción gruesa, esperar ε ≈ 500–3000 mbb/m.
        Con ~100k iters, esperar ε ≈ 50–300 mbb/m.

        Parámetros
        ----------
        num_samples  : int – manos a muestrear (más = más preciso)
        bucket_sims  : int – simulaciones Monte Carlo por bucket

        Retorna
        -------
        float – exploitabilidad en milli-big-blinds por mano (mbb/m)
        """
        START_STACKS = [99.5, 99.0]   # stacks después de pagar blinds

        br_sum = [0.0, 0.0]
        for _ in range(num_samples):
            hand0, hand1, flop, turn, river = _deal()
            hands  = [hand0, hand1]
            boards = [flop, turn, river]
            bkts   = self._precompute_buckets(hands, boards, sims=bucket_sims)

            init = dict(
                bkts=bkts, boards=boards,
                street_idx=0, pot=1.5,
                stacks=list(START_STACKS),
                contribs=[0.5, 1.0],
                bet_hist=[], n_raises=0,
                to_call=0.5, position=0,
                start_stacks=list(START_STACKS),
            )
            for br_player in [0, 1]:
                br_sum[br_player] += self._best_response(br_player, **init)

        eps_bb  = sum(br_sum) / (2.0 * num_samples)
        return eps_bb * 1000.0   # → mbb/mano

    def _best_response(self, br_player,
                       bkts, boards, street_idx,
                       pot, stacks, contribs,
                       bet_hist, n_raises, to_call, position,
                       start_stacks=None) -> float:
        """
        Traversal recursivo de Best Response para br_player.

        - br_player  : elige la acción con máximo valor  (greedy)
        - oponente   : sigue la estrategia promedio del blueprint
        - start_stacks: stacks al inicio de la mano para calcular inversión total
        """
        if start_stacks is None:
            start_stacks = list(stacks)

        active = position
        key    = _fast_key(active, street_idx, bkts, bet_hist)
        mask   = self._mask(to_call, stacks[active], n_raises)

        kw = dict(br_player=br_player, bkts=bkts, boards=boards,
                  start_stacks=start_stacks)

        if active == br_player:
            best = -1e9
            for idx in np.where(mask)[0]:
                v = self._apply_br(
                    ABSTRACT_ACTIONS[idx], **kw,
                    street_idx=street_idx, pot=pot,
                    stacks=stacks, contribs=contribs,
                    bet_hist=bet_hist, n_raises=n_raises,
                    to_call=to_call, position=position,
                )
                if v > best:
                    best = v
            return best
        else:
            avg   = self.get_strategy(key) * mask
            total = avg.sum()
            avg   = avg / total if total > 0.0 else (mask.astype(float) / mask.sum())
            idx   = int(np.random.choice(NUM_ACTIONS, p=avg))
            return self._apply_br(
                ABSTRACT_ACTIONS[idx], **kw,
                street_idx=street_idx, pot=pot,
                stacks=stacks, contribs=contribs,
                bet_hist=bet_hist, n_raises=n_raises,
                to_call=to_call, position=position,
            )

    def _apply_br(self, action, br_player,
                  bkts, boards, street_idx,
                  pot, stacks, contribs,
                  bet_hist, n_raises, to_call, position,
                  start_stacks=None) -> float:
        """
        Aplica una acción y continúa el traversal de Best Response.

        Retorna la utilidad NETA del br_player:
            ganancia_neta = pot × equity − inversión_total_del_br_player
        donde inversión_total = start_stacks[br_player] − stacks_actuales[br_player].
        """
        if start_stacks is None:
            start_stacks = list(stacks)

        active   = position
        opponent = 1 - active

        def invested(s):
            """Inversión total del br_player hasta este momento."""
            return start_stacks[br_player] - s[br_player]

        def recurse(**kw_extra):
            return self._best_response(
                br_player, bkts=bkts, boards=boards,
                start_stacks=start_stacks, **kw_extra)

        if action == FOLD:
            if br_player == opponent:
                # br_player gana: utilidad neta = pot − inversión_propia
                return float(pot) - invested(stacks)
            else:
                # br_player foldea: pierde toda su inversión
                return -invested(stacks)

        elif action == CALL:
            amount = min(to_call, stacks[active])
            ns = list(stacks);  nc = list(contribs)
            ns[active]  -= amount
            nc[active]  += amount
            new_pot = pot + amount

            if nc[0] == nc[1] or ns[active] == 0.0:
                if street_idx == 0 and active == 0 and not bet_hist:
                    return recurse(street_idx=0, pot=new_pot, stacks=ns,
                                   contribs=nc, bet_hist=[CALL],
                                   n_raises=0, to_call=0.0, position=opponent)
                if street_idx < 3:
                    return recurse(street_idx=street_idx + 1, pot=new_pot,
                                   stacks=ns, contribs=[0.0, 0.0],
                                   bet_hist=[], n_raises=0,
                                   to_call=0.0, position=1)
                else:
                    # Showdown: utilidad neta con inversión total acumulada
                    b0 = bkts[(0, 3)]; b1 = bkts[(1, 3)]
                    if   b0 > b1: eq0 = 0.5 + 0.5 * (b0 - b1) / POSTFLOP_BUCKETS
                    elif b1 > b0: eq0 = 0.5 - 0.5 * (b1 - b0) / POSTFLOP_BUCKETS
                    else:         eq0 = 0.5
                    eq = eq0 if br_player == 0 else (1.0 - eq0)
                    return eq * new_pot - invested(ns)
            return recurse(street_idx=street_idx, pot=new_pot, stacks=ns,
                           contribs=nc, bet_hist=bet_hist + [CALL],
                           n_raises=n_raises,
                           to_call=nc[opponent] - nc[active], position=opponent)

        elif action == ALLIN:
            amount = stacks[active]
            ns = list(stacks);  nc = list(contribs)
            ns[active]  = 0.0
            nc[active] += amount
            new_pot     = pot + amount
            new_to_call = max(0.0, nc[active] - nc[opponent])
            return recurse(street_idx=street_idx, pot=new_pot, stacks=ns,
                           contribs=nc, bet_hist=bet_hist + [ALLIN],
                           n_raises=n_raises + 1, to_call=new_to_call,
                           position=opponent)

        else:
            ratio       = RAISE_RATIOS.get(action, 1.0)
            raise_extra = min(ratio * pot, stacks[active] - to_call)
            total_add   = to_call + raise_extra
            if total_add >= stacks[active]:
                total_add = stacks[active]
                action    = ALLIN
            ns = list(stacks);  nc = list(contribs)
            ns[active]  -= total_add
            nc[active]  += total_add
            new_pot      = pot + total_add
            new_to_call  = nc[active] - nc[opponent]
            return recurse(street_idx=street_idx, pot=new_pot, stacks=ns,
                           contribs=nc, bet_hist=bet_hist + [action],
                           n_raises=n_raises + 1, to_call=new_to_call,
                           position=opponent)


