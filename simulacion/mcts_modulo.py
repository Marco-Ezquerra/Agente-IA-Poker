import math
import random
import logging
from copy import deepcopy

from poker_engine import PokerCoreEngine, default_action_callback, compact_card
from montecarlo import advanced_ia_action_callback, human_bot_action_callback

# Importar preflop range setup
from tablas_preflop import (
    determine_action_key_from_history,
    determine_subkey_from_history,
    get_rival_hand
)

# Silenciar logs durante MCTS
_original_log_level = logging.root.level
logging.disable(logging.CRITICAL)

# Acumulador global de logs
LOGS_ROLLOUTS = []
LOGS_REWARDS = []


class NodoMCTS:
    def __init__(self, state, jugador_id, parent=None, action=None, untried_actions=None):
        self.state = deepcopy(state)
        self.jugador_id = jugador_id
        self.parent = parent
        self.action = action
        self.visits = 0
        self.total_reward = 0.0
        self.children = {}
        if untried_actions is None:
            raise ValueError("untried_actions debe proporcionarse en el nodo raíz")
        self.untried_actions = list(untried_actions)

    def uct_score(self, child):
        C = math.sqrt(2)
        if child.visits == 0:
            return float('inf')
        return (child.total_reward / child.visits) + C * math.sqrt(math.log(self.visits) / child.visits)

    def seleccionar(self):
        if self.untried_actions:
            return self
        if not self.children:
            return self
        best = max(self.children.values(), key=lambda c: self.uct_score(c))
        return best.seleccionar()

    def expandir(self):
        if not self.untried_actions:
            return self
        accion = self.untried_actions.pop()
        child = NodoMCTS(self.state, self.jugador_id, parent=self,
                         action=accion, untried_actions=[])
        self.children[accion] = child
        return child

    def simular(self):
        return simular_desde_estado(
            self.state,
            self.jugador_id,
            rival_policy=human_bot_action_callback,
            forced_action=self.action
        )

    def retropropagar(self, reward):
        self.visits += 1
        self.total_reward += reward
        if self.parent:
            self.parent.retropropagar(reward)

    def mejor_accion(self):
        if not self.children:
            return None
        # Elegir la acción con mejor reward promedio
        return max(
            self.children.items(),
            key=lambda x: x[1].total_reward / max(1, x[1].visits)
        )[0]


def run_mcts(state, jugador_id, valid_actions, num_simulaciones=500):
    root = NodoMCTS(state, jugador_id, untried_actions=valid_actions)
    for _ in range(num_simulaciones):
        node = root.seleccionar()
        if node.untried_actions:
            node = node.expandir()
        reward = node.simular()
        node.retropropagar(reward)

    if LOGS_ROLLOUTS:
        import os, json
        os.makedirs("logs_mcts", exist_ok=True)
        with open("logs_mcts/mcts_rollouts.jsonl", "w", encoding="utf-8") as f:
            for entry in LOGS_ROLLOUTS:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        with open("logs_mcts/mcts_rewards.json", "w", encoding="utf-8") as f:
            json.dump(LOGS_REWARDS, f, ensure_ascii=False, indent=2)

    return root.mejor_accion()


def simular_desde_estado(state, jugador_id, rival_policy=None, forced_action=None):
    # ----- Desempaquetado robusto de `state` -----
    try:
        # Lo normal: state == [pot, players, community]
        pot, players, community = state
    except Exception:
        # Si viene envuelto u otro formato, buscamos un sub‑elemento de 3 elementos
        found = False
        if isinstance(state, (list, tuple)):
            for item in state:
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    pot, players, community = item
                    found = True
                    break
        if not found:
            raise ValueError(f"simular_desde_estado: estado inesperado, no pude desempaquetar {state!r}")

    initial_chips = {p[0]: p[1] for p in players}
    if rival_policy is None:
        rival_policy = human_bot_action_callback

    # Inicializar nuevo engine para la simulación
    engine = PokerCoreEngine(
        nombres_jugadores=[f"P{i}" for i in range(len(players))],
        action_callback=None
    )

    # Callback híbrido: fuerza action una vez, luego IA propia y rival
    def hybrid_callback(jugador_obj, st, valid_actions):
        nonlocal forced_action
        if jugador_obj.id == jugador_id and forced_action is not None:
            action = forced_action
            forced_action = None
            return action
        if jugador_obj.id == jugador_id:
            return advanced_ia_action_callback(jugador_obj, st, valid_actions)
        return rival_policy(jugador_obj, st, valid_actions)

    engine.action_callback = hybrid_callback
    engine.reset()
    # Cancelamos reparto automático de manos
    engine.mesa.repartir_manos = lambda: None

    # Función helper para extraer cartas de la baraja interna
    def _take(card_str):
        for c in engine.mesa.baraja.cartas:
            if compact_card(c) == card_str:
                engine.mesa.baraja.cartas.remove(c)
                return c
        raise ValueError(f"Carta {card_str} no encontrada en la baraja")

    # === Asignar las manos iniciales ===
    for j in engine.mesa.jugadores:
        j.fichas = initial_chips[j.id]
        j.mano.clear()
        mano_real = players[j.id][2]
        cartas = []
        for cs in mano_real:
            carta = _take(cs)
            if carta:
                cartas.append(carta)
        j.recibir_cartas(cartas)



    # === Asignar cartas comunitarias ===
    if community:
        engine.mesa.community_cards.clear()
        for cs in community:
            engine.mesa.community_cards.append(_take(cs))

    # Ejecutar la mano completa y calcular reward
    final_state    = engine.play_round()
    _, final_plys, _ = final_state
    final_chips    = {p[0]: p[1] for p in final_plys}
    reward         = final_chips[jugador_id] - initial_chips[jugador_id]

    # Loguear rollout para debug o análisis posterior
    masked_state = deepcopy(state)
    masked_state[1][1][2] = ["??", "??"]
    LOGS_ROLLOUTS.append({
        "initial_state": masked_state,
        "final_state":   final_state,
        "reward":        reward,
        "history":       engine.history
    })
    LOGS_REWARDS.append(reward)

    logging.root.setLevel(_original_log_level)
    return reward
