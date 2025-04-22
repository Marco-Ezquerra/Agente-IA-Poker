import math
import random
import logging
from copy import deepcopy
from poker_engine import PokerCoreEngine, default_action_callback

# Silenciar logs durante MCTS
_original_log_level = logging.root.level
logging.disable(logging.CRITICAL)

class NodoMCTS:
    """
    Nodo para MCTS en póker.
    """
    def __init__(self, state, jugador_id, parent=None, action=None, untried_actions=None):
        self.state = deepcopy(state)
        self.jugador_id = jugador_id
        self.parent = parent
        self.action = action
        self.visits = 0
        self.total_reward = 0.0
        # Inicializar hijos y acciones pendientes
        self.children = {}
        if untried_actions is not None:
            self.untried_actions = list(untried_actions)
        else:
            self.untried_actions = []  # sin acciones adicionales por defecto

    def uct_score(self, child):
        C = math.sqrt(2)
        if child.visits == 0:
            return float('inf')
        return (child.total_reward / child.visits) + C * math.sqrt(math.log(self.visits) / child.visits)

    def seleccionar(self):
        if self.untried_actions:
            return self
        best = max(self.children.values(), key=lambda c: self.uct_score(c))
        return best.seleccionar()

    def expandir(self):
        if not self.untried_actions:
            return self
        accion = self.untried_actions.pop()
        hijo = NodoMCTS(self.state, self.jugador_id, parent=self, action=accion)
        self.children[accion] = hijo
        return hijo

    def simular(self):
        return simular_desde_estado(
            self.state,
            self.jugador_id,
            rival_policy=None,
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
        return max(self.children.items(), key=lambda x: x[1].visits)[0]


def run_mcts(state, jugador_id, valid_actions, num_simulaciones=500):
    root = NodoMCTS(state, jugador_id, untried_actions=valid_actions)
    for _ in range(num_simulaciones):
        node = root.seleccionar()
        if node.untried_actions:
            node = node.expandir()
        reward = node.simular()
        node.retropropagar(reward)
    return root.mejor_accion()


def mcts_action_callback(jugador, state, valid_actions):
    return run_mcts(state, jugador.id, valid_actions, num_simulaciones=500)


def simular_desde_estado(state, jugador_id, rival_policy=None, forced_action=None):
    pot, players, community = state
    initial_chips = {p[0]: p[1] for p in players}
    if rival_policy is None:
        rival_policy = default_action_callback

    env_engine = PokerCoreEngine(
        nombres_jugadores=[f"P{i}" for i in range(len(players))],
        action_callback=None
    )

    def hybrid_callback(jugador_obj, st, valid_actions):
        nonlocal forced_action
        if jugador_obj.id == jugador_id and forced_action is not None:
            action = forced_action
            forced_action = None
            return action
        if jugador_obj.id == jugador_id:
            return default_action_callback(jugador_obj, st, valid_actions)
        return rival_policy(jugador_obj, st, valid_actions)
    env_engine.action_callback = hybrid_callback

    env_engine.reset()
    for j in env_engine.mesa.jugadores:
        j.fichas = initial_chips[j.id]

    final_state = env_engine.play_round()
    _, final_players, _ = final_state
    final_chips = {p[0]: p[1] for p in final_players}

    logging.root.setLevel(_original_log_level)
    return final_chips[jugador_id] - initial_chips[jugador_id]

# Restaurar logs al cargar el módulo
logging.root.setLevel(_original_log_level)

