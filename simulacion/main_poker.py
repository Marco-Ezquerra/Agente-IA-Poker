import math
import random
import logging
from copy import deepcopy
from functools import partial
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
        if not self.children:
            # Nodo terminal sin hijos: se devuelve a sí mismo
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


# ── Modo de juego con blueprint GTO + opponent model ──────────────────────────

def jugar_con_blueprint(num_manos: int = 10,
                        fichas: int = 100,
                        ia_nombre: str = "IA-GTO",
                        rival_nombre: str = "Rival",
                        verbose: bool = True,
                        model_path: str = None):
    """
    Ejecuta num_manos de HUNL usando el blueprint GTO como política de la IA.

    Si no existe blueprint entrenado, usa advanced_ia_action_callback como
    fallback. El rival usa human_bot_action_callback.

    Parámetros
    ----------
    num_manos    : int  – manos a jugar
    fichas       : int  – fichas iniciales de cada jugador (en BBs)
    ia_nombre    : str  – nombre del jugador IA
    rival_nombre : str  – nombre del jugador rival
    verbose      : bool – imprimir resultado de cada mano
    model_path   : str | None – ruta para cargar/guardar el OpponentModel
                  (None = no persistir; 'opp_model.pkl' = persistir entre sesiones)

    Retorna
    -------
    dict con 'ganancia_ia', 'ganancia_rival', 'manos'
    """
    from montecarlo import blueprint_action_callback, human_bot_action_callback
    from opponent_model import OpponentModel

    # ── Cargar OpponentModel persistido o crear uno nuevo ────────────────────
    if model_path and os.path.isfile(model_path):
        try:
            opp_model = OpponentModel.load(model_path)
            if verbose:
                print(f"OpponentModel cargado desde '{model_path}'  "
                      f"({opp_model.stats.hands_seen} manos previas)")
        except Exception:
            opp_model = OpponentModel(opponent_id=1)
    else:
        opp_model = OpponentModel(opponent_id=1)

    # Callback de la IA: blueprint + opponent model
    def ia_callback(jugador, state, valid_actions):
        return blueprint_action_callback(jugador, state, valid_actions,
                                         opp_model=opp_model, opp_samples=6)

    # Callback del rival: bot mediocre (simula jugador humano)
    def rival_callback(jugador, state, valid_actions):
        fase = state.get('fase', 'preflop')
        accion = human_bot_action_callback(jugador, state, valid_actions)
        # Registrar acciones del rival en el opponent model
        opp_model.observe_action(
            accion if isinstance(accion, str) else accion[0],
            fase,
            voluntarily=(fase == 'preflop')
        )
        return accion

    def action_callback(jugador, state, valid_actions):
        if jugador.id == 0:
            return ia_callback(jugador, state, valid_actions)
        return rival_callback(jugador, state, valid_actions)

    engine = PokerCoreEngine(
        nombres_jugadores=[ia_nombre, rival_nombre],
        action_callback=action_callback
    )
    engine.mesa.jugadores[0].fichas = fichas
    engine.mesa.jugadores[1].fichas = fichas

    resultados = []
    for mano_idx in range(num_manos):
        opp_model.new_hand()

        # Guardar fichas antes de la mano
        fichas_antes = {j.nombre: j.fichas for j in engine.mesa.jugadores}

        try:
            engine.reset()
            engine.mesa.jugadores[0].fichas = fichas_antes[ia_nombre]
            engine.mesa.jugadores[1].fichas = fichas_antes[rival_nombre]
            engine.action_callback = action_callback
            final_state = engine.play_round()
        except Exception as e:
            logging.warning("Error en mano %d: %s", mano_idx + 1, e)
            continue

        _, players_finales, _ = final_state
        fichas_despues = {p[0]: p[1] for p in players_finales}
        ganancia_ia = fichas_despues[0] - fichas_antes[ia_nombre]

        resultados.append(ganancia_ia)
        if verbose:
            print(f"Mano {mano_idx+1:>3} | IA: {fichas_despues[0]:6.1f}  "
                  f"Rival: {fichas_despues[1]:6.1f}  "
                  f"Δ IA: {ganancia_ia:+.1f}  "
                  f"Rival: {opp_model.classify()}")

    total_ia    = sum(resultados)
    total_rival = -total_ia
    print(f"\n{'='*55}")
    print(f"Resultado final ({num_manos} manos):")
    print(f"  {ia_nombre}    : {total_ia:+.1f} BBs")
    print(f"  {rival_nombre} : {total_rival:+.1f} BBs")
    print(f"  win-rate IA : {total_ia / max(num_manos, 1):+.2f} BBs/mano")
    print(f"\n{opp_model.summary()}")

    # ── Guardar OpponentModel si se especificó ruta ───────────────────────────
    if model_path:
        try:
            opp_model.save(model_path)
            if verbose:
                print(f"OpponentModel guardado en '{model_path}'")
        except Exception as e:
            logging.warning("No se pudo guardar el OpponentModel: %s", e)

    return {
        'ganancia_ia':    total_ia,
        'ganancia_rival': total_rival,
        'manos':          resultados,
    }


# ── Modo 1v1 interactivo: humano vs IA ────────────────────────────────────────

def jugar_1v1_humano(num_manos: int = 10, fichas: int = 100, model_path: str = None):
    """
    Modo interactivo donde el HUMANO juega contra la IA GTO en la consola.

    En cada turno humano se muestra:
      - Tus cartas  (hole cards)
      - Cartas comunitarias visibles
      - Pot actual y a cuánto sube la apuesta
      - Acciones disponibles numeradas

    El jugador introduce el número de la acción. Para raise debe
    indicar también el tamaño (ej: '3 50' para raise a 50 BBs).
    """
    from montecarlo import blueprint_action_callback
    from opponent_model import OpponentModel

    # Cargar OpponentModel que modela al humano
    if model_path and os.path.isfile(model_path):
        try:
            opp_model = OpponentModel.load(model_path)
            print(f"Historial cargado ({opp_model.stats.hands_seen} manos previas)\n")
        except Exception:
            opp_model = OpponentModel(opponent_id=0)
    else:
        opp_model = OpponentModel(opponent_id=0)

    # Callback IA
    def ia_callback(jugador, state, valid_actions):
        return blueprint_action_callback(jugador, state, valid_actions,
                                         opp_model=opp_model, opp_samples=5)

    # Callback HUMANO — decides tú
    def humano_callback(jugador, state, valid_actions):
        _mostrar_estado(jugador, state)
        return _pedir_accion(valid_actions)

    fichas_ia     = fichas
    fichas_humano = fichas
    total_ia      = 0.0
    total_humano  = 0.0

    print("=" * 60)
    print("  HUNL: TÚ vs IA-GTO")
    print(f"  Fichas: {fichas} BBs cada uno  |  {num_manos} manos")
    print("=" * 60)

    for mano_idx in range(1, num_manos + 1):
        print(f"\n{'─'*50}  MANO {mano_idx}/{num_manos}")

        engine = PokerCoreEngine(
            nombres_jugadores=["Humano", "IA-GTO"],
        )
        engine.jugadores[0].fichas = fichas_humano
        engine.jugadores[1].fichas = fichas_ia

        result = engine.jugar_mano(
            accion_callbacks={
                engine.jugadores[0].id: humano_callback,
                engine.jugadores[1].id: ia_callback,
            }
        )

        delta_humano = result.get('delta', {}).get(engine.jugadores[0].id, 0.0)
        delta_ia     = result.get('delta', {}).get(engine.jugadores[1].id, 0.0)
        fichas_humano += delta_humano
        fichas_ia     += delta_ia
        total_humano  += delta_humano
        total_ia      += delta_ia

        # Actualizar modelo del humano
        for pid, actions in result.get('action_history', {}).items():
            if int(pid) == engine.jugadores[0].id:
                for a in actions:
                    opp_model.observe_action(a)

        signo = "ganas" if delta_humano > 0 else ("empate" if delta_humano == 0 else "pierdes")
        print(f"  → {signo:6s}  {delta_humano:+.1f} BBs  |  "
              f"[Humano: {fichas_humano:.1f}  IA: {fichas_ia:.1f}]")

        if fichas_humano <= 0 or fichas_ia <= 0:
            print("\n  *** Un jugador sin fichas — fin de la partida ***")
            break

    print("\n" + "=" * 60)
    print("  RESULTADO FINAL")
    print(f"  Tú     : {total_humano:+.1f} BBs")
    print(f"  IA-GTO : {total_ia:+.1f} BBs")
    wr = total_humano / max(mano_idx, 1)
    print(f"  Tu win-rate : {wr:+.2f} BBs/mano  ({wr*100:.1f} BB/100)")
    print("=" * 60)

    print(f"\n[IA sobre ti] {opp_model.summary()}")

    if model_path:
        try:
            opp_model.save(model_path)
            print(f"Historial guardado en '{model_path}'")
        except Exception:
            pass


def _mostrar_estado(jugador, state):
    """Imprime el estado visible para el jugador humano en su turno."""
    from poker_engine import compact_card
    cartas = [compact_card(c) for c in jugador.mano]
    community = state.get('community', [])
    pot       = state.get('pot', 0)
    current   = state.get('current_bet', 0)
    contrib   = state.get('contributions', {})
    pagado    = contrib.get(jugador.id, 0)
    to_call   = max(0.0, float(current) - float(pagado))
    fase      = state.get('fase', '').upper()
    stack     = jugador.fichas

    print(f"\n  [{fase}]  Pot: {pot:.1f} BBs  |  Stack: {stack:.1f} BBs")
    print(f"  Tus cartas   : {' '.join(cartas)}")
    if community:
        print(f"  Tablero      : {' '.join(community)}")
    if to_call > 0:
        print(f"  Para igualar : {to_call:.1f} BBs")


def _pedir_accion(valid_actions):
    """Muestra menú numerado y recoge la acción del jugador humano."""
    print("\n  Acciones disponibles:")
    for idx, a in enumerate(valid_actions):
        if isinstance(a, tuple):
            print(f"    {idx+1}. raise {a[1]:.1f} BBs")
        else:
            print(f"    {idx+1}. {a}")
    print("    (Para raise personalizado: 'r <cantidad>')")

    while True:
        raw = input("\n  Tu acción > ").strip().lower()
        if raw.startswith('r '):
            try:
                amount = float(raw.split()[1])
                # Buscar el raise más cercano entre los válidos
                raises = [a for a in valid_actions if isinstance(a, tuple)]
                if raises:
                    # devolver el raise con tamaño más cercano
                    return min(raises, key=lambda x: abs(x[1] - amount))
                print("  No hay raises disponibles.")
                continue
            except (IndexError, ValueError):
                print("  Formato incorrecto. Usa 'r 50' para raise a 50.")
                continue
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(valid_actions):
                return valid_actions[idx]
            print(f"  Número entre 1 y {len(valid_actions)}.")
        except ValueError:
            # Intentar match por texto
            matches = [a for a in valid_actions
                       if isinstance(a, str) and a.startswith(raw)]
            if len(matches) == 1:
                return matches[0]
            print("  Opción no reconocida. Introduce el número o 'r <cantidad>'.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Jugar HUNL con el agente GTO")
    parser.add_argument('--manos',  type=int, default=10,
                        help='Número de manos a jugar (default: 10)')
    parser.add_argument('--fichas', type=int, default=100,
                        help='Fichas iniciales en BBs (default: 100)')
    parser.add_argument('--quiet',  action='store_true',
                        help='Suprimir output por mano')
    parser.add_argument('--model',  type=str, default=None,
                        help='Ruta para persistir el OpponentModel entre sesiones '
                             '(e.g. opp_model.pkl)')
    parser.add_argument('--humano', action='store_true',
                        help='Modo interactivo: tú juegas contra la IA')
    args = parser.parse_args()

    if args.humano:
        jugar_1v1_humano(
            num_manos  = args.manos,
            fichas     = args.fichas,
            model_path = args.model,
        )
    else:
        jugar_con_blueprint(
            num_manos  = args.manos,
            fichas     = args.fichas,
            verbose    = not args.quiet,
            model_path = args.model,
        )

