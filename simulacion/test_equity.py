# test_mcts_vs_human.py

from poker_engine import PokerCoreEngine, compact_card
from montecarlo import get_equity_cached, human_bot_action_callback
from mcts_modulo import run_mcts
from copy import deepcopy


def mostrar_estado(jugador, state, fase):
    mano = [compact_card(c) for c in jugador.mano]
    comunidad = state.get("community", [])
    equity = get_equity_cached(mano, comunidad)
    print(f"\n--- Fase: {fase.upper()} ---")
    print(f"Jugador {jugador.id} ({jugador.nombre})")
    print(f"  Mano:      {mano}")
    print(f"  Comunidad: {comunidad if comunidad else '(vacía)'}")
    print(f"  Equity:    {round(equity * 100, 2)}%")
    return equity


def mcts_vs_human_test(num_partidas=3):
    print(f"\n=== TEST: MCTS vs HumanBot ({num_partidas} partidas) ===")
    engine = PokerCoreEngine(
        nombres_jugadores=["MCTS_IA", "Human_Bot"],
        action_callback=None
    )

    resultados = {0: 0, 1: 0, "empates": 0}

    for partida in range(num_partidas):
        print(f"\n\n======= PARTIDA #{partida + 1} =======")

        def hybrid_callback(jugador, state, valid_actions):
            fase = state.get("fase", "desconocida")
            mostrar_estado(jugador, state, fase)
            print(f"  Acciones válidas: {valid_actions}")
            if jugador.id == 0:
                best = run_mcts(deepcopy(engine.get_state()), jugador.id, valid_actions, num_simulaciones=300)
                print(f"=> MCTS elige: {best}")
                return best
            else:
                action = human_bot_action_callback(jugador, state, valid_actions)
                print(f"=> HumanBot elige: {action}")
                return action

        engine.action_callback = hybrid_callback
        engine.reset()
        final_state = engine.play_round()

        pot = final_state[0]
        fichas_finales = {p[0]: p[1] for p in final_state[1]}
        delta0 = fichas_finales[0] - 100
        delta1 = fichas_finales[1] - 100

        print("\n===== RESULTADO DE LA PARTIDA =====")
        print(f"Fichas finales -> MCTS_IA: {fichas_finales[0]} | Human_Bot: {fichas_finales[1]} | Pot final: {pot}")

        if delta0 > delta1:
            resultados[0] += 1
            print("Ganador: MCTS_IA")
        elif delta1 > delta0:
            resultados[1] += 1
            print("Ganador: Human_Bot")
        else:
            resultados["empates"] += 1
            print("Empate.")

        print("\n--- HISTORIAL COMPLETO ---")
        for evento in engine.history:
            print(evento)

    print("\n====== RESUMEN FINAL ======")
    print(f"Partidas jugadas: {num_partidas}")
    print(f"Ganadas por MCTS_IA:   {resultados[0]}")
    print(f"Ganadas por Human_Bot: {resultados[1]}")
    print(f"Empates:               {resultados['empates']}")


if __name__ == "__main__":
    mcts_vs_human_test(num_partidas=1)
