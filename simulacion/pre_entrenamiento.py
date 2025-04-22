#!/usr/bin/env python3
import os
import json
from poker_engine import PokerCoreEngine
from mcts_modulo import run_mcts
from montecarlo import advanced_ia_action_callback

def guardar_datos_partida(state0, mejor_accion, final_state, history, ruta_salida="dataset_mcts.jsonl"):
    pot_inicial, players, community = state0
    pot_final, players_finales, community_final = final_state

    datos = {
        "state0": state0,
        "mejor_accion": mejor_accion,
        "final_state": final_state,
        "historial": history,
        "ganancia": players_finales[0][1] - players[0][1],
        "mano_ia": players[0][2],
        "mano_rival": players[1][2],
        "community_final": community_final
    }

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    with open(ruta_salida, "a", encoding="utf-8") as f:
        f.write(json.dumps(datos, ensure_ascii=False) + "\n")

def ejecutar_partida_y_guardar(ruta_salida):
    engine = PokerCoreEngine(
        nombres_jugadores=["IA", "Rival"],
        action_callback=None
    )
    engine.reset()
    state0 = engine.get_state()

    ronda = engine.mesa.iniciar_ronda_apuestas("preflop", lambda *args: "check")
    valid_actions = ronda if isinstance(ronda, list) else []

    mejor_accion = run_mcts(state0, jugador_id=0, valid_actions=valid_actions, num_simulaciones=500)

    def callback(jugador, state, valid_actions):
        fase_actual = state.get("fase", "preflop")
        if jugador.id == 0 and fase_actual == "preflop":
            return mejor_accion
        return advanced_ia_action_callback(jugador, state, valid_actions)

    engine.action_callback = callback
    final_state = engine.play_round()

    guardar_datos_partida(state0, mejor_accion, final_state, engine.history, ruta_salida)

def main(num_partidas=10, output_file="dataset_mcts/dataset.jsonl"):
    for i in range(num_partidas):
        print(f"Simulando partida {i+1}/{num_partidas}")
        ejecutar_partida_y_guardar(output_file)

if __name__ == "__main__":
    main()
