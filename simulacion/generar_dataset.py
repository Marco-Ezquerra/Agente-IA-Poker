#!/usr/bin/env python3
"""
Script que simula múltiples partidas completas de póker usando advanced_ia_action_callback en todas las fases.
Imprime todo por pantalla y guarda la salida como string y estructura en dataset_mcts/dataset_advanced_full.jsonl.
"""

import time
import json
import os
import io
import sys
from poker_engine import PokerCoreEngine
from montecarlo import advanced_ia_action_callback

def guardar_datos_partida(state0, final_state, history, salida_pantalla, ruta_salida="dataset_mcts/dataset_advanced_full.jsonl"):
    pot_inicial, players, community = state0
    pot_final, players_finales, community_final = final_state

    datos = {
        "state0": state0,
        "final_state": final_state,
        "historial": history,
        "ganancia": players_finales[0][1] - players[0][1],
        "mano_ia": players[0][2],
        "mano_rival": players[1][2],
        "community_final": community_final,
        "output": salida_pantalla
    }

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    with open(ruta_salida, "a", encoding="utf-8") as f:
        f.write(json.dumps(datos, ensure_ascii=False) + "\n")

def simular_partida():
    buffer = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buffer

    print("\n=== Simulación de partida completa con advanced_ia_action_callback ===\n")

    engine = PokerCoreEngine(
        nombres_jugadores=["IA", "Rival"],
        action_callback=None
    )

    engine.reset()
    state0 = engine.get_state()
    mano_ia = state0[1][0][2]
    print(f"Mano IA (SB): {mano_ia}")
    print(f"Comunidad inicial: {state0[2]}")

    fase_actual = "preflop"

    def callback(jugador, state, valid_actions):
        nonlocal fase_actual
        nueva_fase = state.get("fase", fase_actual)
        if nueva_fase != fase_actual:
            fase_actual = nueva_fase
            comunidad = state.get("community", [])
            print(f"\n[{fase_actual.upper()}] Cartas comunitarias: {comunidad}")
        return advanced_ia_action_callback(jugador, state, valid_actions)

    engine.action_callback = callback

    t0 = time.time()
    final_state = engine.play_round()
    t1 = time.time()

    print("\n--- Fin de la partida ---")
    print(f"Duración: {t1 - t0:.2f} segundos")

    pot, players, community = final_state
    print(f"\nPot total: {pot}")
    for i, (nombre, fichas, mano) in enumerate(players):
        print(f"Jugador {i} ({nombre}) - Fichas finales: {fichas}, Mano: {mano}")
    print("Cartas comunitarias:", community)

    print("\nHistorial completo:")
    for evento in engine.history:
        print("·", evento)

    # Capturar salida e imprimirla de verdad
    sys.stdout = sys_stdout
    salida = buffer.getvalue()
    print(salida)

    guardar_datos_partida(state0, final_state, engine.history, salida)

if __name__ == "__main__":
    for i in range(10):  # Cambia el número de partidas aquí
        print(f"\n========= PARTIDA {i+1} =========")
        simular_partida()
