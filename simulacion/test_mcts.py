#!/usr/bin/env python3
"""
Test que simula una partida completa de póker usando únicamente advanced_ia_action_callback en todas las fases.
"""

import time
from poker_engine import PokerCoreEngine
from montecarlo import advanced_ia_action_callback

def main():
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

if __name__ == "__main__":
    main()
