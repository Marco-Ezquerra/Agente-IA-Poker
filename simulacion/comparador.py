from template import draw_random_cards, eval_hand_from_strings
import sys

# Número de simulaciones (puedes cambiarlo por argumento)
N = 10000

j1_wins = 0
j2_wins = 0
ties = 0

for _ in range(N):
    # Robar 9 cartas sin repetir: 2+2+5
    cartas = draw_random_cards(9)
    #print(cartas)
    j1 = cartas[0:2]
    j2 = cartas[2:4]
    #print(j1)
    #print(j2)
    board = cartas[4:9]
    #print(board)
    score1 = eval_hand_from_strings(j1, board)
    #print(score1)
    score2 = eval_hand_from_strings(j2, board)
    #print(score2)
    if score1 > score2:
        j1_wins += 1
    elif score2 > score1:
        j2_wins += 1
    else:
        ties += 1

# Mostrar resultados
print(f"\nSimulaciones: {N}")
print(f"Jugador 1 gana : {j1_wins}")
print(f"Jugador 2 gana : {j2_wins}")
print(f"Empates        : {ties}")
