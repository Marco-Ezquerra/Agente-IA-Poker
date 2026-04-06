# STATUS

## Resumen Dry Run
- Iteraciones ejecutadas: 1,000
- RAM final: 76.8 MB
- Velocidad final: 11.6 iters/s
- Nodos (regret_sum): 58,781
- Proxy convergencia (regret medio abs): 17.379738

## Estimacion de Memoria
- Si 10k iteraciones ocupan 76.8 MB, 50 millones ocuparian ~374.87 GB (extrapolacion lineal).

## Hand-off al Turn
- Estado: Exito
- Detalle: accion_solver=ai | top5_non_uniform=True
- Top 5 manos rival (reach prob):
  - 2s 2h: 0.001878
  - 2s 2c: 0.001878
  - 2s 7d: 0.001839
  - 2c 7d: 0.001839
  - 2s Jd: 0.001808

## Parametros Actuales
- PREFLOP_BUCKETS: 169
- K-Means/EMD POSTFLOP_BUCKETS: 1000
- Hist bins configurado: 20
- Acciones abstractas: f, c, r1, r2, r3, r4, ai
- Sizings de apuesta (x pot): r1=0.3333x_pot, r2=0.6667x_pot, r3=1x_pot, r4=1.5x_pot
- CHECK disponible: SI (via accion CALL cuando to_call == 0)
