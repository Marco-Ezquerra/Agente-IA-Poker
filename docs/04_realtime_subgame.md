# 04 — Subgame Solving en Tiempo Real

Archivo fuente: [`simulacion/cfr/realtime_search.py`](../simulacion/cfr/realtime_search.py)

---

## 1. Motivación

El blueprint se entrena sobre cartas **abstractas** (buckets). En producción, el agente conoce sus cartas exactas (ej. `As Kd`). El **Subgame Solving** permite re-optimizar la estrategia para el subgame actual, aprovechando esa información adicional con un coste computacional acotado.

### Analogía

El blueprint es como memorizar aperturas de ajedrez. El Subgame Solver es como el motor de búsqueda que analiza en profundidad la posición concreta una vez en la partida real.

---

## 2. Arquitectura de `RealtimeSearch`

```
Para cada muestra de mano del oponente (opp_samples):
  1. _precompute_buckets(hand_agente, hand_oponente_muestreada, board)
     → bkts: dict {(player, street) → bucket}   [O(bucket_sims)]
  2. Para cada iteración CFR (iters_per_sample):
     _search(traverser=0, ...)
     _search(traverser=1, ...)   [O(1) por nodo, sin Monte Carlo]

Extraer estrategia promedio del InfoSet actual del agente
→ acción óptima
```

### Parámetros de `RealtimeSearch`

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `blueprint` | `None` | Blueprint MCCFR preentrenado |
| `depth` | `1` | Calles hacia adelante a explorar |
| `iterations` | `200` | Iteraciones CFR por llamada a `get_action` |

---

## 3. Muestreo de la mano del oponente

Como el agente no conoce las cartas del rival, se muestrean $N$ manos posibles de la baraja residual y se promedian las estrategias:

```python
deck_rem = [c for c in full_deck() if c not in known_cards]
for _ in range(opp_samples):
    random.shuffle(deck_rem)
    opp_hand = deck_rem[:2]
    bkts = _precompute_buckets(my_hand, opp_hand, board, ...)
    # ... iteraciones CFR con esta hipótesis de mano del oponente
```

Este muestreo aproxima el promedio sobre el rango del oponente, que es la cantidad correcta a calcular en juegos con información imperfecta.

---

## 4. Clave de InfoSet — alineación con el blueprint (BUG-5)

**Crítico:** La función `_fast_key` en `realtime_search.py` debe generar exactamente la misma clave que en `mccfr_trainer.py`:

```python
# AMBOS archivos usan exactamente esto:
def _fast_key(player, street_idx, bkts, bet_hist):
    hb = bkts[(player, street_idx)]
    return (player, street_idx, hb, tuple(bet_hist))
```

Si las claves difieren, `blueprint.get_strategy(key)` siempre retorna distribución uniforme (el InfoSet no se encuentra nunca) y el preentrenamiento es completamente inútil en producción.

---

## 5. Limp preflop — consistencia del game tree (BUG-9)

El árbol de juego del realtime debe ser **idéntico al árbol sobre el que se entrenó el blueprint**. Un caso especial es el `limp` (SB iguala el blind sin raise preflop):

```python
# En _step, rama CALL:
if nc[0] == nc[1] or ns[active] == 0.0:
    # Limp: SB iguala, BB recibe opción de check o raise
    if street_idx == 0 and active == 0 and not bet_hist:
        return self._search(
            traverser, bkts, street_idx, new_pot, ns, nc,
            [CALL], 0, 0.0, opponent, depth_left)  # BB actúa con to_call=0
```

Sin este bloque, el BB nunca recibe opción tras el limp → el árbol diverge → la estrategia aprendida para situaciones post-limp es inválida.

---

## 6. Evaluación de nodos hoja — `_leaf_value`

Cuando se supera el horizonte de búsqueda (`depth_left <= 0`), el valor del nodo se estima con:

```python
bucket  = bkts[(traverser, street_idx)]
eq_raw  = bucket / POSTFLOP_BUCKETS         # equity cruda del bucket

# Ponderar con agresividad del blueprint si está disponible
if blueprint:
    strat = blueprint.get_strategy(key)
    aggr  = sum(strat para acciones de raise y allin)
    eq_raw = eq_raw * 0.7 + min(aggr, 1.0) * 0.3

return (2.0 * eq_raw - 1.0) * pot * 0.4
```

El factor `0.4` escala conservadoramente el valor. Un bucket 15 (EHS²≈1.0) con pot=10 retorna $0.35 \times 10 = 3.5$, no el valor teórico de 10. Esto es intencionado: el leaf value es un proxy, no una evaluación exacta.

---

## 7. `_search` — CFR local del subgame

El CFR interno de `RealtimeSearch._search` mantiene sus propias tablas `_regret` y `_strat_sum`, distintas del blueprint:

```python
# Tablas locales, se limpian en cada llamada a get_action
self._regret:    dict = {}   # regrets del subgame actual
self._strat_sum: dict = {}
```

Esto significa que el subgame se resuelve de cero en cada decisión (sin warm-start). Cada llamada a `get_action` parte de regrets cero y acumula durante `iterations` iteraciones. Para decisiones frecuentes (múltiples acciones en la misma mano), cada una se resuelve independientemente.

---

## 8. Flujo completo de `get_action`

```python
action = engine.get_action(
    traverser=0,
    my_hand=['Ah', 'Kd'],
    board=['Jh', 'Ts', '2c'],
    street_str='flop',
    pot=4.5,
    stacks=[97.5, 95.5],
    contribs=[0.0, 0.0],
    bet_hist=[],
    n_raises=0,
    to_call=0.0,
    opp_samples=8,
    bucket_sims=50
)
```

Internamente:

```
1. self._regret.clear(); self._strat_sum.clear()
2. Para cada una de las 8 muestras del oponente:
   a. _precompute_buckets(hand_agente, hand_oponente_muestreada, board) → bkts
   b. 200/8 = 25 iteraciones CFR:
      - _search(traverser=0, ...)
      - _search(traverser=1, ...)
3. Extraer strategy_sum[my_key] → normalizar → argmax → acción
```

---

## 9. Parámetros recomendados por escenario

| Escenario | `opp_samples` | `bucket_sims` | `iterations` | Latencia aprox. |
|-----------|--------------|--------------|--------------|-----------------|
| Juego rápido / testing | 4 | 30 | 100 | < 1 s |
| Juego estándar | 8 | 50 | 200 | 2–5 s |
| Torneos / análisis offline | 16 | 100 | 500 | 10–30 s |

---

## 10. Interacción con el blueprint

El blueprint actúa de dos maneras dentro del subgame solver:

1. **Leaf evaluator:** cuando `depth_left=0`, `_leaf_value` consulta `blueprint.get_strategy(key)` para ponderar la equity con la agresividad GTO.
2. **Warm-start implícito:** aunque las tablas locales empiezan vacías, los nodos del árbol del subgame que coinciden con InfoSets del blueprint recibirán el mismo bucket → la estrategia del blueprint guía la exploración inicial (vía `_leaf_value`).

Para un warm-start explícito (inicializar `_regret` desde el blueprint), sería necesario copiar `blueprint.regret_sum` en `self._regret` antes del CFR. Esto es una posible mejora futura.
