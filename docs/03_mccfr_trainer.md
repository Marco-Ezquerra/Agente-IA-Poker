# 03 — Trainer MCCFR

Archivo fuente: [`simulacion/cfr/mccfr_trainer.py`](../simulacion/cfr/mccfr_trainer.py)

---

## 1. Visión general

`MCCFRTrainer` implementa el algoritmo de **External Sampling MCCFR** (Lanctot et al. 2009) sobre el árbol de juego abstracto de HUNL. El resultado es un **blueprint** — una tabla de estrategias $\bar{\sigma}(I)$ para cada InfoSet visitado — que converge al equilibrio de Nash conforme aumentan las iteraciones.

---

## 2. Árbol de juego abstracto

El árbol se define implícitamente por el estado:

```
estado = (bkts, boards, street_idx, pot, stacks, contribs, bet_hist, n_raises, to_call, position)
```

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `bkts` | dict | Buckets precomputados `{(player, street) → int}` |
| `boards` | list | `[flop, turn, river]` como listas de cartas |
| `street_idx` | int | Calle actual (0–3) |
| `pot` | float | Bote actual en BBs |
| `stacks` | list[float] | `[stack_SB, stack_BB]` |
| `contribs` | list[float] | Contribuciones al bote esta calle |
| `bet_hist` | list[str] | Historial de acciones abstractas |
| `n_raises` | int | Raises realizados en la calle actual |
| `to_call` | float | BBs necesarios para igualar |
| `position` | int | Jugador que debe actuar (0=SB, 1=BB) |

---

## 3. Inicialización: blinds y reparto

Al inicio de cada iteración:

```python
hand0, hand1, flop, turn, river = _deal()   # baraja aleatoria

# Blinds: SB pone 0.5, BB pone 1.0
# pot=1.5, SB tiene que igualar 0.5 más para llegar a 1.0
init_kwargs = dict(
    pot=1.5, stacks=[99.5, 99.0],
    contribs=[0.5, 1.0],
    bet_hist=[], n_raises=0,
    to_call=0.5, position=0,   # SB actúa primero preflop
)
```

Se realizan **dos traversals**: una desde la perspectiva de P0 y otra de P1.

---

## 4. Precompute de buckets

```python
bkts = MCCFRTrainer._precompute_buckets(hands, boards, sims=bucket_sims)
```

Calcula los 8 buckets necesarios (2 jugadores × 4 calles) **una sola vez** antes de iniciar el traversal. Dentro del árbol, los accesos son `bkts[(player, street)]` → $O(1)$.

Este diseño es la optimización más importante del sistema: sin precompute, cada llamada a `_fast_key` requeriría correr Monte Carlo → miles de veces más lento.

---

## 5. Bucle CFR recursivo (`_cfr`)

```python
def _cfr(self, traverser, bkts, boards, street_idx,
         pot, stacks, contribs, bet_hist, n_raises, to_call, position):
```

### 5.1 Nodo del traverser (exploración completa)

```python
if active == traverser:
    # Explorar TODAS las acciones válidas
    for idx in valid_idxs:
        action_vals[idx] = self._apply_action(ABSTRACT_ACTIONS[idx], ...)

    ev = dot(strategy, action_vals)             # valor esperado
    regrets[key] += (action_vals - ev) * mask   # actualizar regrets
    strategy_sum[key] += strategy               # acumular para promedio
    return ev
```

### 5.2 Nodo del oponente (muestreo)

```python
else:
    idx = np.random.choice(NUM_ACTIONS, p=strategy)   # muestrea UNA acción
    strategy_sum[key] += strategy
    return self._apply_action(ABSTRACT_ACTIONS[idx], ...)
```

---

## 6. Transiciones — `_apply_action`

### 6.1 FOLD

```python
if traverser == opponent:
    return float(pot)          # oponente foldea → traverser gana el bote
else:
    return -float(contribs[active])   # traverser foldea → pierde sus contribuciones
```

### 6.2 CALL

El CALL puede desencadenar:

1. **Limp preflop** (`street=0, active=SB, bet_hist=[]`): el BB recibe opción de check/raise.
2. **Fin de calle** (`nc[0] == nc[1]`): avanzar a la siguiente calle.
3. **All-in** (`ns[active] == 0`): cerrar la acción.
4. **Continúa la calle**: el oponente actúa.

```python
# Caso especial: limp SB preflop (BUG-9 fix)
if street_idx == 0 and active == 0 and not bet_hist:
    return self._cfr(..., bet_hist=[CALL], to_call=0.0, position=opponent)
```

### 6.3 RAISE proporcional (BUG-7 fix)

```python
raise_extra = min(ratio * pot, stacks[active] - to_call)

# Guard: si pot==0, raise_extra==0 → ejecutar CALL para no corromper bet_hist
if raise_extra == 0:
    return self._apply_action(CALL, ...)

total_add = to_call + raise_extra
if total_add >= stacks[active]:
    total_add = stacks[active]; action = ALLIN   # colapsar a all-in si no hay fichas
```

---

## 7. Showdown — `_showdown` (BUG-6 fix)

En river o cuando ambos jugadores están all-in, el resultado se estima con los buckets de EHS²:

```python
_denom = max(1, POSTFLOP_BUCKETS - 1)   # =15, no 16
if   b0 > b1: eq0 = 0.5 + 0.5 * (b0 - b1) / _denom
elif b1 > b0: eq0 = 0.5 - 0.5 * (b1 - b0) / _denom
else:         eq0 = 0.5

ev = eq * pot - my_contrib
```

Con `POSTFLOP_BUCKETS=16`, los buckets van de 0 a 15. La diferencia máxima es 15, no 16. Con el denominador correcto, la equity máxima es exactamente 1.0 cuando el bucket es 15 vs 0.

---

## 8. Transición entre calles — `_next_street`

```python
def _next_street(self, traverser, bkts, boards, next_street_idx, pot, stacks):
    return self._cfr(
        traverser, bkts, boards, next_street_idx, pot,
        stacks, [0.0, 0.0], [], 0, 0.0,
        position=1)   # BB actúa primero en postflop
```

El historial de apuestas se reinicia (`[]`), las contribuciones de la calle anterior se resetean (`[0.0, 0.0]`), y en postflop siempre actúa primero el BB (`position=1`).

---

## 9. Estructuras de datos

```python
self.regret_sum:   dict[tuple, np.ndarray(7)]  # regrets acumulados
self.strategy_sum: dict[tuple, np.ndarray(7)]  # estrategia acumulada
```

Cada entrada ocupa $7 \times 8 = 56$ bytes (float64). Con 1M InfoSets:

- `regret_sum`: ~56 MB
- `strategy_sum`: ~56 MB
- **Total**: ~112 MB por 1M InfoSets, ~1.1 GB por 10M InfoSets

---

## 10. Checkpoints automáticos (MEMORY-1)

```python
trainer.train(
    num_iterations=1_000_000,
    save_every=10_000,    # checkpoint cada 10k iters
)
```

Con `save_every > 0`, el blueprint se serializa automáticamente cada `save_every` iteraciones. **Imprescindible** para entrenamientos de producción (>100k iters): un crash en la iteración 90k sin checkpoint perdería todo el trabajo.

---

## 11. Consulta del blueprint

```python
strat = trainer.get_strategy(key)   # np.ndarray(7), suma 1
```

- Si el InfoSet fue visitado: devuelve `strategy_sum[key] / sum(strategy_sum[key])`.
- Si no fue visitado: devuelve distribución uniforme `[1/7, ..., 1/7]`.

La estrategia **promedio** es la que converge a Nash. No usar `regret_sum` directamente para decisiones de juego.

---

## 12. Serialización

```python
trainer.save()           # guarda en cfr/blueprint.pkl
trainer = MCCFRTrainer.load()   # carga desde disco
```

El archivo pickle contiene el dict de `regret_sum`, `strategy_sum` e `iterations`. El protocolo 4 (`pickle.HIGHEST_PROTOCOL`) usa compresión eficiente para arrays NumPy.

---

## 13. Velocidad esperada

Con `bucket_sims=50` en CPU estándar (sin GPU):

| Escenario | Velocidad aprox. |
|-----------|-----------------|
| Primera iteración (cache fría) | ~2–5 s (llenando caché LRU) |
| Iteraciones estables (cache caliente) | ~50–200 iters/s |
| Con `bucket_sims=100` | ~25–100 iters/s |
| Con `bucket_sims=300` | ~8–30 iters/s |

Estas cifras varían según el hardware. El pre-flight check mide la velocidad real y actúa para estimar el tiempo total.
