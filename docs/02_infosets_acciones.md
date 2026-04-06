# 02 — Codificación de Information Sets y Acciones

Archivo fuente: [`simulacion/abstracciones/infoset_encoder.py`](../simulacion/abstracciones/infoset_encoder.py)

---

## 1. ¿Qué es un Information Set?

Un **Information Set (InfoSet)** es el conjunto de todos los nodos del árbol de juego que un jugador no puede distinguir entre sí en un momento dado. En HUNL, el jugador ve:

- Sus propias cartas (2 hole cards)
- Las cartas comunitarias visibles (0–5)
- El historial público de apuestas

Pero **no** ve las cartas del oponente. Todos los estados donde la información visible es idéntica forman un mismo InfoSet.

---

## 2. Clave canónica del InfoSet

La clave es una **tupla hashable** de 4 elementos:

```python
(player, street_idx, hand_bucket, bet_history_tuple)
```

| Campo | Tipo | Rango | Significado |
|-------|------|-------|-------------|
| `player` | int | 0 ó 1 | Posición: 0=SB, 1=BB |
| `street_idx` | int | 0–3 | Calle: 0=preflop, 1=flop, 2=turn, 3=river |
| `hand_bucket` | int | 0–15 | Fortaleza de la mano (EHS²) |
| `bet_history_tuple` | tuple[str] | variable | Secuencia de acciones abstractas de la calle |

### 2.1 Por qué solo 4 elementos

Una clave alternativa podría incluir un bucket por cada calle pasada (`bb_tuple`), añadiendo información sobre la textura histórica del board. Sin embargo esto multiplica el espacio de InfoSets exponencialmente:

- Con `bb_tuple`: $O(B^{\text{calles}} \times |\beta|)$
- Sin `bb_tuple`: $O(B \times |\beta|)$

El `hand_bucket` calculado con EHS² ya incorpora el potencial positivo/negativo relativo al board actual, capturando implícitamente la textura. Esto permite convergencia real con 200k iteraciones en lugar de millones.

### 2.2 Consistencia entre Trainer y Realtime (fix BUG-5)

Es **crítico** que `_fast_key` genere la misma clave en `mccfr_trainer.py` y en `realtime_search.py`. Antes de la corrección, el trainer usaba 4 elementos y el realtime 5, haciendo que toda consulta al blueprint retornara siempre distribución uniforme (miss total). Ambos usan ahora exactamente:

```python
def _fast_key(player, street_idx, bkts, bet_hist):
    hb = bkts[(player, street_idx)]
    return (player, street_idx, hb, tuple(bet_hist))
```

---

## 3. Acciones abstractas

El espacio de acciones continuas de HUNL (apostar cualquier cantidad) se reduce a **7 acciones abstractas**:

| Constante | Código | Ratio pot | Descripción |
|-----------|--------|-----------|-------------|
| `FOLD` | `'f'` | — | Retirarse |
| `CALL` | `'c'` | — | Igualar (incluye check cuando `to_call=0`) |
| `RAISE_THIRD` | `'r1'` | 1/3 × pot | Raise pequeño |
| `RAISE_HALF` | `'r2'` | 1/2 × pot | Raise medio (50% pot) |
| `RAISE_POT` | `'r3'` | 1× pot | Raise estándar |
| `RAISE_2POT` | `'r4'` | 2× pot | Raise grande |
| `ALLIN` | `'ai'` | stack | All-in |

```python
ABSTRACT_ACTIONS = ['f', 'c', 'r1', 'r2', 'r3', 'r4', 'ai']
NUM_ACTIONS = 7
ACTION_IDX  = {'f':0, 'c':1, 'r1':2, 'r2':3, 'r3':4, 'r4':5, 'ai':6}
```

### 3.1 La apuesta del 50% pot (`RAISE_HALF`)

`RAISE_HALF` ya está incorporada como parte del conjunto base de acciones. No es exclusiva del subgame solver — se entrena en el blueprint igual que el resto. Esto es estratégicamente correcto porque el 50% pot es una de las apuestas más comunes en póker moderno (permite mantener rangos amplios con diversas manos).

---

## 4. Conversión acción real → acción abstracta

La función `abstract_action(action, pot)` mapea cualquier acción del motor de juego:

```python
if   ratio <= 0.42:  return RAISE_THIRD   # ≤ 42% pot
elif ratio <= 0.62:  return RAISE_HALF    # 43–62% pot (incluye 50%)
elif ratio <= 1.25:  return RAISE_POT     # 63–125% pot
else:                return RAISE_2POT    # > 125% pot
                                          # ALLIN solo si acción literal 'all in'
```

**Nota:** Un raise de stack completo (ej. `('raise', 95.5)`) se mapea a `RAISE_2POT`, no a `ALLIN`. El mapeo a `ALLIN` solo ocurre cuando la acción es explícitamente `'all in'`. Esta distinción se mantiene porque el árbol CFR trata `ALLIN` como una rama específica con semántica diferente (stack = 0).

---

## 5. Conversión acción abstracta → monto concreto

La función `concrete_raise_amount` materializa un raise abstracto en BBs:

```python
ratio       = RAISE_RATIOS[abstract_act]   # {r1:1/3, r2:1/2, r3:1.0, r4:2.0}
raise_extra = ratio * pot
total       = to_call + raise_extra
return min(total, stack)                   # acotado por stack disponible
```

Para `ALLIN` devuelve directamente `stack`.

---

## 6. Máscara de acciones válidas

No todas las acciones son válidas en todo momento. El método `_mask` genera un vector booleano:

```python
m = np.ones(NUM_ACTIONS, dtype=bool)      # todas válidas por defecto

if to_call == 0.0:
    m[ACTION_IDX[FOLD]] = False           # no hay apuesta que foldar

if n_raises >= raise_max:                 # límite de raises por calle (default 2)
    for a in [RAISE_THIRD, RAISE_HALF, RAISE_POT, RAISE_2POT]:
        m[ACTION_IDX[a]] = False          # no más raises proporcionales

if stack <= to_call:
    for a in [RAISE_THIRD, RAISE_HALF, RAISE_POT, RAISE_2POT]:
        m[ACTION_IDX[a]] = False          # sin fichas para raise
```

**`ALLIN` nunca se bloquea:** incluso al alcanzar `raise_max`, el all-in sigue siendo válido (regla estándar HUNL).

---

## 7. Tamaño del espacio de InfoSets

Con los parámetros actuales:

$$|InfoSets| \approx \underbrace{2}_{\text{posición}} \times \underbrace{4}_{\text{calles}} \times \underbrace{16}_{\text{buckets}} \times \underbrace{|\beta|}_{\text{historiales}}$$

El factor $|\beta|$ (número de historiales distintos) es el componente dominante. Con `raise_max = 2` por calle y 4 calles, el historial máximo tiene ~8 acciones, pero la mayoría de los nodos son más cortos.

En la práctica, después de 200k iteraciones de entrenamiento se observan ~500k–2M InfoSets visitados, acorde con las estimaciones teóricas.

---

## 8. `encode_infoset` vs `_fast_key`

Existen dos funciones para generar la clave:

| Función | Uso | Coste |
|---------|-----|-------|
| `encode_infoset(position, street_str, hand, board, bet_hist)` | API pública, acepta cartas reales | $O(\text{sims})$ si cache miss |
| `_fast_key(player, street_idx, bkts, bet_hist)` | Uso interno en CFR (buckets precomputados) | $O(1)$ |

Durante el entrenamiento siempre se usa `_fast_key`. `encode_infoset` se usa para consultar el blueprint desde código externo (ej. `main_poker.py`).

---

## 9. Historial de apuestas por calle

El `bet_hist` se reinicia en cada calle nueva (en `_next_street` el trainer invoca `_cfr` con `bet_hist=[]`). Solo contiene las acciones abstractas de la calle en curso, no el historial completo de la mano.

Esto es correcto: lo que importa para la estrategia es la textura de la calle actual (quién apostó primero, cuántas veces se re-raisó), no el historial completo de la partida.

**Ejemplo de bet_hist en el flop:**

```
['c']          ← BB checkeó
['c', 'r3']   ← BB checkeó, SB apostó 1× pot
['c', 'r3', 'c']   ← BB llamó
```
