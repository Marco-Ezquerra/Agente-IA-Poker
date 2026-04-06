# 01 — Abstracción de Cartas

Archivo fuente: [`simulacion/abstracciones/card_abstractor.py`](../simulacion/abstracciones/card_abstractor.py)

El espacio de HUNL tiene $\binom{52}{2} \cdot \binom{50}{3} \cdot 47 \cdot 46 \approx 10^{14}$ estados de información posibles. Para hacer el entrenamiento tractable, se reduce mediante **abstracción de cartas**: cada situación se mapea a uno de un número pequeño de **buckets** según la fortaleza esperada de la mano.

---

## 1. Representación interna de cartas

Cada carta se codifica como un entero en $[0, 51]$:

$$\text{card\_int} = \text{rank\_idx} \times 4 + \text{suit\_idx}$$

donde `rank_idx` ∈ [0..12] (2→0, A→12) y `suit_idx` ∈ [0..3] (s/h/d/c).

```python
_CARD2INT = {r+s: RANKS.index(r)*4 + SUITS.index(s)
             for r in RANKS for s in SUITS}
```

A partir del entero: `rank = card // 4`,  `suit = card % 4`.

Esta representación permite operaciones vectorizadas con NumPy sobre batches de $N$ manos simultáneamente.

---

## 2. Evaluación de manos — `_eval5_batch`

Evalúa $N$ manos de 5 cartas en paralelo. Retorna un entero de 64 bits donde **mayor = mejor mano**.

### 2.1 Esquema de puntuación

$$\text{score} = \text{cat} \times 13^5 + \text{kicker}$$

| `cat` | Categoría |
|-------|-----------|
| 8 | Royal/Straight flush |
| 7 | Póker (four of a kind) |
| 6 | Full house |
| 5 | Color (flush) |
| 4 | Escalera (straight) |
| 3 | Trío |
| 2 | Doble pareja |
| 1 | Pareja |
| 0 | Carta alta |

### 2.2 Detección de escalera

```python
is_straight = (
    (r_sorted[:, 4] - r_sorted[:, 0] == 4) &
    (np.diff(r_sorted, axis=1) == 1).all(axis=1)
)
wheel    = np.array([0, 1, 2, 3, 12], dtype=np.int8)  # A2345
is_wheel = (r_sorted == wheel).all(axis=1)
is_straight |= is_wheel
```

### 2.3 Cálculo del kicker (fix aplicado)

El kicker se calcula en base 13:

$$\text{kicker} = \sum_{i=0}^{4} r_i \times 13^i$$

**Corrección wheel (BUG-3):** Para A2345, el As tiene rango 12. Sin corrección, el kicker del wheel sería $12 \times 13^4 = 342{,}732$, haciendo que A2345 supere numéricamente a 23456. La corrección reemplaza el rango del As por $-1$ cuando se detecta wheel:

```python
r_for_kicker = r_sorted.astype(np.int64)
r_for_kicker[is_wheel, 4] = -1   # As actúa como carta baja
```

Resultado verificado: score(A2345) = 1,463,553 < score(23456) = 1,606,358 ✓

---

## 3. EHS — Expected Hand Strength

### 3.1 Definición

$$\text{EHS}(\text{hand}, \text{board}) = P(\text{ganar showdown})$$

más exactamente, incluyendo empates:

$$\text{EHS} = \frac{\text{victorias} + 0.5 \times \text{empates}}{N_{\text{sims}}}$$

### 3.2 Algoritmo Monte Carlo vectorizado

```
Para cada simulación i de 1..N:
    1. Muestrear aleatoriamente 2 cartas para el oponente + cartas para completar el board
    2. Evaluar la mejor mano de 5 del agente sobre el board completo
    3. Evaluar la mejor mano de 5 del oponente sobre el board completo
    4. Registrar victoria / empate / derrota
```

La implementación vectoriza los $N$ muestreos en operaciones NumPy sobre matrices de shape $(N, 7)$, siendo ~20× más rápida que un loop Python.

### 3.3 Función `_best_hand_batch`

Para $N$ jugadores con $n=7$ cartas (2 hole + 5 board), evalúa todas las combinaciones $\binom{7}{5} = 21$ y devuelve el máximo:

```python
combos = np.array(list(combinations(range(n), 5)))  # (21, 5)
hands5 = all_cards[:, combos]                        # (N, 21, 5)
scores = _eval5_batch(hands5.reshape(N*21, 5))       # (N*21,)
return scores.reshape(N, 21).max(axis=1)             # (N,)
```

---

## 4. EHS² — Expected Hand Strength Squared

### 4.1 Fórmula (Johanson et al. 2013)

$$\text{EHS}^2 = \text{EHS}_{\text{cur}} + (1 - \text{EHS}_{\text{cur}}) \cdot \text{ppot} - \text{EHS}_{\text{cur}} \cdot \text{npot}$$

| Término | Significado |
|---------|-------------|
| $\text{EHS}_{\text{cur}}$ | Probabilidad de ganar con las cartas **actuales** (board parcial) |
| $\text{ppot}$ | Potencial positivo: P(pasar de perder/empatar → ganar con cartas futuras) |
| $\text{npot}$ | Potencial negativo: P(pasar de ganar/empatar → perder con cartas futuras) |

**¿Por qué EHS² y no solo EHS?**

EHS mide la fortaleza estática actual. EHS² captura que una mano como el `6d7d` con un board `8h9s2h` tiene EHS bajo pero potencial altísimo (doble gutshot + flush draw). Usar solo EHS subestima estas manos y sobreestima manos fuertes sin mejora posible.

### 4.2 Cálculo de ppot y npot

```python
# Para cada simulación i:
# Estado actual: comparar con board parcial
cur_ahead  = my_cur  > opp_cur   # ganando actualmente
cur_behind = my_cur  < opp_cur   # perdiendo actualmente
cur_tie    = my_cur == opp_cur   # empate actualmente

# Estado futuro: comparar con board completo
fut_win    = my_fut  > opp_fut
fut_lose   = my_fut  < opp_fut

# ppot = P(fut_win | cur_behind OR cur_tie)
ppot_mask  = cur_behind | cur_tie
ppot = (ppot_mask & fut_win).sum() / ppot_mask.sum()

# npot = P(fut_lose | cur_ahead OR cur_tie)
npot_mask  = cur_ahead | cur_tie
npot = (npot_mask & fut_lose).sum() / npot_mask.sum()
```

**Importante:** `EHS_cur` se calcula comparando `my_cur > opp_cur` (con board parcial), **no** usando los resultados finales. Este fue el BUG-2 corregido: el código original usaba EHS futuro donde la fórmula requiere EHS actual.

### 4.3 En river (board completo)

Cuando `len(board) >= 5`, no hay cartas futuras → `ppot = npot = 0` → EHS² = EHS. La función delega directamente en `compute_ehs`.

---

## 5. Buckets preflop — `preflop_bucket`

### 5.1 Isomorfismo de palos

Las 169 formas canónicas preflop se reducen ignorando el palo específico, solo importa si las dos cartas son del mismo palo (**suited**) o no:

```python
canon = (rank_high, rank_low, suited: bool)
# Ejemplos:
# ['Ah','Ks'] → ('A','K',False)  ← AKo
# ['Ah','Kh'] → ('A','K',True)   ← AKs
# ['2h','2d'] → ('2','2',False)  ← 22 (pocket pair)
```

### 5.2 Asignación de bucket

El EHS preflop varía en $[\approx 0.32, \approx 0.86]$. Se normaliza y discretiza en `PREFLOP_BUCKETS = 10` clases:

$$\text{bucket} = \left\lfloor \frac{\text{EHS} - 0.32}{0.86 - 0.32} \times 10 \right\rfloor$$

| Bucket | Rango EHS aprox. | Ejemplo de manos |
|--------|-----------------|-----------------|
| 0 | 0.32 – 0.37 | 72o, 83o |
| 3 | 0.48 – 0.54 | J7o, T6s |
| 6 | 0.63 – 0.69 | AJo, KTs |
| 9 | 0.80 – 0.86 | AA, KK, AKs |

---

## 6. Buckets postflop — `postflop_bucket`

### 6.1 Parámetros

```
POSTFLOP_BUCKETS = 16     # clases 0..15
```

### 6.2 Asignación

$$\text{bucket} = \min(15, \lfloor \text{EHS}^2 \times 16 \rfloor)$$

### 6.3 Caché LRU — optimización crítica

La función `_postflop_bucket_cached` usa una caché LRU de 32.768 entradas:

```python
@lru_cache(maxsize=32768)
def _postflop_bucket_cached(hand_t: tuple, board_t: tuple) -> int:
    ehs2 = compute_ehs2(list(hand_t), list(board_t), _POSTFLOP_CACHE_SIMS)
    return max(0, min(15, int(ehs2 * 16)))
```

**Por qué es critical path:** En cada iteración de entrenamiento se reparte una mano y múltiples ramas del árbol CFR piden el bucket para el mismo par `(mano, board)`. Sin caché, cada llamada tomaría ~5ms (300 sims × evaluaciones). Con caché, toma ~50 ns (hit rate > 90%).

**Diseño de la clave (fix CACHE-1 aplicado):** el parámetro `num_sims` fue eliminado de la clave. Anteriormente, distintos callers con `sims=50/60/150/200` generaban 4 entradas distintas para la misma `(mano, board)`, multiplicando el uso de caché por 4 innecesariamente. Ahora `num_sims = 200` está fijo internamente (`_POSTFLOP_CACHE_SIMS`).

---

## 7. Precompute de buckets por iteración

En `MCCFRTrainer._precompute_buckets`, se calculan **8 buckets por iteración** (2 jugadores × 4 calles) antes de iniciar el traversal del árbol:

```python
for p in [0, 1]:
    bkts[(p, 0)] = preflop_bucket(hand)
    bkts[(p, 1)] = postflop_bucket(hand, flop_board)
    bkts[(p, 2)] = postflop_bucket(hand, turn_board)
    bkts[(p, 3)] = postflop_bucket(hand, river_board)
```

Todos los accesos dentro del árbol son `bkts[(player, street)]` → $O(1)$. Esto es lo que permite que cada iteración tarde milisegundos.

---

## 8. Relación entre EHS, EHS² y la calidad del entrenamiento

Con `POSTFLOP_BUCKETS = 16`, la anchura de cada bucket es `1/16 = 0.0625`. Con `num_sims = 200`, el error estándar de EHS² es $\sigma \approx \frac{0.5}{\sqrt{200}} \approx 0.035$, menor que media anchura de bucket → asignación correcta en más del 95% de los casos.

Con `bucket_sims = 50` (durante entrenamiento rápido), $\sigma \approx 0.07 \approx$ anchura de bucket → ~50% de casos borderline podrían estar en el bucket adyacente. Esto es aceptable porque MCCFR es estocásticamente robusto, pero explica por qué el blueprint definitivo debe entrenarse con `bucket_sims >= 100`.
