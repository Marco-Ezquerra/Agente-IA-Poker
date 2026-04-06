# 00 — Fundamentos Matemáticos

Este documento desarrolla, paso a paso, toda la teoría de juegos necesaria para entender el agente: desde la definición formal de un juego hasta el algoritmo MCCFR que produce la estrategia blueprint.

---

## 1. Juegos de suma cero con información imperfecta

### 1.1 Juego en forma extensiva

Un juego en forma extensiva se define con la tupla:

$$\Gamma = (N,\, H,\, Z,\, A,\, u,\, \sigma_c,\, \mathcal{I})$$

| Símbolo | Significado |
|---------|-------------|
| $N = \{0, 1\}$ | Jugadores (P0 = SB, P1 = BB) |
| $H$ | Conjunto de historias (nodos del árbol de juego) |
| $Z \subset H$ | Historias terminales (showdown o fold) |
| $A(h)$ | Acciones disponibles en el nodo $h$ |
| $u_i : Z \to \mathbb{R}$ | Utilidad del jugador $i$ en cada nodo terminal |
| $\sigma_c$ | Distribución del nodo de azar (reparto de cartas) |
| $\mathcal{I}$ | Partición en **Information Sets** |

Como el juego es de **suma cero**: $u_0(z) + u_1(z) = 0$ para todo $z \in Z$.

### 1.2 Information Set

Un Information Set $I \in \mathcal{I}_i$ agrupa todos los nodos $h$ que el jugador $i$ **no puede distinguir** entre sí dado su conocimiento (sus cartas + historial público). Es el elemento central de la resolución de HUNL.

En nuestro sistema, la clave de un InfoSet es:

```
(position, street_idx, hand_bucket, bet_history_tuple)
```

*El `hand_bucket` resume las cartas privadas del jugador mediante EHS², comprimiendo la información continua de $\binom{52}{2}$ manos posibles en 10 o 16 clases discretas.*

---

## 2. Estrategias

### 2.1 Estrategia de comportamiento

Una **estrategia de comportamiento** $\sigma_i$ asigna a cada InfoSet una distribución de probabilidad sobre las acciones disponibles:

$$\sigma_i(I) : A(I) \to [0,1], \quad \sum_{a \in A(I)} \sigma_i(I)(a) = 1$$

### 2.2 Perfil de estrategias y probabilidades de alcance

Dado el perfil $\sigma = (\sigma_0, \sigma_1)$, la probabilidad de alcanzar el nodo $h$ es:

$$\pi^\sigma(h) = \pi^\sigma_c(h) \cdot \prod_{i \in N} \pi^\sigma_i(h)$$

donde $\pi^\sigma_i(h)$ es el producto de las probabilidades de las acciones del jugador $i$ en el camino hacia $h$.

### 2.3 Valor esperado

El valor esperado del jugador $i$ bajo el perfil $\sigma$ es:

$$v_i(\sigma) = \sum_{z \in Z} \pi^\sigma(z) \cdot u_i(z)$$

---

## 3. Equilibrio de Nash

### 3.1 Definición

Un perfil $\sigma^* = (\sigma^*_0, \sigma^*_1)$ es un **Equilibrio de Nash (NE)** si ningún jugador puede mejorar su valor esperado desviándose unilateralmente:

$$v_i(\sigma^*_i, \sigma^*_{-i}) \geq v_i(\sigma_i, \sigma^*_{-i}) \quad \forall \sigma_i,\, \forall i \in N$$

### 3.2 Explotabilidad

La **explotabilidad** de un perfil $\sigma$ mide cuánto puede perder el peor caso:

$$\varepsilon(\sigma) = \sum_{i \in N} \max_{\sigma_i'} v_i(\sigma_i', \sigma_{-i}) - v_i(\sigma)$$

En un NE exacto, $\varepsilon(\sigma^*) = 0$. En sistemas prácticos, el objetivo es $\varepsilon < 1$ BB/100 manos (definición de "resolución práctica").

---

## 4. Arrepentimiento (Regret)

### 4.1 Arrepentimiento instantáneo

En la iteración $t$, el arrepentimiento instantáneo del jugador $i$ por no haber jugado la acción $a$ en el InfoSet $I$ es:

$$r^t_i(I, a) = v_i(\sigma^t[I \to a],\, \sigma^t_{-i}) - v_i(\sigma^t)$$

donde $\sigma^t[I \to a]$ denota la estrategia modificada que juega $a$ con probabilidad 1 en $I$.

### 4.2 Arrepentimiento acumulado

$$R^T_i(I, a) = \sum_{t=1}^{T} r^t_i(I, a)$$

### 4.3 Regret externo

El **regret externo** del jugador $i$ mide cuánto podría haber ganado con la mejor acción fija en retrospectiva:

$$R^T_{i,\text{ext}} = \frac{1}{T} \max_{a^* \in A(I)} R^T_i(I, a^*)$$

El teorema de Folk garantiza que si ambos jugadores minimizan su regret externo, el promedio de sus estrategias converge a un NE en juegos de suma cero.

---

## 5. Regret Matching

### 5.1 Algoritmo

Dado los regrets acumulados $R^T(I, \cdot)$, la estrategia para la siguiente iteración se calcula mediante **Regret Matching**:

$$\sigma^{T+1}(I, a) = \frac{\max(0,\, R^T(I, a))}{\sum_{a' \in A(I)} \max(0,\, R^T(I, a'))}$$

Si el denominador es 0 (todos los regrets son ≤ 0), se distribuye uniformemente entre las acciones válidas.

### 5.2 Implementación en el código

```python
# card_abstractor.py — _regret_match
def _regret_match(regrets, mask):
    pos   = np.maximum(0.0, regrets) * mask   # max(0, R) por acción válida
    total = pos.sum()
    if total > 0.0:
        return pos / total                     # normalización
    valid = mask.astype(np.float64)
    return valid / valid.sum()                 # uniforme si todos R ≤ 0
```

### 5.3 Convergencia

La **estrategia promedio** $\bar{\sigma}^T(I, a) = \frac{1}{T} \sum_{t=1}^{T} \sigma^t(I, a)$ converge al NE, no la estrategia corriente. Esta es la que se guarda en `strategy_sum` y se usa como blueprint.

---

## 6. CFR — Counterfactual Regret Minimization

### 6.1 Valor contrafactual

El valor contrafactual del jugador $i$ en el InfoSet $I$ dado el perfil $\sigma$ es:

$$v_i^\sigma(I) = \sum_{h \in I,\, z \in Z} \pi^\sigma_{-i}(h) \cdot \pi^\sigma(h, z) \cdot u_i(z)$$

El factor $\pi^\sigma_{-i}(h)$ es la probabilidad de alcanzar $h$ **sin contar las acciones del jugador $i$** (solo azar + oponente). Esto permite aislar la contribución del jugador a la historia.

### 6.2 Arrepentimiento contrafactual

$$r^\sigma_i(I, a) = v_i^\sigma(I \to a) - v_i^\sigma(I)$$

donde $v_i^\sigma(I \to a)$ es el valor contrafactual si el jugador jugara $a$ con certeza en $I$.

### 6.3 Teorema de convergencia

**Teorema (Zinkevich et al. 2007):** Si cada jugador usa Regret Matching con arrepentimientos contrafactuales, entonces:

$$\varepsilon(\bar{\sigma}^T) \leq \frac{C}{\sqrt{T}}$$

donde $C$ es una constante que depende del tamaño del árbol. La explotabilidad decrece como $O(1/\sqrt{T})$.

---

## 7. Monte Carlo CFR (MCCFR) — External Sampling

El árbol de HUNL tiene $\sim 10^{160}$ nodos. CFR exacto es intratable. MCCFR usa muestreo para estimar los valores contrafactuales.

### 7.1 External Sampling

En **External Sampling MCCFR** (Lanctot et al. 2009):

- El **traverser** (jugador cuya perspectiva se optimiza): explora **todas** sus acciones en cada nodo.
- El **oponente**: muestrea **una** acción según su estrategia actual.
- Los nodos de azar: se reduce el árbol al deal específico de la iteración.

### 7.2 Estimador insesgado

El estimador del valor contrafactual en External Sampling es insesgado:

$$\mathbb{E}[\tilde{v}_i(I)] = v_i^\sigma(I)$$

La varianza es acotada porque el oponente sigue su distribución propia, no una distribución uniforme.

### 7.3 Actualización de regrets

Para el traverser $i$ en el InfoSet $I$:

```
Para cada acción válida a:
    action_vals[a] = _apply_action(a)   ← valor recursivo

ev = dot(strategy, action_vals)         ← valor esperado de la estrategia actual

regrets[I] += (action_vals - ev) * mask ← arrepentimiento por cada acción
strategy_sum[I] += strategy             ← acumular para el promedio
```

Esta es exactamente la implementación en `MCCFRTrainer._cfr()`.

---

## 8. Subgame Solving

### 8.1 Motivación

El blueprint se entrena sobre un árbol **abstracto** (10 buckets × 16 buckets × 7 acciones). En producción, se conoce la mano exacta del agente. El **Subgame Solving** (Brown & Sandholm 2017) re-optimiza la estrategia para el subgame actual usando el blueprint como condición de boundary.

### 8.2 Safe Subgame Solving

La garantía de safety requiere que el valor del agente en el subgame sea al menos tan bueno como el valor blueprint. En `RealtimeSearch`, esto se implementa usando `_leaf_value` como proxy del valor terminal cuando se supera el horizonte.

### 8.3 Fórmula de leaf value

$$\text{leaf\_value}(\text{traverser}, \text{bkts}, \text{street}, \text{pot}) = (2 \cdot \text{eq\_raw} - 1) \cdot \text{pot} \cdot 0.4$$

donde $\text{eq\_raw} = \text{bucket}_{\text{traverser}} / \text{POSTFLOP\_BUCKETS}$.

---

## 9. Resumen del flujo matemático

```
Cartas reales (52-card deck)
        │
        ▼ EHS² Monte Carlo (Johanson 2013)
Buckets de abstracción [0..15]
        │
        ▼ Composición en clave (player, street, bucket, bet_hist)
Information Set key (hashable)
        │
        ▼ Regret Matching
σ(I) = distribución sobre {f, c, r1, r2, r3, r4, ai}
        │
        ▼ External Sampling MCCFR (Lanctot 2009)
regret_sum + strategy_sum
        │
        ▼ Promedio de estrategias
Blueprint (strategy_sum / Σ) → Nash ε-equilibrium
        │
        ▼ (producción) Subgame Solving (Brown 2017)
Acción óptima dado el estado actual de la mano
```

---

## Referencias

- Zinkevich et al. (2007) *Regret Minimization in Games with Incomplete Information*
- Lanctot et al. (2009) *Monte Carlo Sampling for Regret Minimization in Extensive Games*, NeurIPS
- Johanson et al. (2013) *Measuring the Size of Large No-Limit Poker Games*
- Brown & Sandholm (2017) *Safe and Nested Subgame Solving for Imperfect-Information Games*, NeurIPS
- Bowling et al. (2015) *Heads-up Limit Hold'em Poker is Solved*, Science
