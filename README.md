# Agente-IA-Poker

Agente de **Heads-Up No-Limit Texas Hold'em** (HUNL) basado en Teoría de Juegos.  
Combina un blueprint GTO entrenado offline con **MCCFR** (Monte Carlo Counterfactual Regret Minimization), búsqueda de subgames en tiempo real y modelado adaptativo del oponente para producir un bot que es simultáneamente difícil de explotar y capaz de explotar rivales débiles.

---

## Contenidos

1. [Fundamentos matemáticos](#1-fundamentos-matemáticos)  
   1.1 [Juegos de información imperfecta y EFG](#11-juegos-de-información-imperfecta-y-efg)  
   1.2 [Information Sets y estrategias conductuales](#12-information-sets-y-estrategias-conductuales)  
   1.3 [Equilibrio de Nash y exploitabilidad](#13-equilibrio-de-nash-y-exploitabilidad)  
   1.4 [Counterfactual Regret Minimization (CFR)](#14-counterfactual-regret-minimization-cfr)  
   1.5 [External Sampling MCCFR](#15-external-sampling-mccfr)  
   1.6 [Abstracción de cartas: EHS y EHS²](#16-abstracción-de-cartas-ehs-y-ehs²)  
   1.7 [Real-time Subgame Search](#17-real-time-subgame-search)  
   1.8 [Modelado del oponente](#18-modelado-del-oponente)  
2. [Arquitectura del sistema](#2-arquitectura-del-sistema)  
3. [Instalación](#3-instalación)  
4. [Cómo usar](#4-cómo-usar)  
   4.1 [Entrenar el blueprint](#41-entrenar-el-blueprint)  
   4.2 [Jugar partidas con el agente](#42-jugar-partidas-con-el-agente)  
   4.3 [Usar los módulos individualmente](#43-usar-los-módulos-individualmente)  
5. [Referencia de archivos](#5-referencia-de-archivos)  
6. [Tests](#6-tests)  

---

## 1. Fundamentos matemáticos

### 1.1 Juegos de información imperfecta y EFG

El póker se modela como un **Extensive Form Game (EFG)** con información imperfecta:

$$G = (N, H, Z, \mathcal{A}, \mathcal{I}, u, P, f_c)$$

| Símbolo | Significado |
|---|---|
| $N = \{0, 1\}$ | Jugadores (SB y BB) |
| $H$ | Conjunto de historias de juego (nodos del árbol) |
| $Z \subseteq H$ | Nodos terminales (fin de mano) |
| $\mathcal{A}(h)$ | Acciones legales en el nodo $h$ |
| $\mathcal{I}_i$ | Partición de $H$ en Information Sets del jugador $i$ |
| $u_i : Z \to \mathbb{R}$ | Utilidad (fichas ganadas/perdidas) |
| $P(h) \in N \cup \{c\}$ | Jugador activo en $h$ (o chance $c$) |
| $f_c$ | Distribución de probabilidad de chance (baraja) |

El árbol de HUNL tiene aproximadamente $10^{164}$ historias terminales: resolver el juego exacto es computacionalmente imposible. Se usan **abstracciones** para reducirlo a un juego tratable manteniendo las propiedades de convergencia.

### 1.2 Information Sets y estrategias conductuales

Un **Information Set** $I \in \mathcal{I}_i$ agrupa todos los nodos que el jugador $i$ no puede distinguir (ve sus propias cartas, el tablero y el historial de apuestas, pero no las cartas del rival):

$$I = (\underbrace{\text{pos}}_{\text{SB/BB}},\ \underbrace{s}_{\text{calle}},\ \underbrace{b_{\text{hand}}}_{\text{bucket mano}},\ \underbrace{(b_{f}, b_{t}, b_{r})}_{\text{buckets tablero}},\ \underbrace{\vec{a}}_{\text{historial apuestas}})$$

Una **estrategia conductual** $\sigma_i(a \mid I)$ asigna una distribución de probabilidad sobre acciones a cada Information Set:

$$\sigma_i : \mathcal{I}_i \times \mathcal{A} \to [0, 1], \quad \sum_{a \in \mathcal{A}(I)} \sigma_i(a \mid I) = 1$$

### 1.3 Equilibrio de Nash y exploitabilidad

Un perfil de estrategias $(\sigma_0^*, \sigma_1^*)$ es un **equilibrio de Nash** si ningún jugador puede mejorar su valor esperado desviándose unilateralmente:

$$v_i(\sigma_i^*, \sigma_{-i}^*) \geq v_i(\sigma_i, \sigma_{-i}^*) \quad \forall \sigma_i, \forall i$$

La **exploitabilidad** de una estrategia $\sigma_i$ mide cuánto puede ganar el mejor rival posible (best response $BR$) contra ella:

$$\text{expl}(\sigma_i) = v_{-i}(BR(\sigma_i), \sigma_i) - v_{-i}(\sigma_{-i}^*, \sigma_i^*)$$

Se expresa en **milli-big-blinds por mano (mbb/m)**. Un bot GTO tiene exploitabilidad 0.

### 1.4 Counterfactual Regret Minimization (CFR)

CFR es el algoritmo que acumula regret contrafactual para converger al equilibrio de Nash.

**Valor contrafactual** de la acción $a$ en el Information Set $I$ para el jugador $i$:

$$v_i^{CF}(\sigma, I, a) = \sum_{h \in I} \underbrace{\pi_{-i}^{\sigma}(h)}_{\text{prob. de llegar a }h\text{ sin }i} \sum_{z \succ h \cdot a} \pi^{\sigma}(h \cdot a, z)\ u_i(z)$$

**Regret contrafactual instantáneo** en la iteración $t$:

$$r^t(I, a) = v_i^{CF}(\sigma^t, I, a) - v_i^{CF}(\sigma^t, I)$$

**Regret acumulado** tras $T$ iteraciones:

$$R^T(I, a) = \sum_{t=1}^{T} r^t(I, a)$$

**Regret Matching** — produce la estrategia del turno siguiente:

$$\sigma^{T+1}(a \mid I) = \frac{\max\!\left(0,\ R^T(I,a)\right)}{\displaystyle\sum_{a'} \max\!\left(0,\ R^T(I,a')\right)}$$

Si todos los regrets son $\leq 0$, se distribuye uniformemente entre acciones válidas.

**Estrategia promedio** — el blueprint final (converge a Nash):

$$\bar{\sigma}^T(a \mid I) = \frac{\displaystyle\sum_{t=1}^{T} \pi_i^{\sigma^t}(I) \cdot \sigma^t(a \mid I)}{\displaystyle\sum_{t=1}^{T} \pi_i^{\sigma^t}(I)}$$

Teorema de convergencia: en juegos de suma cero de dos jugadores, $\bar{\sigma}^T \to \sigma^*$ cuando $T \to \infty$, y la exploitabilidad decrece como $O(1/\sqrt{T})$.

### 1.5 External Sampling MCCFR

El CFR exacto requiere recorrer el árbol completo en cada iteración. **External Sampling** reduce el coste muestreando stocásticamente:

En cada iteración, para el jugador **traverser** $i$:
- En nodos de $i$: se **exploran todas las acciones** y se actualizan regrets
- En nodos del rival $-i$: se **samplea una acción** según $\sigma_{-i}^t$
- En nodos de chance: se samplea una realización aleatoria

El estimador de valor contrafactual es insesgado y la varianza está acotada. El coste por iteración pasa de $O(|H|)$ a $O(|H|^{1/2})$ en media.

**Eficiencia de implementación:** los buckets (EHS / EHS²) se precomputan **una sola vez** al inicio de cada iteración para ambas manos. Dentro del árbol de búsqueda, cada nodo accede al bucket en $O(1)$ mediante `_fast_key`.

### 1.6 Abstracción de cartas: EHS y EHS²

Con las 52 cartas del mazo, hay $\binom{52}{2} = 1{,}326$ manos posibles y $\binom{52}{5} = 2{,}598{,}960$ tableros posibles. Las abstracciones agrupan situaciones matemáticamente similares.

**Expected Hand Strength (EHS):**

$$EHS = \sum_{\text{opp hand}} P(\text{opp hand}) \cdot \mathbf{1}[\text{mi mano gana el showdown}]$$

Se estima via Monte Carlo muestreando manos del rival y tableros restantes. $EHS \in [0, 1]$.

**Potencial positivo y negativo:**

$$ppot = P(\text{pierdo ahora} \land \text{gano final}) \qquad npot = P(\text{gano ahora} \land \text{pierdo final})$$

**EHS² (Hand Strength al cuadrado con potencial):**

$$EHS^2 = EHS + (1 - EHS) \cdot ppot - EHS \cdot npot$$

$EHS^2$ captura tanto la fuerza actual como el potencial de mejora/empeoramiento con las cartas restantes. Es más informativo que EHS solo para decisiones en flop y turn.

**Clustering en buckets:** los valores $EHS^2$ de todas las manos posibles se ordenan por percentil y se asignan a $k$ buckets equiprobables:

| Fase | Métrica | $k$ buckets |
|---|---|---|
| Preflop | EHS (169 manos canónicas) | 10 |
| Flop / Turn / River | EHS² | 50 |

### 1.7 Real-time Subgame Search

Durante la partida real, el agente no usa el blueprint directamente sino que ejecuta una búsqueda de subgame con horizonte $D$ calles hacia adelante. La incertidumbre sobre las cartas del rival se resuelve muestreando $N$ manos del oponente de la baraja residual:

$$\hat{\sigma}^*_i(a \mid I) = \frac{1}{N} \sum_{j=1}^{N} \sigma^*_i(a \mid I^{(j)})$$

donde $I^{(j)}$ es el InfoSet correspondiente a la $j$-ésima mano sampleada del rival.

El blueprint actúa como **leaf evaluator**: por debajo del horizonte $D$, el valor de un nodo se obtiene consultando $\bar{\sigma}$ del blueprint en lugar de seguir expandiendo el árbol.

Referencia: Brown & Sandholm (2017) *"Safe and Nested Subgame Solving for Imperfect-Information Games"*, NeurIPS.

### 1.8 Modelado del oponente

El agente rastrea las siguientes estadísticas del rival para derivar una **estrategia de contra-explotación**:

| Stat | Fórmula | Interpretación |
|---|---|---|
| **VPIP** | $\frac{\text{manos jugadas voluntariamente}}{\text{manos totales}}$ | Looseness |
| **PFR** | $\frac{\text{raises preflop}}{\text{manos totales}}$ | Agresividad preflop |
| **AF** | $\frac{\text{bets} + \text{raises}}{\text{calls}}$ por calle | Agresividad postflop |
| **FTB** | $\frac{\text{folds ante bet/raise}}{\text{bets/raises recibidos}}$ | Fold equity disponible |
| **WTSD** | $\frac{\text{llega a showdown}}{\text{ve flop}}$ | Stickiness |

**Arquetipos y ajustes:**

| Arquetipo | VPIP | AF | Ajuste IA |
|---|---|---|---|
| Loose-passive (fish) | > 0.50 | < 1.5 | Value bet delgado, sin bluffs |
| Loose-aggressive (LAG) | > 0.50 | > 2.5 | Trap con manos top, call-down light |
| Tight-passive (nit) | < 0.25 | < 1.5 | Robar blinds, fold ante resistencia |
| Tight-aggressive (TAG/GTO) | < 0.35 | > 2.0 | Permanecer cerca del blueprint |

Los ajustes se expresan como multiplicadores sobre la frecuencia de bluff y el umbral de value-bet del blueprint.

---

## 2. Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                        AGENTE EN PARTIDA                        │
│                                                                 │
│  ┌──────────────┐    ┌──────────────────┐   ┌───────────────┐  │
│  │ blueprint.pkl│───▶│  RealtimeSearch  │◀──│ OpponentModel │  │
│  │  (MCCFR GTO) │    │  (subgame CFR)   │   │  (stats HUD)  │  │
│  └──────────────┘    └────────┬─────────┘   └───────────────┘  │
│                               │ acción abstracta                │
│                               ▼                                 │
│                    ┌──────────────────┐                         │
│                    │  PokerCoreEngine │                         │
│                    │  (motor de juego)│                         │
│                    └──────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     PIPELINE DE ENTRENAMIENTO                   │
│                                                                 │
│  pre_entrenamiento.py                                           │
│       │                                                         │
│       ▼                                                         │
│  MCCFRTrainer.train(N iters)                                    │
│       │                                                         │
│       ├── _deal() ──▶ preflop_bucket() / postflop_bucket()     │
│       │                    (EHS / EHS²  via Monte Carlo)        │
│       │                                                         │
│       ├── _precompute_buckets()  → tabla O(1) por iteración     │
│       │                                                         │
│       └── _traverse_external()  → actualiza regret_sum         │
│                                    y strategy_sum               │
│                    │                                            │
│                    ▼                                            │
│             blueprint.pkl  ──▶  get_strategy(key)              │
└─────────────────────────────────────────────────────────────────┘
```

**Flujo de decisión en tiempo real:**

```
Estado actual (cartas, tablero, historial, stacks)
        │
        ▼
blueprint_action_callback()
        │
        ├── ¿Existe blueprint.pkl?
        │       Sí ──▶ RealtimeSearch.get_action()
        │               │
        │               ├── Samplea N manos del rival
        │               ├── Precomputa buckets por muestra
        │               ├── CFR a profundidad D
        │               └── Pondera estrategias → acción final
        │
        └── No ──▶ advanced_ia_action_callback()
                        (heurística equity + pot odds)
```

---

## 3. Instalación

### Requisitos

- Python 3.10+
- numpy
- flask (para la interfaz web)

```bash
pip install -r requirements.txt
```

O individualmente:

```bash
pip install numpy flask
```

### Evaluador de manos (opcional pero recomendado)

Por defecto el evaluador es Python puro (`template.py` con `_py_eval_hand`). Para mayor velocidad se puede compilar la extensión en C:

```bash
cd simulacion
gcc -O2 -shared -fPIC -o libhand_eval.so hand_eval.c
```

Si `libhand_eval.so` existe, se carga automáticamente via `ctypes`. Si no existe, el fallback Python se activa sin error.

---

## 4. Cómo usar

### 4.1 Entrenar el blueprint

El script `pre_entrenamiento.py` entrena el blueprint MCCFR y lo guarda en `cfr/blueprint.pkl`.

```bash
cd simulacion

# Prueba rápida de humo (5 000 iteraciones, ~30 s)
python pre_entrenamiento.py --iters 5000 --log 1000

# Entrenamiento de calidad (100 000 iteraciones, ~10 min)
python pre_entrenamiento.py --iters 100000 --log 10000

# Reanudar entrenamiento desde el blueprint existente
python pre_entrenamiento.py --iters 50000 --resume

# Entrenar y validar la estrategia resultante
python pre_entrenamiento.py --iters 20000 --validate

# Ver todas las opciones
python pre_entrenamiento.py --help
```

**Opciones disponibles:**

| Flag | Por defecto | Descripción |
|---|---|---|
| `--iters N` | 10 000 | Número de iteraciones MCCFR |
| `--log N` | 1 000 | Frecuencia de log de progreso |
| `--out PATH` | `cfr/blueprint.pkl` | Ruta de salida del blueprint |
| `--resume` | False | Reanuda desde el blueprint existente |
| `--validate` | False | Imprime muestra de estrategias al terminar |
| `--checkpoint-every N` | 0 | Guarda checkpoint cada N iters (0 = desactivado) |

Para entrenamientos largos (≥100 000 iters) se recomienda activar los checkpoints
para evitar perder progreso si el proceso se interrumpe:

```bash
# Entrenamiento largo con checkpoint cada 25 000 iters
python pre_entrenamiento.py --iters 200000 --checkpoint-every 25000 --resume
```

Los checkpoints se guardan como `cfr/blueprint_ckptNNNNNNN.pkl`. Para reanudar
desde el último checkpoint basta con copiarlo como `cfr/blueprint.pkl` y usar
`--resume`.

**Ejemplo de salida durante el entrenamiento:**

```
[MCCFR]  1000 iters | InfoSets: 3,241 | tiempo: 4.2s
[MCCFR]  2000 iters | InfoSets: 5,817 | tiempo: 8.1s
...
Blueprint guardado en cfr/blueprint.pkl  (12,445 InfoSets)
```

**Ejemplo de salida de validación** (`--validate`):

```
=== Validación de estrategia (muestra) ===
                                          InfoSet  FOLD  CALL  r1/3  r1/2  r1x   r2x   AI
--- pos=0 preflop ['As','Kd'] board=[]   0.00  0.29  0.33  0.18  0.13  0.05  0.02
--- pos=1 preflop ['7h','2c'] board=[]   0.58  0.33  0.06  0.02  0.01  0.00  0.00
```

### 4.2 Interfaz web (jugar contra el bot en el navegador)

El servidor Flask sirve una interfaz visual completa para jugar contra el agente:

```bash
cd simulacion

# Iniciar el servidor web (puerto 5000 por defecto)
python web/app.py

# Puerto personalizado
python web/app.py --port 8080

# Modo debug (recarga automática al editar código)
python web/app.py --debug
```

Abrir el navegador en **`http://localhost:5000`**.

La interfaz muestra:
- Mesa ovalada con cartas del humano y el bot
- Cartas comunitarias por calle (flop, turn, river)
- Panel de acciones: fold / call / check / raise (con slider) / all-in
- Panel lateral izquierdo: historial de manos de la sesión
- Panel lateral derecho: estadísticas del rival (VPIP, PFR, AF, FTB, arquetipo)
- Badge de blueprint (activo/inactivo)

**Nota sobre el blueprint:**  
Si no hay blueprint entrenado (`cfr/blueprint.pkl`), el bot usa una heurística
de equity. Entrena primero con `pre_entrenamiento.py` para obtener el bot GTO:

```bash
# Mínimo recomendado antes de jugar (≈5 min)
python pre_entrenamiento.py --iters 50000
python web/app.py
```

### 4.3 Jugar partidas por CLI con el agente

```bash
cd simulacion

# 20 manos contra el bot (usa blueprint si existe, equity si no)
python main_poker.py --manos 20 --fichas 100

# Sin blueprint (solo heurística de equity)
python main_poker.py --manos 10 --no-blueprint
```

**Opciones de `main_poker.py`:**

| Flag | Por defecto | Descripción |
|---|---|---|
| `--manos N` | 10 | Número de manos a jugar |
| `--fichas N` | 100 | Stack inicial en big blinds |
| `--no-blueprint` | False | Deshabilita el blueprint, usa équity pura |

**Ejemplo de salida:**

```
=== Mano 1 ===
IA: ['Ah', 'Kd'] | Bot: ?? | Board: ['Jh', 'Ts', '2c', '9d', '3s']
Acción IA: raise_pot | Acción Bot: call
Resultado: IA gana pot=4.5

--- Resultado final (10 manos) ---
IA:  +18.5 BB  (ganó 7/10 manos)
Bot: -18.5 BB

--- Estadísticas del rival ---
VPIP: 0.60 | PFR: 0.20 | AF: 0.8 | FTB: 0.55
Arquetipo: loose-passive
Ajuste aplicado: value-bet delgado, bluff reducido al 5%
```

### 4.4 Usar los módulos individualmente

#### Motor de juego

```python
from poker_engine import PokerCoreEngine

engine = PokerCoreEngine(['AgentIA', 'Bot'])
engine.reset()
estado = engine.play_round()
print('Resultado:', estado)
```

#### Calcular equity de una mano

```python
from montecarlo import montecarlo_equity, get_equity_cached

# Equity de AsKs con tablero Jh-Ts-2c (versus 1 rival)
equity = montecarlo_equity(
    hole_cards=['As', 'Ks'],
    community_cards=['Jh', 'Ts', '2c'],
    num_players=2,
    num_simulations=1000,
)
print(f"Equity: {equity:.1%}")   # → ~55%

# Versión cacheada (más rápida en llamadas repetidas)
equity = get_equity_cached(['As', 'Ks'], ['Jh', 'Ts', '2c'])
```

#### Abstracción de cartas

```python
from abstracciones.card_abstractor import preflop_bucket, postflop_bucket

# Bucket de AKs en preflop (0–9, mayor = mejor)
b = preflop_bucket(['As', 'Ks'], sims=200)
print(b)   # → 9  (bucket élite)

# Bucket de 72o en preflop
b = preflop_bucket(['7h', '2c'], sims=200)
print(b)   # → 0  (bucket basura)

# EHS² en flop con AsKs en tablero Jh-Ts-2c
b = postflop_bucket(['As', 'Ks'], ['Jh', 'Ts', '2c'], sims=200)
print(b)   # → bucket alto (tiene outs a escalera real)
```

#### Codificar un InfoSet

```python
from abstracciones.infoset_encoder import encode_infoset

key = encode_infoset(
    position=0,               # SB
    street_str='flop',
    hand_cards=['As', 'Ks'],
    board_cards=['Jh', 'Ts', '2c'],
    bet_history=['r3', 'c'],  # villain bet pot, hero call
    hand_sims=200,
    board_sims=150,
)
print(key)  # → (0, 1, 9, (42,), ('r3', 'c'))
```

#### Consultar el blueprint entrenado

```python
from cfr.mccfr_trainer import MCCFRTrainer
from abstracciones.infoset_encoder import encode_infoset, ABSTRACT_ACTIONS

trainer = MCCFRTrainer.load()  # carga cfr/blueprint.pkl

key = encode_infoset(0, 'preflop', ['As', 'Ks'], [], [])
strategy = trainer.get_strategy(key)  # np.ndarray shape (7,)

for action, prob in zip(ABSTRACT_ACTIONS, strategy):
    print(f"{action}: {prob:.1%}")
# → f: 0.0%  c: 28.4%  r1: 33.1%  r2: 17.8%  r3: 12.6%  r4: 5.9%  ai: 2.2%
```

#### Real-time Search

```python
from cfr.mccfr_trainer  import MCCFRTrainer
from cfr.realtime_search import RealtimeSearch

blueprint = MCCFRTrainer.load()
engine    = RealtimeSearch(blueprint=blueprint, depth=2, iterations=200)

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
)
print(action)  # → 'r3'  (raise pot)
```

#### Modelado del oponente

```python
from opponent_model import OpponentModel

model = OpponentModel(opponent_id=1)

for _ in range(50):
    model.new_hand()
    model.observe_action('call',  'preflop', voluntarily=True)
    model.observe_action('call',  'flop',    voluntarily=False)
    model.observe_fold_to_bet('turn')
    model.observe_showdown(won=False)

adj = model.get_counter_adjustments()
print(model.classify())           # → 'loose-passive'
print(adj['bluff_freq_mult'])     # → 0.3  (reducir bluffs vs fish que llama todo)
print(adj['value_threshold_adj']) # → -0.08 (value-bet más delgado)
```

---

## 5. Referencia de archivos

```
simulacion/
│
├── poker_engine.py          Motor de juego HUNL
│                            └── PokerCoreEngine, Carta, Baraja, Jugador, Mesa
│                                RondaApuestasPreflop, RondaApuestasPostflop
│
├── montecarlo.py            Equity y callbacks de decisión
│                            └── montecarlo_equity, get_equity_cached
│                                advanced_ia_action_callback   (heurística)
│                                human_bot_action_callback     (rival simulado)
│                                blueprint_action_callback     (IA GTO + explotación)
│
├── mcts_modulo.py           MCTS con UCT (legado, sustituido por CFR)
│                            └── NodoMCTS, run_mcts
│
├── main_poker.py            CLI principal del agente
│                            └── jugar_con_blueprint(), jugar_mcts()
│
├── pre_entrenamiento.py     Script de entrenamiento MCCFR
│                            └── main() con argparse, validar_estrategia()
│
├── template.py              Evaluador de manos
│                            └── eval_hand_from_strings()
│                                (usa libhand_eval.so si existe, Python puro si no)
│
├── opponent_model.py        Modelado del oponente
│                            └── OpponentModel, OpponentStats
│
├── tablas_preflop.py        Rangos preflop heurísticos (legacy)
│
├── generar_dataset.py       Generación de datasets de entrenamiento supervisado
│
├── test_equity.py           Tests de cálculo de equity
├── test_mcts.py             Tests del módulo MCTS
├── test_preflop.py          Tests de rangos preflop
├── test_cfr.py              Tests del entrenador MCCFR (7 tests)
├── test_realtime.py         Tests de la búsqueda en tiempo real (7 tests)
├── test_opponent.py         Tests del modelo del oponente (12 tests)
│
├── abstracciones/
│   ├── card_abstractor.py   EHS, EHS², preflop_bucket, postflop_bucket
│   └── infoset_encoder.py   encode_infoset, definición de acciones abstractas
│
├── cfr/
│   ├── mccfr_trainer.py     External Sampling MCCFR (entrenamiento + blueprint)
│   ├── realtime_search.py   Subgame search en tiempo real
│   ├── train_blueprint.py   Script alternativo de entrenamiento (low-level)
│   └── blueprint.pkl        Blueprint entrenado (generado por pre_entrenamiento.py)
│
└── web/
    ├── app.py               Servidor Flask con API REST y sesiones
    ├── templates/
    │   └── index.html       Interfaz HTML de la mesa de póker
    └── static/
        ├── css/poker.css    Estilos de la mesa (mesa ovalada, cartas, HUD)
        └── js/poker.js      Lógica frontend (fetch API, renderizado de cartas)
```

---

## 6. Tests

```bash
cd simulacion

# Tests de equity (Monte Carlo)
python test_equity.py

# Tests del módulo MCTS
python test_mcts.py

# Tests de rangos preflop
python test_preflop.py

# Tests del entrenador MCCFR (requiere numpy)
python test_cfr.py

# Tests de la búsqueda en tiempo real (requiere numpy)
python test_realtime.py

# Tests del modelo del oponente
python test_opponent.py

# Todos los tests a la vez
python -m pytest test_equity.py test_mcts.py test_preflop.py \
                 test_cfr.py test_realtime.py test_opponent.py -v
```

Todos los tests deben pasar antes de entrenar o jugar. Si alguno falla, verifica
que las dependencias estén instaladas:

```bash
pip install -r requirements.txt
```

---

## Referencias

- Lanctot et al. (2009). *Monte Carlo Sampling for Regret Minimization in Extensive Games*. NeurIPS.  
- Johanson et al. (2013). *Measuring the size of large no-limit poker games*. AAAI.  
- Bowling et al. (2015). *Heads-up limit hold'em poker is solved*. Science.  
- Brown & Sandholm (2017). *Safe and Nested Subgame Solving for Imperfect-Information Games*. NeurIPS.  
- Brown & Sandholm (2019). *Superhuman AI for multiplayer poker (Pluribus)*. Science.
- Equity Monte Carlo con caché LRU aproximado
- MCTS con selección UCT, expansión, simulación y retropropagación
- Logs estructurados de partidas en JSON
- Rangos preflop configurables via JSON
- `reset_to_state()` para simular desde cualquier punto de la partida

## En desarrollo

- 🔄 CFR (Counterfactual Regret Minimization)
- 🔄 Filtrado dinámico de rangos coherente con historial completo
- 🔜 Generación automática de datasets para entrenamiento supervisado

