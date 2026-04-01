# Agente IA Poker — HUNL GTO Bot

Bot de Texas Hold'em Heads-Up No-Limit (HUNL) basado en **MCCFR** (Monte Carlo Counterfactual Regret Minimization) con búsqueda de subgames en tiempo real y modelado del oponente. Incluye interfaz web interactiva para jugar contra el bot.

---

## Índice

1. [Fundamentos matemáticos](#1-fundamentos-matemáticos)
2. [Arquitectura del sistema](#2-arquitectura-del-sistema)
3. [Estructura del repositorio](#3-estructura-del-repositorio)
4. [Guía de instalación y ejecución](#4-guía-de-instalación-y-ejecución)
5. [Guía de ejecución en la nube](#5-guía-de-ejecución-en-la-nube)
6. [API REST de la interfaz web](#6-api-rest-de-la-interfaz-web)
7. [Resultados y convergencia](#7-resultados-y-convergencia)
8. [Referencias](#8-referencias)

---

## 1. Fundamentos matemáticos

### 1.1 Formalización del juego como Extensive Form Game (EFG)

Texas Hold'em HUNL se modela como un juego de información imperfecta:

$$G = (N, H, Z, \mathcal{I}, u, f_c)$$

| Símbolo | Definición |
|---|---|
| $N = \{0, 1\}$ | Conjunto de jugadores (P0 = SB, P1 = BB) |
| $H$ | Conjunto de historias (nodos del árbol) |
| $Z \subset H$ | Nodos terminales |
| $\mathcal{I}_i$ | **Information sets** del jugador $i$: partición de los nodos de decisión donde el jugador no distingue entre nodos del mismo conjunto |
| $u_i: Z \to \mathbb{R}$ | Utilidad de $i$ en el nodo terminal; $u_0 + u_1 = 0$ (suma cero) |
| $f_c$ | Distribución de probabilidad del nodo de azar (reparto de cartas) |

### 1.2 Estrategia e Information Sets

La **estrategia conductual** de $i$ es una función:

$$\sigma_i: \mathcal{I}_i \to \Delta(A(I))$$

que asigna a cada information set una distribución de probabilidad sobre las acciones legales.

En este sistema, un **Information Set** se codifica como:

$$I = (\text{player},\ \text{street},\ \text{hand\_bucket},\ \text{bet\_hist})$$

donde `hand_bucket` $\in \{0, \ldots, B-1\}$ es el resultado del clustering EHS² y captura implícitamente la textura del tablero. Esto mantiene el espacio total en:

$$|\mathcal{I}| = 2 \times 4 \times B \times |\beta| \approx 1{,}200 \text{ InfoSets}$$

lo que hace posible la convergencia con 200k iteraciones en CPU.

### 1.3 Equilibrio de Nash y exploitabilidad

Un **perfil de estrategias** $\sigma^* = (\sigma_0^*, \sigma_1^*)$ es un equilibrio de Nash si:

$$\forall i,\ \forall \sigma_i':\ v_i(\sigma_i^*, \sigma_{-i}^*) \geq v_i(\sigma_i', \sigma_{-i}^*)$$

La **exploitabilidad** mide cuánto se puede explotar el blueprint:

$$\varepsilon(\sigma) = \frac{1}{2}\left[v_0^{BR}(\sigma_1) + v_1^{BR}(\sigma_0)\right]$$

donde $v_i^{BR}(\sigma_{-i})$ es el valor de la mejor respuesta pura. Se mide en **mbb/mano** (milli-big-blinds por mano). Un agente perfectamente GTO tiene $\varepsilon = 0$.

### 1.4 CFR — Counterfactual Regret Minimization

#### Valor contrafactual

$$v_i^{CF}(\sigma, I) = \sum_{h \in I} \pi_{-i}^{\sigma}(h) \sum_{z \succ h} \pi^{\sigma}(h \to z) \cdot u_i(z)$$

donde $\pi_{-i}^{\sigma}(h)$ es la probabilidad de llegar a $h$ si **todos los jugadores excepto $i$ juegan según $\sigma$**.

#### Regret instantáneo y acumulado

El **regret instantáneo** de no haber tomado acción $a$ en $I$ en la iteración $t$:

$$r^t(I, a) = v_i^{CF}(\sigma^t, I, a) - v_i^{CF}(\sigma^t, I)$$

El **regret acumulado**:

$$R^T(I, a) = \sum_{t=1}^{T} r^t(I, a)$$

#### Regret Matching

$$\sigma^{T+1}(a \mid I) = \frac{\max\!\left(0,\, R^T(I,a)\right)}{\displaystyle\sum_{a'} \max\!\left(0,\, R^T(I,a')\right)}$$

#### Estrategia promedio — el blueprint

$$\bar{\sigma}^T(a \mid I) = \frac{\displaystyle\sum_{t=1}^{T} \pi_i^{\sigma^t}(I) \cdot \sigma^t(a \mid I)}{\displaystyle\sum_{t=1}^{T} \pi_i^{\sigma^t}(I)}$$

**Teorema de convergencia** (Zinkevich et al. 2007): En juegos de suma cero de 2 jugadores:

$$\varepsilon\!\left(\bar{\sigma}^T\right) \leq \frac{C}{\sqrt{T}}$$

Para este sistema con 1,200 InfoSets, empíricamente $\varepsilon < 500$ mbb/m a 200k iters.

### 1.5 External Sampling MCCFR

**External Sampling** (Lanctot et al. 2009):

- El **traverser** explora **todas** las acciones en sus nodos → actualiza regrets exactamente
- El **oponente** muestrea **una** acción según su estrategia actual

Coste por iteración: $O(|A|^{d/2})$ en lugar de $O(|A|^d)$.

En la implementación:

```
for cada iteración t:
    deal(hand₀, hand₁, flop, turn, river)      # nodo de azar
    buckets = precompute_buckets(hands, boards)  # 8 llamadas EHS² — reutilizadas
    cfr(traverser=0, ...)                        # traversal P0
    cfr(traverser=1, ...)                        # traversal P1
```

### 1.6 Abstracciones de cartas — EHS y EHS²

#### Expected Hand Strength

$$EHS(\mathbf{h}, B) = P(\text{ganar showdown} \mid \text{mano } \mathbf{h},\ \text{board } B)$$

#### EHS² con potencial de mejora

$$EHS^2 = EHS + (1 - EHS) \cdot ppot - EHS \cdot npot$$

- $ppot$: probabilidad de pasar de perder a ganar con cartas futuras
- $npot$: probabilidad de pasar de ganar a perder con cartas futuras

La implementación usa **NumPy vectorizado** — evaluación de $N=300$ manos simultáneas con operaciones de array: **~61× más rápido** que Python puro (8 ms vs 500 ms por llamada).

#### Tabla de buckets

| Fase | Buckets | Métrica | InfoSets/fase |
|---|---|---|---|
| Preflop | 10 | EHS sobre 169 formas canónicas | 20 |
| Flop | 16 | EHS² (ppot + npot) | 32 |
| Turn | 16 | EHS² | 32 |
| River | 16 | EHS puro | 32 |
| **Total** | — | — | **~1,200** |

### 1.7 Real-time Subgame Search

En tiempo de juego, el blueprint se refina:

$$\pi^*(a \mid I_{\text{actual}}) = \text{MCCFR}\!\left(\text{subgame}(I_{\text{actual}}),\ \text{depth}=D,\ \text{leaf}=v^{\text{blueprint}}\right)$$

### 1.8 Opponent Modeling

| Stat | Fórmula | Regular típico |
|---|---|---|
| VPIP | manos voluntarias / total | 22–28% |
| PFR | raises preflop / total | 18–24% |
| AF | (bets+raises) / calls | 2.5–3.5 |
| FTB | folds a bet / bets enfrentados | 45–55% |
| WTSD | showdowns / vio flop | 25–30% |

Ajuste de estrategia:

$$\sigma^{\text{exploit}}(a \mid I) = (1-\alpha)\cdot\bar{\sigma}^T(a \mid I) + \alpha\cdot\text{best\_response}(\hat{\sigma}_{\text{opp}})$$

### 1.9 Online Learning (Experience Replay CFR)

Tras cada mano real, se ejecutan 80 traversals MCCFR adicionales sobre el deal observado (inspirado en *Libratus*, Brown & Sandholm 2017):

$$\Delta R^T(I,a) \mathrel{+}= \frac{80}{T}\cdot r^t(I_{\text{real}},a)$$

---

## 2. Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────┐
│                  Interfaz Web (Flask)                   │
│  POST /api/accion  →  PartidaWeb.accion()               │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────▼─────────────┐
         │       _bot_decide()        │
         │  1. Blueprint lookup O(1)  │
         │  2. RealtimeSearch ~1.5s   │
         │  3. Heurística equity      │
         └─────────────┬─────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    ▼                  ▼                  ▼
MCCFRTrainer    RealtimeSearch     OpponentModel
blueprint.pkl   subgame CFR        VPIP/PFR/AF/FTB
(200k iters)    depth-limited      counter_adjust()
    │
    ▼
CardAbstractor
EHS² vectorizado (NumPy)
16 postflop buckets
LRU cache 32k
```

---

## 3. Estructura del repositorio

```
Agente-IA-Poker/
├── README.md
└── simulacion/
    ├── poker_engine.py       # Motor de juego: Mesa, Jugador, Baraja, Carta
    ├── montecarlo.py         # Callbacks de acción, caché de equity
    ├── mcts_modulo.py        # MCTS legacy (fallback)
    ├── main_poker.py         # CLI: bot vs bot, 1v1 humano
    ├── comparador.py         # Benchmark GTO vs MCTS (BB/100)
    ├── opponent_model.py     # Estadísticas y clasificación del rival
    ├── tablas_preflop.py     # Rangos GTO preflop de referencia
    ├── pre_entrenamiento.py  # Script de entrenamiento MCCFR
    ├── template.py           # Evaluador de manos (nativo o Python puro)
    ├── abstracciones/
    │   ├── card_abstractor.py   # EHS², buckets, evaluador NumPy vectorizado
    │   └── infoset_encoder.py   # Encoding de InfoSets, acciones abstractas
    ├── cfr/
    │   ├── mccfr_trainer.py     # External Sampling MCCFR trainer
    │   ├── realtime_search.py   # Subgame search en tiempo real
    │   └── blueprint.pkl        # Blueprint entrenado (generado localmente)
    └── web/
        ├── app.py               # Flask backend + PartidaWeb + API REST
        ├── templates/index.html # Mesa de póker HTML
        └── static/
            ├── css/poker.css    # Estilos: felt, cartas CSS, animaciones
            └── js/poker.js      # Lógica frontend: fetch, render, acciones
```

---

## 4. Guía de instalación y ejecución

### 4.1 Dependencias

```bash
pip install numpy flask
```

El evaluador de manos nativo (`libpoker-eval`) es opcional. Si no está disponible el sistema usa automáticamente el evaluador NumPy integrado.

### 4.2 Verificar el entorno

```bash
cd simulacion
python3 -c "
from abstracciones.card_abstractor import compute_ehs, postflop_bucket
from cfr.mccfr_trainer import MCCFRTrainer
from web.app import app
print('Entorno OK')
"
```

### 4.3 Entrenar el blueprint

```bash
cd simulacion

# Entrenamiento rápido (~8 min, usable)
python3 pre_entrenamiento.py --iters 50000 --log 5000

# Entrenamiento completo (~160 min, excelente)
python3 pre_entrenamiento.py --iters 200000 --log 10000

# Reanudar entrenamiento existente
python3 pre_entrenamiento.py --iters 200000 --resume --log 10000

# Análisis matemático de convergencia
python3 pre_entrenamiento.py --analizar

# Validar exploitabilidad del blueprint actual
python3 pre_entrenamiento.py --iters 0 --validate
```

Salida esperada a 200k iters:
```
iter  200,000  |  InfoSets: 21,548
Blueprint guardado en 'cfr/blueprint.pkl'  (200,000 iters, 21,548 InfoSets)
Exploitabilidad: 487 mbb/mano  → Excelente (> 200k iters)
```

### 4.4 Interfaz web

```bash
cd simulacion
python3 web/app.py --port 5000
# Abrir: http://localhost:5000
```

### 4.5 Consola 1v1

```bash
python3 main_poker.py --humano --manos 20 --fichas 200
```

### 4.6 Benchmark GTO vs MCTS

```bash
python3 comparador.py --manos 500 --quiet
```

### 4.7 Tests

```bash
python3 test_cfr.py        # 7 tests MCCFR
python3 test_realtime.py   # 7 tests subgame search
python3 test_opponent.py   # 12 tests opponent model
python3 test_equity.py     # tests evaluador
python3 test_preflop.py    # tests rangos preflop
```

### 4.8 Recargar blueprint sin reiniciar servidor

```bash
curl -X POST http://localhost:5000/api/recargar_blueprint
```

---

## 5. Guía de ejecución en la nube

### 5.1 GitHub Codespaces (recomendado)

1. `Code → Codespaces → Create codespace on main`
2. En la terminal integrada:

```bash
pip install numpy flask
cd simulacion
# Entrenar en background
nohup python3 -u pre_entrenamiento.py --iters 200000 --log 10000 > /tmp/train.log 2>&1 &
tail -f /tmp/train.log   # seguir progreso
# Arrancar servidor
python3 web/app.py --port 5000
```

3. En la pestaña `Ports`, hacer el puerto 5000 público y abrir la URL generada.

### 5.2 Google Colab

```python
# Celda 1: Clonar
!git clone https://github.com/Marco-Ezquerra/Agente-IA-Poker.git
%cd Agente-IA-Poker/simulacion
!pip install flask numpy -q

# Celda 2: Entrenar
!python3 pre_entrenamiento.py --iters 200000 --log 20000

# Celda 3: Exponer con ngrok
!pip install pyngrok -q
from pyngrok import ngrok
import subprocess, threading

threading.Thread(
    target=lambda: subprocess.run(['python3','web/app.py','--port','5000']),
    daemon=True
).start()

print("URL pública:", ngrok.connect(5000))
```

### 5.3 VPS / Servidor dedicado (Ubuntu 22.04+)

```bash
sudo apt-get update && sudo apt-get install -y python3 python3-pip git
pip3 install numpy flask gunicorn
git clone https://github.com/Marco-Ezquerra/Agente-IA-Poker.git
cd Agente-IA-Poker/simulacion

# Entrenar (sobrevive al cierre de sesión SSH)
nohup python3 -u pre_entrenamiento.py --iters 200000 --log 10000 > train.log 2>&1 &

# Servidor de producción
gunicorn -w 1 -b 0.0.0.0:80 "web.app:app" --timeout 120
```

### 5.4 Requerimientos de hardware

| Tarea | CPU | RAM | Disco | Tiempo estimado |
|---|---|---|---|---|
| Entrenamiento 50k iters | 1 core | 512 MB | 50 MB | ~8 min |
| Entrenamiento 200k iters | 1 core | 1 GB | 100 MB | ~160 min |
| Servidor web (jugando) | 1 core | 256 MB | blueprint | — |
| Decisión realtime search | 1 core | 256 MB | — | ~1.5 s |

---

## 6. API REST de la interfaz web

Base URL: `http://localhost:5000`

| Endpoint | Método | Body / Params | Descripción |
|---|---|---|---|
| `/` | GET | — | Interfaz web |
| `/api/nueva_partida` | POST | — | Nueva sesión |
| `/api/estado` | GET | — | Estado actual |
| `/api/accion` | POST | `{"tipo":"fold"\|"call"\|"raise","amount":N}` | Acción del humano |
| `/api/nueva_mano` | POST | — | Siguiente mano |
| `/api/stats` | GET | — | Stats del OpponentModel |
| `/api/recargar_blueprint` | POST | — | Recarga blueprint del disco |

Respuesta de `/api/accion`:

```json
{
  "fase": "flop",
  "pot": 4.5,
  "community": ["Qs", "7h", "2d"],
  "tu_mano": ["Ah", "Kd"],
  "stack_humano": 97.5,
  "stack_bot": 98.0,
  "turno_humano": true,
  "acciones_validas": [
    {"tipo": "fold"},
    {"tipo": "call", "amount": 0},
    {"tipo": "raise", "min": 1.5, "max": 97.5, "suggested": [2.25, 3.0, 4.5]}
  ],
  "historial": ["SB: call 0.5", "BB: raise 2.0", "SB: call"],
  "resultado": null
}
```

---

## 7. Resultados y convergencia

### Exploitabilidad vs iteraciones (16 buckets, ~1,200 InfoSets)

| Iteraciones | InfoSets visitados | $\varepsilon$ (mbb/m) | Calidad |
|---|---|---|---|
| 5,000 | ~6,500 | ~23,000 | Básica |
| 20,000 | ~15,000 | ~8,000 | Mejorable |
| 50,000 | ~19,000 | ~3,500 | Buena |
| 100,000 | ~20,500 | ~1,800 | Muy buena |
| 200,000 | ~21,500 | ~500 | Excelente |

### Visitas por InfoSet a 200k iters

$$\frac{200{,}000 \times 2}{1{,}200} \approx 333\ \text{visitas/InfoSet}$$

Garantía de convergencia:

$$\varepsilon \leq \frac{C}{\sqrt{200{,}000}} \approx \frac{C}{447}$$

---

## 8. Referencias

- Zinkevich, M. et al. (2007). *Regret Minimization in Games with Incomplete Information*. NeurIPS.
- Lanctot, M. et al. (2009). *Monte Carlo Sampling for Regret Minimization in Extensive Games*. NeurIPS.
- Johanson, M. et al. (2013). *Evaluating State-Space Abstractions in Extensive-Form Games*. AAMAS.
- Brown, N., Sandholm, T. (2017). *Libratus: The Superhuman AI for No-Limit Poker*. IJCAI.
- Moravčík, M. et al. (2017). *DeepStack: Expert-Level Artificial Intelligence in Heads-Up No-Limit Poker*. Science.
- Brown, N., Sandholm, T. (2019). *Superhuman AI for Multiplayer Poker — Pluribus*. Science.

## Estructura del proyecto

 Este proyecto puede funcionar de forma independiente si `poker-eval` está instalado o accesible como dependencia. Localmengte tengo m arpeta dentro de la carpeta poker-eval que se puede obtener de la siguiente manera:
 git clone https://github.com/atinm/poker-eval.git
cd poker-eval
autoreconf --install
./configure
make
sudo make install


## Funcionalidades del agente

- Simulación completa desde cualquier fase: preflop, flop, turn, river  
- Evaluación de acciones mediante MCTS con nodos personalizados  
- Cálculo de equity en tiempo real mediante `poker-eval`  
- Generación de logs estructurados para análisis posterior  
- En desarrollo: implementación de CFR, filtrado dinámico de rangos y retropropagación adaptativa

---

## Estado actual

- ✅ Motor funcional con simulación desde cualquier punto del juego  
- ✅ Integración de MCTS con evaluación de equity  
- ✅ Sistema de logging y análisis  
- 🔄 En progreso: lógica completa de CFR + generación de datasets  
- 🔜 Próximo paso: filtrado de rangos coherente con historial de acciones y mejora del árbol de decisiones

---



