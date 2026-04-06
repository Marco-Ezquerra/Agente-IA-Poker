# 06 — Guía de Entrenamiento

Esta guía cubre desde la instalación hasta el entrenamiento de producción, con los parámetros recomendados para cada fase.

---

## 1. Requisitos previos

```bash
# Python 3.10+
python --version

# Dependencias
pip install numpy psutil

# Verificar estructura del proyecto
ls simulacion/
# → abstracciones/ cfr/ pre_flight_check.py main_poker.py ...
```

---

## 2. Pre-flight check

**Siempre ejecutar antes del entrenamiento definitivo.** Valida el entorno, mide la velocidad real y genera un informe en `simulacion/STATUS.md`:

```bash
cd simulacion
python pre_flight_check.py
```

El pre-flight check realiza:

1. **Dry run de 10k iteraciones** divididas en bloques de 1k, midiendo RAM y velocidad
2. **Proxy de convergencia**: media de regrets absolutos (debe decrecer en los primeros bloques)
3. **Hand-off test**: verifica que `RealtimeSearch` recibe un rango lógico pasado al turn
4. **Action guardrails**: comprueba que check (`CALL` con `to_call=0`) está habilitado cuando corresponde

Ejemplo de salida esperada:

```
[BLOCK 1k]  iters/s=143  RAM=87 MB  nodes=28,432  regret_mean=1.23
[BLOCK 2k]  iters/s=151  RAM=89 MB  nodes=56,104  regret_mean=1.11
...
[HAND-OFF TEST] street=turn  bucket_agente=11  bucket_oponente=6  ✓
[ACTION GUARDRAILS] check habilitado: ✓  fold sin apuesta: ✓
Estimación entrenamiento 200k iters: ~22 min
```

---

## 3. Estrategia de entrenamiento por fases

Se recomienda un enfoque progresivo en 3 fases, reanudando desde el checkpoint anterior:

### Fase 1 — Calibración rápida (50k iteraciones)

**Objetivo:** verificar que el entrenamiento converge y detectar problemas.

```bash
cd simulacion
python cfr/train_blueprint.py --iters 50000 --log-every 5000
```

**Tiempo esperado:** 5–20 minutos (depende del hardware y `bucket_sims`)

**Señales de buen funcionamiento:**
- Número de InfoSets crece entre 500k y 3M al llegar a 50k iters
- No hay errores ni explosiones de RAM
- La estrategia para AA preflop muestra probabilidades de raise > 0.5

```
  iter    5,000  |  InfoSets:  210,432
  iter   10,000  |  InfoSets:  380,541
  ...
  iter   50,000  |  InfoSets:  822,017
Blueprint guardado en 'cfr/blueprint.pkl'  (50,000 iters, ...)
```

### Fase 2 — Blueprint útil (200k iteraciones)

**Objetivo:** obtener un blueprint con estrategias razonables para uso en juego.

```bash
python cfr/train_blueprint.py --iters 200000 --resume
```

El flag `--resume` carga el blueprint existente y continúa desde la iteración 50k.

**Tiempo esperado:** 20–90 minutos

**Verificación:** Consultar la estrategia para algunas manos conocidas:

```python
from cfr.mccfr_trainer import MCCFRTrainer
from abstracciones.infoset_encoder import encode_infoset, ABSTRACT_ACTIONS

trainer = MCCFRTrainer.load()
key = encode_infoset(0, 'preflop', ['Ah', 'As'], [], [])
strat = trainer.get_strategy(key)

for a, p in zip(ABSTRACT_ACTIONS, strat):
    print(f"  {a}: {p:.3f}")
# f: 0.000  c: 0.021  r1: 0.089  r2: 0.134  r3: 0.312  r4: 0.188  ai: 0.256
# → AA preflop: raise frecuente, nunca fold ✓
```

### Fase 3 — Producción (1M+ iteraciones)

**Objetivo:** blueprint de calidad para competición.

```bash
python cfr/train_blueprint.py --iters 1000000 --resume --log-every 50000
```

**Tiempo estimado:** 1–6 horas según hardware.

**Checkpoints automáticos:** con `save_every=10000` (argumento de `train()`), se guarda automáticamente cada 10k iteraciones. Un crash en la iteración 900k solo pierde las últimas 10k iteraciones.

---

## 4. Parámetros de entrenamiento

| Parámetro | Descripción | Valor rápido | Valor producción |
|-----------|-------------|-------------|-----------------|
| `num_iterations` | Iteraciones totales | 50,000 | 1,000,000+ |
| `bucket_sims` | Simulaciones Monte Carlo por bucket | 30–50 | 100–200 |
| `log_every` | Frecuencia de log | 5,000 | 50,000 |
| `save_every` | Checkpoint automático | 5,000 | 10,000 |

### ¿Cómo afecta `bucket_sims` a la calidad?

- `bucket_sims=50`: $\sigma_{\text{EHS}^2} \approx 0.07$ ≈ anchura de 1 bucket. Hasta 50% de borderlines pueden estar en el bucket adyacente. Aceptable para entrenamiento: MCCFR es robusto al ruido.
- `bucket_sims=100`: $\sigma \approx 0.05 < $ mitad de bucket. >95% de asignaciones correctas.
- `bucket_sims=200`: calidad óptima. Usar si el tiempo lo permite.

---

## 5. Reanudar entrenamiento fine-tuning

```python
from cfr.mccfr_trainer import MCCFRTrainer

# Cargar blueprint existente
trainer = MCCFRTrainer.load('cfr/blueprint.pkl')
print(f"Reanudando desde iteración {trainer.iterations:,}")

# Continuar con mejor calidad de buckets
trainer.train(
    num_iterations=500_000,
    bucket_sims=100,       # mejor calidad para la fase final
    log_every=25_000,
    save_every=10_000,
)
trainer.save()
```

---

## 6. Selección del blueprint en `get_action`

En tiempo de juego, el agente usa `RealtimeSearch` con el blueprint preentrenado:

```python
from cfr.mccfr_trainer import MCCFRTrainer
from cfr.realtime_search import RealtimeSearch

blueprint = MCCFRTrainer.load()                        # cargar blueprint
engine    = RealtimeSearch(blueprint=blueprint,
                           depth=1,
                           iterations=200)

action = engine.get_action(
    traverser=0,                         # 0=SB, 1=BB
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
    bucket_sims=50,
)
# → 'r3' (raise 1× pot), 'c' (call), 'ai' (all-in), etc.
```

---

## 7. Interpretación de la estrategia

El blueprint devuelve una distribución de probabilidad sobre las 7 acciones:

```python
strat = trainer.get_strategy(key)
# array([0.00, 0.05, 0.08, 0.12, 0.28, 0.20, 0.27])
#         f     c    r1    r2    r3    r4    ai
```

- Una estrategia concentrada (ej. `r3=0.95`) indica una situación clara.
- Una estrategia mezclada (ej. `c=0.40, r3=0.40, ai=0.20`) indica mixing óptimo (el oponente no puede explotar ninguna tendencia).
- Si la estrategia es perfectamente uniforme `[0.143 × 7]`, el InfoSet no fue visitado durante el entrenamiento.

---

## 8. Monitoreo del entrenamiento

### Proxy de convergencia

El **regret medio absoluto** debería decrecer monotónicamente:

```python
import numpy as np
mean_regret = np.mean([np.mean(np.abs(v)) for v in trainer.regret_sum.values()])
```

Si el regret crece después de miles de iteraciones, puede indicar un problema con los buckets o el árbol de juego.

### Tamaño del blueprint en disco

```bash
ls -lh simulacion/cfr/blueprint.pkl
```

| Iteraciones | InfoSets aprox. | Tamaño aprox. |
|-------------|----------------|--------------|
| 50k | 800k–1.2M | 90–140 MB |
| 200k | 1.5M–3M | 170–340 MB |
| 1M | 5M–10M | 560 MB–1.1 GB |

### RAM durante entrenamiento

Con `bucket_sims=50` y 200k iters:

```
RAM inicial:      ~50 MB  (Python + NumPy)
RAM peak (50k):  ~200 MB
RAM peak (200k): ~400 MB
RAM peak (1M):   ~800 MB – 1.5 GB
```

Si la RAM es limitada, usar `save_every` más frecuente (ej. 5k) y reiniciar el proceso en bloques.

---

## 9. Solución a problemas comunes

| Síntoma | Causa probable | Solución |
|---------|---------------|----------|
| Estrategia siempre uniforme en juego | Claves incompatibles trainer/realtime | Verificar que `_fast_key` genera 4 elementos en ambos archivos |
| Error `KeyError` en limp preflop | BUG-9 en realtime | Aplicar fix del bloque limp en `_step` |
| RAM > 4 GB en iteración 300k | InfoSets ilimitados | Reducir `bucket_sims` a 30; usar `save_every=5000` |
| `psutil` no encontrado | Falta dependencia | `pip install psutil` |
| Velocidad < 10 iters/s | `bucket_sims` alto + cache fría | Esperar que la caché LRU se caliente (~500 iters) |
| Blueprint `.pkl` no carga | Versión de pickle incompatible | Borrar y reentrenar; no usar Python < 3.8 |
