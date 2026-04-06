# Documentación — Agente IA Póker HUNL

Documentación técnica completa del agente de Texas Hold'em Heads-Up No-Limit (HUNL) basado en Monte Carlo Counterfactual Regret Minimization (MCCFR) con Subgame Solving en tiempo real.

---

## Índice

| Documento | Contenido |
|-----------|-----------|
| [00 — Fundamentos Matemáticos](00_fundamentos_matematicos.md) | Teoría de juegos, Nash, CFR, Regret Matching — paso a paso desde cero |
| [01 — Abstracción de Cartas](01_abstraccion_cartas.md) | EHS, EHS², buckets, evaluación vectorizada, caché LRU |
| [02 — Codificación de InfoSets](02_infosets_acciones.md) | Claves de InfoSet, acciones abstractas, espacio de estados |
| [03 — Trainer MCCFR](03_mccfr_trainer.md) | External Sampling, recorrido del árbol, actualización de regrets |
| [04 — Subgame Solving en Tiempo Real](04_realtime_subgame.md) | RealtimeSearch, leaf evaluator, muestreo del oponente |
| [05 — Modelo del Oponente](05_modelo_oponente.md) | VPIP, PFR, AF, CBet, FTB, arquetipos, contra-explotación |
| [06 — Guía de Entrenamiento](06_guia_entrenamiento.md) | Comandos, fases, checkpoints, parámetros, interpretación |
| [07 — Arquitectura Global](07_arquitectura_global.md) | Diagrama de componentes, flujo de datos, dependencias |

---

## Pila tecnológica

```
Python 3.10+
NumPy           — evaluación vectorizada de manos y EHS Monte Carlo
pickle          — serialización del blueprint
psutil          — monitoreo de RAM durante el pre-flight check
```

## Inicio rápido

```bash
cd simulacion

# 1. Pre-flight check (valida entorno, ~30 s)
python pre_flight_check.py

# 2. Entrenamiento del blueprint (50 000 iteraciones, ~5 min)
python cfr/train_blueprint.py --iters 50000

# 3. Partida de demostración
python main_poker.py
```
