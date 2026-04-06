# 07 — Arquitectura Global

---

## 1. Estructura de carpetas

```
Agente-IA-Poker/
├── docs/                          ← Documentación técnica (esta carpeta)
│   ├── README.md
│   ├── 00_fundamentos_matematicos.md
│   ├── 01_abstraccion_cartas.md
│   ├── 02_infosets_acciones.md
│   ├── 03_mccfr_trainer.md
│   ├── 04_realtime_subgame.md
│   ├── 05_modelo_oponente.md
│   ├── 06_guia_entrenamiento.md
│   └── 07_arquitectura_global.md
│
└── simulacion/
    ├── abstracciones/
    │   ├── __init__.py
    │   ├── card_abstractor.py     ← EHS, EHS², buckets, evaluación vectorizada
    │   └── infoset_encoder.py     ← Claves InfoSet, acciones abstractas
    │
    ├── cfr/
    │   ├── __init__.py
    │   ├── mccfr_trainer.py       ← External Sampling MCCFR, blueprint
    │   ├── realtime_search.py     ← Subgame solver en tiempo real
    │   ├── train_blueprint.py     ← Script CLI de entrenamiento
    │   └── blueprint.pkl          ← (generado) Estrategia entrenada
    │
    ├── poker_engine.py            ← Motor de juego (Baraja, Mesa, Rondas)
    ├── main_poker.py              ← Partida de demostración
    ├── montecarlo.py              ← Estimación de equity (legacy)
    ├── opponent_model.py          ← Modelo y contra-explotación del oponente
    ├── pre_flight_check.py        ← Validación pre-entrenamiento
    ├── tablas_preflop.py          ├── Heurísticas preflop (legacy)
    ├── preflop_ranges.json        │
    ├── template.py                └── Evaluador de manos (legacy)
    │
    ├── test_equity.py             ← Tests de EHS
    ├── test_preflop.py            ← Tests de buckets preflop
    ├── test_opponent.py           ← Tests del modelo de oponente
    ├── test_mcts.py               ← Tests MCTS (legacy)
    └── test_realtime.py           ← Tests de integración realtime
```

---

## 2. Diagrama de componentes

```
┌─────────────────────────────────────────────────────────────────┐
│                        ENTRENAMIENTO                            │
│                                                                 │
│  _deal()                                                        │
│     │                                                           │
│     ▼                                                           │
│  _precompute_buckets()  ←── card_abstractor.py                 │
│  (EHS² × 8 buckets/iter)    (compute_ehs2, lru_cache)         │
│     │                                                           │
│     ▼                                                           │
│  MCCFRTrainer._cfr()    ←── infoset_encoder.py                 │
│  (External Sampling)        (ABSTRACT_ACTIONS, _fast_key)     │
│     │                                                           │
│     ▼                                                           │
│  regret_sum + strategy_sum ──► blueprint.pkl                   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               │ MCCFRTrainer.load()
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      JUEGO EN TIEMPO REAL                       │
│                                                                 │
│  poker_engine.py (Mesa, RondaApuestas)                         │
│     │                                                           │
│     ▼ estado actual (mano, board, pot, historial)               │
│  RealtimeSearch.get_action()                                    │
│     │                                                           │
│     ├── _precompute_buckets() × opp_samples                    │
│     ├── _search() [CFR local, tablas propias]                  │
│     └── _leaf_value() [consulta blueprint como boundary]       │
│     │                                                           │
│     ▼                                                           │
│  acción abstracta ('f','c','r1','r2','r3','r4','ai')           │
│     │                                                           │
│     ▼                                                           │
│  concrete_raise_amount() → monto en BBs                        │
│     │                                                           │
│     ▼                                                           │
│  OpponentModel.observe_action() → ajustes de explotación       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Flujo de datos por iteración de entrenamiento

```
Iteración i:
  1. _deal()
     └─ hand0=[Ah,Kd]  hand1=[Js,Tc]
        flop=[8h,9s,2c]  turn=[Qd]  river=[3s]

  2. _precompute_buckets()
     └─ bkts = {
          (0,0): preflop_bucket([Ah,Kd])      = 9  ← bucket alto
          (0,1): postflop_bucket([Ah,Kd], flop) = 7
          (0,2): postflop_bucket([Ah,Kd], turn) = 8
          (0,3): postflop_bucket([Ah,Kd], river)= 9
          (1,0): preflop_bucket([Js,Tc])      = 6
          (1,1): postflop_bucket([Js,Tc], flop) = 12 ← T9 double pair!
          (1,2): postflop_bucket([Js,Tc], turn) = 13
          (1,3): postflop_bucket([Js,Tc], river)= 13
        }

  3. _cfr(traverser=0, ...)
     Nodo raíz: key=(0, 0, 9, ())   ← SB preflop, bucket 9, sin historial
     strategy = regret_match(regrets[key], mask)
     
     → Explora todas las acciones de SB
     → Para cada acción de SB, muestrea 1 acción de BB
     → Actualiza regrets[(0, 0, 9, ())]
     → Actualiza strategy_sum[(0, 0, 9, ())]

  4. _cfr(traverser=1, ...)
     Simétrico para BB
```

---

## 4. Dependencias entre módulos

```
card_abstractor.py
    └── numpy

infoset_encoder.py
    └── card_abstractor.py (preflop_bucket, postflop_bucket, BUCKETS)

mccfr_trainer.py
    ├── card_abstractor.py (preflop_bucket, postflop_bucket, POSTFLOP_BUCKETS)
    └── infoset_encoder.py (ABSTRACT_ACTIONS, ACTION_IDX, RAISE_RATIOS, ...)

realtime_search.py
    ├── card_abstractor.py (preflop_bucket, postflop_bucket, POSTFLOP_BUCKETS)
    └── infoset_encoder.py (ABSTRACT_ACTIONS, ACTION_IDX, RAISE_RATIOS, ...)
    [consulta mccfr_trainer.MCCFRTrainer en tiempo de ejecución, no en import]

opponent_model.py
    └── (pura Python, sin dependencias externas)

pre_flight_check.py
    ├── card_abstractor.py
    ├── infoset_encoder.py
    ├── mccfr_trainer.py
    ├── realtime_search.py
    └── psutil
```

---

## 5. Gestión del estado del juego

El motor de juego (`poker_engine.py`) mantiene el estado completo: fichas, bote, cartas comunitarias, historial. El agente MCCFR es stateless respecto al juego — recibe el estado como parámetros de entrada y devuelve una acción:

```
Motor de juego ──estado──► Agente MCCFR ──acción──► Motor de juego
                                │
                          [sin memoria del estado previo;
                           el blueprint es la "memoria" estratégica]
```

El `OpponentModel` es el único componente con estado persistente entre manos.

---

## 6. Invariantes del sistema

Estas propiedades deben mantenerse en todo momento:

1. **Consistencia de claves:** `_fast_key` en `mccfr_trainer` y `realtime_search` producen claves idénticas para el mismo estado.

2. **Coherencia del game tree:** `_apply_action` en el trainer y `_step` en el realtime implementan exactamente la misma lógica de transición (incluyendo el bloque de limp preflop).

3. **Caché unificada:** todos los callers de `postflop_bucket` comparten la misma entrada de caché LRU (sin `num_sims` en la clave).

4. **Checkpoints periódicos:** `save_every > 0` durante entrenamientos largos.

5. **`RAISE_HALF` en el conjunto base:** la apuesta 50% pot se entrena con el blueprint, no solo en el subgame solver.

---

## 7. Puntos de extensión futuros

| Objetivo | Dónde cambiar |
|----------|--------------|
| Añadir nueva acción abstracta (ej. 75% pot) | `infoset_encoder.py`: ampliar `ABSTRACT_ACTIONS`, `RAISE_RATIOS` y umbral en `abstract_action()` |
| Aumentar `POSTFLOP_BUCKETS` | `card_abstractor.py`: cambiar `POSTFLOP_BUCKETS`; invalidar blueprint anterior |
| Warm-start del subgame con blueprint | `realtime_search.py`: copiar `blueprint.regret_sum` en `self._regret` antes del CFR |
| Multi-threading del entrenamiento | `mccfr_trainer.py`: convertir `regret_sum/strategy_sum` a estructuras thread-safe; usar `threading.Lock` |
| GPU acceleration | Reescribir `compute_ehs2` con `cupy` (drop-in replacement de NumPy) |
| K-Means++ para buckets adaptativos | `card_abstractor.py`: reemplazar la discretización lineal por clustering con centros aprendidos |
