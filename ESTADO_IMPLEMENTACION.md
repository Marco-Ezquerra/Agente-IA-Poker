# Estado de Implementacion (MCCFR Blueprint)

## Objetivo de esta fase
Cerrar los bloqueos criticos antes del entrenamiento largo del blueprint:
1. Reducir factor de ramificacion durante entrenamiento offline.
2. Evitar ejecucion lenta sin cache EMD precomputado.
3. Validar hand-off blueprint -> subgame solver y estado operativo actual.

## Cambios implementados

### 1) Aislamiento de acciones en entrenamiento offline
- Archivo: simulacion/cfr/mccfr_trainer.py
- Implementacion:
  - Se introdujo modo de entrenamiento controlado por variable de entorno `TRAINING_MODE`.
  - En modo entrenamiento, la mascara de acciones restringe el arbol a 4 acciones:
    - `f` (FOLD)
    - `c` (CALL/check)
    - `r3` (1x pot)
    - `ai` (ALL-IN)
  - En modo normal (produccion/realtime), se mantiene el set completo de 7 acciones.
- Detalle clave:
  - `check` se conserva via `CALL` cuando `to_call == 0`.

### 2) Preparacion obligatoria de cache EMD
- Archivo nuevo: simulacion/setup_training_cache.py
- Implementacion:
  - Script dedicado para generar `equity_clusters.pkl` llamando:
    - `build_equity_clusters(n_samples=100000, bins=20)`

### 3) Bloqueo de seguridad antes de entrenar
- Archivos: simulacion/cfr/mccfr_trainer.py, simulacion/cfr/train_blueprint.py
- Implementacion:
  - Antes de entrenar en modo offline, se valida existencia de:
    - `simulacion/abstracciones/equity_clusters.pkl`
  - Si no existe, se lanza error y el entrenamiento se detiene.
  - El script de entrenamiento activa por defecto `TRAINING_MODE=1` para entorno offline.

### 4) Script de pre-flight y reporte automatico
- Archivo nuevo: simulacion/pre_flight_check.py
- Salida generada: simulacion/STATUS.md
- Implementacion:
  - Dry run de entrenamiento en bloques.
  - Monitoreo de RAM, iters/s, nodos y proxy de convergencia.
  - Prueba de hand-off al turn y top-5 manos por reach probability.
  - Generacion de reporte final en Markdown.

## Ajustes estructurales previos ya aplicados
- Buckets:
  - Preflop exacto: 169 manos canonicas.
  - Postflop: 1000 buckets.
  - Histograma EMD: 20 bins.
- Acciones abstractas estandar (modo completo):
  - `f, c, r1, r2, r3, r4, ai`
  - Sizings: 1/3, 2/3, 1x, 1.5x, all-in.

## Validaciones ejecutadas
- Compilacion/sintaxis de archivos modificados: OK.
- Tests ejecutados en fases anteriores:
  - `simulacion/test_preflop.py`: OK.
  - `simulacion/test_cfr.py`: OK.
  - `simulacion/test_realtime.py`: OK.
- Pre-flight:
  - `simulacion/STATUS.md` generado.
  - Guardrail de check: OK.
  - Hand-off al turn: Exito en smoke test.

## Donde estamos ahora
Estado actual: listo para entrenamiento largo, condicionado a ejecutar primero la preparacion de cache EMD.

Secuencia operativa recomendada:
1. Generar cache:
   - `python simulacion/setup_training_cache.py`
2. Ejecutar pre-flight completo (opcional recomendado):
   - `python simulacion/pre_flight_check.py --iters 10000 --chunk 1000 --bucket-sims 30 --status STATUS.md`
3. Lanzar entrenamiento blueprint offline:
   - `TRAINING_MODE=1 python simulacion/cfr/train_blueprint.py --iters <N>`

## Riesgos y notas
- La extrapolacion lineal de memoria en `STATUS.md` es una aproximacion conservadora, no una garantia exacta.
- Si se cambia la configuracion de buckets/bins, hay que regenerar `equity_clusters.pkl`.
- Blueprints legacy sin schema compatible deben reentrenarse.

## Archivos clave de esta fase
- simulacion/cfr/mccfr_trainer.py
- simulacion/cfr/train_blueprint.py
- simulacion/setup_training_cache.py
- simulacion/pre_flight_check.py
- simulacion/STATUS.md
