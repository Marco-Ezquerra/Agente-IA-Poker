# 05 — Modelo del Oponente

Archivo fuente: [`simulacion/opponent_model.py`](../simulacion/opponent_model.py)

---

## 1. Propósito

El **modelo del oponente** observa el comportamiento histórico del rival y ajusta la estrategia del agente para **contra-explotar** desviaciones del juego óptimo. Un bot GTO puro ignora las debilidades del oponente; este módulo permite explotarlas.

---

## 2. Estadísticas rastreadas

### 2.1 VPIP — Voluntarily Put $ In Pot

$$\text{VPIP} = \frac{\text{veces que entró al bote voluntariamente}}{\text{manos totales}}$$

Se incrementa cuando el oponente hace `call`, `raise` o `all in` preflop (no cuenta el BB que ya está forzado). Mide el **rango de entrada** del oponente.

- Valor típico GTO en HUNL: ~40–55%
- > 60%: loose (range amplio, muchas manos débiles)
- < 25%: tight (range estrecho, pocas manos)

### 2.2 PFR — Pre-Flop Raise

$$\text{PFR} = \frac{\text{raises preflop}}{\text{manos totales}}$$

Mide la **agresividad** preflop. En HU, PFR/VPIP ≈ 0.7–0.9 indica que el jugador rara vez llama (prefiere raise o fold).

### 2.3 AF — Aggression Factor

$$\text{AF} = \frac{\text{apuestas} + \text{raises}}{\text{calls}}$$

Calculado por calle o en total. AF > 2.5 = agresivo; AF < 0.8 = pasivo.

### 2.4 CBet — Continuation Bet

$$\text{CBet\%} = \frac{\text{veces que apostó en flop tras abrir preflop}}{\text{oportunidades de CBet}}$$

**Definición precisa (fix BUG-12):** Una oportunidad de CBet ocurre **exactamente una vez por mano** en la **primera acción del preflop-aggressor en el flop**. El código anterior contaba cualquier acción en el flop (incluyendo check y fold), inflando el denominador artificialmente.

```python
# Solo contar UNA oportunidad por mano, en la primera acción del opener
if s == 'flop' and self._opened_preflop and not self._cbet_opp_counted:
    self._cbet_opp_counted = True
    self.stats.cbet_opp += 1
    if a in ('raise', 'all in'):
        self.stats.cbet_count += 1
```

### 2.5 FTB — Fold To Bet

$$\text{FTB} = \frac{\text{folds ante apuesta del agente}}{\text{veces que el oponente recibió apuesta}}$$

El denominador se registra explícitamente con `observe_bet_faced()`. El numerador se registra con `observe_action('fold', street)`. Si no se llama `observe_bet_faced` antes del fold, el cálculo es incorrecto.

### 2.6 WTSD / WSD

| Estadística | Fórmula | Significado |
|-------------|---------|-------------|
| WTSD% | showdowns / manos | ¿Con qué frecuencia llega al showdown? |
| WSD% | victorias showdown / showdowns | ¿Qué % de showdowns gana? |

WSD > 55% indica manos fuertes al llegar al showdown (tight value). WSD < 45% indica llegar con manos marginales o bluffs.

---

## 3. Arquetipos de jugador

Con `_MIN_HANDS = 12` manos observadas, el modelo clasifica al oponente:

| Arquetipo | VPIP | AF | Estrategia de contra |
|-----------|------|----|---------------------|
| `loose-passive` | > 42% | < 0.8 | Value bet delgado, pocas bluffs, bet sizing grande |
| `loose-aggressive` | > 42% | > 2.5 | Trap con manos fuertes, llamar amplio, sizing pequeño |
| `tight-passive` | < 20% | < 0.8 | Bluffear más, fold ante sus apuestas |
| `tight-aggressive` | < 20% | > 2.5 | Cerca del GTO, desviarse poco del blueprint |
| `unknown` | — | — | < 12 manos observadas; usar blueprint puro |

---

## 4. Ajustes de contra-explotación

`get_counter_adjustments()` combina el arquetipo con el FTB global:

```python
adj = {
    'bluff_freq_mult':     1.0,   # × frecuencia de bluff del blueprint
    'value_threshold_adj': 0.0,   # + umbral de equity para value bet
    'call_threshold_adj':  0.0,   # + umbral de equity para call
    'bet_size_mult':       1.0,   # × tamaño de apuesta
    'archetype':           '...'
}
```

### 4.1 Ajustes por arquetipo

| Arquetipo | `bluff_freq_mult` | `value_threshold_adj` | `bet_size_mult` |
|-----------|------------------|----------------------|-----------------|
| loose-passive | 0.20 (−80%) | −0.08 (apostar más amplio) | 1.35 (+35%) |
| loose-aggressive | 0.35 (−65%) | +0.00 | 0.80 (−20%) |
| tight-passive | 1.80 (+80%) | +0.07 (solo manos fuertes) | 1.00 |
| tight-aggressive | 0.90 (−10%) | +0.00 | 0.92 |

### 4.2 Micro-ajuste por FTB

```python
if ftb > 0.55:   # foldea mucho
    bluff_freq_mult *= 1.60   # bluffear más
elif ftb < 0.20:  # nunca foldea
    bluff_freq_mult *= 0.40   # eliminar bluffs
```

---

## 5. API de observación

```python
model = OpponentModel(opponent_id=1)

# Al inicio de cada mano
model.new_hand()

# Cuando el oponente actúa (en cada calle)
model.observe_action('raise', 'preflop', voluntarily=True)
model.observe_action('call',  'flop')

# Cuando el agente apuesta (para FTB denominator)
model.observe_bet_faced('flop')

# Si el oponente foldea a esa apuesta
model.observe_action('fold', 'flop')

# Si llega a showdown
model.observe_showdown(won=True)

# Consultar ajustes
adj = model.get_counter_adjustments()
```

### Orden correcto para FTB

```
1. agent apuesta/raise              → model.observe_bet_faced('flop')
2a. opponent foldea                 → model.observe_action('fold', 'flop')
2b. opponent llama/raise            → model.observe_action('call', 'flop')
```

---

## 6. Persistencia del modelo

```python
# Guardar entre sesiones
model.save('models/opp_1.pkl')

# Restaurar
model = OpponentModel.load('models/opp_1.pkl')

# Resumen legible
print(model.summary())
# Oponente id=1 | Arquetipo: loose-passive
#   Manos:   45  |  VPIP: 62%  PFR: 31%
#   AF(total): 0.65  |  FTB(total): 22%
#   CBet: 74%  |  WTSD: 38%  WSD: 48%
```

---

## 7. Consideraciones estadísticas

Con pocos datos (`hands_seen < 30`), las estadísticas tienen alta varianza. Guía práctica:

| Estadística | Mínimo recomendado para fiabilidad |
|-------------|-----------------------------------|
| Arquetipo (VPIP+AF) | 20–30 manos |
| CBet% | 15 oportunidades de flop |
| FTB por calle | 10 apuestas enfrentadas por calle |
| WTSD/WSD | 15 showdowns |

Por debajo de `_MIN_HANDS = 12`, el modelo retorna `archetype='unknown'` y no aplica ajustes. Con entre 12 y 30 manos, los ajustes son indicativos pero deben interpretarse con cautela.
