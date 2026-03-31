"""
Modelado del oponente en Heads-Up No-Limit Hold'em.

Rastrea estadísticas de comportamiento del rival para derivar una
estrategia de contra-explotación adaptativa.

Estadísticas rastreadas
-----------------------
VPIP  – Voluntarily Put $ In Pot  (frecuencia de participar en el bote)
PFR   – Pre-Flop Raise            (frecuencia de raise preflop)
AF    – Aggression Factor = (bet+raise) / call  por calle y total
CBet  – Continuation Bet          (apuesta en flop tras haber abierto preflop)
FTB   – Fold To Bet               (fold ante apuesta/raise del rival)
WTSD  – Went To ShowDown
WSD   – Won at ShowDown

Arquetipos de rival
-------------------
loose-passive   : VPIP alto, AF bajo  → value bet delgado, sin bluffs
loose-aggressive: VPIP alto, AF alto  → trap con manos fuertes
tight-passive   : VPIP bajo, AF bajo  → bluffear más, echar cuando apuestan
tight-aggressive: VPIP bajo, AF alto  → cerca del GTO; desviarse poco

Uso
---
    model = OpponentModel(opponent_id=1)

    # Al inicio de cada mano:
    model.new_hand()

    # Cuando el oponente actúa:
    model.observe_action('raise', 'preflop', voluntarily=True)
    model.observe_fold_to_bet('flop')   # si recibió una apuesta

    # Al final de la mano:
    model.observe_showdown(won=True)

    # Ajustes para la IA propia:
    adj = model.get_counter_adjustments()
    # adj['bluff_freq_mult'], adj['value_threshold_adj'], etc.
"""

from dataclasses import dataclass, field
from typing import Dict


# ── Estadísticas del oponente ─────────────────────────────────────────────────

@dataclass
class OpponentStats:
    """Contadores crudos de todas las estadísticas rastreadas."""

    hands_seen:  int = 0
    vpip_count:  int = 0
    pfr_count:   int = 0
    wtsd_count:  int = 0
    wsd_count:   int = 0
    cbet_count:  int = 0    # veces que hizo cbet (flop) tras abrir
    cbet_opp:    int = 0    # veces que podía hacer cbet

    # Por calle: {'preflop': n, 'flop': n, 'turn': n, 'river': n}
    aggressive: Dict[str, int] = field(
        default_factory=lambda: {s: 0 for s in ['preflop', 'flop', 'turn', 'river']})
    passive:    Dict[str, int] = field(
        default_factory=lambda: {s: 0 for s in ['preflop', 'flop', 'turn', 'river']})
    fold_to_bet: Dict[str, int] = field(
        default_factory=lambda: {s: 0 for s in ['preflop', 'flop', 'turn', 'river']})
    bets_faced:  Dict[str, int] = field(
        default_factory=lambda: {s: 0 for s in ['preflop', 'flop', 'turn', 'river']})

    # ── Propiedades derivadas ────────────────────────────────────────────────

    @property
    def vpip(self) -> float:
        return self.vpip_count / self.hands_seen if self.hands_seen else 0.50

    @property
    def pfr(self) -> float:
        return self.pfr_count / self.hands_seen if self.hands_seen else 0.25

    @property
    def wtsd(self) -> float:
        return self.wtsd_count / max(1, self.hands_seen)

    @property
    def wsd(self) -> float:
        return self.wsd_count / max(1, self.wtsd_count)

    @property
    def cbet_freq(self) -> float:
        return self.cbet_count / max(1, self.cbet_opp)

    def af(self, street: str = 'total') -> float:
        """Aggression Factor = (bet+raise) / call."""
        if street == 'total':
            agg = sum(self.aggressive.values())
            pas = sum(self.passive.values())
        else:
            agg = self.aggressive.get(street, 0)
            pas = self.passive.get(street, 0)
        return agg / max(1, pas)

    def ftb(self, street: str = 'total') -> float:
        """Fold To Bet rate para la calle dada."""
        if street == 'total':
            f = sum(self.fold_to_bet.values())
            b = sum(self.bets_faced.values())
        else:
            f = self.fold_to_bet.get(street, 0)
            b = self.bets_faced.get(street, 0)
        return f / max(1, b)


# ── Modelo del oponente ───────────────────────────────────────────────────────

class OpponentModel:
    """
    Observa y modela el comportamiento del oponente para derivar
    ajustes de contra-explotación en tiempo real.

    Parámetros
    ----------
    opponent_id : int – id del jugador que se modela (0=SB, 1=BB)
    """

    # Umbrales de clasificación
    _LOOSE_VPIP   = 0.42
    _TIGHT_VPIP   = 0.20
    _AGGR_AF      = 2.5
    _PASSIVE_AF   = 0.8
    _FOLD_THRESH  = 0.55   # FTB > 55% → bluffear más
    _CALL_THRESH  = 0.20   # FTB < 20% → eliminar bluffs

    # Muestra mínima para confiar en las estadísticas
    _MIN_HANDS    = 12

    def __init__(self, opponent_id: int = 1):
        self.opponent_id      = opponent_id
        self.stats            = OpponentStats()
        self._opened_preflop  = False   # tracking intra-hand

    # ── API de observación ────────────────────────────────────────────────────

    def new_hand(self):
        """Llamar al inicio de cada mano nueva."""
        self.stats.hands_seen += 1
        self._opened_preflop   = False

    def observe_action(self, action_str: str, street: str,
                       voluntarily: bool = True):
        """
        Registra una acción observada del oponente.

        Parámetros
        ----------
        action_str  : str  – 'fold' | 'call' | 'check' | 'raise' | 'all in'
        street      : str  – 'preflop' | 'flop' | 'turn' | 'river'
        voluntarily : bool – True si fue voluntaria (no forzada por blinds)
        """
        a = action_str.lower().strip()
        s = street.lower().strip()

        # VPIP / PFR (solo preflop, solo acciones voluntarias)
        if s == 'preflop' and voluntarily:
            if a in ('call', 'raise', 'all in'):
                self.stats.vpip_count += 1
            if a in ('raise', 'all in'):
                self.stats.pfr_count += 1
                self._opened_preflop  = True

        # CBet: apuesta en el flop tras haber abierto en preflop
        if s == 'flop' and self._opened_preflop:
            self.stats.cbet_opp += 1
            if a in ('raise', 'all in'):
                self.stats.cbet_count += 1

        # Agresividad / pasividad
        if a in ('raise', 'all in'):
            self.stats.aggressive[s] = self.stats.aggressive.get(s, 0) + 1
        elif a in ('call', 'check'):
            self.stats.passive[s] = self.stats.passive.get(s, 0) + 1
        elif a == 'fold':
            # Solo registra el fold; el denominador (bets_faced) debe
            # haberse registrado previamente con observe_bet_faced().
            self.stats.fold_to_bet[s] = self.stats.fold_to_bet.get(s, 0) + 1

    def observe_bet_faced(self, street: str):
        """
        Registrar que el oponente recibió una apuesta (denominator de FTB).
        Llamar cuando el agente apuesta/raise y el oponente tiene que responder.
        """
        s = street.lower()
        self.stats.bets_faced[s] = self.stats.bets_faced.get(s, 0) + 1

    def observe_showdown(self, won: bool):
        """Registrar resultado de showdown del oponente."""
        self.stats.wtsd_count += 1
        if won:
            self.stats.wsd_count += 1

    # ── Clasificación ─────────────────────────────────────────────────────────

    def classify(self) -> str:
        """
        Clasifica al rival en uno de cuatro arquetipos:
          'loose-passive' | 'loose-aggressive' |
          'tight-passive' | 'tight-aggressive' | 'unknown'
        """
        if self.stats.hands_seen < self._MIN_HANDS:
            return 'unknown'

        loose = self.stats.vpip > self._LOOSE_VPIP
        aggr  = self.stats.af() > self._AGGR_AF

        if   loose and aggr:  return 'loose-aggressive'
        elif loose:           return 'loose-passive'
        elif aggr:            return 'tight-aggressive'
        else:                 return 'tight-passive'

    # ── Ajustes de contra-explotación ────────────────────────────────────────

    def get_counter_adjustments(self) -> dict:
        """
        Devuelve ajustes sobre los umbrales de la IA para contrarrestar al rival.

        Claves devueltas
        ----------------
        bluff_freq_mult     : float  – multiplicador frecuencia de bluff  (×1.0 = neutro)
        value_threshold_adj : float  – ajuste aditivo al umbral de equity para value bet
        call_threshold_adj  : float  – ajuste aditivo al umbral de equity para call
        bet_size_mult       : float  – multiplicador de tamaño de apuesta
        archetype           : str    – etiqueta del arquetipo detectado
        """
        adj = {
            'bluff_freq_mult':     1.0,
            'value_threshold_adj': 0.0,
            'call_threshold_adj':  0.0,
            'bet_size_mult':       1.0,
            'archetype':           self.classify(),
        }

        archetype = adj['archetype']

        if archetype == 'loose-passive':
            # Llama con cualquier cosa → value bet delgado, apostamos más
            adj['bluff_freq_mult']     = 0.20
            adj['value_threshold_adj'] = -0.08   # apostar con hands más débiles
            adj['bet_size_mult']       = 1.35

        elif archetype == 'loose-aggressive':
            # Bluffea y re-raise mucho → trampear con manos fuertes, llamar amplio
            adj['bluff_freq_mult']     = 0.35
            adj['call_threshold_adj']  = -0.10   # llamar con rango más amplio
            adj['bet_size_mult']       = 0.80    # apostar menos para inducir raise

        elif archetype == 'tight-passive':
            # Muy selectivo → bluffear más, fold ante sus apuestas
            adj['bluff_freq_mult']     = 1.80
            adj['value_threshold_adj'] = 0.07    # solo value con manos muy fuertes
            adj['bet_size_mult']       = 1.00

        elif archetype == 'tight-aggressive':
            # Cerca del GTO → desviarse poco del blueprint
            adj['bluff_freq_mult']     = 0.90
            adj['bet_size_mult']       = 0.92

        # Micro-ajuste por Fold To Bet global
        ftb = self.stats.ftb()
        if ftb > self._FOLD_THRESH:
            # Foldea mucho → aumentar bluffs
            adj['bluff_freq_mult'] = min(adj['bluff_freq_mult'] * 1.60, 3.5)
        elif ftb < self._CALL_THRESH:
            # Nunca foldea → eliminar bluffs
            adj['bluff_freq_mult'] = max(adj['bluff_freq_mult'] * 0.40, 0.0)

        return adj

    # ── Representación ────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Resumen legible de las estadísticas del oponente."""
        s = self.stats
        lines = [
            f"Oponente id={self.opponent_id} | Arquetipo: {self.classify()}",
            f"  Manos: {s.hands_seen:>4}  |  VPIP: {s.vpip:.0%}  PFR: {s.pfr:.0%}",
            f"  AF(total): {s.af():.2f}  |  FTB(total): {s.ftb():.0%}",
            f"  CBet: {s.cbet_freq:.0%}  |  WTSD: {s.wtsd:.0%}  WSD: {s.wsd:.0%}",
        ]
        return "\n".join(lines)

    # ── Persistencia ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Serializa el modelo del oponente en disco (pickle).

        Guarda tanto las estadísticas acumuladas como el opponent_id,
        para poder reanudar el tracking entre sesiones.

        Parámetros
        ----------
        path : str – ruta del archivo destino (e.g. 'opp_model.pkl')
        """
        import pickle, os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'opponent_id':     self.opponent_id,
                'stats':           self.stats,
                '_opened_preflop': self._opened_preflop,
            }, f, protocol=4)

    @classmethod
    def load(cls, path: str) -> 'OpponentModel':
        """
        Carga un OpponentModel previamente guardado con save().

        Parámetros
        ----------
        path : str – ruta del archivo pickle

        Retorna
        -------
        OpponentModel con las estadísticas restauradas

        Raises
        ------
        FileNotFoundError si el archivo no existe
        """
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls(opponent_id=data['opponent_id'])
        model.stats            = data['stats']
        model._opened_preflop  = data.get('_opened_preflop', False)
        return model
