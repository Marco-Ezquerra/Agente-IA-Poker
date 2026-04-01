"""
Servidor Flask para jugar al poker contra el agente IA.

Endpoints
---------
GET  /                   → Sirve la interfaz HTML
POST /api/nueva_partida  → Reinicia sesión y reparte primera mano
GET  /api/estado         → Estado JSON de la mano actual
POST /api/accion         → Aplica acción del humano (fold/call/check/raise/all_in)
POST /api/nueva_mano     → Siguiente mano tras ver resultado
GET  /api/stats          → Stats del oponente model (VPIP, PFR, AF, …)
POST /api/recargar_blueprint → Recarga el blueprint desde disco

Diseño
------
- Un objeto PartidaWeb por sesión (UUID en cookie), guardado en _sessions dict.
- PartidaWeb es una máquina de estados no-bloqueante: no usa el loop ejecutar()
  de poker_engine. Gestiona la secuencia de acciones paso a paso.
- Cuando toca al bot, decide automáticamente (RealtimeSearch + blueprint).
- Cuando toca al humano, devuelve estado JSON con acciones válidas y espera.
"""

import os
import sys
import uuid
import random
import secrets
import logging
import threading
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM  = os.path.dirname(_HERE)
if _SIM not in sys.path:
    sys.path.insert(0, _SIM)

from collections import deque
from flask import Flask, request, jsonify, render_template, session

from poker_engine  import Carta, Baraja, Jugador, compact_card
from opponent_model import OpponentModel

# Importaciones opcionales (pueden no estar disponibles sin blueprint entrenado)
try:
    from cfr.mccfr_trainer   import MCCFRTrainer
    from cfr.realtime_search import RealtimeSearch
    _CFR_AVAILABLE = True
except Exception:
    _CFR_AVAILABLE = False

try:
    from abstracciones.card_abstractor import preflop_bucket, postflop_bucket
    _ABSTRACTOR_AVAILABLE = True
except Exception:
    _ABSTRACTOR_AVAILABLE = False

try:
    from template import eval_hand_from_strings
    _EVAL_AVAILABLE = True
except Exception:
    _EVAL_AVAILABLE = False

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
FICHAS_INICIALES = 200.0
SB_AMOUNT        = 0.5
BB_AMOUNT        = 1.0
MAX_RAISES       = 4
BLUEPRINT_PATH   = os.path.join(_SIM, "cfr", "blueprint.pkl")

# ── Blueprint global (compartido entre sesiones) ──────────────────────────────
_blueprint      = None
_realtime_proto = None   # prototipo; cada sesión crea su propia instancia

def _load_blueprint():
    global _blueprint, _realtime_proto
    if not _CFR_AVAILABLE:
        return
    try:
        _blueprint = MCCFRTrainer.load(BLUEPRINT_PATH)
        _realtime_proto = RealtimeSearch(
            blueprint=_blueprint, depth=1, iterations=120
        )
        log.warning("Blueprint cargado: %s InfoSets", len(_blueprint.regret_sum))
    except Exception as e:
        log.warning("Blueprint no disponible: %s", e)
        _blueprint      = None
        _realtime_proto = None

_load_blueprint()

# Lock para escrituras concurrentes al blueprint (online learning thread-safe)
_blueprint_lock = threading.Lock()
# Contador global de manos jugadas (para guardado periódico)
_hands_since_save  = 0
_SAVE_EVERY_N_HANDS = 20   # guardar blueprint cada 20 manos aprendidas

# ═══════════════════════════════════════════════════════════════════════════════
#  ONLINE LEARNING — Aprendizaje continuo basado en experiencia
# ═══════════════════════════════════════════════════════════════════════════════

def _online_learn(hand_sb, hand_bb, community, deck_fill,
                  n_iters: int = 80, bucket_sims: int = 60):
    """
    Actualiza el blueprint CFR con la mano real recién jugada.

    Fundamento matemático (Experience Replay CFR):
      En CFR est\u00e1ndar, cada iteraci\u00f3n usa un deal ALEATORIO y contribuye
      igualmente a todos los InfoSets del juego abstracto.

      Con Online/Experience-Replay CFR (Brown & Sandholm, 2019 — Pluribus
      Appendix), se hacen traversals adicionales en deals REALES para
      reforzar los InfoSets m\u00e1s frecuentes en la pr\u00e1ctica:
        - El InfoSet (SB, preflop, bucket(AKs), ...) recibe m\u00e1s visitas
          si el humano llega frecuentemente con ese tipo de mano.
        - La estrategia promedio converge al mismo equilibrio de Nash,
          pero con menor varianza en los caminos frecuentes.

    El deal completo se reconstruye llenando las cartas faltantes con el
    deck sobrante de la mano. Si faltan cartas (e.g. el juego termin\u00f3 en
    preflop), se completa el board aleatoriamente.

    Parámetros
    ----------
    hand_sb    : list[str]  – 2 cartas del SB
    hand_bb    : list[str]  – 2 cartas del BB
    community  : list[str]  – cartas comunitarias ya reveladas (0..5)
    deck_fill  : list[str]  – cartas sobrantes sin revelar
    n_iters    : int        – traversals MCCFR sobre este deal
    bucket_sims: int        – sims Monte Carlo para precomputar buckets
    """
    global _hands_since_save

    if _blueprint is None or not _CFR_AVAILABLE:
        return
    try:
        from cfr.mccfr_trainer import MCCFRTrainer

        # Completar board hasta 5 cartas con el deck sobrante
        full_community = list(community)
        fill_cards     = list(deck_fill)
        random.shuffle(fill_cards)
        while len(full_community) < 5 and fill_cards:
            full_community.append(fill_cards.pop(0))
        # Si aún faltan, completar con cartas nuevas aleatorias del deck completo
        if len(full_community) < 5:
            known = set(hand_sb + hand_bb + full_community)
            extra = [c for c in _full_deck() if c not in known]
            random.shuffle(extra)
            while len(full_community) < 5 and extra:
                full_community.append(extra.pop(0))

        if len(full_community) < 5:
            return   # imposible completar el board

        boards = [
            full_community[:3],
            full_community[3:4],
            full_community[4:5],
        ]
        hands  = [hand_sb, hand_bb]

        with _blueprint_lock:
            bkts = MCCFRTrainer._precompute_buckets.__func__(
                None, hands, boards, sims=bucket_sims
            ) if False else _blueprint._precompute_buckets(hands, boards, sims=bucket_sims)

            init_kwargs = dict(
                bkts=bkts, boards=boards,
                street_idx=0, pot=1.5,
                stacks=[99.5, 99.0],
                contribs=[0.5, 1.0],
                bet_hist=[], n_raises=0,
                to_call=0.5, position=0,
            )
            for _ in range(n_iters // 2):
                for traverser in [0, 1]:
                    _blueprint._cfr(traverser, **init_kwargs)
                _blueprint.iterations += 1

        _hands_since_save += 1
        if _hands_since_save >= _SAVE_EVERY_N_HANDS:
            _hands_since_save = 0
            # Guardar en hilo separado para no bloquear
            threading.Thread(
                target=lambda: _blueprint.save(BLUEPRINT_PATH),
                daemon=True,
            ).start()
            log.warning("Blueprint guardado  (%d iters, %d InfoSets)",
                        _blueprint.iterations, len(_blueprint.regret_sum))

    except Exception as e:
        log.warning("Online learning error: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
#  PARTIDA WEB — Máquina de estados no-bloqueante
# ═══════════════════════════════════════════════════════════════════════════════

class PartidaWeb:
    """
    Máquina de estados que gestiona una partida HUNL completa
    sin bloquear el hilo HTTP.

    Estados posibles del juego (self.game_state):
        'IDLE'       – sin mano activa
        'HUMAN_TURN' – esperando acción del humano via API
        'BOT_TURN'   – el bot está calculando (transitorio, no retorna al cliente)
        'HAND_OVER'  – mano terminada, esperando /api/nueva_mano

    Posiciones (se alternan cada mano):
        sb_idx = índice del jugador que es Small Blind esta mano
        bb_idx = 1 - sb_idx
    """

    # ── Constructor ───────────────────────────────────────────────────────────

    def __init__(self):
        # Jugadores permanentes (stacks persisten entre manos)
        self.players = [
            Jugador("Tú",  FICHAS_INICIALES, id=0),
            Jugador("Bot", FICHAS_INICIALES, id=1),
        ]
        self.human_idx = 0
        self.bot_idx   = 1

        # Posiciones (rotan)
        self.hand_number = 0
        self.sb_idx      = 0  # SB en la primera mano = Humano
        self.bb_idx      = 1

        # Estado de la mano actual
        self.game_state = 'IDLE'
        self.street     = None          # 'preflop' | 'flop' | 'turn' | 'river'
        self.community  = []            # lista de compact cards visibles
        self._baraja_cards = []         # cartas de la baraja (como compact strings)
        self._deck_pool    = []         # cartas disponibles tras repartir

        # Estado de la calle actual
        self.pot               = 0.0
        self.current_bet       = 0.0
        self.contributions     = {0: 0.0, 1: 0.0}
        self.raise_count       = 0
        self.all_in_occurred   = False
        self._players_to_act   = deque()  # cola de quién actúa
        self._last_raiser      = None
        self._street_actions   = []       # [(actor_idx, tipo, amount), …]

        # Resultado de la mano
        self.result            = None     # dict cuando HAND_OVER

        # Historial de manos (resumen por mano)
        self.session_results   = []

        # IA
        self.opp_model = OpponentModel(opponent_id=self.bot_idx)
        self._search   = (
            RealtimeSearch(blueprint=_blueprint, depth=1, iterations=80)
            if _CFR_AVAILABLE else None
        )

    # ── Pública: nueva mano ───────────────────────────────────────────────────

    def nueva_mano(self):
        """Reparte nueva mano, posta blinds y avanza hasta el turno del humano."""
        # Recargar fichas si alguien se quedó sin chips
        for p in self.players:
            if p.fichas < BB_AMOUNT * 2:
                p.fichas = FICHAS_INICIALES

        self.hand_number += 1

        # Alternar posiciones cada mano
        self.sb_idx = (self.hand_number - 1) % 2
        self.bb_idx = 1 - self.sb_idx

        # Reset estado de mano
        self.community        = []
        self.pot              = 0.0
        self.current_bet      = 0.0
        self.contributions    = {0: 0.0, 1: 0.0}
        self.raise_count      = 0
        self.all_in_occurred  = False
        self._street_actions  = []
        self._last_raiser     = None
        self.result           = None
        self.game_state       = 'IDLE'
        for p in self.players:
            p.mano = []

        # Repartir cartas
        baraja = list(_full_deck())
        random.shuffle(baraja)
        self.players[0].mano = baraja[:2]
        self.players[1].mano = baraja[2:4]
        self._deck_pool       = baraja[4:]

        # Postear blinds
        sb = self.players[self.sb_idx]
        bb = self.players[self.bb_idx]
        sb_post = min(SB_AMOUNT, sb.fichas)
        bb_post = min(BB_AMOUNT, bb.fichas)
        sb.fichas                   -= sb_post
        bb.fichas                   -= bb_post
        self.pot                    += sb_post + bb_post
        self.contributions[self.sb_idx] = sb_post
        self.contributions[self.bb_idx] = bb_post
        self.current_bet             = bb_post

        # Iniciar calle preflop
        self.street = 'preflop'
        # Preflop: SB actúa primero, luego BB (BB tiene "opción" si SB llama)
        self._players_to_act = deque([self.sb_idx, self.bb_idx])
        self._last_raiser    = self.bb_idx  # BB es el "agresor" inicial (big blind)

        self.opp_model.new_hand()
        self.game_state = 'IN_PROGRESS'   # permite que _advance() no salga inmediatamente
        self._advance()
        return self.get_state_json()

    # ── Pública: acción del humano ────────────────────────────────────────────

    def accion(self, tipo, amount=None):
        """
        Aplica la acción del humano y avanza el juego.

        Parámetros
        ----------
        tipo   : 'fold' | 'call' | 'check' | 'raise' | 'all_in'
        amount : float | None – chips adicionales (para 'raise')

        Retorna
        -------
        dict – estado JSON actualizado
        """
        if self.game_state != 'HUMAN_TURN':
            return self.get_state_json()

        # Validar que el siguiente en actuar es el humano
        if not self._players_to_act or self._players_to_act[0] != self.human_idx:
            return self.get_state_json()

        # Registrar observación para el opponent model
        self._observe_human_action(tipo)

        # Aplicar la acción (solo actualiza el estado, no llama _advance)
        self.game_state = 'IN_PROGRESS'
        self._apply_action(self.human_idx, tipo, amount)
        # Avanzar hasta siguiente turno humano o fin de mano
        if self.game_state not in ('HAND_OVER',):
            self._advance()
        return self.get_state_json()

    # ── Pública: estado JSON ──────────────────────────────────────────────────

    def get_state_json(self):
        """Serializa el estado completo de la partida a dict JSON."""
        human  = self.players[self.human_idx]
        bot    = self.players[self.bot_idx]
        pos    = "SB" if self.human_idx == self.sb_idx else "BB"

        to_call = max(0.0, self.current_bet - self.contributions[self.human_idx])

        valid_actions = []
        if self.game_state == 'HUMAN_TURN':
            valid_actions = self._compute_valid_actions(self.human_idx)

        # Cartas del bot: solo visibles en showdown
        bot_hand = None
        if self.result and self.result.get('tipo') == 'showdown':
            bot_hand = bot.mano

        return {
            "game_state"    : self.game_state,
            "street"        : self.street,
            "pot"           : round(self.pot, 2),
            "community"     : self.community,
            "tu_mano"       : human.mano,
            "tu_stack"      : round(human.fichas, 2),
            "bot_stack"     : round(bot.fichas, 2),
            "tu_pos"        : pos,
            "turno_humano"  : self.game_state == 'HUMAN_TURN',
            "current_bet"   : round(self.current_bet, 2),
            "to_call"       : round(to_call, 2),
            "valid_actions" : valid_actions,
            "resultado"     : self.result,
            "bot_mano"      : bot_hand,
            "mano_numero"   : self.hand_number,
            "sb_idx"        : self.sb_idx,
            "resultados_sesion": self.session_results[-10:],
        }

    # ── Privado: avanzar hasta turno humano o fin ─────────────────────────────

    def _advance(self):
        """Avanza el juego automáticamente mientras toca al bot."""
        while True:
            if self.game_state in ('HAND_OVER', 'IDLE', 'HUMAN_TURN'):
                break
            if not self._players_to_act:
                # Calle terminada → siguiente calle
                self._next_street()
                if self.game_state == 'HAND_OVER':
                    break
                continue
            next_actor = self._players_to_act[0]
            if next_actor == self.human_idx:
                self.game_state = 'HUMAN_TURN'
                break
            # Es turno del bot → decidir y aplicar
            self.game_state = 'BOT_TURN'
            tipo, amount = self._bot_decide()
            self._apply_action(self.bot_idx, tipo, amount)

    # ── Privado: aplicar acción ───────────────────────────────────────────────

    def _apply_action(self, actor_idx, tipo, amount=None):
        """Aplica una acción al estado del juego y actualiza la cola."""
        if not self._players_to_act or self._players_to_act[0] != actor_idx:
            return

        player  = self.players[actor_idx]
        to_call = max(0.0, self.current_bet - self.contributions[actor_idx])

        if tipo == 'fold':
            winner_idx = 1 - actor_idx
            self.players[winner_idx].fichas += self.pot
            self._street_actions.append((actor_idx, 'fold', 0))
            self._end_hand('fold', winner_idx)
            return

        elif tipo in ('call', 'check'):
            call_amt = min(to_call, player.fichas)
            player.fichas -= call_amt
            self.contributions[actor_idx] += call_amt
            self.pot += call_amt
            if player.fichas == 0:
                self.all_in_occurred = True
            self._street_actions.append((actor_idx, 'call', call_amt))
            self._players_to_act.popleft()
            # all-in sin más jugadores pendientes → siguiente calle gestionada por _advance

        elif tipo == 'raise':
            # amount = chips ADICIONALES por encima de la apuesta actual
            min_raise = max(BB_AMOUNT, self.current_bet)
            if amount is None or amount < min_raise:
                amount = min_raise
            # Clampar al stack del jugador
            total_add = min(to_call + amount, player.fichas)
            if total_add >= player.fichas:
                self.all_in_occurred = True
            player.fichas -= total_add
            self.contributions[actor_idx] += total_add
            self.pot += total_add
            new_bet = self.contributions[actor_idx]
            self.current_bet  = new_bet
            self.raise_count += 1
            self._last_raiser = actor_idx
            self._street_actions.append((actor_idx, 'raise', total_add))
            self._players_to_act.popleft()
            # El otro jugador debe actuar
            opponent = 1 - actor_idx
            self._players_to_act = deque([opponent])

        elif tipo == 'all_in':
            total = player.fichas
            player.fichas = 0
            self.contributions[actor_idx] += total
            self.pot += total
            if self.contributions[actor_idx] > self.current_bet:
                self.current_bet = self.contributions[actor_idx]
            self.all_in_occurred = True
            self._street_actions.append((actor_idx, 'all_in', total))
            self._players_to_act.popleft()
            # Si el oponente ya ha igualado o está all-in también → showdown
            opponent = 1 - actor_idx
            opp_to_call = max(0.0, self.current_bet - self.contributions[opponent])
            if opp_to_call > 0 and self.players[opponent].fichas > 0:
                self._players_to_act = deque([opponent])

    # ── Privado: siguiente calle ──────────────────────────────────────────────

    def _next_street(self):
        """Avanza a la siguiente calle o resuelve el showdown."""
        streets = ['preflop', 'flop', 'turn', 'river']
        idx     = streets.index(self.street)

        # All-in o river completado → completar cartas y resolver
        if self.all_in_occurred or idx >= 3:
            # Completar tablero hasta 5 cartas
            while len(self.community) < 5 and self._deck_pool:
                self.community.append(self._deck_pool.pop(0))
            self._resolve_showdown()
            return

        # Avanzar a la siguiente calle
        self.street          = streets[idx + 1]
        self.current_bet     = 0.0
        self.contributions   = {0: 0.0, 1: 0.0}
        self.raise_count     = 0
        self._last_raiser    = None
        self._street_actions = []

        # Repartir cartas comunitarias
        if self.street == 'flop':
            for _ in range(3):
                if self._deck_pool:
                    self.community.append(self._deck_pool.pop(0))
        else:
            if self._deck_pool:
                self.community.append(self._deck_pool.pop(0))

        # Postflop: BB actúa primero (fuera de posición)
        self._players_to_act = deque([self.bb_idx, self.sb_idx])

    # ── Privado: showdown y fin de mano ──────────────────────────────────────

    def _resolve_showdown(self):
        """Evalúa las manos y distribuye el pot."""
        h0 = self.players[0].mano
        h1 = self.players[1].mano
        board = self.community

        if _EVAL_AVAILABLE and len(board) >= 3:
            try:
                rank0 = eval_hand_from_strings(h0 + board)
                rank1 = eval_hand_from_strings(h1 + board)
                if rank0 < rank1:
                    winner = 0
                elif rank1 < rank0:
                    winner = 1
                else:
                    winner = -1  # split
            except Exception:
                winner = self._showdown_by_buckets()
        else:
            winner = self._showdown_by_buckets()

        pot = self.pot
        if winner == -1:
            split = pot / 2
            self.players[0].fichas += split
            self.players[1].fichas += split
            result_str = "split"
        else:
            self.players[winner].fichas += pot
            result_str = "win" if winner == self.human_idx else "lose"

        name = self.players[winner].nombre if winner != -1 else "empate"
        self._end_hand('showdown', winner if winner != -1 else self.human_idx,
                       extra={"tipo": "showdown", "ganador": name,
                              "resultado_humano": result_str, "pot": round(pot, 2)})

    def _showdown_by_buckets(self):
        """Fallback: decide ganador por bucket de EHS²."""
        if not _ABSTRACTOR_AVAILABLE:
            return random.choice([0, 1])
        h0, h1 = self.players[0].mano, self.players[1].mano
        board   = self.community
        b0 = postflop_bucket(h0, board, num_sims=30) if board else preflop_bucket(h0)
        b1 = postflop_bucket(h1, board, num_sims=30) if board else preflop_bucket(h1)
        if b0 > b1:   return 0
        if b1 > b0:   return 1
        return -1

    def _end_hand(self, tipo, winner_idx, extra=None):
        """Marca la mano como terminada y registra resultado."""
        result_humano = ("win" if winner_idx == self.human_idx
                         else "fold_win" if tipo == 'fold' and winner_idx == self.human_idx
                         else "lose")
        if tipo == 'showdown':
            result_humano = extra.get("resultado_humano", result_humano)

        self.result = extra or {
            "tipo"           : tipo,
            "ganador"        : self.players[winner_idx].nombre,
            "resultado_humano": result_humano,
            "pot"            : round(self.pot, 2),
        }
        if "tipo" not in self.result:
            self.result["tipo"] = tipo

        self.session_results.append({
            "mano"    : self.hand_number,
            "resultado": result_humano,
            "pot"     : round(self.pot, 2),
            "stacks"  : [round(p.fichas, 2) for p in self.players],
        })
        self.game_state = 'HAND_OVER'

        # ── Online learning: actualizar blueprint con las cartas reales \u2500\u2500\u2500\u2500\u2500\u2500\u2500
        # Se ejecuta en un hilo separado para no bloquear la respuesta HTTP.
        if _CFR_AVAILABLE and _blueprint is not None:
            deck_fill = list(self._deck_pool)
            hand_sb   = list(self.players[self.sb_idx].mano)
            hand_bb   = list(self.players[self.bb_idx].mano)
            comm_snap = list(self.community)
            threading.Thread(
                target=_online_learn,
                args=(hand_sb, hand_bb, comm_snap, deck_fill),
                daemon=True,
            ).start()

    # ── Privado: decisión del bot ─────────────────────────────────────────────

    def _bot_decide(self):
        """Calcula la acción del bot usando el pipeline:

          1. Blueprint CFR  (estrategia GTO trained offline)
          2. Ajuste opponent model  (explotar desviaciones del rival)
          3. Heurística equity  (fallback cuando el InfoSet no está en blueprint)

        El override de equity-pura SOLO se activa cuando to_call es un
        true all-in (>=95% stack), no para cualquier apuesta grande.
        Usos del override anterior (50% stack) causaban que el blueprint
        fuese ignorado en casi toda situación postflop → decisiones aleatorias.
        """
        bot     = self.players[self.bot_idx]
        to_call = max(0.0, self.current_bet - self.contributions[self.bot_idx])
        adj     = self.opp_model.get_counter_adjustments()

        # Override SOLO para auténticos all-in calls (>=95% stack)
        # Justificación: con 8 buckets la heurística es más fiable que el
        # blueprint en spots all-in de alta varianza.
        if bot.fichas > 0 and to_call >= bot.fichas * 0.95:
            return self._heuristic_bot_action(to_call, bot.fichas, adj)

        # ── 1. Consulta directa al blueprint (estrategia GTO) ────────────────
        abstract = self._blueprint_query(bot)
        if abstract is not None:
            return self._translate_abstract_action(abstract, to_call, bot.fichas, adj)

        # ── 2. Fallback heurístico ───────────────────────────────────────────
        return self._heuristic_bot_action(to_call, bot.fichas, adj)

    def _to_abstract_bet_hist(self):
        """
        Convierte _street_actions (lista de (actor, tipo, amount)) a la
        secuencia de acciones abstractas que usa el blueprint como clave.

        Bug previo: usaba self.pot (pot final acumulado) para calcular
        pot_before de cada acción → pot_before era incorrecto para cualquier
        calle con >1 acción.

        Fix: reconstruye el pot progresivamente action-by-action y calcula
        solo la porción de raise_extra (chips por encima de calling) vs
        el pot antes de la acción.
        """
        from abstracciones.infoset_encoder import (
            FOLD, CALL, RAISE_THIRD, RAISE_HALF, RAISE_POT, RAISE_2POT, ALLIN,
        )
        # Pot al inicio de la calle actual (excluye lo ya apostado en esta calle)
        amt_this_street = sum(
            amt for (_, t, amt) in self._street_actions if t != 'fold'
        )
        running_pot = max(self.pot - amt_this_street, BB_AMOUNT)
        rc = [0.0, 0.0]  # contribuciones de cada jugador en esta calle

        bet_hist = []
        for (actor, tipo, amount) in self._street_actions:
            if tipo == 'fold':
                bet_hist.append(FOLD)
                continue

            pot_before = max(running_pot, BB_AMOUNT)

            if tipo in ('call', 'check'):
                bet_hist.append(CALL)
            elif tipo == 'raise':
                # amount = total_add = to_call_portion + raise_extra
                # to_call_portion = lo que el actor adeudaba antes de actuar
                to_call_portion = max(0.0, rc[1 - actor] - rc[actor])
                raise_extra     = max(0.0, amount - to_call_portion)
                ratio           = raise_extra / pot_before
                if ratio <= 0.42:
                    bet_hist.append(RAISE_THIRD)
                elif ratio <= 0.62:
                    bet_hist.append(RAISE_HALF)
                elif ratio <= 1.2:
                    bet_hist.append(RAISE_POT)
                else:
                    bet_hist.append(RAISE_2POT)
            elif tipo == 'all_in':
                bet_hist.append(ALLIN)

            running_pot   += amount
            rc[actor]     += amount

        return bet_hist

    def _blueprint_query(self, bot):
        """
        Consulta directamente el blueprint CFR para obtener la estrategia
        en el InfoSet actual y muestrea una acción de esa distribución.

        Retorna una acción abstracta string ('f','c','r1',…) o None si el
        InfoSet no está en el blueprint (InfoSet no visitado durante el train).
        """
        if _blueprint is None or not _ABSTRACTOR_AVAILABLE:
            return None
        try:
            from abstracciones.card_abstractor import (
                preflop_bucket, postflop_bucket, PREFLOP_BUCKETS, POSTFLOP_BUCKETS,
            )
            from abstracciones.infoset_encoder import ABSTRACT_ACTIONS

            street_idx = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}[self.street]

            # ── Bucket de la mano del bot en la calle actual ─────────────────
            if self.street == 'preflop':
                hand_bucket   = preflop_bucket(bot.mano, num_sims=80)
                board_buckets = ()
            else:
                hand_bucket   = postflop_bucket(bot.mano, self.community, num_sims=80)
                # Board buckets de calles ya reveladas (excluye la actual)
                bb_list = []
                if street_idx >= 1 and len(self.community) >= 3:
                    bb_list.append(postflop_bucket(bot.mano, self.community[:3], num_sims=50))
                if street_idx >= 2 and len(self.community) >= 4:
                    bb_list.append(postflop_bucket(bot.mano, self.community[:4], num_sims=50))
                if street_idx >= 3 and len(self.community) >= 5:
                    bb_list.append(postflop_bucket(bot.mano, self.community[:5], num_sims=50))
                board_buckets = tuple(bb_list)

            bet_hist = tuple(self._to_abstract_bet_hist())

            # ── Posición abstracta: SB=0, BB=1  (NO el índice del jugador) ──
            # Bug crítico previo: usaba self.bot_idx (siempre 1) como posición.
            # En manos donde el bot es SB (posición 0), el blueprint entrenado
            # con SB=0 devolvía la estrategia BB → decisiones completamente
            # erróneas en el 50% de las manos.
            abstract_position = 0 if self.bot_idx == self.sb_idx else 1
            key      = (abstract_position, street_idx, hand_bucket, board_buckets, bet_hist)

            # ── Lookup en blueprint con fallback progresivo ──────────────────
            # 1. Historial exacto
            # 2. Historial vacío (misma calle, misma mano, pero inicio de calle)
            # 3. Bucket vecino ±1 (interpolación implícita)
            ss = _blueprint.strategy_sum.get(key)
            if ss is None or ss.sum() == 0:
                key_fb = (abstract_position, street_idx, hand_bucket, board_buckets, ())
                ss = _blueprint.strategy_sum.get(key_fb)
            if ss is None or ss.sum() == 0:
                # Bucket vecino (suavizado: si bucket=5 no visitado, probar 4 y 6)
                for delta in (1, -1, 2, -2):
                    nb = hand_bucket + delta
                    if 0 <= nb < (10 if self.street == 'preflop' else 8):
                        key_nb = (abstract_position, street_idx, nb, board_buckets, ())
                        ss = _blueprint.strategy_sum.get(key_nb)
                        if ss is not None and ss.sum() > 0:
                            break

            if ss is None or ss.sum() == 0:
                return None  # InfoSet no visitado → fallback heurístico

            # ── Samplear de la distribución (NO argmax: el bot debe ser mixto) ─
            avg     = ss / ss.sum()
            chosen  = int(np.random.choice(len(avg), p=avg))
            return ABSTRACT_ACTIONS[chosen]

        except Exception as e:
            log.warning("Blueprint query falló: %s", e)
            return None

    def _translate_abstract_action(self, abstract, to_call, stack, adj):
        """Traduce acción abstracta del CFR a acción concreta del juego.

        Bet sizing basado en las convenciones del entrenamiento:
          raise_extra = ratio × pot  (pot antes de actuar)

        Caps basados en Stack-to-Pot Ratio (SPR) para evitar raises
        irracionales en spots deep-stack donde el blueprint no ha convergido:
          SPR = stack / pot
          SPR > 6 → máximo 1× pot (RAISE_POT)
          SPR > 3 → máximo 2× pot (RAISE_2POT)
          SPR ≤ 3 → all-in permitido

        Ajuste del opponent model: aumentar/reducir frecuencia de bluff
        según el tipo de rival detectado.
        """
        from abstracciones.infoset_encoder import (
            FOLD, CALL, RAISE_THIRD, RAISE_HALF, RAISE_POT, RAISE_2POT, ALLIN,
            RAISE_RATIOS,
        )

        bluff_mult = adj.get('bluff_freq_mult', 1.0)
        spr        = stack / max(self.pot, BB_AMOUNT)   # Stack-to-Pot Ratio

        if abstract == FOLD:
            # Con rival demasiado fold: convertir algunos folds en calls (light call)
            if bluff_mult > 1.3 and to_call < self.pot * 0.3 and random.random() < 0.20:
                abstract = CALL
            else:
                return ('fold', None)

        if abstract == CALL:
            return ('call', None)

        # ── SPR-based downgrade de apuestas grandes ──────────────────────────
        # Con blueprint poco convergido, ALLIN y RAISE_2POT aparecen demasiado
        # frecuentemente. Los caps de SPR los convierten a tamaños razonables.
        if abstract == ALLIN:
            if spr <= 2.5:                    # Short stack: all-in correcto
                return ('all_in', None)
            elif spr <= 5.0:                  # Stack medio: raise 2× pot
                abstract = RAISE_2POT
            else:                             # Deep stack: raise pot
                abstract = RAISE_POT

        if abstract == RAISE_2POT and spr > 6.0:
            abstract = RAISE_POT             # cap 2× pot en spots muy deep

        # ── Calcular size concreto ───────────────────────────────────────────
        ratio       = RAISE_RATIOS.get(abstract, 1.0)
        # Fórmula idéntica al entrenamiento: raise_extra = ratio × pot
        raise_extra = ratio * self.pot
        # Cap al stack disponible (evitar all-in implícito)
        available   = stack - to_call - 0.01
        raise_extra = min(raise_extra, available)

        if raise_extra < BB_AMOUNT:
            return ('call', None)             # size demasiado pequeño → check/call
        if raise_extra + to_call >= stack:
            return ('all_in', None)           # naturalmente all-in
        return ('raise', round(raise_extra, 2))

    def _heuristic_bot_action(self, to_call, stack, adj):
        """
        Heurística de equity cuando no hay blueprint disponible.

        Equity normalizada correctamente:
          preflop : bucket ∈ [0,9]  → eq = bucket / 9   → [0, 1]
          postflop: bucket ∈ [0,49] → eq = bucket / 49  → [0, 1]

        Decisión basada en pot-odds:
          Call   si eq > pot_odds  (EV ≥ 0)
          Raise  si eq > 0.60 + value_adj (mano fuerte)
          Bluff  según bluff_freq_mult del opponent model
          Fold   si eq < pot_odds y sin bluff
        """
        bot = self.players[self.bot_idx]

        if not _ABSTRACTOR_AVAILABLE:
            pot_odds_simple = to_call / max(to_call + self.pot, 0.01)
            return ('call', None) if pot_odds_simple < 0.40 else ('fold', None)

        try:
            from abstracciones.card_abstractor import (
                preflop_bucket, postflop_bucket,
                PREFLOP_BUCKETS, POSTFLOP_BUCKETS,
            )
            if self.community:
                raw = postflop_bucket(bot.mano, self.community, num_sims=80)
                eq  = raw / max(POSTFLOP_BUCKETS - 1, 1)   # [0, 1]
            else:
                raw = preflop_bucket(bot.mano, num_sims=80)
                eq  = raw / max(PREFLOP_BUCKETS - 1, 1)    # [0, 1]
        except Exception:
            eq = 0.45

        pot_odds   = to_call / max(to_call + self.pot, 0.01)
        bluff_mult = adj.get('bluff_freq_mult', 1.0)
        value_adj  = adj.get('value_threshold_adj', 0.0)
        value_thr  = 0.60 + value_adj

        # Para situaciones de todo el stack (all-in o near all-in), exigir
        # un pequeño margen extra sobre las pot-odds para compensar la varianza
        # de la estimación Monte Carlo (80 sims) del bucket de equity.
        #   bet_fraction > 50 % del stack → margen 4 %
        #   bet_fraction > 15 % del stack → margen 2 %
        allin_margin = 0.0
        if bot.fichas > 0:
            bet_fraction = to_call / max(bot.fichas + to_call, 0.01)
            if bet_fraction > 0.50:
                allin_margin = 0.04
            elif bet_fraction > 0.15:
                allin_margin = 0.02

        if to_call == 0:
            # Sin apuesta: check o bet
            if eq > value_thr:
                bet = round(min(self.pot * 0.65, stack), 2)
                if self.raise_count < MAX_RAISES and bet >= BB_AMOUNT:
                    return ('raise', bet)
            # Bluff semi-polarizado
            if random.random() < 0.22 * bluff_mult:
                bet = round(min(self.pot * 0.45, stack), 2)
                if self.raise_count < MAX_RAISES and bet >= BB_AMOUNT:
                    return ('raise', bet)
            return ('call', None)  # check

        # Facing a bet
        if eq > pot_odds + allin_margin:
            # Raise por valor si la mano es fuerte
            if eq > value_thr and self.raise_count < MAX_RAISES:
                bet = round(min(self.pot * 0.65, stack - to_call), 2)
                if bet >= BB_AMOUNT:
                    return ('raise', bet)
            return ('call', None)

        # Fold (con pequeña frecuencia de bluff-call para no ser explotable)
        if random.random() < min(0.12 * bluff_mult, 0.25):
            return ('call', None)

        return ('fold', None)

    # ── Privado: acciones válidas ─────────────────────────────────────────────

    def _compute_valid_actions(self, player_idx):
        """Devuelve lista de acciones válidas para el jugador dado."""
        player  = self.players[player_idx]
        to_call = max(0.0, self.current_bet - self.contributions[player_idx])
        actions = []

        if to_call > 0:
            actions.append({"tipo": "fold"})
            call_amt = min(to_call, player.fichas)
            actions.append({"tipo": "call", "amount": round(call_amt, 2)})
        else:
            actions.append({"tipo": "check", "amount": 0})

        if player.fichas > to_call and self.raise_count < MAX_RAISES:
            min_raise = max(BB_AMOUNT, self.current_bet)
            remaining = player.fichas - to_call
            presets = []
            for ratio in [1/3, 1/2, 2/3, 1.0, 1.5, 2.0]:
                amt = round(ratio * self.pot, 2)
                if BB_AMOUNT <= amt <= remaining:
                    presets.append({"tipo": "raise", "amount": amt,
                                    "label": f"{ratio:.0%} pot" if ratio < 1 else
                                             ("pot" if ratio == 1.0 else f"{ratio}× pot")})
            if presets:
                actions.append({
                    "tipo"    : "raise_options",
                    "presets" : presets,
                    "min"     : round(min_raise, 2),
                    "max"     : round(remaining, 2),
                })

        actions.append({"tipo": "all_in", "amount": round(player.fichas, 2)})
        return actions

    # ── Privado: observar acción del humano ───────────────────────────────────

    def _observe_human_action(self, tipo):
        voluntarily = tipo in ('call', 'raise', 'all_in')
        is_raise    = tipo in ('raise', 'all_in')
        self.opp_model.observe_action(
            'raise' if is_raise else tipo,
            self.street,
            voluntarily=voluntarily,
        )


# ── Utilidad: baraja completa ─────────────────────────────────────────────────

def _full_deck():
    """Retorna las 52 cartas en formato compact (e.g. 'Ah', 'Ks')."""
    ranks = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
    suits = ['s','h','d','c']
    return [r + s for r in ranks for s in suits]


# ═══════════════════════════════════════════════════════════════════════════════
#  FLASK APP
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__, template_folder='templates', static_folder='static')

# Usar una SECRET_KEY estable (desde variable de entorno o archivo local).
# Un valor fijo garantiza que las cookies de sesión sobreviven reinicios del
# servidor; un valor aleatorio invalida todas las sesiones al reiniciar.
_SECRET_KEY_FILE = os.path.join(_HERE, '.secret_key')

def _load_or_create_secret_key() -> str:
    """Lee la clave de POKER_SECRET_KEY, del archivo .secret_key, o la crea."""
    key = os.environ.get('POKER_SECRET_KEY')
    if key:
        return key
    if os.path.exists(_SECRET_KEY_FILE):
        with open(_SECRET_KEY_FILE) as f:
            key = f.read().strip()
        if key:
            return key
    key = secrets.token_hex(32)
    try:
        with open(_SECRET_KEY_FILE, 'w') as f:
            f.write(key)
    except OSError:
        pass
    return key

app.secret_key = _load_or_create_secret_key()

_sessions: dict = {}   # sid → PartidaWeb


def _get_session() -> PartidaWeb:
    """Obtiene o crea la PartidaWeb de la sesión actual."""
    sid = session.get('sid')
    if not sid or sid not in _sessions:
        sid = str(uuid.uuid4())
        session['sid']  = sid
        _sessions[sid]  = PartidaWeb()
    return _sessions[sid]


# ── Rutas ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/nueva_partida', methods=['POST'])
def nueva_partida():
    """Crea sesión nueva y reparte la primera mano."""
    sid = str(uuid.uuid4())
    session['sid'] = sid
    partida = PartidaWeb()
    _sessions[sid] = partida
    estado = partida.nueva_mano()
    return jsonify(estado)


@app.route('/api/estado', methods=['GET'])
def estado():
    """Retorna el estado actual de la sesión."""
    partida = _get_session()
    return jsonify(partida.get_state_json())


@app.route('/api/accion', methods=['POST'])
def accion():
    """Aplica la acción del humano: {tipo, amount?}."""
    partida = _get_session()
    data    = request.get_json(silent=True) or {}
    tipo    = data.get('tipo', 'fold')
    amount  = data.get('amount', None)
    if amount is not None:
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            amount = None
    estado = partida.accion(tipo, amount)
    return jsonify(estado)


@app.route('/api/nueva_mano', methods=['POST'])
def nueva_mano():
    """Reparte la siguiente mano (llamar tras ver resultado)."""
    partida = _get_session()
    estado  = partida.nueva_mano()
    return jsonify(estado)


@app.route('/api/stats', methods=['GET'])
def stats():
    """Devuelve estadísticas del opponent model."""
    partida = _get_session()
    s = partida.opp_model.stats
    clasificacion = partida.opp_model.classify()
    ajustes = partida.opp_model.get_counter_adjustments()
    return jsonify({
        "hands_seen"   : s.hands_seen,
        "vpip"         : round(s.vpip, 3),
        "pfr"          : round(s.pfr, 3),
        "af_total"     : round(s.af(), 3),
        "ftb_total"    : round(s.ftb(), 3),
        "wtsd"         : round(s.wtsd, 3),
        "clasificacion": clasificacion,
        "ajustes"      : ajustes,
    })


@app.route('/api/recargar_blueprint', methods=['POST'])
def recargar_blueprint():
    """Recarga el blueprint desde disco (útil tras continuar entrenamiento)."""
    _load_blueprint()
    return jsonify({
        "ok"     : _blueprint is not None,
        "infosets": len(_blueprint.regret_sum) if _blueprint else 0,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Servidor Poker IA')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    print(f"\n  Servidor iniciado en http://localhost:{args.port}")
    print(f"  Blueprint: {'Cargado (%d InfoSets)' % len(_blueprint.regret_sum) if _blueprint else 'No disponible (modo heurístico)'}")
    print(f"  Entrenamiento: python pre_entrenamiento.py --iters 200000 --resume\n")
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)
