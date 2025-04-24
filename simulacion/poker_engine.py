import random
import json
import logging
logging.disable(logging.CRITICAL)   #comentamos todos los log que hemos metido para depurar
from template import eval_hand_from_strings

# Configuración global de logging (se eliminan muchos DEBUG de iteración)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
    force=True
)

def compact_card(carta):
    valor_map = {'10': 'T', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A'}
    v = valor_map.get(carta.valor, str(carta.valor))
    palo_map = {'Corazones': 'h', 'Picas': 's', 'Trebol': 'c', 'Diamante': 'd'}
    return v + palo_map[carta.palo]

### Clases básicas (Carta, Baraja, Jugador)
class Carta:
    def __init__(self, palo, valor):
        self.palo = palo
        self.valor = valor
    def __str__(self):
        return f"{self.valor} de {self.palo}"
    def __repr__(self):
        return self.__str__()

class Baraja:
    def __init__(self):
        palos = ["Corazones", "Picas", "Trebol", "Diamante"]
        valores = [str(n) for n in range(2, 11)] + ["J", "Q", "K", "A"]
        self.cartas = [Carta(p, v) for p in palos for v in valores]
        self.mezclar()
    def mezclar(self):
        random.shuffle(self.cartas)
    def repartir(self, cantidad):
        if cantidad > len(self.cartas):
            raise ValueError("No hay suficientes cartas para repartir")
        return [self.cartas.pop() for _ in range(cantidad)]

class Jugador:
    def __init__(self, nombre, fichas=100, id=0):
        self.nombre = nombre
        self.fichas = fichas
        self.mano = []
        self.id = id
    def recibir_cartas(self, cartas):
        self.mano.extend(cartas)
    def get_state(self):
        return [self.id, self.fichas, [compact_card(c) for c in self.mano]]

### Clase Mesa (no modificada esencialmente)
class Mesa:
    def __init__(self, nombres_jugadores):
        self.baraja = Baraja()
        self.jugadores = [Jugador(nombre, id=i) for i, nombre in enumerate(nombres_jugadores)]
        self.community_cards = []
        self.pot = 0.0
        self.history = []
    def log_event(self, event):
        self.history.append(event)
    def repartir_manos(self):
        for j in self.jugadores:
            cartas = self.baraja.repartir(2)
            j.recibir_cartas(cartas)
            self.log_event(["d", j.id, compact_card(cartas[0]), compact_card(cartas[1])])
    def get_state(self):
        return [self.pot, [j.get_state() for j in self.jugadores],
                [compact_card(c) for c in self.community_cards]]
    def repartir_flop(self):
        self.community_cards = self.baraja.repartir(3)
        self.log_event(["D", 0, *[compact_card(c) for c in self.community_cards]])
    def repartir_turn(self):
        turn_card = self.baraja.repartir(1)
        self.community_cards.extend(turn_card)
        self.log_event(["T", 0, compact_card(turn_card[0])])
    def repartir_river(self):
        river_card = self.baraja.repartir(1)
        self.community_cards.extend(river_card)
        self.log_event(["R", 0, compact_card(river_card[0])])
    def iniciar_ronda_apuestas(self, fase, action_callback):
        # Para fases: se crea la ronda de apuestas.
        if fase.lower() == "preflop":
            ronda = RondaApuestasPreflop(self.jugadores[0], self.jugadores[1], fase, self.pot, action_callback)
        else:
            ronda = RondaApuestasPostflop(self.jugadores[0], self.jugadores[1], fase, self.pot, action_callback)
        outcome = ronda.ejecutar()
        self.pot = ronda.pot
        self.history.extend(ronda.history)
        return outcome

class RondaApuestasPreflop:
    def __init__(self, sb, bb, fase, initial_pot, action_callback):
        self.small_blind = sb
        self.big_blind = bb
        self.fase = fase.lower()
        self.pot = initial_pot
        self.current_bet = 0.0
        self.contributions = {sb.id: 0.0, bb.id: 0.0}
        self.raise_counter = {sb.id: 0, bb.id: 0}
        self.raise_total_max = 2
        self.phase_outcome = None
        self.all_in_occurred = False
        self.history = []
        self.action_callback = action_callback if action_callback else default_action_callback
        self._inicializar_preflop()

    def log(self, event):
        self.history.append(event)

    def get_state(self):
        return {
            "fase": self.fase,
            "pot": self.pot,
            "current_bet": self.current_bet,
            "contributions": self.contributions,
            "stack_small_blind": self.small_blind.fichas,
            "stack_big_blind": self.big_blind.fichas,
            "hand_small_blind": [compact_card(c) for c in self.small_blind.mano],
            "hand_big_blind": [compact_card(c) for c in self.big_blind.mano],
            "community": []
        }

    def _inicializar_preflop(self):
        sb_amount = 0.5
        bb_amount = 1.0
        self.current_bet = bb_amount
        self.small_blind.fichas -= sb_amount
        self.big_blind.fichas -= bb_amount
        self.pot += (sb_amount + bb_amount)
        self.contributions[self.small_blind.id] += sb_amount
        self.contributions[self.big_blind.id] += bb_amount
        self.log(["s", self.small_blind.id, sb_amount])
        self.log(["b", self.big_blind.id, bb_amount])
        self.log(["p", self.pot])
        self.estado = "SB_TURN"

    def get_valid_actions(self, jugador):
        to_call = self.current_bet - self.contributions[jugador.id]
        base_actions = ["fold", "call", "all in"] if to_call > 0 else ["check", "all in"]

        if self.raise_counter.get(jugador.id, 0) < self.raise_total_max:
            tamanos = [self.pot / 3, self.pot / 2, (2 * self.pot) / 3, self.pot, 1.5 * self.pot, 2 * self.pot]
            raise_opciones = []
            for tam in tamanos:
                raise_total = round(to_call + tam, 2)
                if raise_total >= 2.0:
                    raise_opciones.append(("raise", raise_total))
            return base_actions + raise_opciones
        else:
            return base_actions

    def _procesar_apuesta(self, jugador, tipo, all_in=False, raise_amount=None):
        to_call = self.current_bet - self.contributions[jugador.id]

        if tipo == "call":
            total = min(to_call, jugador.fichas)
            if total == jugador.fichas:
                all_in = True
            jugador.fichas -= total
            self.contributions[jugador.id] += total
            self.pot += total
            if self.contributions[jugador.id] > self.current_bet:
                self.current_bet = self.contributions[jugador.id]
            self.all_in_occurred |= all_in
            self.log(["c", jugador.id, total])
            return {"tipo": "call", "all_in": all_in, "to_call": to_call, "total": total}

        elif tipo == "raise":
            raise_total = raise_amount
            total = to_call + raise_total
            if total >= jugador.fichas:
                total = jugador.fichas
                raise_total = total - to_call
                all_in = True
            jugador.fichas -= total
            self.contributions[jugador.id] += total
            self.pot += total
            self.current_bet = self.contributions[jugador.id]
            self.all_in_occurred |= all_in

            if all_in:
                self.log(["all in", jugador.id, total])
                return {"tipo": "all in", "all_in": True, "to_call": to_call, "total": total}
            else:
                self.log(["r", jugador.id, raise_total, total])
                return {"tipo": "raise", "all_in": False, "to_call": to_call, "raise": raise_total, "total": total}

        elif tipo == "all in":
            total = jugador.fichas
            jugador.fichas = 0
            self.contributions[jugador.id] += total
            self.pot += total
            if self.contributions[jugador.id] > self.current_bet:
                self.current_bet = self.contributions[jugador.id]
            self.all_in_occurred = True
            self.log(["all in", jugador.id, total])
            return {"tipo": "all in", "all_in": True, "to_call": to_call, "total": total}

        else:  # fold
            return {"tipo": "fold"}

    def _accion_sb(self):
        valid_actions = self.get_valid_actions(self.small_blind)
        accion = self.action_callback(self.small_blind, self.get_state(), valid_actions)
        if self.current_bet - self.contributions[self.small_blind.id] == 0 and accion == "call":
            accion = "check"
        if accion == "fold":
            self.phase_outcome = "fold"
            self.big_blind.fichas += self.pot
            return False
        elif accion in ["call", "check"]:
            self._procesar_apuesta(self.small_blind, "call")
            self.estado = "BB_TURN"
            return True
        elif accion == "all in":
            self._procesar_apuesta(self.small_blind, "all in")
            self.phase_outcome = "all in"
            return False
        elif isinstance(accion, tuple) and accion[0] == "raise":
            self._procesar_apuesta(self.small_blind, "raise", raise_amount=accion[1])
            self.raise_counter[self.small_blind.id] += 1
            self.estado = "BB_TURN"
            return True
        else:
            self.phase_outcome = "fold"
            self.big_blind.fichas += self.pot
            return False

    def _accion_bb_preflop(self):
        valid_actions = self.get_valid_actions(self.big_blind)
        accion = self.action_callback(self.big_blind, self.get_state(), valid_actions)
        if self.current_bet - self.contributions[self.big_blind.id] == 0 and accion == "call":
            accion = "check"
        if accion == "fold":
            self.phase_outcome = "fold"
            self.small_blind.fichas += self.pot
            return False
        elif accion in ["call", "check"]:
            self._procesar_apuesta(self.big_blind, "call")
            if self.current_bet > self.contributions[self.small_blind.id]:
                self.estado = "SB_TURN"
                return True
            self.phase_outcome = self.fase
            return False
        elif accion == "all in":
            self._procesar_apuesta(self.big_blind, "all in")
            self.phase_outcome = "all in"
            return False
        elif isinstance(accion, tuple) and accion[0] == "raise":
            self._procesar_apuesta(self.big_blind, "raise", raise_amount=accion[1])
            self.raise_counter[self.big_blind.id] += 1
            self.estado = "SB_TURN"
            return True
        else:
            self._procesar_apuesta(self.big_blind, "call")
            self.phase_outcome = self.fase
            return False

    def _ejecutar_preflop(self):
        while True:
            if self.estado == "SB_TURN":
                if not self._accion_sb():
                    break
            elif self.estado == "BB_TURN":
                if not self._accion_bb_preflop():
                    break

    def ejecutar(self):
        self._ejecutar_preflop()
        if self.phase_outcome != "fold" and self.all_in_occurred:
            self.phase_outcome = "all in"
        self.log(["e", self.phase_outcome, self.pot])
        return self.phase_outcome


### Ronda de apuestas Postflop (diferente: turno inicia en BB, luego SB)
class RondaApuestasPreflop:
    def __init__(self, sb, bb, fase, initial_pot, action_callback):
        self.small_blind = sb
        self.big_blind = bb
        self.fase = fase.lower()
        self.pot = initial_pot
        self.current_bet = 0.0
        self.contributions = {sb.id: 0.0, bb.id: 0.0}
        self.raise_counter = {sb.id: 0, bb.id: 0}
        self.raise_total_max = 2
        self.phase_outcome = None
        self.all_in_occurred = False
        self.history = []
        self.action_callback = action_callback if action_callback else default_action_callback
        self._inicializar_preflop()

    def log(self, event):
        self.history.append(event)

    def get_state(self):
        return {
            "fase": self.fase,
            "pot": self.pot,
            "current_bet": self.current_bet,
            "contributions": self.contributions,
            "stack_small_blind": self.small_blind.fichas,
            "stack_big_blind": self.big_blind.fichas,
            "hand_small_blind": [compact_card(c) for c in self.small_blind.mano],
            "hand_big_blind": [compact_card(c) for c in self.big_blind.mano],
            "community": []
        }

    def _inicializar_preflop(self):
        sb_amount = 0.5
        bb_amount = 1.0
        self.current_bet = bb_amount
        self.small_blind.fichas -= sb_amount
        self.big_blind.fichas -= bb_amount
        self.pot += (sb_amount + bb_amount)
        self.contributions[self.small_blind.id] += sb_amount
        self.contributions[self.big_blind.id] += bb_amount
        self.log(["s", self.small_blind.id, sb_amount])
        self.log(["b", self.big_blind.id, bb_amount])
        self.log(["p", self.pot])
        self.estado = "SB_TURN"

    def get_valid_actions(self, jugador):
        to_call = self.current_bet - self.contributions[jugador.id]
        base_actions = ["fold", "call", "all in"] if to_call > 0 else ["check", "all in"]

        if self.raise_counter.get(jugador.id, 0) < self.raise_total_max:
            tamanos = [self.pot / 3, self.pot / 2, (2 * self.pot) / 3, self.pot, 1.5 * self.pot, 2 * self.pot]
            raise_opciones = []
            for tam in tamanos:
                raise_total = round(to_call + tam, 2)
                if raise_total >= 2.0:
                    raise_opciones.append(("raise", raise_total))
            return base_actions + raise_opciones
        else:
            return base_actions

    def _procesar_apuesta(self, jugador, tipo, all_in=False, raise_amount=None):
        to_call = self.current_bet - self.contributions[jugador.id]

        if tipo == "call":
            total = min(to_call, jugador.fichas)
            if total == jugador.fichas:
                all_in = True
            jugador.fichas -= total
            self.contributions[jugador.id] += total
            self.pot += total
            if self.contributions[jugador.id] > self.current_bet:
                self.current_bet = self.contributions[jugador.id]
            self.all_in_occurred |= all_in
            self.log(["c", jugador.id, total])
            return {"tipo": "call", "all_in": all_in, "to_call": to_call, "total": total}

        elif tipo == "raise":
            raise_total = raise_amount
            total = to_call + raise_total
            if total >= jugador.fichas:
                total = jugador.fichas
                raise_total = total - to_call
                all_in = True
            jugador.fichas -= total
            self.contributions[jugador.id] += total
            self.pot += total
            self.current_bet = self.contributions[jugador.id]
            self.all_in_occurred |= all_in

            if all_in:
                self.log(["all in", jugador.id, total])
                return {"tipo": "all in", "all_in": True, "to_call": to_call, "total": total}
            else:
                self.log(["r", jugador.id, raise_total, total])
                return {"tipo": "raise", "all_in": False, "to_call": to_call, "raise": raise_total, "total": total}

        elif tipo == "all in":
            total = jugador.fichas
            jugador.fichas = 0
            self.contributions[jugador.id] += total
            self.pot += total
            if self.contributions[jugador.id] > self.current_bet:
                self.current_bet = self.contributions[jugador.id]
            self.all_in_occurred = True
            self.log(["all in", jugador.id, total])
            return {"tipo": "all in", "all_in": True, "to_call": to_call, "total": total}

        else:
            return {"tipo": "fold"}

    def _accion_sb(self):
        valid_actions = self.get_valid_actions(self.small_blind)
        accion = self.action_callback(self.small_blind, self.get_state(), valid_actions)
        if self.current_bet - self.contributions[self.small_blind.id] == 0 and accion == "call":
            accion = "check"
        if accion == "fold":
            self.phase_outcome = "fold"
            self.big_blind.fichas += self.pot
            return False
        elif accion in ["call", "check"]:
            self._procesar_apuesta(self.small_blind, "call")
            self.estado = "BB_TURN"
            return True
        elif accion == "all in":
            self._procesar_apuesta(self.small_blind, "all in")
            self.phase_outcome = "all in"
            return False
        elif isinstance(accion, tuple) and accion[0] == "raise":
            self._procesar_apuesta(self.small_blind, "raise", raise_amount=accion[1])
            self.raise_counter[self.small_blind.id] += 1
            self.estado = "BB_TURN"
            return True
        else:
            self.phase_outcome = "fold"
            self.big_blind.fichas += self.pot
            return False

    def _accion_bb_preflop(self):
        valid_actions = self.get_valid_actions(self.big_blind)
        accion = self.action_callback(self.big_blind, self.get_state(), valid_actions)
        if self.current_bet - self.contributions[self.big_blind.id] == 0 and accion == "call":
            accion = "check"
        if accion == "fold":
            self.phase_outcome = "fold"
            self.small_blind.fichas += self.pot
            return False
        elif accion in ["call", "check"]:
            self._procesar_apuesta(self.big_blind, "call")
            if self.current_bet > self.contributions[self.small_blind.id]:
                self.estado = "SB_TURN"
                return True
            self.phase_outcome = self.fase
            return False
        elif accion == "all in":
            self._procesar_apuesta(self.big_blind, "all in")
            self.phase_outcome = "all in"
            return False
        elif isinstance(accion, tuple) and accion[0] == "raise":
            self._procesar_apuesta(self.big_blind, "raise", raise_amount=accion[1])
            self.raise_counter[self.big_blind.id] += 1
            self.estado = "SB_TURN"
            return True
        else:
            self._procesar_apuesta(self.big_blind, "call")
            self.phase_outcome = self.fase
            return False

    def _ejecutar_preflop(self):
        while True:
            if self.estado == "SB_TURN":
                if not self._accion_sb():
                    break
            elif self.estado == "BB_TURN":
                if not self._accion_bb_preflop():
                    break

    def ejecutar(self):
        self._ejecutar_preflop()
        if self.phase_outcome != "fold" and self.all_in_occurred:
            self.phase_outcome = "all in"
        self.log(["e", self.phase_outcome, self.pot])
        return self.phase_outcome


class RondaApuestasPostflop(RondaApuestasPreflop):
    def __init__(self, sb, bb, fase, initial_pot, action_callback):
        super().__init__(sb, bb, fase, initial_pot, action_callback)
        self.estado = "BB_TURN"

    def _accion(self, jugador, rival):
        valid_actions = self.get_valid_actions(jugador)
        accion = self.action_callback(jugador, self.get_state(), valid_actions)
        if self.current_bet - self.contributions[jugador.id] == 0 and accion == "call":
            accion = "check"

        if accion == "fold":
            self.phase_outcome = "fold"
            rival.fichas += self.pot
            return False

        elif accion == "check":
            self._procesar_apuesta(jugador, "call")
            if jugador.id == self.small_blind.id and self.estado == "SB_TURN":
                self.phase_outcome = self.fase
                return False
            else:
                self.estado = "SB_TURN" if jugador.id == self.big_blind.id else "BB_TURN"
                return True

        elif accion == "call":
            self._procesar_apuesta(jugador, "call")
            self.phase_outcome = self.fase
            return False

        elif accion == "all in":
            self._procesar_apuesta(jugador, "all in")
            self.phase_outcome = "all in"
            return False

        elif isinstance(accion, tuple) and accion[0] == "raise":
            self._procesar_apuesta(jugador, "raise", raise_amount=accion[1])
            self.raise_counter[jugador.id] += 1
            self.estado = "SB_TURN" if jugador.id == self.big_blind.id else "BB_TURN"
            return True

        else:
            self._procesar_apuesta(jugador, "call")
            self.phase_outcome = self.fase
            return False

    def _ejecutar_postflop(self):
        while True:
            if self.estado == "BB_TURN":
                if not self._accion(self.big_blind, self.small_blind):
                    break
            elif self.estado == "SB_TURN":
                if not self._accion(self.small_blind, self.big_blind):
                    break

    def ejecutar(self):
        self._ejecutar_postflop()
        if self.phase_outcome != "fold" and self.all_in_occurred:
            self.phase_outcome = "all in"
        self.log(["e", self.phase_outcome, self.pot])
        return self.phase_outcome


class PokerCoreEngine:
    """
    Core engine de póker optimizado.
    Provee métodos: reset, get_state, play_round y save_history.
    """
    def __init__(self, nombres_jugadores=["ALAVES", "LASPALMAS"], action_callback=None):
        self.nombres_jugadores = nombres_jugadores
        self.action_callback = action_callback if action_callback else default_action_callback
        self.mesa = Mesa(nombres_jugadores)
        self.history = []

    def reset(self):
        self.mesa = Mesa(self.nombres_jugadores)
        self.history = []
        self.mesa.log_event(["R"])  # Registro de reinicio
        logging.info("Juego reiniciado.")
        return self.get_state()

    def get_state(self):
        return self.mesa.get_state()

    def save_history(self, filename="poker_history.json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4, ensure_ascii=False)
        logging.info("Historial guardado en %s.", filename)

    def play_round(self):
        # Preflop
        self.mesa.log_event(["phase", "preflop"])
        self.mesa.repartir_manos()
        state_preflop = self.get_state()
        self.mesa.log_event(["state", state_preflop])
        outcome = self.mesa.iniciar_ronda_apuestas("preflop", self.action_callback)

        if outcome == "fold":
            self.mesa.log_event(["phase_end", "preflop", "fold"])
        else:
            sb = self.mesa.jugadores[0]
            bb = self.mesa.jugadores[1]
            postflop_sb = bb  # fuera de posición
            postflop_bb = sb  # en posición

            # Flop
            self.mesa.repartir_flop()
            r_flop = RondaApuestasPostflop(postflop_sb, postflop_bb, "flop", self.mesa.pot, self.action_callback)
            outcome_flop = r_flop.ejecutar()
            self.mesa.history.extend(r_flop.history)
            self.mesa.pot = r_flop.pot

            if outcome_flop == "fold":
                self.mesa.log_event(["phase_end", "flop", "fold"])
            elif outcome_flop == "all in":
                self.mesa.repartir_turn()
                self.mesa.repartir_river()
            else:
                # Turn
                self.mesa.repartir_turn()
                r_turn = RondaApuestasPostflop(postflop_sb, postflop_bb, "turn", self.mesa.pot, self.action_callback)
                outcome_turn = r_turn.ejecutar()
                self.mesa.history.extend(r_turn.history)
                self.mesa.pot = r_turn.pot

                if outcome_turn == "all in":
                    self.mesa.repartir_river()
                else:
                    # River
                    self.mesa.repartir_river()
                    r_river = RondaApuestasPostflop(postflop_sb, postflop_bb, "river", self.mesa.pot, self.action_callback)
                    outcome_river = r_river.ejecutar()
                    self.mesa.history.extend(r_river.history)
                    self.mesa.pot = r_river.pot

        # Showdown
        if len(self.mesa.community_cards) == 5:
            self.mesa.log_event(["phase", "showdown"])
            board = [compact_card(c) for c in self.mesa.community_cards]
            scores = []
            for j in self.mesa.jugadores:
                mano = [compact_card(c) for c in j.mano]
                score = eval_hand_from_strings(mano, board)
                scores.append(score)
                self.mesa.log_event(["x", j.id, score])

            if scores[0] == scores[1]:
                self.mesa.log_event(["z", self.mesa.pot])
                self.mesa.jugadores[0].fichas += self.mesa.pot / 2
                self.mesa.jugadores[1].fichas += self.mesa.pot / 2
            elif scores[0] > scores[1]:
                self.mesa.log_event(["w", self.mesa.jugadores[0].id, self.mesa.pot])
                self.mesa.jugadores[0].fichas += self.mesa.pot
            else:
                self.mesa.log_event(["w", self.mesa.jugadores[1].id, self.mesa.pot])
                self.mesa.jugadores[1].fichas += self.mesa.pot

        state_final = self.get_state()
        self.mesa.log_event(["state_final", state_final])
        self.history.extend(self.mesa.history)
        logging.info("Partida finalizada. Estado final: %s", state_final)
        return state_final

    def reset_to_state(self, pot, jugadores_info, community_cards, fase="preflop"):
        """
        Reinicia el engine a un estado arbitrario.
        - pot: valor numérico del bote.
        - jugadores_info: lista de dicts con claves 'id', 'fichas' y 'mano' (lista de strings tipo 'Ah','Kd' o None).
        - community_cards: lista de strings (['As','7d',…]) o [].
        - fase: 'preflop', 'flop', 'turn' o 'river'.
        """
        self.mesa = Mesa(self.nombres_jugadores)
        self.history = []
        self.mesa.pot = pot
        for info in jugadores_info:
            j = self.mesa.jugadores[info["id"]]
            j.fichas = info["fichas"]
            j.mano = []

        self.mesa.baraja.mezclar()

        def _take(card_str):
            for c in self.mesa.baraja.cartas:
                if compact_card(c) == card_str:
                    self.mesa.baraja.cartas.remove(c)
                    return c
            raise ValueError(f"Carta {card_str} no encontrada en la baraja")

        for info in jugadores_info:
            if info.get("mano"):
                j = self.mesa.jugadores[info["id"]]
                for cs in info["mano"]:
                    carta = _take(cs)
                    j.recibir_cartas([carta])

        self.mesa.community_cards = []
        for cs in community_cards:
            carta = _take(cs)
            self.mesa.community_cards.append(carta)

        self.mesa.log_event(["phase", fase])
        self.mesa.log_event(["state", self.get_state()])
        return self.get_state()



# --- Callback de acción por defecto ---
def default_action_callback(jugador, state, valid_actions):
    """
    Devuelve una acción aleatoria de valid_actions.
    Si la acción es raise, retorna ("raise", monto) aleatorio.
    """
    choice = random.choice(valid_actions)
    if isinstance(choice, tuple) and choice[0] == "raise":
        min_raise = choice[1]
        max_raise = jugador.fichas if jugador.fichas > min_raise else min_raise
        monto = round(random.uniform(min_raise, max_raise), 2)
        return ("raise", monto)
    else:
        return choice
