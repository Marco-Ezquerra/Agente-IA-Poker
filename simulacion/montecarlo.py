# montecarlo.py

import random
from itertools import combinations
from template import eval_hand_from_strings
from poker_engine import compact_card

# === CACHE GLOBAL DE EQUITY ===
_equity_cache = {}

# Funciones de equity Montecarlo

def generar_baraja_compacta():
    palos = ['h', 'd', 'c', 's']
    valores = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    return [v + p for v in valores for p in palos]


def montecarlo_equity(hole_cards, community_cards, num_players=2, num_simulations=500):
    wins = ties = losses = 0
    deck_base = generar_baraja_compacta()
    # eliminar cartas visibles
    for card in hole_cards + community_cards:
        if card in deck_base:
            deck_base.remove(card)
    for _ in range(num_simulations):
        deck = deck_base.copy()
        # repartir a oponentes
        opponents = []
        for _ in range(num_players - 1):
            opp = random.sample(deck, 2)
            for c in opp:
                deck.remove(c)
            opponents.append(opp)
        # completar board
        board = community_cards.copy()
        board += random.sample(deck, 5 - len(board))
        # evaluar
        my_score = eval_hand_from_strings(hole_cards, board)
        best = my_score
        tie = False
        for opp in opponents:
            score = eval_hand_from_strings(opp, board)
            if score > best:
                best = score; tie = False
            elif score == best:
                tie = True
        if my_score == best and not tie:
            wins += 1
        elif tie:
            ties += 1
        else:
            losses += 1
    total = wins + ties + losses
    return (wins + 0.5 * ties) / total if total else 0


def get_equity_cached(hole_cards, community_cards, num_players=2, num_simulations=500):
    key = (tuple(sorted(hole_cards)), tuple(sorted(community_cards)), num_players, num_simulations)
    if key in _equity_cache:
        return _equity_cache[key]
    eq = montecarlo_equity(hole_cards, community_cards, num_players, num_simulations)
    _equity_cache[key] = eq
    return eq

# === IA avanzada ===

def advanced_ia_action_callback(jugador, state, valid_actions):
    """
    IA basada en equity Montecarlo ajustada por fase (preflop, flop, turn, river).
    Usa presupuestos de simulación diferentes por fase.
    """
    my_cards = [compact_card(c) for c in jugador.mano]
    fase = state.get('fase', 'preflop')
    community = state.get('community', [])
    pot = state.get('pot', 0)
    current = state.get('current_bet', 0)
    contrib = state.get('contributions', {})
    to_call = current - contrib.get(jugador.id, 0)

    # Presupuesto de simulaciones por fase
    sims_map = {
        'preflop': 500,
        'flop': 1000,
        'turn': 2000,
        'river': 5000
    }
    num_sims = sims_map.get(fase, 500)

    # Calcular equity según fase y board actual
    equity = get_equity_cached(my_cards, community, num_players=2, num_simulations=num_sims)

    # Acciones de raise disponibles
    raises = [a for a in valid_actions if isinstance(a, tuple) and a[0] == 'raise']
    allowed = sorted([amt for _, amt in raises])
    pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0

    def pick_raise(ratio):
        if not allowed:
            return None
        target = ratio * pot
        return min(allowed, key=lambda x: abs(x - target))

    # Lógica de decisión según fase
    if to_call > 0:
        if fase == 'preflop':
            if equity > 0.65:
                action = ('raise', pick_raise(1.0)) if allowed else 'call'
            elif equity > 0.45:
                action = 'call'
            else:
                action = 'fold'
        elif fase == 'flop':
            if equity > 0.75:
                action = ('raise', pick_raise(1.2)) if allowed else 'call'
            elif equity < pot_odds:
                action = ('raise', pick_raise(0.4)) if allowed and random.random() < 0.35 else 'fold'
            elif equity > 0.5:
                action = 'call'
            else:
                action = 'fold'
        elif fase == 'turn':
            if equity > 0.80:
                action = ('raise', pick_raise(1.3)) if allowed else 'call'
            elif equity < pot_odds:
                action = ('raise', pick_raise(0.5)) if allowed and random.random() < 0.20 else 'fold'
            else:
                action = 'call'
        else:  # river
            if equity > 0.85:
                action = ('raise', pick_raise(1.5)) if allowed else 'call'
            elif equity < pot_odds:
                action = ('raise', pick_raise(0.5)) if allowed and random.random() < 0.10 else 'fold'
            else:
                action = 'call'
    else:
        # Sin coste de llamada (to_call == 0)
        if equity > 0.5:
            action = ('raise', pick_raise(0.5)) if allowed and random.random() < 0.60 else 'check'
        elif equity < 0.10:
            # bluff ocasionales
            prob_bluff = {'flop': 0.25, 'turn': 0.15, 'river': 0.10}.get(fase, 0.10)
            if random.random() < prob_bluff:
                action = ('raise', pick_raise(1.0)) if allowed else 'check'
            else:
                action = 'check'
        else:
            action = 'check'

    # Etiquetado de tipo de jugada
    tipo = (
        'valor' if isinstance(action, tuple) and action[0] == 'raise' and equity > 0.65 else
        'bluff' if isinstance(action, tuple) and action[0] == 'raise' else
        'neutral' if action == 'call' and 0.4 < equity < 0.7 else
        'defensivo' if action == 'call' and equity < 0.4 else
        'pasivo' if action in ['check', 'fold'] else 'otro'
    )

    log_entry = ['ia', jugador.id, fase, round(equity, 4), my_cards, community,
                 action if isinstance(action, str) else action[0], tipo]
    state.setdefault('history', []).append(log_entry)
    state.setdefault('ia_logs', []).append(log_entry)
    return action

# BOT humano simple

def human_bot_action_callback(jugador, state, valid_actions):
    my_cards=[compact_card(c) for c in jugador.mano]
    community=state.get('community',[])
    pot=state.get('pot',0)
    current=state.get('current_bet',0)
    contrib=state.get('contributions',{})
    to_call=current-contrib.get(jugador.id,0)
    equity=get_equity_cached(my_cards,community,2,1000)
    pot_odds=to_call/(pot+to_call) if (pot+to_call)>0 else 0
    raises=[a for a in valid_actions if isinstance(a,tuple) and a[0]=='raise']
    if to_call>0:
        if equity<pot_odds:
            if random.random()<0.25 and raises:
                allowed=sorted([amt for _,amt in raises])
                decision=('raise',random.choice(allowed[:3]))
            else: decision='call'
        elif equity<pot_odds+0.05: decision='call'
        else:
            if raises:
                allowed=sorted([amt for _,amt in raises])
                extra=(equity-pot_odds)*pot
                decision=('raise',max(allowed[0],min(extra,allowed[-1])))
            else: decision='call'
    else:
        if equity<0.5 and random.random()<0.2 and raises:
            allowed=sorted([amt for _,amt in raises])
            decision=('raise',random.choice(allowed[:3]))
        else: decision='check'
    return decision
