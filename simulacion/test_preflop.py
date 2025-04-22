# test_preflop_ranges.py

from tablas_preflop import (
    preflop_ranges,
    get_rival_hand,
    determine_action_key_from_history,
    determine_subkey_from_history
)

def test_get_rival_hand():
    print("===== Testing get_rival_hand =====")
    for position in ["SB", "BB"]:
        print(f"-- Position: {position} --")
        for action_key, val in preflop_ranges[position].items():
            if isinstance(val, dict):
                # subkeys
                for sub in val.keys():
                    hand = get_rival_hand(position, action_key, sub)
                    print(f"{position} {action_key}.{sub} -> {hand}")
            else:
                hand = get_rival_hand(position, action_key)
                print(f"{position} {action_key} -> {hand}")
    print("===== Testing history key parsers =====")
    # ejemplo de historial
    history = [
        ["phase","preflop"],
        ["r",0,0.5,0.5],   # SB open
        ["c",1,0.5],      # BB call
        ["r",1,1.0,1.5],  # BB 3bet
        ["c",0,1.0]       # SB call vs 3bet
    ]
    ak = determine_action_key_from_history(history)
    sk = determine_subkey_from_history(history)
    print(f"determine_action_key_from_history -> {ak}")
    print(f"determine_subkey_from_history -> {sk}")

if __name__ == "__main__":
    test_get_rival_hand()
