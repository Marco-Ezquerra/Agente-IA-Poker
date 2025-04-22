

#include <poker_defs.h>
#include <inlines/eval.h>

int eval_hand(int rank1, int suit1, int rank2, int suit2,
              int br1, int bs1, int br2, int bs2,
              int br3, int bs3, int br4, int bs4, int br5, int bs5) {

    StdDeck_CardMask hand;
    StdDeck_CardMask_RESET(hand);

    // Añadir las 7 cartas a la máscara
    StdDeck_CardMask_OR(hand, hand, StdDeck_MASK(StdDeck_MAKE_CARD(rank1, suit1)));
    StdDeck_CardMask_OR(hand, hand, StdDeck_MASK(StdDeck_MAKE_CARD(rank2, suit2)));
    StdDeck_CardMask_OR(hand, hand, StdDeck_MASK(StdDeck_MAKE_CARD(br1, bs1)));
    StdDeck_CardMask_OR(hand, hand, StdDeck_MASK(StdDeck_MAKE_CARD(br2, bs2)));
    StdDeck_CardMask_OR(hand, hand, StdDeck_MASK(StdDeck_MAKE_CARD(br3, bs3)));
    StdDeck_CardMask_OR(hand, hand, StdDeck_MASK(StdDeck_MAKE_CARD(br4, bs4)));
    StdDeck_CardMask_OR(hand, hand, StdDeck_MASK(StdDeck_MAKE_CARD(br5, bs5)));

    // Evaluar la mano de 7 cartas
    return Hand_EVAL_N(hand, 7);
}

