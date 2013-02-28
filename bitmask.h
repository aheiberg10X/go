#ifndef BITMASK_H
#define BITMASK_H

#include "constants.h"

struct BitMask {
    bool mask[BOARDSIZE];
    int count;

    BitMask();
    
    /*void initBitMask();*/
    void clear();

    void set( int bit, bool value );

    bool get( int bit );
    
    void copyInto( BitMask* bm );

};

#endif
