#ifndef BITMASK_H
#define BITMASK_H

#include "constants.h"

struct BitMask {
    //for quick get/setting of specific board intersections
    bool mask[BOARDSIZE];
    int count;

    BitMask();
    /*~BitMask();*/
    
    /*void initBitMask();*/
    void clear();

    void set( int bit, bool value );

    bool get( int bit );
    
    void copyInto( BitMask* bm );

    void Or( BitMask bm );

};

#endif
