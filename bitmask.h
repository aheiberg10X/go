#ifndef BITMASK_H
#define BITMASK_H

#include "constants.h"

struct BitMask {
    int masks[BITMASK_SIZE];

    BitMask();
    
    /*void initBitMask();*/
    void clear();

    void set( int bit, bool value );

    bool get( int bit );

};


#endif
