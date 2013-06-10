#ifndef ZOBRIST 
#define ZOBRIST

#include "constants.h"

//http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt64.html

struct ZobristHash{
    int int_bits;

    //TODO: change to 64 bit repr
    //make a random int for each position and each color
    //lay it out as below:
    //[0:BLACK,1:BLACK,...,BOARDSIZE-1:BLACK, ... WHITE]
    int values[NUM_ZOBRIST_VALUES];

    void ctor();

    //TODO: not crucial, set up once at start, never alter again

    void copyInto( ZobristHash* target );

    int getValue( char color, int ix );

    int updateHash( int hash,
                    int position, 
                    char color );
    
    int sizeOf();

};

#endif
