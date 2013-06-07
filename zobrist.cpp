#include "zobrist.h"
#include <string.h>
#include <stdlib.h>

//TODO 64 bit impl
void ZobristHash::ctor(){
    int_bits = 32;
    for( int ix=0; ix<NUM_ZOBRIST_VALUES; ix++ ){
        values[ix] = rand();
    }
}

//__device__
//void ZobristHash::ctor( curandState* states, int block_id ){
//for( int ix=0; ix<NUM_ZOBRIST_VALUES; ix++ ){
//values[ix] = generate( states, block_id ) * INT_MAX
//}

void ZobristHash::copyInto( ZobristHash* target ){
    memcpy( target->values, values, NUM_ZOBRIST_VALUES*sizeof(int) );
}

int ZobristHash::getValue( char color, int ix ){
    int i;
    if(      color == BLACK ){ i = ix; }
    else if( color == WHITE ){ i = BOARDSIZE+ix; }
    return values[i];
}

//empty is the default.
//things go B/W->empty, or B/W->empty
int ZobristHash::updateHash( int hash, int position, char color ){
    return hash ^ getValue( color, position );
}

int ZobristHash::sizeOf(){ return int_bits; }


