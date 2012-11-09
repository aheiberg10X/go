#include "bitmask.h"
#include <stdio.h>

BitMask::BitMask(){
    clear();
}

void BitMask::clear(){
    for( int i=0; i<BITMASK_SIZE; i++ ){
        masks[i] = 0;
    }
}

void BitMask::set(int bit, bool value ) {
    if( bit >= BOARDSIZE ){ return; }
    int row = bit / MOD;
    int col = bit % MOD;
    if( value == 1){
        masks[row] |= (int) value << col;
    }
    else if( value == 0 ){
        masks[row] &= (int) value << col;
    }
    else{
        return;
    }
}

bool BitMask::get( int bit ){
    int row = bit / MOD;
    int col = bit % MOD;
    return (bool) ((masks[row] >> col) & 1);
}

