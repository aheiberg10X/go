#include "bitmask.h"
#include <string.h>

/////////////////////////////////////////////////////////////////////////////
//                     BitMask.cu
/////////////////////////////////////////////////////////////////////////////
BitMask::BitMask(){
    clear();
}

void BitMask::copyInto( BitMask* target ){
    memcpy( target->mask, mask, BOARDSIZE*sizeof(bool) );
    target->count = count;
}

void BitMask::clear(){
    memset( mask, false, BOARDSIZE );
    count = 0;
}

void BitMask::set(int bit, bool value ) {
    mask[bit] = value;
    if( value ){ count++;}
    else{        count--;}

}

bool BitMask::get( int bit ){
    return mask[bit];
}

// Have deprecated the old,smaller but slower BitMask into bitmask_small.h
// This is it's impl, just not renamed yet
/*
BitMask::BitMask(){
    clear();
}

void BitMask::copyInto( BitMask* target ){
    memcpy( target->masks, masks, BITMASK_SIZE*sizeof(int) );
    target->count = count;
}

void BitMask::clear(){
    memset( masks, 0, sizeof masks );
    //for( int i=0; i<BITMASK_SIZE; i++ ){
    //masks[i] = 0;
    //}
    count = 0;
}

void BitMask::set(int bit, bool value ) {
    if( bit >= BOARDSIZE ){ return; }
    int row = bit / MOD;
    int col = bit % MOD;
    if( value == 1){
        masks[row] |= (int) value << col;
        count++;
    }
    else if( value == 0 ){
        masks[row] &= (int) value << col;
        count--;
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

void BitMask::Or( BitMask bm ){
    for( int i=0; i<BITMASK_SIZE; i++ ){
        masks[i] |= bm.masks[i];
    }
}
*/

////////////////////

