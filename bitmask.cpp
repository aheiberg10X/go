Rifndef BITMASK_H
#define BITMASK_H

#include <assert.h>

class BitMask {
public:
    int size;
    int mod;
    int num_rows;
    int* masks;

    BitMask( int asize ){
        size = asize;
        mod = 32;
        if( asize % mod == 0 ){
            num_rows = asize/mod;
        }
        else{
            num_rows = asize/mod+1;
        }
        masks = new int[num_rows];
        for( int i=0; i<num_rows; i++ ){
            masks[i] = 0;
        }
    }

    ~BitMask(){
        delete masks;
    }

    void set( int bit, bool value ) {
        if( bit >= size ){ return; }
        int row = bit / mod;
        int col = bit % mod;
        if( value == 1){
            masks[row] |= (int) value << col;
        }
        else if( value == 0 ){
            masks[row] &= (int) value << col;
        }
        else{
            assert(false);
        }
    }

    bool get( int bit ){
        int row = bit / mod;
        int col = bit % mod;
        return (bool) ((masks[row] >> col) & 1);
    }
};

#endif
