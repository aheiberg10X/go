#ifndef BITMASK_H
#define BITMASK_H

#include "constants.h"
#include <cuda.h>
#include <cuda_runtime.h>

struct BitMask {
    int masks[BITMASK_SIZE];
    int count;

    __device__ __host__ BitMask();
    
    /*void initBitMask();*/
    __device__ __host__ void clear();

    __device__ __host__ void set( int bit, bool value );

    __device__ __host__ bool get( int bit );

};


#endif
