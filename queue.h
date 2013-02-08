#ifndef QUEUE
#define QUEUE

#include "constants.h"
#include <cuda.h>
#include <cuda_runtime.h>

struct Queue{
    //int length;
    int array[BOARDSIZE];
    int begin, end;
    int nElems;

    __device__ __host__ Queue();

    __device__ __host__ int ringInc( int i );

    /*void initQueue();*/

    __device__ __host__ void clear();

    __device__ __host__ void push( int a );

    __device__ __host__ int pop();

    __device__ __host__ bool isEmpty();

};

#endif
