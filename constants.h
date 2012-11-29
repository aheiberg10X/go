#ifndef CONSTANTS
#define CONSTANTS

#include <math.h>

//Use the GPU?
#define USE_GPU true 

//How many leaf simulations should be performed
#define NUM_SIMULATIONS 1000

//How many itertions of MCTS before best move selected?
#define NUM_ITERATIONS 1000

#define BALANCING_CONSTANT (1/sqrt(2))

//GoStateStruct Constants
//set these
#define DIMENSION 9
#define NUM_PAST_STATES 2 

//Leave these alone
#define BIGDIM (DIMENSION+2)
#define BOARDSIZE (BIGDIM*BIGDIM)
#define PAST_STATE_SIZE (BOARDSIZE*NUM_PAST_STATES) 
#define ADJACENCY 8
#define ADJ_PLUS_ONE (ADJACENCY+1)
#define MAX_MOVES BOARDSIZE

//bitmask constants
#define MOD 32
#define BITMASK_SIZE (BOARDSIZE/MOD + 1)

//useful
#define BLACK 'b'
#define WHITE 'w'
#define EMPTY 'e'
#define OFFBOARD 'o'
#define PASS 0
#define EXCLUDED_ACTION -123


enum DIRECTION {
    N = 0,
    S = 1,
    E = 2,
    W = 3,
    NW = 4,
    NE = 5,
    SW = 6,
    SE = 7
};

#endif
