#ifndef CONSTANTS
#define CONSTANTS

#include <math.h>

//Use the GPU? Moved on a l0ong time ago...
/*#define USE_GPU false*/
#define N_ROOT_THREADS 1
#define N_VALUE_THREADS 4

//How many leaf simulations should be performed
#define NUM_SIMULATIONS 10000

//How many itertions of MCTS before best move selected?
#define NUM_ITERATIONS 100

//when doing valuePolicy, use (one of) the best valued action(s) 
//EGREEDY % of the time
#define EGREEDY .9

#define BALANCING_CONSTANT (1/sqrt(2))

//GoStateStruct Constants
//set these
#define DIMENSION 19 

//Leave these alone
#define MAX_EMPTY (DIMENSION*DIMENSION)
#define BIGDIM (DIMENSION+2)
#define BOARDSIZE (BIGDIM*BIGDIM)
#define PAST_STATE_SIZE (BOARDSIZE*NUM_PAST_STATES) 
#define ADJACENCY 8
#define ADJ_PLUS_ONE (ADJACENCY+1)
#define MAX_MOVES BOARDSIZE

//past board state constants
#define NUM_ZOBRIST_VALUES (BOARDSIZE*2)
#define NUM_PAST_STATES 20

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
