#ifndef CONSTANTS
#define CONSTANTS

#define NUM_PAST_STATES 2 
#define BIGDIM 21 
#define BOARDSIZE (BIGDIM*BIGDIM)
#define MAX_ITERATIONS BOARDSIZE
#define PAST_STATE_SIZE (BOARDSIZE*NUM_PAST_STATES) 
#define ADJACENCY 8
#define ADJ_PLUS_ONE (ADJACENCY+1)

#define MOD 32
#define BITMASK_SIZE (BOARDSIZE/MOD + 1)

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
