#ifndef CONSTANTS
#define CONSTANTS

#define NUM_PAST_STATES 2
#define BIGDIM 11
#define BOARDSIZE 121
#define PAST_STATE_SIZE 242 //BOARDSIZE*NUM_PASTSTATES

#define MOD 32
#define BITMASK_SIZE 4  //BOARDSIZE/MOD + 1

#define BLACK 'b'
#define WHITE 'w'
#define EMPTY 'e'
#define OFFBOARD 'o'

#define PASS 0

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
