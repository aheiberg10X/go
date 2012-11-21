#ifndef GOSTATE
#define GOSTATE

#include <stdlib.h>
#include "constants.h"
#include "queue.h"
#include "bitmask.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <string>
#include <sstream>
#include <assert.h>

using namespace std;

struct GoStateStruct{
    char board[BOARDSIZE];
    //storage space for current board when checking move legality
    char frozen_board[BOARDSIZE];
    int frozen_num_open;
    int action;
    int num_open;
    char player;

    char past_boards[PAST_STATE_SIZE]; 
    char past_players[NUM_PAST_STATES];
    int past_actions[NUM_PAST_STATES];

    //scratch space for floodFill
    //TODO
    //save space by treating this as char array. Numbers [0,121] < 2^8
    //alternatively, save it as a BitMask, and iterate through each time to 
    //get the marked elements
    int floodfill_array[BOARDSIZE];
    int neighbor_array[8];
    int internal_neighbor_array[8];
    int filtered_array[8];
    int internal_filtered_array[8];
    char color_array[3];

    //data structure for floodFill
    BitMask marked;
    BitMask on_queue;
    Queue queue;

    GoStateStruct();
    /*__device__ __host__ GoStateStruct( void** pointers );*/
    
    int numElementsToCopy();

    /*void cudaAllocateAndCopy( void** pointers );*/

    __device__ __host__
    void freezeBoard();

    __device__ __host__
    void thawBoard();

    __host__
    void* copy( );
    
    __device__ __host__ void copyInto( GoStateStruct* target );

    __device__ __host__
    char flipColor( char c );

    __device__ __host__
    bool sameAs( char* board, char player );

    /*bool sameAs( GoStateStruct* gss2 );*/

    __device__ __host__
    void togglePlayer( );

    __host__
    string toString( );

    __host__
    string boardToString( char* board );

    __device__ __host__ 
    int neighbor( int ix, DIRECTION dir);

    __device__ __host__
    int ix2action( int ix, char player );

    __device__ __host__
    int action2ix( int action );

    __device__ __host__
    char action2color( int action );

    __device__ __host__
    int ix2color( int ix );

    __device__ __host__
    int coord2ix( int i, int j );

    __device__ __host__
    int ixColor2Action( int ix, char color );

    __device__ __host__
    int coordColor2Action( int i, int j, char color );

    __device__ __host__
    bool isPass( int action );

    __device__ __host__
    void setBoard( int ix, char color );

    __device__ __host__
    void setBoard( int* ixs, int len, char color );

    __device__ __host__
    void neighborsOf( int* to_fill, int ix, int adjacency );

    __device__ __host__
    void filterByColor(  
                        int* to_fill, 
                        int* to_fill_len,
                        int* neighbs,
                        int adjacency,
                        char* color_array,
                        int filter_len );


    __device__ __host__
    bool floodFill(  
                    int* to_fill,
                    int* to_fill_len,
                    int epicenter_ix,
                    int adjacency,
                    char* flood_color_array,
                    int filter_len,
                    char* stop_color_array,
                    int stop_len );


    __device__ __host__
    bool isSuicide( int action );

    __device__ __host__
    void freezeBoard( char* target );

    __device__ __host__
    void setBoard( char* target );

    __device__ __host__
    bool isDuplicatedByPastState();

    __device__ __host__
    void advancePastStates( char* past_board,
                            char past_player,
                            int past_action );

    __device__ __host__ 
    bool applyAction( int action, bool side_effects );

    __device__ 
    int randomAction( curandState* crs, int tid, BitMask* to_exclude );

    __host__ 
    int randomAction( BitMask* to_exclude );

    __device__ __host__
    bool isTerminal();

    __device__ __host__
    void getRewards( int* to_fill );

};

#endif
