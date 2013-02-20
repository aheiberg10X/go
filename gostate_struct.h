#ifndef GOSTATE
#define GOSTATE

#include <stdlib.h>
#include "constants.h"
#include "queue.h"
#include "bitmask.h"
#include "zobrist.h"

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
    ZobristHash* zhasher;
    int zhash;
    int action;
    int num_open;
    char player;


    //storage space for current board when checking move legality
    char frozen_board[BOARDSIZE];
    int frozen_num_open;
    int frozen_zhash;

    int past_zhashes[NUM_PAST_STATES];
    //TODO: do we ever use the past actions other than the most recent?
    //No, only use the last action in check inside isTerminal...
    /*int past_actions[NUM_PAST_STATES];*/
    int past_action;

    //scratch space for floodFill
    //TODO
    //save space by treating this as char array. Numbers [0,121] < 2^8
    //alternatively, save it as a BitMask, and iterate through each time to 
    //get the marked elements
    /*int floodfill_array[BOARDSIZE];*/
    int neighbor_array[8];
    int internal_neighbor_array[8];
    int filtered_array[8];
    int internal_filtered_array[8];
    char color_array[3];

    //data structure for floodFill
    BitMask marked;
    BitMask connected_to_lib;
    Queue queue;

    __host__
    void ctor( ZobristHash* zh);
    
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
    void capture( BitMask* bm );

    __device__ __host__ 
    bool isBorder( int ix );

    __device__ __host__
    void neighborsOf( int* to_fill, int ix, int adjacency );

    __device__ __host__ 
    void neighborsOf2( int* to_fill, int* to_fill_len,
                       int ix, int adjacency, char filter_color );

    __device__ __host__
    void filterByColor(  
                        int* to_fill, 
                        int* to_fill_len,
                        int* neighbs,
                        int adjacency,
                        char* color_array,
                        int filter_len );


    /*
    __device__ __host__
    bool floodFill(  
                    int* to_fill,
                    int* to_fill_len,
                    int epicenter_ix,
                    int adjacency,
                    char* flood_color_array,
                    int filter_len,
                    char* stop_color_array,
                    int stop_len );*/

    __device__ __host__
    bool floodFill2( 
            /*int* to_fill,*/
            /*int* to_fill_len,*/
            /*BitMask* connected_to_*/
            /*BitMask* flooded, */
                    int epicenter_ix,
                    int adjacency,
                    char flood_color,
                    char stop_color );


    __device__ __host__
    bool isSuicide( int action );

    __device__ __host__
    void freezeBoard( char* target );

    __device__ __host__
    void setBoard( char* target );

    __device__ __host__
    bool isDuplicatedByPastState();

    __device__ __host__
    void advancePastStates( int past_zhash, //char* past_board,
            /*char past_player,*/
                            int past_action );

    /*__device__ __host__ */
    /*bool isNaivelyLegal( int ix, char COLOR );*/

    __device__ __host__ 
    bool applyAction( int action, bool side_effects );

    __device__ 
    int randomAction( curandState* crs, int tid, BitMask* to_exclude, bool side_effects );

    __host__ 
    int randomAction( BitMask* to_exclude, bool side_effects );

    __host__ __device__
    int randomActionBase( BitMask* to_exclude, bool side_effects, int* empty_ixs);

    __device__ __host__
    bool isTerminal();

    __device__ __host__
    void getRewards( int* to_fill );

    //an version for checking territory status that does not rely on fact that only single poitions will be left empty when game ends
    /*__device__ __host__*/
    /*void getRewardsComplete( int* to_fill );*/

};

#endif


/*
void GoStateStruct::cudaAllocateAndCopy( void** pointers ){
    int* dev_action;
    int* dev_num_open;
    char* dev_board;
    char* dev_player;
    char* dev_past_boards;
    char* dev_past_players;
    int* dev_past_actions;
    char* dev_frozen_board;
    int* dev_frozen_num_open;

    cudaMalloc( (void**)&dev_action, sizeof(int) );
    cudaMemcpy( dev_action, &(action), sizeof(int), cudaMemcpyHostToDevice );
    pointers[0] = (void*) dev_action;

    cudaMalloc( (void**)&dev_num_open, sizeof(int) );
    cudaMemcpy( dev_num_open, &(num_open), sizeof(int), cudaMemcpyHostToDevice );
    pointers[1] = (void*) dev_num_open;

    cudaMalloc( (void**)&dev_board, BOARDSIZE*sizeof(char) );
    cudaMemcpy( dev_board, board, BOARDSIZE*sizeof(char), cudaMemcpyHostToDevice );
    pointers[2] = (void*) dev_board;

    cudaMalloc( (void**)&dev_player, sizeof(char) );
    cudaMemcpy( dev_player, &(player), sizeof(char), cudaMemcpyHostToDevice );
    pointers[3] = (void*) dev_player;

    cudaMalloc( (void**)&dev_past_boards, sizeof(char)*PAST_STATE_SIZE );
    cudaMemcpy( dev_past_boards, past_boards, PAST_STATE_SIZE*sizeof(char), cudaMemcpyHostToDevice );
    pointers[4] = (void*) dev_past_boards;

    cudaMalloc( (void**)&dev_past_players, sizeof(int)*NUM_PAST_STATES );
    cudaMemcpy( dev_past_players, past_players, sizeof(int)*NUM_PAST_STATES, cudaMemcpyHostToDevice );
    pointers[5] = (void*) dev_past_players;
    
    cudaMalloc( (void**)&dev_past_actions, sizeof(int)*NUM_PAST_STATES );
    cudaMemcpy( dev_past_actions, past_actions, sizeof(int)*NUM_PAST_STATES, cudaMemcpyHostToDevice );
    pointers[6] = (void*) dev_past_actions;
    
    cudaMalloc( (void**)&dev_frozen_board, sizeof(char)*BOARDSIZE );
    cudaMemcpy( dev_frozen_board, frozen_board, BOARDSIZE*sizeof(char), cudaMemcpyHostToDevice );
    pointers[7] = (void*) dev_frozen_board;
    
    cudaMalloc( (void**)&dev_frozen_num_open, sizeof(int) );
    cudaMemcpy( dev_frozen_num_open, &frozen_num_open, sizeof(int), cudaMemcpyHostToDevice );
    pointers[8] = (void*) dev_frozen_num_open;
}
*/
/*
GoStateStruct::GoStateStruct( void** pointers ){
    action = *((int*) pointers[0]);
    num_open = *((int*) pointers[1]);
    for( int i=0; i<BOARDSIZE; i++ ){
        board[i] = ((char*) pointers[2])[i];
    }
    player = *((char*) pointers[3]);
    for( int i=0; i<PAST_STATE_SIZE; i++ ){
        past_boards[i] = ((char*) pointers[4])[i];
    }
    for( int i=0; i<NUM_PAST_STATES; i++ ){
        past_players[i] = ((char*) pointers[5])[i];
    }
    for( int i=0; i<NUM_PAST_STATES; i++ ){
        past_actions[i] = ((int*) pointers[6])[i];
    }
    for( int i=0; i<BOARDSIZE; i++ ){
        frozen_board[i] = ((char*) pointers[7])[i];
    }
    frozen_num_open = *((int*) pointers[8]);
}*/
/*
int GoStateStruct::numElementsToCopy(){
    return 9;
}
*/
