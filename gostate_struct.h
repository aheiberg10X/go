#ifndef GOSTATE
#define GOSTATE

#include <stdlib.h>
#include "constants.h"
#include "queue.h"
#include "bitmask.h"

/*#include <cuda.h>*/
/*#include <cuda_runtime.h>*/

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
    int filtered_array[8];
    char color_array[3];

    //data structure for floodFill
    BitMask marked;
    BitMask on_queue;
    Queue queue;

    GoStateStruct();
    void initGSS( );

    int numElementsToCopy();

    void cudaAllocateAndCopy( void** pointers );

    void freezeBoard();
    void thawBoard();
    void* copy( );

    char flipColor( char c );

    bool sameAs( char* board, char player );

    /*bool sameAs( GoStateStruct* gss2 );*/

    void togglePlayer( );

    string toString( );
    string boardToString( char* board );

    int neighbor( int ix, DIRECTION dir);

    int ix2action( int ix, char player );

    int action2ix( int action );

    char action2color( int action );

    int ix2color( int ix );

    int coord2ix( int i, int j );

    int ixColor2Action( int ix, char color );

    int coordColor2Action( int i, int j, char color );

    bool isPass( int action );

    void setBoard( int ix, char color );

    void setBoard( int* ixs, int len, char color );

    void neighborsOf( int* to_fill, int ix, int adjacency );

    void filterByColor(  
                        int* to_fill, 
                        int* to_fill_len,
                        int* neighbs,
                        int adjacency,
                        char* color_array,
                        int filter_len );


    bool floodFill(  
                    int* to_fill,
                    int* to_fill_len,
                    int epicenter_ix,
                    int adjacency,
                    char* flood_color_array,
                    int filter_len,
                    char* stop_color_array,
                    int stop_len );


    bool isSuicide( int action );

    void freezeBoard( char* target );
    void setBoard( char* target );

    bool isDuplicatedByPastState();
    void advancePastStates( char* past_board,
                            char past_player,
                            int past_action );

};

#endif
