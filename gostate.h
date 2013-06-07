#ifndef GOSTATE
#define GOSTATE

#include "constants.h"
#include "queue.h"
#include "bitmask.h"
#include "zobrist.h"
#include "mcts_state.h"

#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <assert.h>
#include <stdint.h>
#include <time.h>
#include <stdio.h>
#include <iostream>

using namespace std;

class GoState : public MCTS_State {
private :
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

    /*int known_illegal[BOARDSIZE];*/
    BitMask black_known_illegal;
    BitMask white_known_illegal;

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

    //
    uint16_t empty_intersections[MAX_EMPTY];
    uint16_t frozen_empty_intersections[MAX_EMPTY];

    void freezeBoard();

    void thawBoard();

    char flipColor( char c );

    void togglePlayer( );

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

    void setKnownIllegal( int ix );

    bool isKnownIllegal( int ix );
    
    void capture( BitMask* bm );

    bool isBorder( int ix );

    void neighborsOf( int* to_fill, int ix, int adjacency );

    void neighborsOf2( int* to_fill, int* to_fill_len,
                       int ix, int adjacency, char filter_color );
    bool neighborsOf3( int* to_fill, int* to_fill_len,
                       int ix, int adjacency, char filter_color );

    void filterByColor(  
                        int* to_fill, 
                        int* to_fill_len,
                        int* neighbs,
                        int adjacency,
                        char* color_array,
                        int filter_len );

    bool floodFill (int epicenter_ix,
                    int adjacency,
                    char flood_color,
                    char stop_color );

    void freezeBoard( char* target );

    void setBoard( char* target );

    bool isDuplicatedByPastState();

    void advancePastStates( int past_zhash, 
                            int past_action );


public :
    //MCTState Interface
    const int getNumPlayers();

    int getNumActions();

    int getPlayerIx();

    int movesMade();

    void deleteState();

    void copyInto( MCTS_State* target );
    void copyInto( GoState* target );

    MCTS_State* copy( );

    bool fullyExpanded( int action );

    bool isChanceAction();

    bool applyAction( int action, bool side_effects );

    int randomAction( BitMask* to_exclude, bool side_effects );
    
    bool isTerminal();

    void getRewards( int* to_fill );

    /////////////////////////////////////////////////////////

    //TODO fix up all ctor uses
    GoState( ZobristHash* zh);

    string toString( );

    string boardToString( char* board );

    void board2MATLAB( double* matlab_board );

    void MATLAB2board( double* matlab_board );

    static int bufferix2nobufferix( int ix );

    static int nobufferix2bufferix( int ix );

    //an version for checking territory status that does not rely on fact that only single poitions will be left empty when game ends
    /*void getRewardsComplete( int* to_fill );*/

};

#endif

