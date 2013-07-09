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
#include <vector>
#include <cmath>

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

    //work space for floodFill (hold over from GPU, memory tight days)
    int neighbor_array[8];
    int internal_neighbor_array[8];
    int filtered_array[8];
    int internal_filtered_array[8];
    char color_array[3];

    //data structure for floodFill
    BitMask marked;
    BitMask connected_to_lib;
    Queue queue;
    //for group detection, we need floodFillForGroups to track intersections
    //that get marked.  
    vector<int> marked_group;

    //
    uint16_t empty_intersections[MAX_EMPTY];
    uint16_t frozen_empty_intersections[MAX_EMPTY];

    void freezeBoard();

    void thawBoard();

    void togglePlayer( );

    int neighbor( int ix, DIRECTION dir);

    int ix2action( int ix, char player );

    int action2ix( int action );

    char action2color( int action );

    int coord2ix( int i, int j );

    int ixColor2Action( int ix, char color );

    int coordColor2Action( int i, int j, char color );

    bool isPass( int action );

    void setBoard( int ix, char color );

    void setBoard( int* ixs, int len, char color );

    void setKnownIllegal( int ix );

    bool isKnownIllegal( int ix );
    
    void capture();

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


    void freezeBoard( char* target );

    void setBoard( char* target );

    bool isDuplicatedByPastState();

    void advancePastStates( int past_zhash, 
                            int past_action );


public :
    ///////////////////////////////////////////////////
    //MCTS_State Interface
    //
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
    
    string toString( );
    string featuresToString(int* features, int nfeatures);

    static string prettyBoard( string* board, int gap );

    /////////////////////////////////////////////////////////
    //Helpers
    //

    GoState( ZobristHash* zh);

    string boardToString( char* board );

    void board2MATLAB( double* matlab_board );

    void MATLAB2board( double* matlab_board );

    static int bufferix2nobufferix( int ix );

    static int nobufferix2bufferix( int ix );
    
    char ix2color( int ix );

    bool floodFill (int epicenter_ix,
                    int adjacency,
                    char flood_color,
                    char stop_color );
    
    //need accessor to FF's "marked" BitMask
    int floodFillSize();
    
    vector<int> getMarkedGroup();

    char flipColor( char c );

    //an version for checking territory status that does not rely on fact that only single poitions will be left empty when game ends
    /*void getRewardsComplete( int* to_fill );*/

};

#endif

