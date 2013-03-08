#ifndef GODOMAIN_H
#define GODOMAIN_H

#include "domain.h"
//#include "stonestring.h"
#include <string>
//#include <map>
//#include "gostate.h"
#include "gostate_struct.h"
#include <assert.h>
#include <iostream>

//TODO: need this for rand()
//replace with #include <curand.h>
//example
//https://gist.github.com/803377
#include <algorithm>

using namespace std;

class GoDomain : public Domain {
public :

    int getNumActions( void* state ){
        return BOARDSIZE;
        //return ((GoState*) state)->boardsize;
    }

    int getNumPlayers( void* state ){
        return 2; 
    }

    int getPlayerIx( void* state ){
        char player = ((GoStateStruct*) state)->player;
        assert( player == WHITE || player == BLACK );
        if( player == WHITE ){
            return 0;
        }
        else {//if( player == BLACK ){
            return 1;
        }
    }

    void copyStateInto( void* source, void* target ){
        GoStateStruct* gs_source = (GoStateStruct*) source;
        GoStateStruct* gs_target = (GoStateStruct*) target;
        gs_source->copyInto( gs_target );
    };

    void* copyState( void* source ){
        GoStateStruct* gs_source = (GoStateStruct*) source;
        GoStateStruct* gs_target = (GoStateStruct*) malloc(sizeof(GoStateStruct));
        gs_source->copyInto(gs_target);
        return (void*) gs_target;
    }


    void deleteState( void* state ) {
        free(state);
    }

    /* See domain.h */
    bool applyAction( void* uncast_state, 
                      int action,
                      bool side_effects ){

        assert( action >= 0 );
        GoStateStruct* state = (GoStateStruct*) uncast_state;
        return state->applyAction( action, side_effects );
    }
    

    //deprecated, in kernel
    void getRewards( int* to_fill,
                     void* uncast_state ){
        ((GoStateStruct*) uncast_state)->getRewards( to_fill );
    }
    
    //return an unsigned action, i.e an ix in the board
    //deprecated, in kernel
    int randomAction( void* uncast_state, 
                      BitMask* to_exclude ){
        bool side_effects = false;
        return ((GoStateStruct*) uncast_state)->randomAction( to_exclude, 
                                                              side_effects);
    }

    bool fullyExpanded( int action ){
        return action == EXCLUDED_ACTION;
    }

    bool isChanceAction( void* state ){
        return false;
    }

    bool isTerminal( void* uncast_state ){
        return ((GoStateStruct*) uncast_state)->isTerminal();
    }

};

#endif





 

