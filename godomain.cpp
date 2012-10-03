#ifndef GODOMAIN_H
#define GODOMAIN_H

#include "domain.h"
#include <string>
#include "gostate.h"
#include <set>
#include <assert.h>
#include <iostream>
/*#include "goaction.h"*/

using namespace std;

class GoDomain : public Domain {
public :
    int getPlayerIx( void* state ){
        return 42;
    }

    void* copyState( void* state ){
        GoState* gs = (GoState*) state;
        return (void*) gs->copy( false);
    };

    void applyAction( void* uncast_state, 
                      int action,
                      bool side_effects ){

        GoState* state = (GoState*) uncast_state;
        bool legal = true;
        GoState* frozen = state->copy(false);

        if( ! state->isPass(action) ){
            int ix = state->action2ix(action);
            COLOR color = state->action2color(action);
            state->setBoard( ix, color );

            //resolve captures
            int adjacency = 4;
            int neighbs[adjacency];
            state->neighborsOf( neighbs, ix, adjacency );
            COLOR opp_color = (color == WHITE) ? BLACK : WHITE;

            int opp_neighbs[adjacency];
            int opp_len = 0;
            COLOR filter_array[1] = {opp_color};
            state->filterByColor( opp_neighbs, &opp_len,
                                  neighbs, adjacency,
                                  filter_array, 1 );

            int num_removed = 0;
            for( int onix=0; onix < opp_len; onix++ ){
                int floodfill_len = 0;
                COLOR stop_color_array[1] = {EMPTY};
                state->floodFill( state->floodfill_array, &floodfill_len,
                                  opp_neighbs[onix],
                                  adjacency,
                                  filter_array, 1,
                                  stop_color_array, 1 );
                state->setBoard( state->floodfill_array,
                                 floodfill_len, 
                                 EMPTY );
            }

            if( state->isSuicide( action ) ){
                legal = false;
            }

            //check past states for duplicates
            //for()
            
            if( legal ){
                if( side_effects ){
                }
                else{
                    state = frozen;
                }
            }
            else{
                if( side_effects ){
                    cout << "action: " << action << endl;
                    assert(false);
                }
                else{
                    state = frozen;
                }
            }  

                                  
        }

        return;
    }
    
    void getRewards( int* to_fill,
                     void* state ){
        return;
    }

    string randomAction( void* state,
                        set<string> to_exclude ){
        return "adsfasdf";
    }
    
    bool fullyExpanded( int action ){
        return false;
    }

    bool isTerminal( void* state ){
        return true;
    }

};

#endif





 

