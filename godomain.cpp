#ifndef GODOMAIN_H
#define GODOMAIN_H

#include "domain.h"
#include "stonestring.h"
#include <string>
#include <map>
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

    bool applyAction( void* uncast_state, 
                      int action,
                      bool side_effects ){


        GoState* state = (GoState*) uncast_state;
        assert( state->action2color(action) == state->player );

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
            COLOR opp_color = state->flipColor(color); 

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
            for( int i=0; i < NUM_PAST_STATES; i++ ){
                GoState* past_state = state->past_states[i];
                if( state->sameAs( past_state->board,
                                   state->flipColor( past_state->player ) ) ){
                    legal = false;
                    break;
                }
            }
        }

        if( legal ){
            if( side_effects ){
                state->action = action;
                delete state->past_states[0];
                for( int i=0; i<NUM_PAST_STATES-1; i++ ){
                    //TODO del the state we are jettisoning?
                    state->past_states[i] = state->past_states[i+1];
                }
                state->togglePlayer();
                state->past_states[NUM_PAST_STATES-1] = frozen;
            }
            else{
                state = frozen;
            }
            return true;
        }
        else{
            if( side_effects ){
                cout << "action: " << action << endl;
                assert(false);
            }
            else{
                state = frozen;
            }
            return false;
        }  
    }
    
    void getRewards( int* to_fill,
                     void* uncast_state ){

        GoState* state = (GoState*) uncast_state;

        set<int> marked;
        int string_id = 0;
        //ix : stonestring_id
        map<int,int> string_lookup;
        // stonestring_id : StoneString()
        map<int,StoneString*> stone_strings;

        //TODO
        //this is unnecessary
        //we just need to go through and look for empty spots that become
        //territory.  Count them up and then count the stones
        //for( int ix=0; ix < boardsize; ix++ ){
            //if( board[ix] == OFFBOARD ||
                //board[ix] == EMPTY ||
                //marked.find(ix) != marked.end() ){
                //continue;
            //}

            //int floodfill_array[boardsize];
            //COLOR color = state->ix2color(ix);
            //COLOR* flood_colors = {color};
            //COLOR* stop_colors;
            //int floodfill_len = 0;
            //state->floodFill( floodfill_array,
                              //&floodfill_len,
                              //ix,
                              //8,
                              //flood_colors, 1,
                              //stop_colors,  0 );

            //if( floodfill_len > 0 ){
                //StoneString* ss = new StoneString( string_id, 
                                                   //floodfill_array, 
                                                   //floodfill_len, 
                                                   //color );
                //stone_strings[string_id] = ss;
                //for( int i=0; i < floodfill_len; i++ ){
                    //int ix = floodfill_array[i];
                    //marked.insert( ix );
                    //stonestring_lookup[ix] = string_id;
                //}
                //string_id++;
            //}
        //}
        //


        marked.clear();
        for( int ix=0; ix < boardsize; ix++ ){
            if( board[ix] == OFFBOARD ||
                board[ix] == WHITE    ||
                board[ix] == BLACK    ||
                marked.find(ix) != marked.end() ){
                continue;
            }
            //find if the ix is neighbored by stones of only one color
            //if yeah, floodfill it, on the lookup for the opp color
            //mark the returned territory
            //TODO performance leak here, say FF finds an opp color and 
            //returns nothing.  All the EMPTIES looked at by floodfill
            //will be reexamined as epicenters themselves, waste
            //consider reworking signature, so that it sets a bool whether the
            //stop colors were found.  If true, then floodfill_array can
            //still be accessed to mark the useless nodes 



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





 

