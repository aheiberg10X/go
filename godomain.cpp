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

    void* copyState( void* state ){
        GoStateStruct* gs = (GoStateStruct*) state;
        return gs->copy();
    };

    void deleteState( void* state ) {
        cout << "commented out deleteState" << endl;
        //GoStateStruct* gs = (GoStateStruct*) state;
        //delete gs;
        //return;
    }

    bool applyAction( void* uncast_state, 
                      int action,
                      bool side_effects ){

        //cout << "inside applyAction" << endl;
        assert( action >= 0 );
        GoStateStruct* state = (GoStateStruct*) uncast_state;
        return state->applyAction( action, side_effects );
    }

    //deprecated, in kernel
    /*
    void getRewards( int* to_fill,
                     void* uncast_state ){
        //cout << "start getRewards" << endl;
        GoStateStruct* state = (GoStateStruct*) uncast_state;

        //set<int> marked;
        BitMask marked; //( state->boardsize );
        //cout << "ere1" << endl;
        //int string_id = 0;
        //ix : stonestring_id
        //map<int,int> string_lookup;
        // stonestring_id : StoneString()
        //map<int,StoneString*> stone_strings;

        //TODO
        //this building StoneStrings is unnecessary
        //we just need to go through and look for empty spots that become
        //territory.  Count them up and then count the stones
        //for( int ix=0; ix < boardsize; ix++ ){
            //if( board[ix] == OFFBOARD ||
                //board[ix] == EMPTY ||
                //marked.find(ix) != marked.end() ){
                //continue;
            //}

            //int floodfill_array[boardsize];
            //char color = state->ix2color(ix);
            //char* flood_colors = {color};
            //char* stop_colors;
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
        //marked.clear();

        int white_score = 0;
        int black_score = 0;
        for( int ix=0; ix < BOARDSIZE; ix++ ){
            //cout << "marekd at: " << ix << " : " << marked.get(ix) << endl;
            //cout << state->board[ix] << endl;
            if( state->board[ix] == OFFBOARD ||
                marked.get( ix ) ){
                    //marked.find(ix) != marked.end() ){
                continue;
            }
            if( state->board[ix] == WHITE ){
                white_score++;
                continue;
            }
            if( state->board[ix] == BLACK ){
                black_score++;
                continue;
            }

            //find if ix has a neighbors of {WHITE,EMPTY} or {BLACK,EMPTY}
            //if so, set ncolor to be WHITE or BLACK
            //       set nix to be the ix of one such stone
            //else, the ix is not anybody's territory
            char ncolor;
            int nix;

            int adjacency = 4;
            int neighbs[adjacency];
            state->neighborsOf( neighbs,
                                ix,
                                adjacency );
            int white_neighbs[adjacency];
            int num_white_neighbs = 0;
            char filter_colors[1] = {WHITE};
            state->filterByColor(
                           white_neighbs, &num_white_neighbs,
                                  neighbs, adjacency, 
                                  filter_colors, 1 );

            int black_neighbs[adjacency];
            int num_black_neighbs = 0;
            filter_colors[0] = BLACK;
            state->filterByColor( 
                                  black_neighbs, &num_black_neighbs,
                                  neighbs, adjacency,
                                  filter_colors, 1 );

            bool has_white = num_white_neighbs > 0;
            bool has_black = num_black_neighbs > 0;
            if(      has_white && ! has_black ) { ncolor = WHITE; }
            else if( has_black && ! has_white ) { ncolor = BLACK; }
            else                                { ncolor = EMPTY; }

            //set nix to the first neighbor of the char ncolor
            for( int j=0; j<adjacency; j++ ){
                nix = neighbs[j];
                if( state->ix2color( nix ) == ncolor ){
                    break;
                }
            }

            if( ncolor == BLACK || ncolor == WHITE ){

                //this is overkill given how we are moving
                //is enough to just see a color adjacent to an empty
                //assuming the rest bug free, it will be that colors territory
                int floodfill_len = 0;
                char flood_colors[1] = {EMPTY};
                char stop_colors[1] = {state->flipColor(ncolor)};
                bool are_territories = 
                    state->floodFill(  
                                      state->floodfill_array, &floodfill_len,
                                      ix, 
                                      adjacency,
                                      flood_colors, 1,
                                      stop_colors, 1 );

                //mark these empty positions regardless of their territory 
                //status
                for( int i=0; i<floodfill_len; i++ ){
                    marked.set( state->floodfill_array[i], true );
                    //marked.insert( state->floodfill_array[i] );
                    if( are_territories ){
                        if( ncolor == WHITE ){
                            white_score++;
                        }
                        else if( ncolor == BLACK ){
                            black_score++;
                        }
                        else{ assert(false); }
                    }
                }
            }
        }
        white_score *= 2;
        black_score *= 2;
        white_score += 11; //5.5*2

        to_fill[0] = white_score > black_score ? 1 : 0;
        to_fill[1] = black_score > white_score ? 1 : 0;
        //cout << "end getRewards" << endl;
        return;
    }
    */

    //return an unsigned action, i.e an ix in the board
    //deprecated, in kernel
    /* 
    int randomAction( void* uncast_state,
                      BitMask* to_exclude ){
        //bool* to_exclude ){
        //cout << "inside randomAction" << endl;
        GoStateStruct* state = (GoStateStruct*) uncast_state;
        //get a random shuffle of the empty intersections
        //set<int>::iterator it;
        int size = state->num_open; //state->open_positions.size();
        int empty_ixs[ BOARDSIZE ];
        //cout << "size: " << size << endl;

        int i = 0;
        int j;
        //can shuffle randomly as we insert...
        for( int ix=0; ix<BOARDSIZE; ix++ ){
            //cout << "random shuffle i: " << i << endl;
            if( state->board[ix] == EMPTY ){
                if( i == 0 ){
                    empty_ixs[0] = ix;
                }
                else{
                    //TODO
                    //replace with device rand
                    j = rand() % i;
                    empty_ixs[i] = empty_ixs[j];
                    empty_ixs[j] = ix;
                }
                i++;
            }
        }
        //cout << "after shuffled" << endl;

        //try each one to see if legal
        bool legal_moves_available = false;
        int candidate;
        for( int j=0; j<size; j++ ){
            candidate = empty_ixs[j];
            //cout << "legality test begin" << endl;
            bool is_legal = applyAction( uncast_state, candidate, false );
            //cout << "legaity test conclude" << endl;
            //bool is_legal = applyAction( state, candidate, false );

            if( is_legal ){
                legal_moves_available = true;
                //if( to_exclude[candidate] == false ){
                if( !to_exclude->get( candidate ) ){
                    //return action;
                    return candidate;
                }
            }
        }

        if( legal_moves_available ){ //but all were excluded...
            return EXCLUDED_ACTION;
        }
        else {
            return PASS;
        }
    }*/

    bool fullyExpanded( int action ){
        return action == EXCLUDED_ACTION;
    }

    bool isChanceAction( void* state ){
        return false;
    }

    //deprecated, in knernel 
    /*
    bool isTerminal( void* uncast_state ){
        //cout << "whoa there" << endl;
        GoStateStruct* state = (GoStateStruct*) uncast_state;
        //TODO
        //rework for new abstraction
        //GoStateStruct* last_state = state->past_states[NUM_PAST_STATES-1];
        bool r = state->action == PASS && 
               state->past_actions[NUM_PAST_STATES-1] == PASS; //last_state->action == PASS;
        //cout << "wtf" << endl;
        return r;
    }
    */

};

#endif





 

