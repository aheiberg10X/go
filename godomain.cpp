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
#include <algorithm>
/*#include "goaction.h"*/

using namespace std;

static int excluded_action = -123;

class GoDomain : public Domain {
public :

    int getPlayerIx( void* state ){
        return 42;
    }

    void* copyState( void* state ){
        GoState* gs = (GoState*) state;
        return (void*) gs->copy( false);
    };

    bool applyAction( void** p_uncast_state, 
                      int action,
                      bool side_effects ){


        GoState* state = (GoState*) *p_uncast_state;
        //cout << state->toString() << endl;
        //cout << "actoin: " << action << " state->player: " << state->player << endl;

        bool legal = true;
        GoState* frozen = state->copy(false);
        //cout << "froxqne toString: " << frozen->toString() << endl;

        if( ! state->isPass(action) ){
            assert( state->action2color(action) == state->player );
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
                bool fill_completed =
                state->floodFill( state->floodfill_array, &floodfill_len,
                                  opp_neighbs[onix],
                                  adjacency,
                                  filter_array, 1,
                                  stop_color_array, 1 );
                if( fill_completed ){
                    state->setBoard( state->floodfill_array,
                                     floodfill_len, 
                                     EMPTY );
                }
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
                //TODO why does this cause segfault?
                //delete state;
                *p_uncast_state = (void*) frozen;
            }
            return true;
        }
        else{
            if( side_effects ){
                cout << "action: " << action << endl;
                assert(false);
            }
            else{
                *p_uncast_state = (void*) frozen;
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

        int white_score = 0;
        int black_score = 0;
        for( int ix=0; ix < state->boardsize; ix++ ){
            if( state->board[ix] == OFFBOARD ||
                marked.find(ix) != marked.end() ){
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
            COLOR ncolor;
            int nix;

            int adjacency = 4;
            int neighbs[adjacency];
            state->neighborsOf( neighbs,
                                ix,
                                adjacency );
            int white_neighbs[adjacency];
            int num_white_neighbs = 0;
            COLOR filter_colors[1] = {WHITE};
            state->filterByColor( white_neighbs, &num_white_neighbs,
                                  neighbs, adjacency, 
                                  filter_colors, 1 );

            int* black_neighbs;
            int num_black_neighbs = 0;
            filter_colors[0] = BLACK;
            state->filterByColor( black_neighbs, &num_black_neighbs,
                                  neighbs, adjacency,
                                  filter_colors, 1 );

            bool has_white = num_white_neighbs > 0;
            bool has_black = num_black_neighbs > 0;
            if(      has_white && ! has_black ) { ncolor = WHITE; }
            else if( has_black && ! has_white ) { ncolor = BLACK; }
            else                                { ncolor = EMPTY; }

            //set nix to the first neighbor of the COLOR ncolor
            for( int j=0; j<adjacency; j++ ){
                nix = neighbs[j];
                if( state->ix2color( nix ) == ncolor ){
                    break;
                }
            }

            if( ncolor == BLACK || ncolor == WHITE ){
                int floodfill_len = 0;
                COLOR flood_colors[1] = {EMPTY};
                COLOR stop_colors[1] = {state->flipColor(ncolor)};
                bool are_territories = 
                    state->floodFill( state->floodfill_array, &floodfill_len,
                                      nix, 
                                      adjacency,
                                      flood_colors, 1,
                                      stop_colors, 1 );

                //mark these empty positions regardless of their territory 
                //status
                for( int i=0; i<floodfill_len; i++ ){
                    marked.insert( state->floodfill_array[i] );
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

        to_fill[0] = white_score;
        to_fill[1] = black_score;
        return;
    }

    int randomAction( void** p_uncast_state,
                      set<int> to_exclude ){
        GoState* state = (GoState*) *p_uncast_state;

        //get a random shuffle of the empty intersections
        set<int>::iterator it;
        int size = state->open_positions.size();
        int empty_ixs[ size ];

        int i = 0;
        for( it = state->open_positions.begin();
             it != state->open_positions.end();
             it++ ){
            empty_ixs[i++] = *it;
        }
        srand(time(NULL));
        random_shuffle( &empty_ixs[0], 
                        &empty_ixs[ size ] );

        //try each one to see if legal
        bool legal_moves_available = false;
        int candidate, action;
        for( int j=0; j<size; j++ ){
            candidate = empty_ixs[j];
            action = state->ix2action( candidate, state->player );
            //cout << "player: " << state->player << endl;
            //cout << "action to test: " << action << endl;
            //cout << "GoState* before AA: " << *p_uncast_state << endl;
            bool is_legal = applyAction( p_uncast_state, action, false );
            //cout << "GoState* after AA: " << *p_uncast_state << endl;
            if( is_legal ){
                legal_moves_available = true;
                if( to_exclude.find(action) == to_exclude.end() ){
                    return action;
                }
            }
        }

        if( legal_moves_available ){ //but all were excluded...
            return excluded_action;
        }
        else {
            return PASS;
        }
    }
    
    bool fullyExpanded( int action ){
        return action == excluded_action;
    }

    bool isChanceAction( void* state ){
        return false;
    }

    bool isTerminal( void* uncast_state ){
        GoState* state = (GoState*) uncast_state;
        GoState* last_state = state->past_states[NUM_PAST_STATES-1];
        cout << "this act: " << state->action << " past act: " << last_state->action << endl;
        return state->action == PASS && last_state->action == PASS;
    }

};

#endif





 

