#include <assert.h>
#include <iostream>
#include <math.h>
#include <vector>

#include "mcts.h"
//break in abstraction for valuePolicy 
#include "gostate.h"
#include "weights.h"

//value function
#include "value_functions/value2.h"

#include "matrix.h"
#include <time.h>

using namespace std;

//perform NUM_ITERATIONS iterations of the 4-step MCTS algorithm
//returns the search_tree, rooted at root_node
MCTS_Node* MCTS::search( MCTS_State* root_state ){
    
    int num_players = root_state->getNumPlayers();
    int num_actions = root_state->getNumActions();
    MCTS_Node* root_node = new MCTS_Node( num_players, num_actions );

    //create a copy of root_state, this will be our scratch state-space
    //as we traverse down the tree.  Every MCTS iteration, restore the 
    //scratch space to the original root
    MCTS_State* state = root_state->copy();
    MCTS_Node* node;

    int rewards[num_players];

    int iterations = 0;

    clock_t tree_policy_time, simulation_time;
    clock_t t1,t2;
    t1 = clock();

    while( iterations < NUM_ITERATIONS ){
        root_state->copyInto( state );
        cout << endl << endl << "iteration: " << iterations << endl;
        //cout << "\n\nroot state: " << state->toString() << endl;

        //(1) and (2)
        node = treePolicy( root_node, state );
        //cout << "state after tree policy: " << state->toString() << endl;

        //(3)
        defaultPolicy( rewards, state );
        //cout << "rewards w/b: " << rewards[0] << "/" << rewards[1] << endl;

        //(4)
        backprop( node, rewards, num_players );
        iterations += 1; 
    }
    state->deleteState();
    t2 = clock();

    assert( root_node->is_root );
    cout << "Search time: " << ((float) t2-t1) / CLOCKS_PER_SEC << endl;
    
    return root_node;
}

//use this to toggle between the different tree policys (random, UCT, value)
MCTS_Node* MCTS::treePolicy( MCTS_Node* node, 
                             MCTS_State*     state ){

    //return randomPolicy( node, state );

    //the following makes BLACK BLACK uses valueFucntion
    int player = state->getPlayerIx();
    if( player == 1 ){
        return valuePolicy( node, state );
    }
    else{
        return randomPolicy( node, state );
    }
}

//value function approach to the treePolicy
//currently using fred's MATLAB impl
MCTS_Node* MCTS::valuePolicyMATLAB( MCTS_Node* node, 
                              MCTS_State* state ){
    //TODO: do this once for the class at the beginning
    //make them class members
    //make the weights vector, from Fred's Training0202.mat
    mxArray *w = mxCreateNumericMatrix(WEIGHTS_SIZE, 1, mxDOUBLE_CLASS, mxREAL);
    memcpy( (double*) mxGetPr(w), WEIGHTS, WEIGHTS_SIZE*sizeof(double) );

    //will hold the -1,0,1 board representation (no outer buffer)
    mxArray *x = mxCreateNumericMatrix(MAX_EMPTY, 1, mxDOUBLE_CLASS, mxREAL);

    //output valueation
    mxArray* V = mxCreateDoubleScalar(0); 

    //Will hold the un-border-buffered version of some gostate board
    //use doubles instead of chars
    //this overhead is very small compared to the time spent in mlfValue
    double matlab_board[MAX_EMPTY];

    GoState* gss = (GoState *) state;

    GoState** working_gss = new GoState* [N_VALUE_THREADS];
    for( int i=0; i<N_VALUE_THREADS; i++ ){
       working_gss[i] = (GoState*) gss->copy();
       //new GoState( gss->zhasher );
    }

    while( node->marked && !state->isTerminal() ){
        cout << "valuePolicy node: " << node << endl;
        //cout << "what now?" << endl;
        //TODO, what is min valueFunction will take?
        
        //int legal_actions[BOARDSIZE];
        float action_values[BOARDSIZE];

        int ta = clock();
        #pragma omp parallel for shared(action_values, gss, working_gss ) \
                                 num_threads(N_VALUE_THREADS)
        for( int action=0; action<BOARDSIZE; action++ ){
            if( gss->ix2color(action) != EMPTY ){ 
                action_values[action] = -1;
                continue; 
            }
            //cout << "starting on action: " << action << endl;

            //determine which thread is working
            int actions_per_thread = BOARDSIZE / N_VALUE_THREADS;
            int tid = action / actions_per_thread;
            //dump all rounding remainder into the last thread
            if( tid == N_VALUE_THREADS ){
                tid = N_VALUE_THREADS-1;
            }

            //scratch space to try out the action
            gss->copyInto( working_gss[tid] );
    
            //the valuation for this action
            double value;

            //if have a value already
            if( node->tried_actions->get(action) ){
                //cout << "have action" << endl;
                value = node->children[action]->value;
            }
            //else no value/node yet
            else{
                //cout << "don't have value for action: " << action << endl;
                bool is_legal = working_gss[tid]->applyAction( action, true );
                if( !is_legal ){ 
                    //cout << action << " is illegal, skipping" << endl;
                    action_values[action] = -1;
                    continue; 
                }
                else{
                    MCTS_Node* child_node = new MCTS_Node( node, action );
                    working_gss[tid]->board2MATLAB(matlab_board);
                    memcpy( (double*) mxGetPr(x), 
                            matlab_board, 
                            MAX_EMPTY * sizeof(double) );
                    bool success = mlfValue2(1, &V, x, w);
                    //cout << "success: " << success << endl;
                    double* r = mxGetPr(V);
                    value = r[0];
                    if( value < 0 ){
                        cout << "weird value: " << value << endl;
                    }
                    child_node->value = value;
                }
            }
            action_values[action] = value;
        }//for action
        

        //segregate the legal and maximum value moves
        //so that they can be randomly chosen from
        double max_value = -999999;
        int max_actions[BOARDSIZE];
        int max_actions_end = 0;
        int legal_actions[BOARDSIZE];
        int legal_actions_end = 0;

        for( int action=0; action < BOARDSIZE; ++action ){
            int value = action_values[action];
            bool action_legal = value >= 0;
            if( action_legal ){
                legal_actions[legal_actions_end++] = action;
                //cout << "value is: " << value << "\n" << endl;
                if( value == max_value ){
                    max_actions[max_actions_end++] = action;
                }
                else if( value > max_value ){
                    max_value = value;
                    max_actions[0] = action;
                    max_actions_end = 1;
                }
            }
        }
        int tb = clock(); 

        //cout << "evaling: " << legal_actions_end << " legal actions took: " << (float (tb-ta))/CLOCKS_PER_SEC << "s" << endl;

        //the action we choose to take
        int chosen_action;
        //cout << max_actions_end << " actions with max value" << endl;
        double r = (double) rand() / RAND_MAX;
        //cout << "random: " << r << endl;

        int rix;
        if( r < EGREEDY ){
            //choose randomly amongst the best valued actions
            rix = rand() % max_actions_end;
            //cout << "n max actions: " << max_actions_end << endl;
            //cout << "going greedy, using action: " << max_actions[rix] << " (rix =" << rix << ")" << endl;
            chosen_action = max_actions[rix];
        }
        else{
            //cout << "going random" << endl;
            rix = rand() % legal_actions_end;
            chosen_action = legal_actions[rix];
        }
        cout << "chosen action: " << chosen_action << endl;
        state->applyAction( chosen_action, true );
        //cout << gss->toString() << endl;
        node = node->children[chosen_action];
        //cout << "new node: " << node << " marked: " << node->marked << endl;
        //cout << "hit any key" << endl;
        //cin.ignore();*/
    }//while not terminal

    mxDestroyArray(V);
    mxDestroyArray(x);
    mxDestroyArray(w);
    node->marked = true;
    return node;
}

//breaking abstraction
//but get over it
MCTS_Node* MCTS::valuePolicy( MCTS_Node* node,
                              MCTS_State* state ){

    GoState* gs = (GoState*) state;

    //compute the groups
    //TODO make these all true, not false.  except for offboards and emtpy
    BitMask probe_starts;
    for( int ix=0; ix < BOARDSIZE; ++ix ){
        char ixcolor = gs->ix2color(ix);
        if( ixcolor == WHITE || ixcolor == BLACK ){
            probe_starts.set(ix,true);
        }
    }


    //the group numbers for every intersection.  -1 for offboards and empties
    int group_id = 0;
    int group_assignments[BOARDSIZE];
    memset( group_assignments, -1, BOARDSIZE );

    //for now, just int size info
    vector<int> group_info;

    for( int ix=0; ix < BOARDSIZE; ++ix ){
        if( probe_starts.get(ix) ){
            char fill_color = gs->ix2color(ix);
            //don't want any stopping, 'n' will never be seen
            char stop_color = 'n';
            bool fill_completed = gs->floodFill( ix, 8, 
                                                 fill_color, stop_color );
            assert( fill_completed );
            //want to remove the flooded intersections from probe_starts
            //assign them group id in group_info
            //right now would have to iterate through entire BitMask
            //to find the marked ones
            //pass floodFill an optional vector that gets marked nodes
            //pushed to it?
            cout << "FF size: " << gs->floodFillSize() << endl ;
            //do stuff
        }
        else{
            //either an empty, offboard, or already assigned a group
            //assumes no intersection can be part of only one group
        }
    }



}

MCTS_Node* MCTS::randomPolicy( MCTS_Node* root_node,
                               MCTS_State*     state ){
    int action;
    MCTS_Node* node = root_node;
    BitMask empty_to_exclude;
    empty_to_exclude.clear();

    while( node->marked && !state->isTerminal() ){
        action = state->randomAction( &empty_to_exclude, true );
        if( node->tried_actions->get(action) ){
            node = node->children[action];
        }
        else{
            node = new MCTS_Node( node, action );
        }
    }
    node->marked = true;
    return node;
}

MCTS_Node* MCTS::uctPolicy( MCTS_Node* node, 
                            MCTS_State*    state ){
    while( node->marked and !state->isTerminal() ){
        int action = state->randomAction( node->tried_actions, true );
        //TODO break in abstraction
        bool fully_expanded = action == EXCLUDED_ACTION; 
        if( !fully_expanded ){
            //move already applied
            node = new MCTS_Node( node, action );
            //node = expand( node, state, action );
        }
        else{
            int player_ix = state->getPlayerIx();
            node = bestChild( node, player_ix, false );
            bool is_legal = state->applyAction( node->action, true );
        }
    }    
    node->marked = true;
    return node;
}

float MCTS::scoreNode( MCTS_Node* node, 
                     MCTS_Node* parent, 
                     int        player_ix, 
                     bool just_exploitation ){
    int creward = node->total_rewards[player_ix];
    int cvisit = node->visit_count;
    int pvisit = parent->visit_count;
    //cout << "creward: " << creward << " cvisit: " << cvisit << endl;
    float exploitation = creward / ((float) cvisit);
    //cout << "exploitation: " << exploitation << endl;
    float exploration = BALANCING_CONSTANT*sqrt( 2*log(pvisit) / cvisit );
    if( just_exploitation ){
        return exploitation;
    }
    else {
        return exploitation + exploration;
    }
}

MCTS_Node* MCTS::bestChild( MCTS_Node* parent, 
                            int        player_ix, 
                            bool just_exploitation ){
    bool uninit = true;
    int max_ix = 0;
    float max_score = -1;
    float score;
    MCTS_Node* child;

    int vcount = 0;

    int max_ixs[BOARDSIZE];
    int max_ixs_len = 0;

    //cout << "player_ix: " << player_ix << endl;
    for( int ix=0; ix < parent->num_actions; ix++ ){
        if( parent->tried_actions->get(ix) ){
            child = parent->children[ix];
            vcount += child->visit_count;
            score = scoreNode( child, parent, player_ix, just_exploitation );
                //((double) child->total_rewards[player_ix]) / child->visit_count;
                //cout << "child action: " << child->action << " score: " << score << endl;
            if( score > max_score ){
                max_ixs[0] = ix;
                max_ixs_len = 1;
                max_score = score;
            }
            else if( score == max_score ){
                max_ixs[ max_ixs_len++ ] = ix;
            }


        }
    }
    int r = rand() % max_ixs_len;
    max_ix = max_ixs[r];
    //cout << "vcount: " << vcount << endl;
    //cout << "max ix: " << max_ix << endl;
    return parent->children[max_ix];

}

void MCTS::defaultPolicy( int* total_rewards, 
                          MCTS_State* state ){
    const int nplayers = state->getNumPlayers();
    int total_results[nplayers];

    MCTS_State* scratch_state = state->copy();

    //a decent guess at the number of actions taken thus far
    int n_moves_made = state->movesMade();
    for( int i=0; i<NUM_SIMULATIONS; i++ ){
        state->copyInto( scratch_state );

        //NOTE: Turn timing off when OpenMP used, clock() synchronizes
        //      in the kernel and slows everything down
        BitMask to_exclude;
        int move_count = 0;
        //TODO: MAX_MOVES is domain dependent, break in abstraction
        //HUGE time difference between MAX_EMPTY and MAX_MOVES
        //is MAX_EMPTY enough moves for a game to be "resolved"
        //while( move_count < MAX_MOVES-n_moves_made && 
        while( move_count < MAX_EMPTY-n_moves_made && 
               !scratch_state->isTerminal() ){
            int action = scratch_state->randomAction( &to_exclude, true );

            //printf( "%s\n\n", scratch_state->toString().c_str() );
            //cout << "hit any key..." << endl;
            //cin.ignore();

            ++move_count;
        }

        int rewards[nplayers];
        scratch_state->getRewards( rewards );
        for( int p=0; p<nplayers; ++p ){
            total_rewards[p] += rewards[p];
        }
    }
    delete scratch_state;
}


void MCTS::backprop( MCTS_Node* node, 
                     int*       rewards,
                     int        num_players ){
    while( !node->is_root ){
        node->visit_count++;
        for( int i=0; i < num_players; i++ ){
            //cout << "rewardSSS: " << node->total_rewards[i] << endl;
            node->total_rewards[i] += rewards[i];
        }
        node = node->parent;
    }
    return;
}

//deprecated
/*
void MCTS::launchSimulationKernel( GoState* gss, int* rewards ){
    int white_win = 0;
    int black_win = 0;

    //a decent guess at the number of actions taken thus far
    int n_moves_made = gss->movesMade(); //MAX_EMPTY - gss->num_open;
    GoState* linear = (GoState*) gss->copy();
    for( int i=0; i<NUM_SIMULATIONS; i++ ){
        gss->copyInto( linear );

        //NOTE: Turn timing off when OpenMP used, clock() synchronizes
        //      in the kernel and slows everything down
        BitMask to_exclude;
        int move_count = 0;
        while( move_count < MAX_MOVES-n_moves_made && 
               !linear->isTerminal() ){
            int action = linear->randomAction( &to_exclude, true );

            //printf( "%s\n\n", linear->toString().c_str() );
            //cout << "hit any key..." << endl;
            //cin.ignore();
            ++move_count;
        }

        int rewards[2];
        linear->getRewards( rewards );
        if( rewards[0] == 1 ){
            white_win++;
        }
        else if( rewards[1] == 1 ){
            black_win++;
        }
    }
    delete linear;
    assert( white_win+black_win == NUM_SIMULATIONS );
    rewards[0] = white_win;
    rewards[1] = black_win;
}
*/


