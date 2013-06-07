#include "mcts.h"
#include <assert.h>
#include <iostream>
#include <math.h>

//for debugging
#include "gostate.h"
#include "kernel.h"

//value function
#include "value_functions/value2.h"
#include "matrix.h"
#include "weights.h"
#include <time.h>

using namespace std;

//ctor
MCTS::MCTS( Domain* d ){
    domain = d;
}

//perform NUM_ITERATIONS iterations of the 4-step MCTS algorithm
//returns the search_tree, rooted at root_node
MCTS_Node* MCTS::search( void* root_state ){
    assert( !domain->isChanceAction( root_state ) );

    int num_players = domain->getNumPlayers( root_state );
    int num_actions = domain->getNumActions( root_state );
    MCTS_Node* root_node = new MCTS_Node( num_players, num_actions );

    //create a copy of root_state, this will be our scratch space
    MCTS_Node* node;
    int rewards[num_players];

    int iterations = 0;
    clock_t tree_policy_time, simulation_time;
    clock_t t1,t2;
    t1 = clock();
    while( iterations < NUM_ITERATIONS ){
        void* state = domain->copyState( root_state );
        cout << endl << endl << "iteration: " << iterations << endl;
        //cout << "\n\nroot state: " << ((GoStateStruct*) state)->toString() << endl;

        node = treePolicy( root_node, state );
        //cout << "state after tree policy: " << endl; //((GoStateStruct*) state)->toString() << endl;

        //TODO
        //break in domain/state interface
        //keep for speed?  less abstraction to call through this way
        defaultPolicy( rewards, state );
        //launchSimulationKernel( (GoStateStruct*) state, rewards );
        //cout << "rewards w/b: " << rewards[0] << "/" << rewards[1] << endl;

        backprop( node, rewards, num_players );
        iterations += 1; 
        domain->deleteState( state );

    }
    t2 = clock();

    assert( root_node->is_root );
    cout << "Search time: " << ((float) t2-t1) / CLOCKS_PER_SEC << endl;
    
    return root_node;
}

MCTS_Node* MCTS::treePolicy( MCTS_Node* node, 
                             void*     state ){

    //return randomPolicy( node, state );

    int player = domain->getPlayerIx(state);
    //BLACK uses valueFucntion
    if( player == 1 ){
        return valuePolicy( node, state );
    }
    else{
        return randomPolicy( node, state );
    }
}

//this go specific from the beginning, so won't mind breaking abstraction
MCTS_Node* MCTS::valuePolicy( MCTS_Node* node, 
                              void* uncast_state ){
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

    GoStateStruct* gss = (GoStateStruct *) uncast_state;

    GoStateStruct** working_gss = new GoStateStruct* [N_VALUE_THREADS];
    for( int i=0; i<N_VALUE_THREADS; i++ ){
       working_gss[i] = new GoStateStruct;
    }

    while( node->marked && !domain->isTerminal( uncast_state ) ){
        cout << "valuePolicy node: " << node << endl;
        //cout << "what now?" << endl;
        //TODO, what is min valueFunction will take?
        
        //int legal_actions[BOARDSIZE];
        float action_values[BOARDSIZE];

        int ta = clock();
        #pragma omp parallel for shared(action_values, gss, working_gss ) \
                                 num_threads(N_VALUE_THREADS)
        for( int action=0; action<BOARDSIZE; action++ ){
            if( gss->board[action] != EMPTY ){ 
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
        domain->applyAction(uncast_state, chosen_action, true );
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

MCTS_Node* MCTS::randomPolicy( MCTS_Node* root_node,
                               void*     uncast_state ){
    int action;
    MCTS_Node* node = root_node;
    BitMask empty_to_exclude;
    empty_to_exclude.clear();

    while( node->marked && !domain->isTerminal( uncast_state ) ){
        action = domain->randomAction( uncast_state, 
                                       &empty_to_exclude,
                                       true );
        //cout << "state->action : " << ((GoStateStruct* ) uncast_state)->action << endl;
        if( node->tried_actions->get(action) ){
            node = node->children[action];
        }
        else{
            node = new MCTS_Node( node, action );
        }
        //cout << "node: " << node << endl;
    }
    node->marked = true;
    return node;
}

MCTS_Node* MCTS::uctPolicy( MCTS_Node* node, 
                            void*    state ){
    while( node->marked and !domain->isTerminal( state ) ){
        int action = domain->randomAction( state, node->tried_actions, true );
        //TODO break in abstraction
        bool fully_expanded = action == EXCLUDED_ACTION; 
        if( !fully_expanded ){
            //move already applied
            node = new MCTS_Node( node, action );
            //node = expand( node, state, action );
        }
        else{
            int player_ix = domain->getPlayerIx(state);
            node = bestChild( node, player_ix, false );
            bool is_legal = domain->applyAction( state, node->action, true );
        }
    }    
    node->marked = true;
    return node;
}

//MCTS_Node* MCTS::expand( MCTS_Node* parent, 
//void*      pstate, 
//int        action ){
//bool is_legal = domain->applyAction( pstate, action, true );
//assert(is_legal);
//return new MCTS_Node( parent, action );
//}

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

/*deprecated by launchSimulationKernel */
void MCTS::defaultPolicy( int* total_rewards, 
                          void* uncast_state ){
    const int nplayers = domain->getNumPlayers( uncast_state );
    int total_results[nplayers];

    void* scratch_state = domain->copyState( uncast_state );

    //a decent guess at the number of actions taken thus far
    int n_moves_made = domain->movesMade( uncast_state );
    //int n_moves_made = MAX_EMPTY - gss->num_open;

    for( int i=0; i<NUM_SIMULATIONS; i++ ){
        //GoStateStruct* linear = (GoStateStruct*) gss->copy();
        domain->copyStateInto( uncast_state, scratch_state );

        //NOTE: Turn timing off when OpenMP used, clock() synchronizes
        //      in the kernel and slows everything down
        BitMask to_exclude;
        int move_count = 0;
        //TODO: MAX_MOVES is domain dependent, break in abstraction
        while( move_count < MAX_MOVES-n_moves_made && 
               !domain->isTerminal( scratch_state ) ){
            //!linear->isTerminal() ){
            //int action = linear->randomAction( &to_exclude, true );
            int action = domain->randomAction( scratch_state, &to_exclude, true );

            //printf( "%s\n\n", linear->toString().c_str() );
            //cout << "hit any key..." << endl;
            //cin.ignore();

            ++move_count;
        }

        int rewards[nplayers];
        //linear->getRewards( rewards );
        domain->getRewards( rewards, scratch_state );
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

