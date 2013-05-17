#include "mcts.h"
#include <assert.h>
#include <iostream>
#include <math.h>

//for debugging
#include "gostate_struct.h"
#include "kernel.h"

//value function
#include "value_functions/value2.h"
#include "matrix.h"
#include "weights.h"
#include <time.h>

using namespace std;

MCTS::MCTS( Domain* d ){
    domain = d;
}

//returns the search_tree
MCTS_Node* MCTS::search( void* root_state ){
    assert( !domain->isChanceAction( root_state ) );
    int num_players = domain->getNumPlayers( root_state );
    int num_actions = domain->getNumActions( root_state );
    MCTS_Node* root_node = new MCTS_Node( num_players, num_actions );

    //create a copy of root_state, this will be our scratch space

    MCTS_Node* node;
    int rewards[num_players];

    int iterations = 0;
    float tree_policy_time, simulation_time;
    clock_t t1,t2;
    while( iterations < NUM_ITERATIONS ){
        void* state = domain->copyState( root_state );
        //cout << "\n\ncopied state: " << ((GoStateStruct*) state)->toString() << endl;
        //start from the beginning...
        //cout << endl << endl << "iteration: " << iterations << endl;
        //cout << "\n\nroot state: " << ((GoStateStruct*) state)->toString() << endl;

        t1 = clock();
        node = treePolicy( root_node, state );
        t2 = clock();
        tree_policy_time += ((float) t2-t1)/CLOCKS_PER_SEC;
        //cout << "state after tree policy: " << endl; //((GoStateStruct*) state)->toString() << endl;

        //TODO
        //break in domain/state interface
        //mcts should assume nothing about the uncast_state and always
        //go through domain
        t1 = clock();
        launchSimulationKernel( (GoStateStruct*) state, rewards );
        t2 = clock();
        simulation_time += ((float) t2-t1)/CLOCKS_PER_SEC;
        //cout << "rewards w/b: " << rewards[0] << "/" << rewards[1] << endl;

        backprop( node, rewards, num_players );
        iterations += 1; 
        //cout << "before delete root state zhasher->values[0]: " << ((GoStateStruct* )root_state)->zhasher->values[10] << endl;
        domain->deleteState( state );
        //cout << "after delete root state zhasher->values[0]: " << ((GoStateStruct* )root_state)->zhasher->values[10] << endl;

    }

    assert( root_node->is_root );
    cout << "Tree policy time: " << tree_policy_time << endl;
    cout << "Simulation Simte: "<< simulation_time << endl;
    //print out info on each child for debugging
    /*
    MCTS_Node* child;
    for( int i=0; i<root_node->num_actions; i++){
        if( root_node->tried_actions->get(i) ){
            child = root_node->children[i];
            //cout << "action: " << i << " EV:";
            //<< child->visit_count << " total_rewards: ";
            for( int j=0; j< num_players; j++ ){
                cout << (double) child->total_rewards[j] / child->visit_count << ", ";
            }
            //cout << endl;
        }
        else{
            //cout << "didn't try: " << i << endl;
        }
    }*/

    /*
    int player_ix = domain->getPlayerIx(state);
    MCTS_Node* best_child = bestChild( root_node, player_ix, true );
    int best_action = best_child->action;
    */

    //delete root_node;

    //return best_action;
    return root_node;
}

MCTS_Node* MCTS::treePolicy( MCTS_Node* node, 
                             void*     state ){

    return randomPolicy( node, state );

    int player = domain->getPlayerIx(state);
    //BLACK uses valueFucntion
    if( player == 1 ){
        cout << "using value policy" << endl;
        return valuePolicy( node, state );
    }
    else{
        return randomPolicy( node, state );
    }
}

//this go specific from the beginning, so won't mind breaking abstraction
MCTS_Node* MCTS::valuePolicy( MCTS_Node* node, 
                              void* uncast_state ){

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
    GoStateStruct* working_gss = new GoStateStruct;

    while( node->marked && !domain->isTerminal( uncast_state ) ){
        cout << "what now?" << endl;
        //TODO, what is min valueFunction will take?
        double max_value = -999999;
        int max_actions[BOARDSIZE];
        int max_actions_end = 0;
        int legal_actions[BOARDSIZE];
        int legal_actions_end = 0;

        int ta = clock();
        for( int action=0; action<BOARDSIZE; action++ ){
            cout << "starting on action: " << action << endl;

            //just applied the previous action, restore state to parent
            gss->copyInto( working_gss );
    
            if( gss->board[action] == OFFBOARD ){ continue; }

            //the valuation for this action
            double value;

            //if have a value already
            if( node->tried_actions->get(action) ){
                cout << "have action" << endl;
                value = node->children[action]->value;
            }
            //else no value/node yet
            else{
                cout << "don't have value for action: " << action << endl;
                bool is_legal = working_gss->applyAction( action, true );
                if( !is_legal ){ 
                    cout << action << " is illegal, skipping" << endl;
                    continue; 
                }
                else{
                    legal_actions[legal_actions_end++] = action;
                    MCTS_Node* child_node = new MCTS_Node( node, action );
                    working_gss->board2MATLAB(matlab_board);
                    memcpy( (double*) mxGetPr(x), 
                            matlab_board, 
                            MAX_EMPTY * sizeof(double) );
                    bool success = mlfValue2(1, &V, x, w);
                    cout << "success: " << success << endl;
                    double* r = mxGetPr(V);
                    value = r[0];
                    child_node->value = value;
                }
                
            }
            cout << "value is: " << value << "\n" << endl;
            if( value == max_value ){
                max_actions[max_actions_end++] = action;
            }
            else if( value > max_value ){
                max_value = value;
                max_actions[0] = action;
                max_actions_end = 1;
            }

        }// for action
        int tb = clock(); 
        cout << "evaling: " << legal_actions_end << " legal actions took: " << (float (tb-ta))/CLOCKS_PER_SEC << "s" << endl;
        break;

        //the action we choose to take
        int chosen_action;
        cout << max_actions_end << " actions with max value" << endl;
        double r = (double) rand() / RAND_MAX;
        cout << "random: " << r << endl;


        int rix;
        if( r < EGREEDY ){
            //choose randomly amongst the best valued actions
            rix = rand() % max_actions_end;
            cout << "going greedy, using action: " << max_actions[rix] << " (rix =" << rix << ")" << endl;
        }
        else{
            cout << "going random" << endl;
            rix = rand() % legal_actions_end;
        }
        chosen_action = max_actions[rix];
        cout << "chosen action: " << chosen_action << endl;
        node = node->children[chosen_action];
        //cout << "hit any key" << endl;
        //cin.ignore();
    }//while not terminal

    mxDestroyArray(V);
    cout << "hello" << endl;
    mxDestroyArray(x);
    cout << "there" << endl;
    mxDestroyArray(w);
    cout << "returning node" << endl;
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
                                       &empty_to_exclude );

        //cout << "state->action : " << ((GoStateStruct* ) uncast_state)->action << endl;
        //if( abs(action) == abs(((GoStateStruct* ) uncast_state)->action) ){
        //cout << "randPol action: " << action << endl;
        //cout << "state->action : " << ((GoStateStruct* ) uncast_state)->action << endl;
        //assert(false);
        //
        //}
        //cout << "randPol action: " << action << endl;
        domain->applyAction( uncast_state, 
                             action, 
                             true );

        //if( node->tried.find(action) != node->tried.end() ){
        //int ix = domain
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
        int action = domain->randomAction( state, node->tried_actions );
        if( !domain->fullyExpanded(action) ){
            node = expand( node, state, action );
        }
        else{
            node->fully_expanded = true;
            int player_ix = domain->getPlayerIx(state);
            node = bestChild( node, player_ix, false );
        }
    }    
    node->marked = true;
    return node;
}

MCTS_Node* MCTS::expand( MCTS_Node* parent, 
                         void*      pstate, 
                         int        action ){
    bool is_legal = domain->applyAction( pstate, action, true );
    assert(is_legal);
    return new MCTS_Node( parent, action );
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

/*deprecated by launchSimulationKernel */
void MCTS::defaultPolicy( int* rewards, 
                          void* uncast_state ){
    
    //cout << "inside defaultPolicy" << endl;
    int count = 0;
    int action;

    BitMask to_exclude;

    while( !domain->isTerminal( uncast_state) ){
        if( count > 1000 ){
            cout << "probably in loop" << endl;
            break;
        }
        action = domain->randomAction( uncast_state, &to_exclude );
        //cout << "actionasdf: " << action << endl;
        //GoStateStruct* state = (GoStateStruct*) uncast_state;
        
        domain->applyAction( uncast_state, action, true );
        //cout << "applied" << endl;
        count++;
    }
   
    domain->getRewards( rewards, uncast_state );

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

