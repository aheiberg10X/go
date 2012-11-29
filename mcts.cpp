#include "mcts.h"
#include <assert.h>
#include <iostream>
#include <math.h>

//for debugging
#include "gostate_struct.h"
#include "kernel.h"

using namespace std;

MCTS::MCTS( Domain* d ){
    domain = d;
}

//returns an action
int MCTS::search( void* root_state ){
    assert( !domain->isChanceAction( root_state ) );
    int num_players = domain->getNumPlayers( root_state );
    int num_actions = domain->getNumActions( root_state );
    MCTS_Node* root_node = new MCTS_Node( num_players, num_actions );

    MCTS_Node* node;
    void* state;
    int rewards[num_players];

    int iterations = 0;
    while( iterations < NUM_ITERATIONS ){
        state = domain->copyState(root_state);
        //cout << endl << endl << "iteration: " << iterations << endl;
        //cout << "\n\nroot state: " << ((GoStateStruct*) state)->toString() << endl;

        node = treePolicy( root_node, state );
        //cout << "state after tree policy: " << ((GoStateStruct*) state)->toString() << endl;

        launchSimulationKernel( (GoStateStruct*) state, rewards );

        backprop( node, rewards, num_players );
        iterations += 1; 
        domain->deleteState( state );
    }

    assert( root_node->is_root );
    
    //print out info on each child for debugging
    MCTS_Node* child;
    for( int i=0; i<root_node->num_actions; i++){
        if( root_node->tried_actions->get(i) ){
            child = root_node->children[i];
            cout << "action: " << i << " EV:";
            // << child->visit_count << " total_rewards: ";
            for( int j=0; j< num_players; j++ ){
                cout << (double) child->total_rewards[j] / child->visit_count << ", ";
            }
            cout << endl;
        }
        else{
            cout << "didn't try: " << i << endl;
        }
    }

    int player_ix = domain->getPlayerIx(state);
    MCTS_Node* best_child = bestChild( root_node, player_ix, true );
    int best_action = best_child->action;

    delete root_node;

    return best_action;
}

MCTS_Node* MCTS::treePolicy( MCTS_Node* node, 
                             void*     state ){
    //return valueFunction( node, state )
    return randomPolicy( node, state );
}

MCTS_Node* MCTS::valueFunction( MCTS_Node* node, 
                                void* state ){
    //TODO
    return node;
}

MCTS_Node* MCTS::randomPolicy( MCTS_Node* root_node,
                               void*     uncast_state ){
    int action;
    MCTS_Node* node = root_node;
    BitMask empty_to_exclude;

    while( node->marked && !domain->isTerminal( uncast_state ) ){
        action = domain->randomAction( uncast_state, 
                                       &empty_to_exclude );
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

int MCTS::scoreNode( MCTS_Node* node, 
                     MCTS_Node* parent, 
                     int        player_ix, 
                     bool just_exploitation ){
    int creward = node->total_rewards[player_ix];
    int cvisit = node->visit_count;
    int pvisit = parent->visit_count;
    double exploitation = creward / ((double) cvisit);
    double exploration = BALANCING_CONSTANT*sqrt( 2*log(pvisit) / cvisit );
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
    double max_score = 0;
    double score;
    MCTS_Node* child;

    cout << "player_ix: " << player_ix << endl;
    for( int ix=0; ix < parent->num_actions; ix++ ){
        if( parent->tried_actions->get(ix) ){
            child = parent->children[ix];
            score = scoreNode( child, parent, player_ix, just_exploitation );
                //((double) child->total_rewards[player_ix]) / child->visit_count;
            if( uninit ){
                max_ix = ix;
                max_score = score;
                uninit = false;
            }
            else{
                if( score > max_score ){
                    max_ix = ix;
                    max_score = score;
                }
            }
        }
    }
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

