#include "mcts.h"
#include <assert.h>
#include <iostream>

//for debugging
#include "gostate.h"

using namespace std;

MCTS::MCTS( Domain* d ){
    domain = d;
}

//returns an action
int MCTS::search( void* root_state ){
    assert( !domain->isChanceAction( root_state ) );
    
    int num_players = domain->getNumPlayers( root_state );
    MCTS_Node* root_node = new MCTS_Node(num_players);

    MCTS_Node* node;
    void* state;
    int rewards[num_players];

    int iterations = 0;
    while( iterations < 1000 ){
        cout << "iteration: " << iterations << endl;
        state = domain->copyState(root_state);
        //cout << "\n\nroot state: " << ((GoState*) state)->toString() << endl;
        node = treePolicy( root_node, (void**) &state );
        //cout << "state after tree policy: " << ((GoState*) state)->toString() << endl;
        defaultPolicy( rewards, (void**) &state );
        cout << "state after simulation: " << ((GoState*) state)->toString() << endl;

        cout << "rewards: " << rewards[0] << "," << rewards[1] << endl;

        backprop( node, rewards, num_players );
        iterations += 1; 
    }

    //TODO
    //print out info on each child for debugging
    assert( root_node->is_root );
    map<int,MCTS_Node*>::iterator it;
    for( it  = root_node->d_action_child.begin(); 
         it != root_node->d_action_child.end();
         it++ ){
        //cout << "action: " << (*it).first <<
        //"\nvisit count: " << (*it).second->visit_count << endl;
        //cout << "rewards: ";
        for( int i=0; i < num_players; i++) {
            //cout << (*it).second->total_rewards[i] << ", ";
        }
        //cout << endl;
    }

    int player_ix = domain->getPlayerIx(root_state);
    MCTS_Node* best_child = bestChild( root_node, root_state, player_ix );
    return best_child->action;
}

MCTS_Node* MCTS::treePolicy( MCTS_Node* node, 
                             void**     state ){
    return randomPolicy( node, state );
}

MCTS_Node* MCTS::randomPolicy( MCTS_Node* node,
                               void**     p_uncast_state ){
    int action;
    GoState* state = (GoState*) *p_uncast_state;
    //cout << "marked: " << node->marked << "terminal: " << domain->isTerminal( state ) << endl;
    set<int> empty_to_exclude;
    while( node->marked && !domain->isTerminal( state ) ){
        action = domain->randomAction( p_uncast_state, 
                                       empty_to_exclude );
        //if( !domain->isChanceAction ){
        //}
        //else{
        //}
        //cout << "applying action: " << action << endl;
        domain->applyAction( p_uncast_state, 
                             action, 
                             true );
        if( node->tried.find(action) != node->tried.end() ){
            node = node->d_action_child[action];
        }
        else{
            node = new MCTS_Node( node, action );
        }

    }
    return node;
}

MCTS_Node* MCTS::uctPolicy( MCTS_Node* node, 
                            void**     state ){
    return node;
}

//side effect: updates pstate by applying action
MCTS_Node* MCTS::expand( MCTS_Node* parent, 
                         void*      pstate, 
                         int        action ){
    return parent;
}

int MCTS::scoreNode( void*      state, 
                     MCTS_Node* node, 
                     MCTS_Node* parent, 
                     int        player_ix, 
                     double     balancing_constant ){
    return 42;
}

MCTS_Node* MCTS::bestChild( MCTS_Node* parent, 
                            void*      pstate, 
                            int        player_ix ){
    return parent;
}


void MCTS::defaultPolicy( int* rewards, 
                          void** p_uncast_state ){
    
    //cout << "inside defaultPolicy" << endl;
    int count = 0;
    int action;
    set<int> to_exclude;
    while( !domain->isTerminal( *p_uncast_state) ){
        if( count > 1000 ){
            cout << "probably in loop" << endl;
            break;
        }
        action = domain->randomAction( p_uncast_state, to_exclude );
        //GoState* state = (GoState*) *p_uncast_state;
        //cout << "applying: " << action << " to:" << state->toString() << endl;
        domain->applyAction( p_uncast_state, action, true );
        count++;
    }
    
    domain->getRewards( rewards, *p_uncast_state );

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
