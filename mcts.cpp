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
    int num_actions = domain->getNumActions( root_state );
    MCTS_Node* root_node = new MCTS_Node( num_players, num_actions );

    MCTS_Node* node;
    void* state;
    int rewards[num_players];

    int iterations = 0;
    while( iterations < 1000 ){
        state = domain->copyState(root_state);
        //cout << endl << endl << "iteration: " << iterations << endl;
        //cout << "\n\nroot state: " << ((GoState*) state)->toString() << endl;

        node = treePolicy( root_node, (void**) &state );

        //cout << "state after tree policy: " << ((GoState*) state)->toString() << endl;
        defaultPolicy( rewards, (void**) &state );
        //cout << "state after simulation: " << ((GoState*) state)->toString() << endl;

        //cout << "rewards: " << rewards[0] << "," << rewards[1] << endl;

        backprop( node, rewards, num_players );
        iterations += 1; 
        domain->deleteState( state );
    }

    //TODO
    //print out info on each child for debugging
    assert( root_node->is_root );
    //MCTS_Node* child;
    //for( int i=0; i<root_node->num_actions; i++){
    //if( root_node->tried_actions->get(i) ){
    //child = root_node->children[i];
    //cout << "action: " << i << " EV:";
    //// << child->visit_count << " total_rewards: ";
    //for( int j=0; j< num_players; j++ ){
    //cout << (double) child->total_rewards[j] / child->visit_count << ", ";
    //}
    //cout << endl;
    //}
    //else{
    //cout << "didn't try: " << i << endl;
    //}
    //}

    int player_ix = domain->getPlayerIx(state);
    MCTS_Node* best_child = bestChild( root_node, player_ix );
    int best_action = best_child->action;
    cout << "deleteing root_node" << endl;
    delete root_node;
    cout << "num_nodes created: " << MCTS_Node::num_nodes_created << endl;
    cout << "num_nodes destoryed: " << MCTS_Node::num_nodes_destroyed << endl;
    return best_action;
}

MCTS_Node* MCTS::treePolicy( MCTS_Node* node, 
                             void**     state ){
    return randomPolicy( node, state );
}

MCTS_Node* MCTS::randomPolicy( MCTS_Node* root_node,
                               void**     p_uncast_state ){
    int action;
    //cout << "marked: " << node->marked << "terminal: " << domain->isTerminal( state ) << endl;
    MCTS_Node* node = root_node;
    //bool empty_to_exclude[node->num_actions];
    //for(int i=0; i<node->num_actions; i++ ){
    //empty_to_exclude[i] = false;
    //}
    BitMask empty_to_exclude (node->num_actions);

    while( node->marked && !domain->isTerminal( *p_uncast_state ) ){
        action = domain->randomAction( p_uncast_state, 
                                       &empty_to_exclude );
        //if( !domain->isChanceAction ){
        //}
        //else{
        //}
        domain->applyAction( p_uncast_state, 
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
    assert(false);
    return 42;
}

MCTS_Node* MCTS::bestChild( MCTS_Node* parent, 
        //void*      pstate, 
                            int        player_ix ){
    bool uninit = true;
    int max_ix = 0;
    double max_score = 0;
    double score;
    //GoState* gs = (GoState*) pstate;
    MCTS_Node* child;

    cout << "player_ix: " << player_ix << endl;
    for( int ix=0; ix < parent->num_actions; ix++ ){
        if( parent->tried_actions->get(ix) ){
            child = parent->children[ix];
            score = ((double) child->total_rewards[player_ix]) / child->visit_count;
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


void MCTS::defaultPolicy( int* rewards, 
                          void** p_uncast_state ){
    
    //cout << "inside defaultPolicy" << endl;
    int count = 0;
    int action;

    GoState* s = ((GoState*) *p_uncast_state);
    BitMask to_exclude( s->boardsize );
    //bool to_exclude[s->boardsize];
    //for( int i=0; i<s->boardsize; i++ ){
    //to_exclude[i] = false;
    //}
    while( !domain->isTerminal( *p_uncast_state) ){
        if( count > 1000 ){
            cout << "probably in loop" << endl;
            break;
        }
        action = domain->randomAction( p_uncast_state, &to_exclude );

        GoState* state = (GoState*) *p_uncast_state;
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

