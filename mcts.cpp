#include "mcts.h"
#include <assert.h>

using namespace std;

MCTS::MCTS( Domain* d ){
    domain = d;
}

//returns an action
int MCTS::search( void* root_state ){
    assert( domain->isChanceAction( root_state ) );
    MCTS_Node* root_node = new MCTS_Node();

    MCTS_Node* node = root_node;
    void* state;
    int rewards[ domain->getNumPlayers( state ) ];

    int iterations = 0;
    while( iterations < 100 ){
        state = domain->copyState(root_state);
        node = treePolicy( root_node, root_state );
        defaultPolicy( rewards, state );
        backprop( node, rewards );
        iterations += 1; 
    }
    return 42;
}

MCTS_Node* MCTS::treePolicy( MCTS_Node* node, 
                             void*      state ){
    return node;
}

MCTS_Node* MCTS::uctPolicy( MCTS_Node* node, 
                            void*     state ){
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
                          void* state ){
    return;
}

void MCTS::backprop( MCTS_Node* node, 
                     int*       rewards ){
    return;
}

