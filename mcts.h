#ifndef MCTS_H
#define MCTS_H

#include <string>
#include "mcts_node.cpp"
#include "domain.h"

class MCTS {
public :
    std::string domain_name;
    Domain* domain;
    int num_players;
    
    MCTS( Domain* domain );

    //returns an action
    int search( void* root_state );

    MCTS_Node* treePolicy( MCTS_Node* node, 
                           void**     state );

    MCTS_Node* randomPolicy( MCTS_Node* node, 
                             void**     state );

    MCTS_Node* uctPolicy( MCTS_Node* node, 
                          void**    state );

    //side effect: updates pstate by applying action
    MCTS_Node* expand( MCTS_Node* parent, 
                       void*      pstate, 
                       int        action );

    int scoreNode( void*      state, 
                   MCTS_Node* node, 
                   MCTS_Node* parent, 
                   int        player_ix, 
                   double     balancing_constant );

    MCTS_Node* bestChild( MCTS_Node* parent, 
                          void*      pstate, 
                          int        player_ix );

    void defaultPolicy( int*    rewards, 
                        void** state );

    void backprop( MCTS_Node* node, 
                   int*       rewards,
                   int        num_rewards );
};

#endif
