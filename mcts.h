#ifndef MCTS_H
#define MCTS_H

#include <string>
#include "mcts_node.h"
#include "domain.h"

class MCTS {
public :
    std::string domain_name;
    Domain* domain;
    int num_players;
    
    MCTS( Domain* domain );

    //returns an action
    MCTS_Node* search( void* root_state );

    MCTS_Node* treePolicy( MCTS_Node* node, 
                           void*      state );

    MCTS_Node* randomPolicy( MCTS_Node* node, 
                             void*      state );

    MCTS_Node* valuePolicy( MCTS_Node* node,
                              void*      state );

    MCTS_Node* uctPolicy( MCTS_Node* node, 
                          void*      state );

    //side effect: updates pstate by applying action
    MCTS_Node* expand( MCTS_Node* parent, 
                       void*      pstate, 
                       int        action );

    float scoreNode( MCTS_Node* node, 
                   MCTS_Node* parent, 
                   int        player_ix, 
                   bool       just_exploitation );

    MCTS_Node* bestChild( MCTS_Node* parent, 
                          int        player_ix,
                          bool       just_exploitation );

    void defaultPolicy( int*    rewards, 
                        void*   state );

    /*void launchSimulationKernel();*/

    void backprop( MCTS_Node* node, 
                   int*       rewards,
                   int        num_rewards );
};

#endif
