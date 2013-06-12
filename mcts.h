#ifndef MCTS_H
#define MCTS_H

#include <string>
#include "mcts_node.h"
#include "mcts_state.h"
#include "gostate.h"

class MCTS {
public :
    MCTS_Node* search( MCTS_State* root_state );

    MCTS_Node* treePolicy( MCTS_Node*  node, 
                           MCTS_State* state );

    MCTS_Node* randomPolicy( MCTS_Node*  node, 
                             MCTS_State* state );

    MCTS_Node* valuePolicy( MCTS_Node*  node,
                            MCTS_State* state );
    
    MCTS_Node* valuePolicyMATLAB( MCTS_Node*  node,
                                  MCTS_State* state );
  
    MCTS_Node* uctPolicy( MCTS_Node*  node, 
                          MCTS_State* state );

    float scoreNode( MCTS_Node* node, 
                     MCTS_Node* parent, 
                     int        player_ix, 
                     bool       just_exploitation );

    MCTS_Node* bestChild( MCTS_Node* parent, 
                          int        player_ix,
                          bool       just_exploitation );

    void defaultPolicy( int*    rewards, 
                        MCTS_State*   state );

    void backprop( MCTS_Node* node, 
                   int*       rewards,
                   int        num_rewards );
    
    /*void launchSimulationKernel( GoState* gss, int* rewards );*/
};

#endif
