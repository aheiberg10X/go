#ifndef MCTS_H
#define MCTS_H

#include <assert.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <time.h>
#include <string>

#include "mcts_node.h"
#include "mcts_state.h"

//learned feature weights
#include "weights.h"

//MATLAB compiled implementation.  Deprecated (bceause slow) by FeatureFuncs
/*#include "value_functions/value2.h"*/
#include "feature_funcs.h"

using namespace std;

class MCTS {
public :
    
    //perform NUM_ITERATIONS iterations of the 4-step MCTS algorithm
    //returns the search_tree, rooted at root_node
    MCTS_Node* search( MCTS_State* root_state );

    ////////////////////////////////////////////////////////////
    //   Node selection and expansion policies
    ////////////////////////////////////////////////////////////
    //treePolicy calls one (or more) of the optoins
    MCTS_Node* treePolicy( MCTS_Node*  node, 
                           MCTS_State* state );

    MCTS_Node* randomPolicy( MCTS_Node*  node, 
                             MCTS_State* state );

    MCTS_Node* valuePolicy( MCTS_Node*  node,
                            MCTS_State* state );
    
    /*MCTS_Node* valuePolicyMATLAB( MCTS_Node*  node,*/
    /*MCTS_State* state );*/
  
    MCTS_Node* uctPolicy( MCTS_Node*  node, 
                          MCTS_State* state );

    ///////////////////////////////////////////////////////////
    //  Based on MCTS simulations, get the values of a node
    ///////////////////////////////////////////////////////////
    float scoreNode( MCTS_Node* node, 
                     MCTS_Node* parent, 
                     int        player_ix, 
                     bool       just_exploitation );

    MCTS_Node* bestChild( MCTS_Node* parent, 
                          int        player_ix,
                          bool       just_exploitation );

    ////////////////////////////////////////////////////////////
    //  Simulate play from the edge of the search tree
    ///////////////////////////////////////////////////////////
    void defaultPolicy( int*    rewards, 
                        MCTS_State*   state );

    //Backprop the rewards from the simulation/playout
    void backprop( MCTS_Node* node, 
                   int*       rewards,
                   int        num_rewards );
    
};

#endif
