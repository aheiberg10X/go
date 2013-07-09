#ifndef MCTS_NODE_H
#define MCTS_NODE_H

#include "bitmask.h"
#include <iostream>
#include <stdlib.h>

using namespace std;

class MCTS_Node {
public :
    bool is_root;
    MCTS_Node* parent;
    int action;  //action taken from parent
    int visit_count;
    int* total_rewards;
    int num_players;
    int num_actions;
    double value;  //if using a value function, store it here

    //If tried_actions.get(ix) == 1, then there is a pointer to that child
    //in children[ix]
    BitMask* tried_actions;
    MCTS_Node** children;

    bool marked;

    //for initializing the root node
    MCTS_Node(int anum_players, int anum_actions);

    //for initializing non-root nodes
    MCTS_Node( MCTS_Node* aparent, int aaction );

    ~MCTS_Node();

};
#endif
