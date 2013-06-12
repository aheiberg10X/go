#ifndef MCTS_NODE_H
#define MCTS_NODE_H

/*#include <map>*/
/*#include <set>*/
#include "bitmask.h"
#include <iostream>

class MCTS_Node {
public :
    bool is_root;
    MCTS_Node* parent;
    int action;
    int visit_count;
    int* total_rewards;
    int num_players;
    int num_actions;
    double value;

    //If tried_actions.get(ix) == 1, then there is a pointer to that child
    //in children[ix]
    BitMask* tried_actions;
    MCTS_Node** children;

    bool marked;

    MCTS_Node(int anum_players, int anum_actions);

    MCTS_Node( MCTS_Node* aparent, int aaction );

    ~MCTS_Node();
    
    //used when tracking mem leaks
    /*
    static int num_nodes_created;
    static int num_nodes_destroyed;
    static void incCreated();
    static void incDestroyed();
    */

};
#endif
