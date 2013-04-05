#ifndef MCTS_NODE_H
#define MCTS_NODE_H

/*#include <map>*/
/*#include <set>*/
#include "bitmask.h"
#include <iostream>

class MCTS_Node {
public :
    static int num_nodes_created;
    static int num_nodes_destroyed;
    bool is_root;
    MCTS_Node* parent;
    int action;
    int visit_count;
    int* total_rewards;
    int num_players;
    int num_actions;
    double value;

    int avoid_false_sharing[128];

    //TODO
    //mem waste to maintain both a tried and action:child map
    //but we want ot be able to give random action a to_exclude list
    //if all we have is the map, the domain needs to know something about
    //MCTS_Nodes, which is an abstraction leak...
    /*std::set<int> tried;*/
    /*std::map<int,MCTS_Node*> d_action_child;*/

    //If tried_actions[ix] == true, then there is a pointer to that child
    //in children[ix]
    /*bool* tried_actions;*/
    BitMask* tried_actions;
    MCTS_Node** children;

    bool marked;

    //tentative
    /*int value;*/
    bool fully_expanded;

    MCTS_Node(int anum_players, int anum_actions);

    MCTS_Node( MCTS_Node* aparent, int aaction );

    ~MCTS_Node();
    
    static void incCreated();
    static void incDestroyed();

};
#endif
