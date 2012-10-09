#ifndef MCTS_NODE_H
#define MCTS_NODE_H

#include <map>

using namespace std;

class MCTS_Node {
public :
    bool is_root;
    MCTS_Node* parent;
    int action;
    int visit_count;
    int* total_rewards;
    int num_players;
    map<int,MCTS_Node*> d_action_child;
    bool marked;

    //tentative
    int value;
    bool fully_expanded;

    MCTS_Node(){
        is_root = true;
        marked = true;
    }

    MCTS_Node( MCTS_Node* aparent, int aaction ){
        is_root = false;
        marked = false;
        parent = aparent;
        action = aaction;
        visit_count = 0;

        num_players = aparent->num_players;
        total_rewards = new int[num_players];
        for( int i=0; i < num_players; i++ ){
            total_rewards[i] = 0;
        }

        //tentative?
        value = 0;
        fully_expanded = false;
    }
};

#endif
