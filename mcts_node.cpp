#ifndef MCTS_NODE_H
#define MCTS_NODE_H

#include <map>
#include <set>
#include <iostream>

using namespace std;

class MCTS_Node {
public :
    bool is_root;
    MCTS_Node* parent;
    int action;
    int visit_count;
    int* total_rewards;
    int num_players;

    //TODO
    //mem waste to maintain both a tried and action:child map
    //but we want ot be able to give random action a to_exclude list
    //if all we have is the map, the domain needs to know something about
    //MCTS_Nodes, which is an abstraction leak...
    set<int> tried;
    map<int,MCTS_Node*> d_action_child;
    bool marked;

    //tentative
    int value;
    bool fully_expanded;

    MCTS_Node(int anum_players){
        num_players = anum_players;
        is_root = true;
        marked = true;
    }

    MCTS_Node( MCTS_Node* aparent, int aaction ){
        is_root = false;
        marked = false;
        parent = aparent;
        action = aaction;

        //TODO
        //this compiles?!?!?!?
        parent->d_action_child[action] = this;
        parent->tried.insert(action);

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

    ~MCTS_Node(){
        delete total_rewards;
        tried.clear();
        map<int,MCTS_Node*>::iterator it;
        for( it=d_action_child.begin(); it!=d_action_child.end(); it++){
            delete (*it).second;
        }
        d_action_child.clear();
    }
};

#endif
