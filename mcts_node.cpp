#include <iostream>
#include "mcts_node.h"

#include <stdlib.h>
using namespace std;

MCTS_Node::MCTS_Node(int anum_players, int anum_actions){
    incNumNodes();
    num_players = anum_players;
    num_actions = anum_actions;
    tried_actions = new bool[anum_actions];
    for( int i=0; i<num_actions; i++ ){
        tried_actions[i] = false;
    }   

    children = new MCTS_Node* [anum_actions];
    is_root = true;
    marked = true;
}

MCTS_Node::MCTS_Node( MCTS_Node* aparent, int aaction ){
    incNumNodes();
    is_root = false;
    marked = false;
    parent = aparent;
    action = aaction;
    
    num_players = aparent->num_players;
    num_actions = aparent->num_actions;

    tried_actions = new bool[num_actions];
    for( int i=0; i<num_actions; i++ ){
        tried_actions[i] = false;
    }   

    children = new MCTS_Node* [num_actions];

    //TODO
    //this compiles?!?!?!?
    //parent->d_action_child[action] = this;
    //parent->tried.insert(action);
    //
    //TODO: cant call state->action2ix but need to
    tried_actions[action] = true;
    children[action] = this;



    visit_count = 0;

    total_rewards = new int[num_players];
    for( int i=0; i < num_players; i++ ){
        total_rewards[i] = 0;
    }

    //tentative?
    value = 0;
    fully_expanded = false;
}

int MCTS_Node::num_nodes = 0;

void MCTS_Node::incNumNodes(){
    num_nodes++;
}
//void MCTS_Node::setNumNodes(){
//num_nodes = 0;
//}

MCTS_Node::~MCTS_Node(){
    cout << "NODE DELETED" << endl;
    delete total_rewards;
    //tried.clear();
    //map<int,MCTS_Node*>::iterator it;
    //for( it=d_action_child.begin(); it!=d_action_child.end(); it++){
    //delete (*it).second;
    //}
    //d_action_child.clear();
    delete tried_actions;
    delete children;
}
