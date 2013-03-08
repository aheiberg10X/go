#include <iostream>
#include "mcts_node.h"

#include <stdlib.h>
using namespace std;

//for initializing the root node
MCTS_Node::MCTS_Node(int anum_players, int anum_actions){
    incCreated();
    num_players = anum_players;
    num_actions = anum_actions;
    tried_actions = new BitMask; 

    children = new MCTS_Node* [anum_actions];
    is_root = true;
    marked = true;
    
    total_rewards = new int[num_players];
    for( int i=0; i < num_players; i++ ){
        total_rewards[i] = 0;
    }
    fully_expanded = false;
}

//for initializing non-root nodes
MCTS_Node::MCTS_Node( MCTS_Node* aparent, int aaction ){
    incCreated();
    is_root = false;
    marked = false;
    parent = aparent;
    action = aaction;
    
    num_players = aparent->num_players;
    num_actions = aparent->num_actions;

    tried_actions = new BitMask; 

    children = new MCTS_Node* [num_actions];

    parent->tried_actions->set(action, true);
    parent->children[action] = this;

    visit_count = 0;

    total_rewards = new int[num_players];
    for( int i=0; i < num_players; i++ ){
        total_rewards[i] = 0;
    }

    value = 0;
    fully_expanded = false;
}

int MCTS_Node::num_nodes_created = 0;
int MCTS_Node::num_nodes_destroyed = 0;

void MCTS_Node::incCreated(){
    num_nodes_created++;
}

void MCTS_Node::incDestroyed(){
    num_nodes_destroyed++;
}
    
MCTS_Node::~MCTS_Node(){
    incDestroyed();
    //cout << "NODE DELETED" << endl;
    delete[] total_rewards;
    //cout << "wat" << endl;
    for( int i=0; i<num_actions; i++ ){
        if( tried_actions->get(i) ){
            //cout << "going" << endl;
            delete children[i];
            //cout << "on" << endl;
        }    
        //else{
        //delete children[i];
        //}
    }
    delete[] children;
    delete tried_actions;
    //delete children;
}
