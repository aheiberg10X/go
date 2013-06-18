//#include "gostate.h"
#include "gostate.h"
//#include "godomain.cpp"
#include "mcts.h"
#include "queue.h"
#include "bitmask.h"
#include "zobrist.h"

//parallel
#include <omp.h>

#include <assert.h>
#include <iostream>
#include <time.h>
#include <stdint.h>
#include <time.h>

//hard coded data
#include "weights.h"
#include "board_100th.h"

//value function
#include "value_functions/value2.h"

using namespace std;

void testValuePolicy(){
    ZobristHash* zh = new ZobristHash;
    zh->ctor();
    MCTS mcts;
    MCTS_Node* dummy_node = new MCTS_Node( 2, BOARDSIZE );
    MCTS_State* gss = new GoState(zh);
    ((GoState*) gss)->MATLAB2board( game1234 );
    //gss->applyAction(BIGDIM+1,true);
    //gss->applyAction(BIGDIM+3,true);
    //gss->applyAction(BIGDIM+2,true);
    //gss->applyAction(BIGDIM+4,true);
    //gss->applyAction(2*BIGDIM+3,true);
    //gss->applyAction(BIGDIM+5,true);
    //gss->applyAction(2*BIGDIM+4,true);
    //cout << gss->toString() << endl;

    //mcts.valuePolicy( dummy_node, gss );
    string dummy[BOARDSIZE];
    cout << GoState::prettyBoard( dummy, 4 );
};

void testDefaultPolicy(){
    ZobristHash* zh = new ZobristHash;
    zh->ctor();
    MCTS mcts;
    GoState* gss = new GoState(zh);

    int rewards[2];
    clock_t t1,t2;

    //gss->MATLAB2board( game1234 );
    //BitMask* to_exclude = new BitMask;
    //for( int i=0; i<300; i++ ){
    //int action = gss->randomAction( to_exclude, true );
    //}
    //cout << "random board with 100 played" << gss->toString() << endl;
    t1 = clock();
    mcts.defaultPolicy( rewards, gss );
    //mcts.launchSimulationKernel( gss, rewards );
    t2 = clock();
    cout << "time taken: " << ((float) (t2-t1)) / CLOCKS_PER_SEC << endl;
};

void playGame(/*params for treePolicy configuration*/){
    ZobristHash* zh = new ZobristHash;
    zh->ctor();
    MCTS mcts;
    GoState* gs = new GoState(zh);

    int tid;
    int best_action;

    MCTS_Node** search_trees = new MCTS_Node* [N_ROOT_THREADS];
    GoState** states = new GoState* [N_ROOT_THREADS];

    int nmoves = 0;
    clock_t total_time_1 = clock();
    while( nmoves <= MAX_MOVES && !gs->isTerminal() ){
        //initialize each thread's copy of state
        clock_t t1 = clock();
        for( int i=0; i<N_ROOT_THREADS; i++ ){
            states[i] = (GoState*) gs->copy();
        }

        //do parallel tree search
        //pass in the search tree, sans estimates, from laste iteration?
        #pragma omp parallel for shared(states, search_trees) \
                                 private(tid) \
                                 num_threads(N_ROOT_THREADS)
        for( tid=0; tid<N_ROOT_THREADS; tid++){
            search_trees[tid] = mcts.search( states[tid] );
        }

        //aggregate results
        for( int ix=0; ix<search_trees[0]->num_actions; ix++){
            //put aggregate results into first thread: search_trees[0]
            MCTS_Node* agg_child;
            if( ! search_trees[0]->tried_actions->get(ix) ){
                agg_child = new MCTS_Node( search_trees[0], ix );
            }
            else{
                agg_child = search_trees[0]->children[ix];
            }
            
            for( int tid=0; tid<N_ROOT_THREADS; tid++ ){
                if( search_trees[tid]->tried_actions->get(ix) ){
                    MCTS_Node* child = search_trees[tid]->children[ix];
                    agg_child->total_rewards[0] += child->total_rewards[0];
                    agg_child->total_rewards[1] += child->total_rewards[1];
                    agg_child->visit_count += child->visit_count;
                }
            }
        }

        int pix = gs->getPlayerIx();
        int best_action = mcts.bestChild( search_trees[0], 
                                          pix, 
                                          true )->action;

        //cout << "best action for player: " << pix << " is: " << best_action << endl;

        //depending on NUM_ITERATIONS and players' tree policies,
        //deleting the search trees may be wasteful
        //but for now it is no loss to delete
        for( int tid=0; tid<N_ROOT_THREADS; tid++ ){
            delete search_trees[tid];
        }
        
        //apply to uncast_state
        //assert best_action is legal
        bool legal = gs->applyAction( best_action, true );
        cout << "After the " << nmoves << " move, " << endl << gs->toString() << endl;
        assert(legal);
        clock_t t2 = clock();
        //cout << "time taken: " << ((float) (t2-t1)) / CLOCKS_PER_SEC << endl;

        //cout << "hit any key..." << endl;
        //cin.ignore();
        nmoves++;
    }
    int rewards[2];
    gs->getRewards( rewards );
    cout << "Black Wins: " << rewards[1] << endl;
    cout << "total time taken: " << ((float) (clock()-total_time_1)) / CLOCKS_PER_SEC << endl;
    //do something with the results
};


//based on 5x5 board
void testManhattanDist(){
    ZobristHash* zh = new ZobristHash;
    zh->ctor();
    MCTS mcts;
    MCTS_Node* dummy_node = new MCTS_Node( 2, BOARDSIZE );
    GoState* gss;
    pair<int,int> dists;

    //1
    gss = new GoState(zh);
    gss->applyAction(3*BIGDIM+3,true);
    gss->applyAction(BIGDIM+5,true);
    gss->applyAction(3*BIGDIM+2,true);
    cout << gss->toString() << endl;
    dists = gss->getManhattanDistPair( 3*BIGDIM+3 );
    assert( dists.first == 1 && dists.second == 4 );
    delete gss;

    //2
    gss = new GoState(zh);
    gss->applyAction(BIGDIM+1,true);
    gss->applyAction(BIGDIM+5,true);
    gss->applyAction(5*BIGDIM+5,true);
    cout << gss->toString() << endl;
    dists = gss->getManhattanDistPair( 5*BIGDIM+5 );
    assert( dists.first == 8 && dists.second == 4 );
    delete gss;


}

void testSetBinaryFeatures(){
    const int nfeatures = 31;
    int features[ nfeatures * MAX_EMPTY ] = {0};

    ZobristHash* zh = new ZobristHash;
    zh->ctor();
    GoState gss(zh);

    gss.applyAction(3*BIGDIM+3,true);
    gss.applyAction(3*BIGDIM+2,true);
    gss.applyAction(2*BIGDIM+3,true);
    gss.setBinaryFeatures( features, nfeatures );

    cout << "done setting" << endl;

    string feature_str = gss.featuresToString( features, nfeatures );
    cout << feature_str << endl;

}

void timeSetBinaryFeatures(){
    clock_t t1 = clock();
    
    ZobristHash* zh = new ZobristHash;
    zh->ctor();
    GoState* gss = new GoState(zh);
    gss->MATLAB2board( game1234 );
    cout << gss->toString() << endl;

    const int nfeatures = 31;
    int features[ nfeatures * MAX_EMPTY ] = {0};

    for( int i=0; i<10000; ++i ){
        gss->setBinaryFeatures( features, nfeatures ); 
        //if( i == 0 ){
        //cout << gss->featuresToString( features, nfeatures ) << endl;
        //}
    }

    clock_t t2 = clock();
    cout << "time taken: " << ((float) (t2-t1)) / CLOCKS_PER_SEC << endl;
}

int main(){
    srand(time(0));
    cout << sizeof(GoState) << endl;
    
    //mclInitializeApplication(NULL,0);
    //value2Initialize();

    //testValuePolicy();
    //testDefaultPolicy();
    //playGame();
    //testManhattanDist();
    //testSetBinaryFeatures();
    timeSetBinaryFeatures();
   
    //value2Terminate();
    //mclTerminateApplication();

    return 0;

};
