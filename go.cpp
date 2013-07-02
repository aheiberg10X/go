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
#include <fstream>

//hard coded data
#include "weights.h"
#include "board_100th.h"

//value function
#include "value_functions/value2.h"

#include "gaussian/gaussianiir2d.h"
#include "feature_funcs.h"

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

    clock_t t1 = clock();
    for( int i = 0; i<1000; ++i ){
        mcts.valuePolicy( dummy_node, gss );
    }
    clock_t t2 = clock();
    cout << "time taken: " << ((float) (t2-t1)) / CLOCKS_PER_SEC << endl;

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
    dists = FeatureFuncs::getManhattanDistPair( gss, 3*BIGDIM+3 );
    assert( dists.first == 1 && dists.second == 4 );
    delete gss;

    //2
    gss = new GoState(zh);
    gss->applyAction(BIGDIM+1,true);
    gss->applyAction(BIGDIM+5,true);
    gss->applyAction(5*BIGDIM+5,true);
    cout << gss->toString() << endl;
    dists = FeatureFuncs::getManhattanDistPair( gss, 5*BIGDIM+5 );
    assert( dists.first == 8 && dists.second == 4 );
    delete gss;


}

void testSetBinaryFeatures(){
    int features[ NFEATURES * MAX_EMPTY ] = {0};

    ZobristHash* zh = new ZobristHash;
    zh->ctor();
    GoState gss(zh);
    gss.MATLAB2board( game1234 );
    cout << gss.toString() << endl;
    FeatureFuncs::setBinaryFeatures( &gss, features );

    cout << "done setting" << endl;

    float features_copy[NFEATURES * MAX_EMPTY];
    copy( features, features + NFEATURES*MAX_EMPTY, features_copy );

    //string feature_str = gss.featuresToString( features, NFEATURES );
    //cout << feature_str << endl;

    int feat = 21;
    //GoState::board2csv( &features_copy[feat*MAX_EMPTY], MAX_EMPTY, DIMENSION, "feat21_unconvolved.csv" );
    gaussianiir2d( &features_copy[feat*MAX_EMPTY], DIMENSION, DIMENSION, 1, 2 );
    FeatureFuncs::board2csv( &features_copy[feat*MAX_EMPTY], MAX_EMPTY, DIMENSION, "feat21_convolved_sigma1_2steps.csv" );


}

void timeSetBinaryFeatures(){
    clock_t t1 = clock();
    
    ZobristHash* zh = new ZobristHash;
    zh->ctor();
    GoState* gss = new GoState(zh);
    gss->MATLAB2board( game1234 );
    cout << gss->toString() << endl;

    int features[ NFEATURES * MAX_EMPTY ] = {0};
    float features_copy[NFEATURES * MAX_EMPTY];

    for( int i=0; i<10000; ++i ){
        FeatureFuncs::setBinaryFeatures( gss, features ); 
        copy( features, features + NFEATURES*MAX_EMPTY, features_copy );
        for( int feat=1; feat<NFEATURES; ++feat ){
            gaussianiir2d( &features_copy[feat*MAX_EMPTY], DIMENSION, DIMENSION, 1, 2 );
        }
        //if( i == 0 ){
        //cout << gss->featuresToString( features, NFEATURES ) << endl;
        //}
    }

    clock_t t2 = clock();
    cout << "time taken: " << ((float) (t2-t1)) / CLOCKS_PER_SEC << endl;
}

void testConvolution(){
    float image[9] = {1,1,1,2,2,2,3,3,3};
    int width = 3;
    int height = 3;
    float sigma = 1;
    int numsteps = 100;
    gaussianiir2d( image, width, height, sigma, numsteps );
    for( int i=0; i<width*height; i++ ){
        if( i % width == 0 ){
            cout << endl;
        }
        cout << image[i] << ", ";
    }

    FeatureFuncs::board2csv( image, 9, width, "test_convolution.csv" );


}

void testGabor(){
    ZobristHash* zh = new ZobristHash;
    zh->ctor();
    GoState gss(zh);
    //gss.MATLAB2board( game8 );
    //gss.MATLAB2board( game23 );
    gss.MATLAB2board( game112 );
    //gss.MATLAB2board( game1001 );
    //gss.MATLAB2board( game1234 );
    cout << gss.toString() << endl;

    int features[ NFEATURES * MAX_EMPTY ] = {0};
    FeatureFuncs::setBinaryFeatures( &gss, features );

    int feat = 1;
    float output_board[MAX_EMPTY];
    
    //string empty_str = gss.featuresToString( &features[feat*MAX_EMPTY], 1 );
    //cout << empty_str << endl;

    float feat_copy[MAX_EMPTY];
    copy( &features[feat*MAX_EMPTY], &features[feat*MAX_EMPTY] + MAX_EMPTY,
          feat_copy );
    FeatureFuncs::board2csv( feat_copy, MAX_EMPTY, DIMENSION, "game112_feat1.csv" );

    FeatureFuncs::setEdges( &features[feat*MAX_EMPTY], output_board );

    //string feature_str = gss.featuresToString( output_board, 1 );
    //cout << feature_str << endl;

    float output_copy[MAX_EMPTY];
    copy( output_board, output_board + MAX_EMPTY, output_copy );
    FeatureFuncs::board2csv( output_copy, MAX_EMPTY, DIMENSION, "game112_feat1_edges.csv" );
}

void pointMultBenchmark(){
    //19*19*31*31*10*361
    //const int size = 1252384810;
    const int size=    100000000;
    float* naive_a = new float[size];
    float* naive_b = new float[size];
    float* interleaved = new float[2*size];

    for( int i=0; i<size; ++i ){
        naive_a[i] = .4;
        naive_b[i] = .33;
    }
    for( int i=0; i<2*size; i+=2 ){
        interleaved[i] = .4;
        interleaved[i+1] = .33;
    }

    clock_t s = clock();
    float n = FeatureFuncs::naivePointMult( naive_a, naive_b, size );
    clock_t e = clock();
    cout << "Naive: " << ((float) (e-s)) / CLOCKS_PER_SEC << endl;
    cout << "results: " << n << endl;

    s = clock();
    float i = FeatureFuncs::interleavedPointMult( interleaved, size );
    e = clock();
    cout << "Interleaved: " << ((float) (e-s)) / CLOCKS_PER_SEC << endl;
    cout << "results: " << i << endl;

}

void crossCorrelateTest(){
    int binary_features[ NFEATURES*MAX_EMPTY ];
    fill_n( binary_features, NFEATURES*MAX_EMPTY, 1 );

    float convolved_features[ NCONVOLUTIONS*NFEATURES*MAX_EMPTY];
    fill_n( convolved_features, NCONVOLUTIONS*NFEATURES*MAX_EMPTY, .5 );

    float results[ NFEATURES*NFEATURES*NCONVOLUTIONS ] = {0};
    FeatureFuncs::crossCorrelate( binary_features, convolved_features, results );
    cout << "bf[0]: " << binary_features[0];
    cout << "results[0]: " << results[0] << endl;
}

int main(){
    srand(time(0));
    cout << sizeof(GoState) << endl;
    
    //mclInitializeApplication(NULL,0);
    //value2Initialize();

    //crossCorrelateTest();
    testValuePolicy();
    //testDefaultPolicy();
    //playGame();
    //testManhattanDist();
    //testSetBinaryFeatures();
    //timeSetBinaryFeatures();
    
    //testConvolution();
    
    //testGabor();
    //
    //pointMultBenchmark();
   
    //value2Terminate();
    //mclTerminateApplication();

    return 0;

};
