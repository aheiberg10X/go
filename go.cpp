//#include "gostate.h"
#include "gostate_struct.h"
#include "godomain.cpp"
#include "mcts.h"
#include "queue.h"
#include "bitmask.h"
#include "zobrist.h"

//just for direct timing/testing launchSimulationKernel
#include "kernel.h"

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

int main(){
    srand(time(0));
    //cout << sizeof(COLOR) << endl;
    //cout << sizeof(int) << endl;
    //cout << sizeof(char) << endl;
    cout << sizeof(bool) << endl;
    cout << sizeof(GoStateStruct) << endl;
    cout << sizeof(uint8_t) << endl;
    cout << sizeof(uint64_t) << endl;
    //cout << PAST_STATE_SIZE << endl;
    

    //zobrist testing
    ZobristHash* zh = new ZobristHash;
    zh->ctor();
    
    mclInitializeApplication(NULL,0);
    value2Initialize();


    //testing the linking of MATLAB compiled code to do value computation
    if( false ){
        //input
        //TODO, do this once somewhere, weights.h?
        //Wasteful right now, but not the bottleneck
        //
        //
        Domain* domain = (Domain*) new GoDomain();
        MCTS mcts(domain);
        MCTS_Node* dummy_node = new MCTS_Node( 2, BOARDSIZE );
        GoStateStruct* gss = new GoStateStruct;
        gss->ctor(zh);
        gss->MATLAB2board( game1234 );
        cout << gss->toString() << endl;
        ///set board to example?
        cout << "here" << endl;
        MCTS_Node* return_node1 = mcts.valuePolicy( dummy_node, gss );

    }
        //
        //
        //playout simulation performance timing
    if( true ){
        GoStateStruct* gss = new GoStateStruct;
        gss->ctor(zh);

        Domain* domain = (Domain*) new GoDomain();
        MCTS mcts(domain);

        int rewards[2];
        clock_t t1,t2;

        //gss->MATLAB2board( game1234 );
        //BitMask* to_exclude = new BitMask;
        //for( int i=0; i<300; i++ ){
        //int action = gss->randomAction( to_exclude, true );
        //}
        //cout << "random board with 100 played" << gss->toString() << endl;
        cout << "num open: " << gss->num_open << endl;
        t1 = clock();
        mcts.defaultPolicy( rewards, (void *) gss );
        //launchSimulationKernel( gss, rewards );
        t2 = clock();
        cout << "time taken: " << ((float) (t2-t1)) / CLOCKS_PER_SEC << endl;

    }
    
    //play a full MCTS game
    if( false ){
        Domain* domain = (Domain*) new GoDomain();
        MCTS mcts(domain);

        GoStateStruct* gs = new GoStateStruct;
        gs->ctor(zh);
        //void* uncast_state = (void*) gs;

        int tid;
        int best_action;

        MCTS_Node** search_trees = new MCTS_Node* [N_ROOT_THREADS];
        GoStateStruct** states = new GoStateStruct* [N_ROOT_THREADS];

        int nmoves = 0;
        clock_t total_time_1 = clock();
        while( nmoves <= MAX_MOVES && !domain->isTerminal( gs ) ){
            //initialize each thread's copy of state
            clock_t t1 = clock();
            for( int i=0; i<N_ROOT_THREADS; i++ ){
                states[i] = gs->copy();
            }
 
            //do parallel tree search
            //pass in the search tree, sans estimates, from laste iteration?
            #pragma omp parallel for shared(states, search_trees) \
                                     private(tid) \
                                     num_threads(N_ROOT_THREADS)
            for( tid=0; tid<N_ROOT_THREADS; tid++){
                search_trees[tid] = mcts.search( (void*) states[tid] );
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

            int pix = domain->getPlayerIx( gs );
            int best_action = mcts.bestChild( search_trees[0], 
                                              pix, 
                                              true )->action;

            //cout << "best action for player: " << pix << " is: " << best_action << endl;

            //TODO: deleting the search tries is wasteful
            //we don't want the other players estimates
            //but reusing the valuations would be smart
            //clean up
            for( int tid=0; tid<N_ROOT_THREADS; tid++ ){
                delete search_trees[tid];
            }
            
            //apply to uncast_state
            //assert best_action is legal
            bool legal = domain->applyAction( gs, best_action, true );
            cout << "After the " << nmoves << " move, " << endl << gs->toString() << endl;
            assert(legal);
            clock_t t2 = clock();
            //cout << "time taken: " << ((float) (t2-t1)) / CLOCKS_PER_SEC << endl;

            //cout << "hit any key..." << endl;
            //cin.ignore();
            nmoves++;
        }
        int rewards[2];
        domain->getRewards( rewards, (void *) gs );
        cout << "Black Wins: " << rewards[1] << endl;
        cout << "total time taken: " << ((float) (clock()-total_time_1)) / CLOCKS_PER_SEC << endl;
        //do something with the results
    }
   
    ////Causing double free errors...???
    value2Terminate();
    mclTerminateApplication();

    //test basic random and apply functionality on pre-configured boards
    if( false ){
        
        GoStateStruct* gss = new GoStateStruct;
        gss->ctor(zh);
        int wi[3] = {1,2,2};
        int wj[3] = {2,1,2};
        for( int i=0; i<3; i++ ){
            gss->setBoard( gss->coord2ix( wi[i], wj[i] ), WHITE );
        }
        cout << gss->toString() << endl;
        BitMask empty;
        int action = gss->randomAction( &empty, true );
        cout << gss->toString() << endl;
    }


    return 0;

};
