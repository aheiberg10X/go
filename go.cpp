//#include "gostate.h"
#include "gostate_struct.h"
#include "godomain.cpp"
#include "mcts.h"
#include "queue.h"
#include "bitmask.h"
#include "zobrist.h"

//just for direct timing/testing launchSimulationKernel
#include "kernel.h"

#include <omp.h>

#include <assert.h>
#include <iostream>
#include <time.h>
#include <stdint.h>

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

    //playout simulation performance timing
    if( true ){
        GoStateStruct* gss = new GoStateStruct;
        gss->ctor(zh);

        //BitMask* to_exclude = new BitMask;
        //int action = gss->randomAction( to_exclude, false );
        int rewards[2];
        launchSimulationKernel( gss, rewards );
    }

    //play a full MCTS game
    if( false ){
        Domain* domain = (Domain*) new GoDomain();
        MCTS mcts(domain);

        GoStateStruct* gs = new GoStateStruct;
        gs->ctor(zh);
        //void* uncast_state = (void*) gs;

        int nthreads = 4;
        int tid;
        int best_action;

        MCTS_Node** search_trees = new MCTS_Node* [nthreads];
        GoStateStruct** states = new GoStateStruct* [nthreads];

        while( !domain->isTerminal( gs ) ){
            //initialize each thread's copy of state
            for( int i=0; i<nthreads; i++ ){
                states[i] = gs->copy();
            }

            //do parallel tree search
            #pragma omp parallel for shared(states, search_trees) \
                                     private(tid) \
                                     num_threads(nthreads)
            for(int tid=0; tid<nthreads; tid++){
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
                
                for( int tid=0; tid<nthreads; tid++ ){
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

            cout << "best action for player: " << pix << " is: " << best_action << endl;
            cout << "hit any key..." << endl;
            cin.ignore();

            //clean up
            for( int tid=0; tid<nthreads; tid++ ){
                delete search_trees[tid];
            }
            
            //apply to uncast_state
            //assert best_action is legal
            bool legal = domain->applyAction( gs, best_action, true );
            assert(legal);
        }
    }
    
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

   
    
    //domain testing
    //randomAction and applyAction, the crucial parts of the simulation kernel,
    //work in fixed memory
    /*
    if( false ){

        //string name = "original";
        //GoState* state = new GoState( false );
        GoStateStruct* state = new GoStateStruct();

        const int l = 6;
        int is[l] = {2,2,3,3,4,4};
        int js[l] = {1,3,2,3,1,2};
        for( int i=0; i<l; i++ ){
            state->setBoard( state->coord2ix( is[i], js[i] ), WHITE );
        }
        cout << state->toString() << endl;
        
        GoStateStruct* state2 = new GoStateStruct();
        //state->initGSS( );

        const int ll = 6;
        int is2[ll] = {2,2,3,3,4,4};
        int js2[ll] = {1,3,2,3,1,2};
        for( int i=0; i<ll; i++ ){
            state2->setBoard( state2->coord2ix( is2[i], js2[i] ), BLACK );
        }

        //cout << state->boardToString( state2->board ) << endl;
        state->advancePastStates( state2->board,
                                  state2->player,
                                  state2->action );

        //cout << state->boardToString( &(state->past_boards[PAST_STATE_SIZE-BOARDSIZE]) ) << endl;

        cout << "duplicated: " << state->isDuplicatedByPastState() << endl;
          
    }
    */


    return 0;

};
