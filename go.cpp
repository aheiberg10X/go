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

    cout << "asdfasd" << endl;
    //play a full MCTS game
    if( false ){
        Domain* domain = (Domain*) new GoDomain();
        MCTS mcts(domain);

        GoStateStruct* gs = new GoStateStruct;
        gs->ctor(zh);
        void* uncast_state = (void*) gs;

        int ntrees = 1;
        int tid;
        int* best_actions = new int[ntrees];
        void** uncast_states = new void*[ntrees];
        //void** MCTSs = new MCTS*[ntrees];

        for( int i=0; i<ntrees; i++ ){
            uncast_states[i] = domain->copyState( uncast_state );
            //MCTSs[i] = new MCTS;
            //MCTSs[i]->domain = domain;
        }
        #pragma omp parallel shared(uncast_states, best_actions) \
                             private(tid) \
                             num_threads(ntrees)
        {
            tid = omp_get_thread_num();
            cout << "hello" << endl;
            while( !domain->isTerminal( uncast_states[tid] ) ){
                int ta = clock();
                                                    
                best_actions[tid] = mcts.search( uncast_states[tid] );
                int tb = clock();
                cout << "time taken is: " << ((float) tb-ta)/CLOCKS_PER_SEC << endl;
                cout << "Best Action: " << best_actions[tid] << endl;
                domain->applyAction( uncast_states[tid], best_actions[tid], true );
                cout << "Applying action: " << best_actions[tid] << endl;
                cout << "Resulting state: " << ((GoStateStruct* ) uncast_states[tid])->toString(  ) << endl;
                cout << "hit any key..." << endl;
                cin.ignore();
            }

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
