//#include "gostate.h"
#include "gostate_struct.h"
#include "godomain.cpp"
#include "mcts.h"
#include "queue.h"
#include "bitmask.h"

#include <assert.h>
#include <iostream>
#include <time.h>

using namespace std;

int main(){
    //cout << sizeof(COLOR) << endl;
    //cout << sizeof(int) << endl;
    //cout << sizeof(char) << endl;
    //cout << sizeof(bool) << endl;
    //cout << sizeof(GoStateStruct) << endl;
    //cout << PAST_STATE_SIZE << endl;

    srand(time(NULL));

    /*
    GoStateStruct state;
    int bi[9] = {1,1,2,2,2,2,3,3,4};
    int bj[9] = {2,3,1,2,3,4,3,4,3};
    int wi[3] = {3,3,4};
    int wj[3] = {1,2,2};
    for( int i=0; i<9; i++ ){
        state.setBoard( state.coord2ix( bi[i], bj[i] ), BLACK );
    }
    for( int i=0; i<3; i++ ){
        state.setBoard( state.coord2ix( wi[i], wj[i] ), WHITE );
    }

    cout << state.toString() << endl;
    BitMask empty;
    int action = state.randomAction( &empty );
    cout << action << endl;
    */

    if( true ){
        Domain* domain = (Domain*) new GoDomain();
        GoStateStruct* gs = new GoStateStruct;
        void* uncast_state = (void*) gs;
        MCTS mcts(domain);

        while( !domain->isTerminal( uncast_state ) ){
            int ta = clock();
            int best_action = mcts.search( uncast_state );
            int tb = clock();
            cout << "time taken is: " << ((float) tb-ta)/CLOCKS_PER_SEC << endl;
            cout << "Best Action: " << best_action << endl;
            domain->applyAction( uncast_state, best_action, true );
            cout << "Applying action: " << best_action << endl;
            cout << "Resulting state: " << ((GoStateStruct* ) uncast_state)->toString(  ) << endl;
            cout << "hit any key..." << endl;
            
            cin.ignore();
        }
        
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
        GoDomain gd;
        BitMask bm;
        while( !gd.isTerminal( (void*) state ) ){
            cout << state->toString() << endl;
            int action = gd.randomAction( (void*) state, &bm );
            bool legal = gd.applyAction( (void*) state, action, true );
        }
        cout << state->toString() << endl;
         
    }
    */


    return 0;

};
