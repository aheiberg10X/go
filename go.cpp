//#include "gostate.h"
#include "gostate_struct.h"
#include "godomain.cpp"
#include "mcts.h"
#include "queue.h"
#include "bitmask.h"

#include <assert.h>
#include <iostream>

using namespace std;

int main(){
    //cout << sizeof(COLOR) << endl;
    cout << sizeof(int) << endl;
    cout << sizeof(char) << endl;
    cout << sizeof(bool) << endl;
    cout << sizeof(GoStateStruct) << endl;

    srand(time(NULL));

    /*
    if( false ){
        Queue q (4);
        q.push(7);
        q.push(8);
        q.push(9);
        q.push(10);
        cout << q.pop() << endl;
        cout << q.pop() << endl;
        cout << q.pop() << endl;
        cout << q.pop() << endl;
        cout << q.isEmpty() << endl;
    }*/

    
    if( false ){
        BitMask bm;
        bm.set(1,false);
        cout << bm.get(1) << endl;
        bm.set(1,true);
        cout << bm.get(1) << endl;
        bm.set(44,true);
        cout << bm.get(44) << endl;
        bm.set(44,false);
        cout << bm.get(44) << endl;


    }


    if( true ){
        Domain* domain = (Domain*) new GoDomain();
        GoStateStruct* gs = new GoStateStruct;
        gs->initGSS();
        void** p_uncast_state = (void**) &gs;
        MCTS mcts(domain);

        cout << "yeesh" << endl;

        while( !domain->isTerminal( *p_uncast_state ) ){
            int best_action = mcts.search( *p_uncast_state );
            cout << "Best Action: " << best_action << endl;
            domain->applyAction( p_uncast_state, best_action, true );
            cout << "Applying action: " << best_action << endl;
            cout << "Resulting state: " << ((GoStateStruct* ) *p_uncast_state)->toString(  ) << endl;
            cout << "hit any key..." << endl;

            cin.ignore();
        }
        
    }
    
    //domain testing
    //randomAction and applyAction, the crucial parts of the simulation kernel,
    //work in fixed memory
    if( false ){

        //string name = "original";
        //GoState* state = new GoState( false );
        GoStateStruct* state = new GoStateStruct();
        //state->initGSS( );
        void** p_uncast_state = (void**) &state;

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
            state2->setBoard( state2->coord2ix( is2[i], js2[i] ), WHITE );
        }
        state->advancePastStates( state2 );

        cout << state->boardToString( &(state->past_boards[BOARDSIZE]) ) << endl;

        cout << "duplicated: " << state->isDuplicatedByPastState() << endl;

/*
        GoDomain gd;

        BitMask to_exclude;
        to_exclude.initBitMask();
        while( ! gd.isTerminal( *p_uncast_state ) ){
            //cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << endl;
            //cout << "current state: " << state->toString() << endl;
            //cout << "last state: " << state->past_states[NUM_PAST_STATES-1]->toString() << endl;
            int raction = gd.randomAction( p_uncast_state, &to_exclude );
            cout << "\napplying rand act: " << raction << " to: \n" << state->toString() << endl;
            gd.applyAction( p_uncast_state, raction, true );

        }
        cout << "finished" << endl;

        int rewards[2];
        state = (GoStateStruct*) *p_uncast_state;
        gd.getRewards( rewards, *p_uncast_state );
        cout << "white_score: " << rewards[0] << " black_score: " << rewards[1] << endl;
*/

        //cout << "isTerminal: " << gd.isTerminal( (void*) s&tate ) << endl;
        //action = s.coordColor2Action( 4,3,WHITE );
        //gd.applyAction( (void*) &s, action, true );
        //s.setBoard( s.coord2ix(2,1), WHITE );
        //cout << "is suicide: " << s.isSuicide( s.coord2ix(2,1) ) << endl;
    }



    return 0;

};
