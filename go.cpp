#include "gostate.h"
#include "godomain.cpp"
#include "mcts.h"

#include <assert.h>
#include <iostream>

using namespace std;

int main(){

    //mcts testing
    if( false ){
        GoState* gs = new GoState( "s", 4, false );
        Domain* domain = (Domain*) new GoDomain();
        MCTS mcts(domain);
        int best_action = mcts.search( (void*) gs );
        
    }
    
    //domain testing
    if( true ){

        string name = "original";
        GoState* state = new GoState( name, 9, false );

        //int filtered_array[4];
        //int filtered_len = 0;
        //
        //int floodfill_len = 0;
        //COLOR flood_array[1] = {EMPTY};
        //COLOR stop_array[1] = {BLACK};
        //
        //int count = 0;
        //
        //while( count < 100000 ){
        ////state->neighborsOf( state->floodfill_array,
        ////48,
        ////4 );
        ////state->filterByColor( filtered_array,
        ////&filtered_len,
        ////state->floodfill_array,
        ////4,
        ////flood_array, 1 );
        //
        ////state->floodFill( state->floodfill_array,
        ////&floodfill_len,
        ////55,
        ////4,
        ////flood_array, 1,
        ////stop_array, 1 );
        ////
        //state->isSuicide( 15 );
        //count++;
        //}
                              

        //const int l = 6;
        //int is[l] = {2,2,3,3,4,4};
        //int js[l] = {1,3,2,3,1,2};
        //for( int i=0; i<l; i++ ){
            //s.setBoard( s.coord2ix( is[i], js[i] ), WHITE );
        //}
        
        //const int ll = 6;
        //int is2[ll] = {1,1,2,2,3,4};
        //int js2[ll] = {1,3,2,4,4,3};
        //for( int i=0; i<ll; i++ ){
            //s.setBoard( s.coord2ix( is2[i], js2[i] ), BLACK );
        //}

        //state->togglePlayer();
        
        GoDomain gd;

        //cout << state->toString() << endl;
        //int count = 0;
        //while( count < 100000 ){
        //int action = state->coordColor2Action(2,1,BLACK);
        //gd.applyAction( (void**) &state, action, false );
        //count++;
        //}
        
        set<int> to_exclude;
        while( ! gd.isTerminal( state ) ){
            //cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << endl;
            //cout << "current state: " << state->toString() << endl;
            //cout << "last state: " << state->past_states[NUM_PAST_STATES-1]->toString() << endl;
            int raction = gd.randomAction( (void**) &state, to_exclude );
            cout << "\napplying rand act: " << raction << " to: \n" << state->toString() << endl;
            gd.applyAction( (void**) &state, raction, true );
        }
        cout << "finished" << endl;

        //int rewards[2];
        //gd.getRewards( rewards, (void*) &s );
        //cout << "white_score: " << rewards[0] << " black_score: " << rewards[1] << endl;


        //cout << "isTerminal: " << gd.isTerminal( (void*) state ) << endl;
        //action = s.coordColor2Action( 4,3,WHITE );
        //gd.applyAction( (void*) &s, action, true );
        //s.setBoard( s.coord2ix(2,1), WHITE );
        //cout << "is suicide: " << s.isSuicide( s.coord2ix(2,1) ) << endl;
    }



    return 0;

};
