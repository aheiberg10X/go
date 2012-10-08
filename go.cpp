#include "gostate.h"
#include <iostream>
#include "godomain.cpp"
#include <assert.h>

using namespace std;

int main(){
    string name = "s";
    GoState s( name, 4, false );
    GoState* state = &s;

    const int l = 6;
    int is[l] = {2,2,3,3,4,4};
    int js[l] = {1,3,2,3,1,2};
    for( int i=0; i<l; i++ ){
        s.setBoard( s.coord2ix( is[i], js[i] ), WHITE );
    }
    
    const int ll = 6;
    int is2[ll] = {1,1,2,2,3,4};
    int js2[ll] = {1,3,2,4,4,3};
    for( int i=0; i<ll; i++ ){
        s.setBoard( s.coord2ix( is2[i], js2[i] ), BLACK );
    }

    s.togglePlayer();
    
    GoDomain gd;

    cout << state->toString() << endl;
    
    set<int> to_exclude;
    while( ! gd.isTerminal( state ) ){
        cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << endl;
        cout << "current state: " << state->toString() << endl;
        cout << "last state: " << state->past_states[NUM_PAST_STATES-1]->toString() << endl;
        int raction = gd.randomAction( (void**) &state, to_exclude );
        cout << "\nrand act: " << raction << endl;
        gd.applyAction( (void**) &state, raction, true );
    }

    //int rewards[2];
    //gd.getRewards( rewards, (void*) &s );
    //cout << "white_score: " << rewards[0] << " black_score: " << rewards[1] << endl;


    cout << "isTerminal: " << gd.isTerminal( (void*) state ) << endl;
    //action = s.coordColor2Action( 4,3,WHITE );
    //gd.applyAction( (void*) &s, action, true );
    //s.setBoard( s.coord2ix(2,1), WHITE );
    //cout << "is suicide: " << s.isSuicide( s.coord2ix(2,1) ) << endl;



    return 0;

};
