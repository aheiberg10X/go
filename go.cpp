#include <iostream>
#include "gostate.h"
#include "godomain.h"
#include <assert.h>

using namespace std;

int main(){
    string name = "s";
    GoState s( name, 5, false );

    const int l = 7;
    int is[l] = {1,1,2,2,3,4,1};
    int js[l] = {1,3,2,3,2,4,2};
    for( int i=0; i<l; i++ ){
        s.setBoard( s.coord2ix( is[i], js[i] ), WHITE );
    }
    
    const int ll = 5;
    int is2[ll] = {1,2,3,4,3};
    int js2[ll] = {4,4,3,2,1};
    for( int i=0; i<ll; i++ ){
        s.setBoard( s.coord2ix( is2[i], js2[i] ), BLACK );
    }
    
    //name = "ps";
    //GoState ps( name, 5, false );
    //const int lll = 3;
    //int is3[lll] = {1,2,3};
    //int js3[lll] = {3,2,1};
    //for( int i=0; i<lll; i++ ){
    //ps.setBoard( ps.coord2ix( is3[i], js3[i] ), WHITE );
    //}
    //
    //s.past_states[NUM_PAST_STATES-1] = &ps;

    //GoState* s2 = s.copy(false);
    //cout << s2->past_states[NUM_PAST_STATES-1]->toString();



    //int adjacency = 4;
    //s.color_array[0] = WHITE;
    //s.neighborsOf( s.neighbor_array, s.coord2ix(3,3), adjacency);
    //int filtered_len = 0;
    //s.filterByColor(
            //s.filtered_array, &filtered_len,
            //s.neighbor_array, adjacency,
            //s.color_array   , 1 );

    ////cout << "filtered_len: " << filtered_len << endl;
    ////for( int i=0; i < filtered_len; i++ ){
    ////cout << s.ix2coord( s.filtered_array[i] ) << endl;
    ////}

    //int flood_len = 0;
    //COLOR stop_array[1] = {BLACK};
    //s.floodFill( s.floodfill_array, &flood_len,
                 //s.coord2ix(2,2),
                 //adjacency,
                 //s.color_array, 1,
                 //stop_array, 1 );
 
    //for( int i=0; i < flood_len; i++ ){
        //cout << s.ix2coord( s.floodfill_array[i] ) << endl;
    //}    

    s.setBoard( s.coord2ix(2,1), WHITE );
    cout << s.toString() << endl;
    cout << "is suicide: " << s.isSuicide( s.coord2ix(2,1) ) << endl;

    GoDomain gd();


    return 0;

};
