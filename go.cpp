#include <iostream>
#include "state.h"

using namespace std;

int main(){
    State s( 5, false );
    cout << s.toString() << endl;
    s.neighborsOf(8,4);
    //int* na = s.neighbor_array;

    int ix = 15;
    s.setBoard( &ix, 1, WHITE );

    s.color_array[0] = WHITE;
    s.filterByColor( s.neighbor_array, 4, s.color_array, 1 );
    int* na = s.filtered_array;
    cout << "filtered len: " << s.filtered_len << endl;
    for( int i=0; i<s.filtered_len; i++ ){
        cout << na[i] << endl;
    }

    //State copy = s.copy();
    //
    //copy.board[1] = WHITE;
    //cout << copy.dim << endl;
    //cout << copy.player << endl;
    //cout << copy.board[1] << endl;


    return 0;

};
