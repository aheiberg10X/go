#include <iostream>
#include "state.h"

using namespace std;

int main(){
    State s( 5, false );
    State copy = s.copy();
    cout << s.dim << endl;
    cout << s.player << endl;
    cout << s.board[1] << endl;

    copy.board[1] = WHITE;
    cout << copy.dim << endl;
    cout << copy.player << endl;
    cout << copy.board[1] << endl;

    return 0;

};
