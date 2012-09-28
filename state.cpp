#include "state.h"

#include <stdlib.h>
#include <queue>
#include <assert.h>
#include <iostream>

State::State( int dimension, bool shallow ){
    dim = dimension;
    bigdim = dim+2;
    boardsize = (bigdim)*(bigdim);
    board = new COLOR[boardsize];
    for( int i=0; i<boardsize; i++ ){
        if( i < bigdim || 
            i % bigdim == 0 || 
            i % bigdim == bigdim-1 || 
            i >= boardsize - bigdim ) {
                board[i] = OFFBOARD;
        }
        else{
            board[i] = EMPTY; 
            open_positions.insert(i);
        }
    }
    player = BLACK;

}

State State::copy() {
    return State( dim, false );
}

string State::toString(){
    string out;
    for( int i=0; i<boardsize; i++ ){
        if( board[i] == BLACK ){
            out += "x";
        }
        else if( board[i] == WHITE ){
            out += "o";
        }
        else if( board[i] == OFFBOARD ){
            out += "_";
        }
        else if( board[i] == EMPTY ){
            out += ".";
        }
        else{
            assert(false);
        }
        out += " ";
        if( i % bigdim == bigdim-1 ){
            out += "\n";
        }
    }
    return out;
}

int State::neighbor(int ix, DIRECTION dir){
    if( board[ix] == OFFBOARD ){
        return OFFBOARD;
    }
    else{
        if(      dir == N ){  return ix - bigdim; }
        else if( dir == S ){  return ix + bigdim; }
        else if( dir == E ){  return ix + 1;}
        else if( dir == W ){  return ix -1; }
        else if( dir == NW ){ return ix - bigdim - 1; }
        else if( dir == NE ){ return ix - bigdim + 1; }
        else if( dir == SW ){ return ix + bigdim - 1; }
        else if( dir == SE ){ return ix + bigdim + 1; }
        else{
            assert(false);
        }
    }
}

//int State::action2ix( int action ){
    //return abs(action)-1;
//}
//int State::action2color( int action ){
    //return (action > 0) ? WHITE : BLACK;
//}
//int State::ix2color( int ix ){
    //return (ix == OFFBOARD) ? OFFBOARD : board[ix];
//}

void State::setBoard( int* ixs, int len, COLOR color ){
    for( int i=0; i<=len; i++ ){
        int ix = ixs[i];
        if( color == EMPTY ){
            open_positions.erase(ix);
        }
        else{
            open_positions.insert(ix);
        }
        board[ix] = color;
    }
}


void State::neighborsOf( int ix, int adjacency ){
    assert( adjacency==4 || adjacency==8 );
    for( int dir=0; dir<adjacency; dir++ ){
        neighbor_array[dir] = neighbor(ix, (DIRECTION) dir);
    }
    //return neighbor_array;
}

void State::filterByColor( int* neighbor_array, 
                           int adjacency, 
                           COLOR* color_array,
                           int filter_len ){

    filtered_len = 0;
    int faix = 0;
    COLOR color;
    bool matched;
    for(int naix=0; naix<adjacency; naix++){
        color = board[neighbor_array[naix]];
        //cout << "color of neighbor " << neighbor_array[naix] << " is: " << color << endl;
        for( int caix=0; caix<filter_len; caix++ ){
            if( color == color_array[caix] ){ 
                //cout << "match!" << endl;
                filtered_array[faix] = neighbor_array[naix];
                filtered_len++;
            }
        }
    }
            //return filtered_array;
}



//TODO: incomplete
//int* floodFill( int ix, 
//COLOR* color_array, 
//COLOR* stop_colors, 
//int adjacency ){
//set<int> marked;
//queue<int> q;
//while( !q.empty() ){
//int ix = q.front();
//q.pop();
//marked.insert(ix);
//neighbs = neighborsOf( ix, adjacency );
//}
//}
