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

    floodfill_array = new int[boardsize];

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


void* State::neighborsOf( int* to_fill, int ix, int adjacency ){
    assert( adjacency==4 || adjacency==8 );
    for( int dir=0; dir<adjacency; dir++ ){
        to_fill[dir] = neighbor(ix, (DIRECTION) dir);
    }
    //return neighbor_array;
}

void State::filterByColor( int* to_fill, 
                           int* to_fill_len,
                           int* neighbor_array, 
                           int adjacency, 
                           COLOR* color_array,
                           int filter_len ){

    *to_fill_len = 0;
    int fillix = 0;
    COLOR color;
    for(int naix=0; naix<adjacency; naix++){
        color = board[neighbor_array[naix]];
        for( int caix=0; caix<filter_len; caix++ ){
            if( color == color_array[caix] ){ 
                to_fill[fillix] = neighbor_array[naix];
                (*to_fill_len)++;
            }
        }
    }
}

//TODO: incomplete
void State::floodFill( int* to_fill,
                       int* to_fill_len, 
                       int epicenter_ix,
                       int* neighbor_array,
                       int adjacency,
                       COLOR* filter_color_array, 
                       int filter_len,
                       COLOR* stop_color_array, 
                       int stop_len ){

    set<int> marked;
    queue<int> q;
    q.insert(epicenter_ix);

    while( !q.empty() ){
        int ix = q.front();
        q.pop();
        marked.insert(ix);

        neighborsOf( neighbor_array, 
                     ix, 
                     adjacency );

        //find if neighbors that are a cause to stop the flood fill
        filterByColor( filterer_array,
                       filtered_len,
                       neighbor_array,
                       adjacency,
                       stop_color_array,
                       stop_len);

        bool stop_color_in_neighbs = &filtered_array > 0;
        if( stop_color_in_neights ){
            marked.clear();
            break;
        }
        else {
            //find connector neighbors
            filterByColor( filtered_array, 
                           filtered_len,
                           neighbors_array,
                           adjacency,
                           filter_color_array,
                           filter_len );
            //see if connector neighbors are already in marked
            //if not, add them
            for( int faix=0; faix<&filtered_len; faix++ ){
                int ix = filtered_array[faix];
                if( marked.find( ix ) != marked::end ){
                    q.insert(ix);
                }
            }
        }
    }
    
    //populate to_fill; kinda of unnecassay just did to keep pattern same
    int i = 0;
    for( int it=marked.begin(); it != marked.end(); i++ ){
        to_fill[i++] = it;
    }
    *to_fill_len = i;

}
