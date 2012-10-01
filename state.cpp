#include "state.h"

#include <stdlib.h>
#include <queue>
#include <assert.h>
#include <iostream>
#include <sstream>

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
    int a = 0;
    int b = 0; 
    int c = 0;
    floodfill_len = &a;
    color_len = &b;
    filtered_len = &c;

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

int State::action2ix( int action ){
    return abs(action);
}
COLOR State::action2color( int action ){
    return (action > 0) ? WHITE : BLACK;
}
int State::ix2color( int ix ){
    return (ix == OFFBOARD) ? OFFBOARD : board[ix];
}

int State::coord2ix( int i, int j ){
    return bigdim*i + j;
}

string State::ix2coord( int ix ){
    int j = ix % bigdim;
    int i = ix / bigdim;
    stringstream out;
    out << i << "," << j;
    return out.str();
}

void State::setBoard( int* ixs, int len, COLOR color ){
    for( int i=0; i<len; i++ ){
        int ix = ixs[i];
        setBoard( ix, color );
    }
}

void State::setBoard( int ix, COLOR color ){
    if( ix >= boardsize || board[ix] == OFFBOARD ){ return; }

    if( color == EMPTY ){
        open_positions.erase(ix);
    }
    else{
        open_positions.insert(ix);
    }
    board[ix] = color;
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
                           int* neighbs, 
                           int adjacency, 
                           COLOR* color_array,
                           int filter_len ){

    *to_fill_len = 0;
    int fillix = 0;
    COLOR ncolor;
    for(int naix=0; naix<adjacency; naix++){
        int nix = neighbs[naix];
        ncolor = board[nix];
        //cout << "nix: " << nix << " ncolor: " << ncolor << endl;
        for( int caix=0; caix<filter_len; caix++ ){
            if( ncolor == color_array[caix] ){ 
                to_fill[(*to_fill_len)] = nix;
                *to_fill_len = *to_fill_len + 1;
            }
        }
    }
}

void State::floodFill( int* to_fill,
                       int* to_fill_len, 
                       int epicenter_ix,
                       int adjacency,
                       COLOR* filter_color_array, 
                       int filter_len,
                       COLOR* stop_color_array, 
                       int stop_len ){
    set<int> marked;
    queue<int> q;
    q.push(epicenter_ix);

    int neighbs[8];

    while( !q.empty() ){
        int ix = q.front();
        q.pop();
        marked.insert(ix);
 
        neighborsOf( neighbs, 
                     ix, 
                     adjacency );

        //find if there are neighbors that cause flood fill to stop
        filterByColor( filtered_array,
                       filtered_len,
                       neighbs,
                       adjacency,
                       stop_color_array,
                       stop_len);

        bool stop_color_is_in_neighbs = *filtered_len > 0;
        if( stop_color_is_in_neighbs ){
            marked.clear();
            break;
        }
        else {
            //find connected neighbors
            filterByColor( filtered_array, 
                           filtered_len,
                           neighbs,
                           adjacency,
                           filter_color_array,
                           filter_len );
            //see if connector neighbors are already in marked
            //if not, add them
            for( int faix=0; faix < *filtered_len; faix++ ){
                int ix = filtered_array[faix];
                if( marked.find( ix ) == marked.end() ){
                    q.push(ix);
                }
            }
        }
    }
    
    //populate to_fill; kinda of unnecassay just did to keep use pattern same
    int i = 0;
    set<int>::iterator it;
    for( it = marked.begin(); it != marked.end(); it++ ){
        to_fill[i++] = *it;
    }
    *to_fill_len = i;

}

//assumes setBoard(action) already applied
bool State::isSuicide( int action ){
    COLOR color = action2color( action );
    int ix = action2ix( action );
    cout << "ix: " << ix << endl;
    int adjacency = 4;

    neighborsOf( neighbor_array, ix, adjacency );

    //same colored neighbors
    COLOR colors[1] = {color};
    int filtered[adjacency];
    int filtered_len;
    filterByColor( filtered, &filtered_len,
                   neighbor_array, adjacency,
                   colors, 1 );

    //floodfill each neighbor stopping if an adjacent EMPTY is found
    //TODO: in python impl., the origial ix is added to filtered_array ??
    //If one of the groups has no liberties, the move is illegal
    //(Not considering moves that make space by capturing opponent first
    // these pieces should be removed beforehand)
    bool left_with_no_liberties = false;
    set<int> marked;
    COLOR stop_array[1] = {EMPTY};

    for( int i=0; i < filtered_len; i++ ){
        int nix = filtered[i];
        //cout << "nix: " << nix << endl;
        if( marked.find(nix) != marked.end() ){ continue; }

        int flood_len = 0;
        floodFill( floodfill_array, &flood_len,
                   nix,
                   adjacency,
                   colors, 1,
                   stop_array, 1 );
        for( int j=0; j < flood_len; j++ ){
            //cout << "marking: " << floodfill_array[j] << endl;
            marked.insert( floodfill_array[j] );
        }
        //cout << "nix: " << nix << "no_liber: " << (flood_len > 0) << endl;
        left_with_no_liberties |= flood_len > 0;
    }

    bool surrounded_by_kin = true;
    for( int i=0; i<adjacency; i++ ){
        int ncolor = ix2color( neighbor_array[i] );
        surrounded_by_kin &= ncolor == color || ncolor == OFFBOARD;
    }
                    
    return left_with_no_liberties || surrounded_by_kin;
}
