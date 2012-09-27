#include "state.h"

#include <stdlib.h>
#include <queue>
#include <assert.h>

State::State( int dimension, bool shallow ){
    dim = dimension;
    boardsize = dim*dim;
    board = new int[boardsize];
    for( int i=1; i<=boardsize; i++ ){
        open_positions.insert(i);
    }
    player = BLACK;
}

State State::copy() {
    return State( dim, false );
}


//TODO: a way to do this with macros?  see GNUgo
int State::getNorth( int ix ){
    if( ix < dim || ix == OFFBOARD ) return OFFBOARD;
    else                             return ix - dim;
}
int State::getSouth( int ix ){
    if( ix >= boardsize-dim || ix == OFFBOARD ) return OFFBOARD;
    else                                        return ix + dim;
}
int State::getEast( int ix ){
    if( ix % dim == dim-1 || ix < OFFBOARD ) return OFFBOARD;
    else                                     return ix + 1;
}
int State::getWest( int ix ){
    if( ix % dim == 0 || ix == OFFBOARD ) return OFFBOARD;
    else                                  return ix - 1;
}

int State::getNorthWest( int ix ){
    return getNorth( getWest( ix ) );
}
int State::getNorthEast( int ix ){
    return getNorth( getEast( ix ) );
}
int State::getSouthEast( int ix ){
    return getSouth( getEast( ix ) );
}
int State::getSouthWest( int ix ){
    return getSouth( getWest( ix ) );
}
int State::action2ix( int action ){
    return abs(action)-1;
}
int State::action2color( int action ){
    return (action > 0) ? WHITE : BLACK;
}
int State::ix2color( int ix ){
    return (ix == OFFBOARD) ? OFFBOARD : board[ix];
}

void State::setBoard( int* ixs, int len, COLORS color ){
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


//TODO: this is not working, and gross to makeit work,
//consider reworking the directions into one function indexes by enum DIR ints
int* State::neighborsOf( int ix, int adjacency ){
    if( adjacency == 4){
        int* r = new int[4];
        r = {getNorth(ix),
                     getSouth(ix),
                             getEast(ix),
                             getWest(ix)};
        return r;
    }
    else if( adjacency == 8 ){
        int r[8] = {getNorth(ix),
                             getSouth(ix),
                             getEast(ix),
                             getWest(ix), 
                             getNorthWest(ix),
                             getNorthEast(ix),
                             getSouthWest(ix),
                             getSouthEast(ix)};
        return r;
    }
    else{ assert(false); }
}

//TODO: incomplete
//int* floodFill( int ix, 
//COLORS* filter_colors, 
//COLORS* stop_colors, 
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
