#include "gostate.h"

#include <stdlib.h>
#include <queue>
#include <assert.h>
#include <iostream>
#include <sstream>

GoState::GoState( string name, int dimension, bool shallow ){
    this->name = name;
    dim = dimension;
    bigdim = dim+2;
    boardsize = (bigdim)*(bigdim);
    board = new COLOR[boardsize];
    action = 42;
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

    if( ! shallow ){
        shallow = false;
        for( int i=0; i < NUM_PAST_STATES; i++ ){
            stringstream ss;
            ss << "ps" << i;
            past_states[i] = new GoState( ss.str(), dim, true);
        } 
    }
    else{
        shallow = true;
    }
}

GoState::~GoState(){
    //cout << "deleteing: " << name << endl;
    delete board;
    delete floodfill_array;
    open_positions.clear();

    if( shallow ){
        for( int i=0; i < NUM_PAST_STATES; i++ ){
            delete past_states[i];
        }
    }
    //TODO why does this cause such havoc?
    //delete past_states;
}

GoState* GoState::copy( bool shallow ) {
    GoState* s = new GoState( "copy", dim, false );
    
    for( int i=0; i<boardsize; i++ ){
        s->board[i] = board[i];
    }

    set<int> op (open_positions.begin(), open_positions.end());
    s->open_positions = op;

    s->player = player;
    s->action = action;

    if( ! shallow ){
        for( int i=0; i < NUM_PAST_STATES; i++ ){
            GoState* psc = past_states[i]->copy(true);
            s->past_states[i] = psc;
        }
    }

    return s;
}

COLOR GoState::flipColor( COLOR c ){
    assert( c == WHITE || c == BLACK );
    return (c == WHITE) ? BLACK : WHITE;
}

bool GoState::sameAs( COLOR* board, COLOR player ){
    if( this->player != player ){
        return false;
    }
    else{
        for( int i=0; i < boardsize; i++ ){
            if( this->board[i] != board[i] ){
                return false;
            }
        }
        return true;
    }
}

bool GoState::sameAs( GoState s ){
    assert( s.dim == dim );
    return sameAs( s.board, s.player );
}

void GoState::togglePlayer() {
    (player == WHITE) ? player = BLACK : player = WHITE;
}

string GoState::toString(){
    string out;

    stringstream ss;
    ss << "Player: " << player << endl;
    out += ss.str();

    for( int i=0; i<boardsize; i++ ){
        if(      board[i] == BLACK    ){ out += "x"; }
        else if( board[i] == WHITE    ){ out += "o"; }
        else if( board[i] == OFFBOARD ){ out += "_"; }
        else if( board[i] == EMPTY    ){ out += "."; }
        else{                            assert(false); }

        out += " ";
        if( i % bigdim == bigdim-1 ){
            out += "\n";
        }
    }
    ss << "Action : " << action << endl;
    out += ss.str();
    return out;
}

int GoState::neighbor(int ix, DIRECTION dir){
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

int GoState::ix2action( int ix, COLOR player ){
    return ix * (int) player;
}

int GoState::action2ix( int action ){
    return abs(action);
}

COLOR GoState::action2color( int action ){
    assert( action != 0 );
    return (action > 0) ? WHITE : BLACK;
}

int GoState::ix2color( int ix ){
    return (ix == OFFBOARD) ? OFFBOARD : board[ix];
}

int GoState::coordColor2Action( int i, int j, COLOR color ){
    int ix = coord2ix(i,j);
    return ixColor2Action(ix, color);
}

int GoState::ixColor2Action( int ix, COLOR color ){
    assert( color==WHITE || color==BLACK );
    int mod = (color == WHITE) ? 1 : -1;
    return ix*mod;
}

int GoState::coord2ix( int i, int j ){
    return bigdim*i + j;
}

string GoState::ix2coord( int ix ){
    int j = ix % bigdim;
    int i = ix / bigdim;
    stringstream out;
    out << i << "," << j;
    return out.str();
}

bool GoState::isPass( int action ){
    return action == 0;
}

void GoState::setBoard( int* ixs, int len, COLOR color ){
    for( int i=0; i<len; i++ ){
        int ix = ixs[i];
        setBoard( ix, color );
    }
}

void GoState::setBoard( int ix, COLOR color ){
    if( ix >= boardsize || board[ix] == OFFBOARD ){ return; }

    if( color == EMPTY ){
        open_positions.insert(ix);
    }
    else{
        assert( board[ix] == EMPTY );
        open_positions.erase(ix);
    }
    board[ix] = color;
}

void* GoState::neighborsOf( int* to_fill, int ix, int adjacency ){
    assert( adjacency==4 || adjacency==8 );
    for( int dir=0; dir<adjacency; dir++ ){
        to_fill[dir] = neighbor(ix, (DIRECTION) dir);
    }
    //return neighbor_array;
}

void GoState::filterByColor( int* to_fill, 
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

bool GoState::floodFill( int* to_fill,
                       int* to_fill_len, 
                       int epicenter_ix,
                       int adjacency,
                       COLOR* flood_color_array, 
                       int filter_len,
                       COLOR* stop_color_array, 
                       int stop_len ){

    set<int> marked;
    set<int> on_queue;
    queue<int> q;
    q.push(epicenter_ix);
    
    //for(int i=0; i<boardsize; i++){
    //q.push(i);
    //}
    int neighbs[adjacency];
    bool stop_color_not_encountered = true;

    while( !q.empty() ){
        int ix = q.front();
        q.pop();
        marked.insert(ix);
        //cout << "inserting: " << ix << endl;
 
        neighborsOf( neighbs, 
                     ix, 
                     adjacency );

        //find if there are neighbors that cause flood fill to stop
        int filtered_len = 0;
        filterByColor( filtered_array,
                       &filtered_len,
                       neighbs,
                       adjacency,
                       stop_color_array,
                       stop_len);

        bool stop_color_is_in_neighbs = filtered_len > 0;
        if( stop_color_is_in_neighbs ){
            //marked.clear();
            stop_color_not_encountered = false;
            break;
        }
        else {
            //find connected neighbors
            filterByColor( filtered_array, 
                           &filtered_len,
                           neighbs,
                           adjacency,
                           flood_color_array,
                           filter_len );
            //see if connector neighbors are already in marked
            //if not, add them
            assert( filtered_len <= 4 );
            for( int faix=0; faix < filtered_len; faix++ ){
                int ix = filtered_array[faix];
                if( marked.find( ix ) == marked.end() && 
                    on_queue.find(ix) == on_queue.end() ){
                    //cout << "pushing: " << ix << endl;
                    q.push(ix);
                    on_queue.insert(ix);
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

    marked.clear();
    queue<int> empty;
    std::swap( q, empty );
    //delete neighbs;
    cout << "\n\n\n\nFINISHING\n\n\n\n";
    return stop_color_not_encountered;
}

//assumes setBoard(action) already applied
bool GoState::isSuicide( int action ){
    COLOR color = action2color( action );
    int ix = action2ix( action );
    //cout << "ix: " << ix << endl;
    int adjacency = 4;

    neighborsOf( neighbor_array, ix, adjacency );

    //same colored neighbors
    COLOR colors[1] = {color};
    int filtered[adjacency+1];
    int filtered_len;
    filterByColor( filtered, &filtered_len,
                   neighbor_array, adjacency,
                   colors, 1 );
    //add the origial ix to the filtered array
    filtered[filtered_len] = ix;
    filtered_len++;


    //floodfill each neighbor stopping if an adjacent EMPTY is found
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
        bool fill_completed = 
        floodFill( floodfill_array, &flood_len,
                   nix,
                   adjacency,
                   colors, 1,
                   stop_array, 1 );
        //delete colors;

        if( fill_completed ){
            for( int j=0; j < flood_len; j++ ){
                //cout << "marking: " << floodfill_array[j] << endl;
                marked.insert( floodfill_array[j] );
            }
        }
        //cout << "nix: " << nix << "no_liber: " << (flood_len > 0) << endl;
        left_with_no_liberties |= fill_completed;
    }
    //delete filtered;
    //delete stop_array;
    bool surrounded_by_kin = true;
    for( int i=0; i<adjacency; i++ ){
        int ncolor = ix2color( neighbor_array[i] );
        surrounded_by_kin &= ncolor == color || ncolor == OFFBOARD;
    }
                    
    return left_with_no_liberties || surrounded_by_kin;
}
