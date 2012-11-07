#include <stdlib.h>
//#include <assert.h>
//#include <iostream>
//#include <sstream>

#include "queue.cpp"
#include "bitmask.cpp"

#define NUM_PASTSTATES 2
#define BIGDIM 11
#define BOARDSIZE 121
#define PASTSTATE_SIZE 244 //BOARDSIZE*NUM_PASTSTATES+2

#define BLACK 'b'
#define WHITE 'w'
#define EMPTY 'e'
#define OFFBOARD 'o'

typedef struct {
    char board[BOARDSIZE];
    int action;
    int num_open;
    char player;

    char past_states[PASTSTATE_SIZE]; 

    //TODO
    //save space by treating this as char array. Numbers [0,121] < 2^8
    int floodfill_array[BOARDSIZE];
    int neighbor_array[8];
    int filtered_array[8];
    char color_array[3];

} GoStateStruct;

//void initBoard( bool shallow ){
//}

int numElementsToCopy(){
    return 5;
}

void GoState::cudaAllocateAndCopy( void** pointers ){
    int* dev_bigdim;
    int* dev_boardsize;
    int* dev_action;
    int* dev_num_open;
    char* dev_board;
    char* dev_player;
    bool* dev_shallow;

    //cudaMalloc( (void**)&dev_dim, sizeof(int) );
    cudaMalloc( (void**)&dev_bigdim, sizeof(int) );
    cudaMemcpy( dev_bigdim, &bigdim, sizeof(int), cudaMemcpyHostToDevice );
    pointers[0] = (void*) dev_bigdim;

    cudaMalloc( (void**)&dev_boardsize, sizeof(int) );
    cudaMemcpy( dev_boardsize, &boardsize, sizeof(int), cudaMemcpyHostToDevice );
    pointers[1] = (void*) dev_boardsize;

    cudaMalloc( (void**)&dev_action, sizeof(int) );
    cudaMemcpy( dev_action, &action, sizeof(int), cudaMemcpyHostToDevice );
    pointers[2] = (void*) dev_action;

    cudaMalloc( (void**)&dev_num_open, sizeof(int) );
    cudaMemcpy( dev_num_open, &num_open, sizeof(int), cudaMemcpyHostToDevice );
    pointers[3] = (void*) dev_num_open;

    cudaMalloc( (void**)&dev_board, boardsize*sizeof(char) );
    cudaMemcpy( dev_board, board, boardsize*sizeof(char), cudaMemcpyHostToDevice );
    pointers[4] = (void*) dev_board;

    cudaMalloc( (void**)&dev_player, sizeof(char) );
    cudaMemcpy( dev_player, &player, sizeof(char), cudaMemcpyHostToDevice );
    pointers[5] = (void*) dev_player;

    cudaMalloc( (void**)&dev_shallow, sizeof(bool) );
    cudaMemcpy( dev_shallow, &shallow, sizeof(bool), cudaMemcpyHostToDevice );
    pointers[6] = (void*) dev_shallow;
}

GoStateStruct* copy( GoStateStruct* gss ){
    GoStateStruct* s = malloc(sizeof(GoStateStruct));
    for( int i=0; i<BOARDSIZE; i++ ){
        s->board[i] = gss->board[i];
    }

    s->num_open = gss->num_open;
    s->player = gss->player;
    s->action = gss->action;

    for( int i=0; i<PASTSTATE_SIZE; i++){
        s->past_states[i] = gss->past_states[i];
    }
    return s; 
};

char flipColor( char c ){
    assert( c == WHITE || c == BLACK );
    return (c == WHITE) ? BLACK : WHITE;
}

bool sameAs( GoStateStruct* gss, char* board, char player ){
    if( gss->player != player ){
        return false;
    }
    else{
        for( int i=0; i < BOARDSIZE; i++ ){
            if( gss->board[i] != board[i] ){
                return false;
            }
        }
        return true;
    }
}

bool sameAs( GoStateStruct* gss1, GoStateStruct* gss2 ){
    return sameAs( gss1, gss2->board, gss2->player );
}

void togglePlayer( GoStateStruct* gss ) {
    (gss->player == WHITE) ? gss->player = BLACK : gss->player = WHITE;
}

string GoState::toString( GoStateStruct* gss ){
    string out;

    stringstream ss;
    ss << "Player: " << gss->player << endl;
    out += ss.str();

    for( int i=0; i<BOARDSIZE; i++ ){
        if(      gss->board[i] == BLACK    ){ out += "x"; }
        else if( gss->board[i] == WHITE    ){ out += "o"; }
        else if( gss->board[i] == OFFBOARD ){ out += "_"; }
        else if( gss->board[i] == EMPTY    ){ out += "."; }
        else{                                 assert(false); }

        out += " ";
        if( i % BIGDIM == BIGDIM-1 ){
            out += "\n";
        }
    }
    ss << "Action : " << gss->action << endl;
    out += ss.str();
    return out;
}

int neighbor(GoStateStruct* gss, int ix, DIRECTION dir){
    if( gss->board[ix] == OFFBOARD ){
        return OFFBOARD;
    }
    else{
        if(      dir == N ){  return ix - BIGDIM; }
        else if( dir == S ){  return ix + BIGDIM; }
        else if( dir == E ){  return ix + 1;}
        else if( dir == W ){  return ix -1; }
        else if( dir == NW ){ return ix - BIGDIM - 1; }
        else if( dir == NE ){ return ix - BIGDIM + 1; }
        else if( dir == SW ){ return ix + BIGDIM - 1; }
        else {//if( dir == SE ){ 
            return ix + BIGDIM + 1;
        }
    }
}

int ix2action( GoStateStruct* gss, int ix, char player ){
    int parity;
    if( gss->player == WHITE ){
        parity = 1;
    }
    else{
        parity = -1;
    }
    return ix * parity;
}

int action2ix( int action ){
    return abs(action);
}

char action2color( int action ){
    assert( action != 0 );
    return (action > 0) ? WHITE : BLACK;
}

int ix2color( GoStateStruct* gss, int ix ){
    return (ix == OFFBOARD) ? OFFBOARD : gss->board[ix];
}

int coordColor2Action( int i, int j, char color ){
    int ix = coord2ix(i,j);
    return ixColor2Action(ix, color);
}

int ixColor2Action( int ix, char color ){
    assert( color==WHITE || color==BLACK );
    int mod = (color == WHITE) ? 1 : -1;
    return ix*mod;
}

int coord2ix( int i, int j ){
    return BIGDIM*i + j;
}

