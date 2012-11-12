#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "../go/gostate_struct.h"
#include "godomain.cpp"

#include <time.h>
//#include "simpleclass.cu"
//#include "simplestruct.cu"

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error 
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__) 
inline void __checkCudaErrors(cudaError err, const char *file, const int line ) 
{ 
    if(cudaSuccess != err) 
    { 
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) ); 
        exit(-1);         
    } 
}

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
      printf("Error at %s:%d\n",__FILE__,__LINE__);            \
      return EXIT_FAILURE;}} while(0)

const int num_blocks = 10;
const int num_elems = 10;
//generate a random number for the ind'th block
__device__ float generate( curandState* globalState, int ind ){
    //int tid = blockIdx.x;
    curandState localState = globalState[ind];
    float rnd = curand_uniform( &localState );
    globalState[ind] = localState;
    return rnd;
}
//////////////////////////////////////////////////////////////////////////////
//                     GoStructState.cu
/////////////////////////////////////////////////////////////////////////////
GoStateStruct::GoStateStruct(){
    player = BLACK;
    action = 42;
    num_open = 0;
    frozen_num_open = 0;
    for( int i=0; i<BOARDSIZE; i++ ){
        if( i < BIGDIM || 
            i % BIGDIM == 0 || 
            i % BIGDIM == BIGDIM-1 || 
            i >= BOARDSIZE - BIGDIM ) {
                board[i] = OFFBOARD;
                frozen_board[i] = OFFBOARD;
        }
        else{
            board[i] = EMPTY; 
            frozen_board[i] = EMPTY;
            num_open++;
        }
    }
    for( int i=0; i<PAST_STATE_SIZE; i++ ){
        past_boards[i] = OFFBOARD;
    }
    for( int i=0; i<NUM_PAST_STATES; i++ ){
        past_actions[i] = -1;
        past_players[i] = OFFBOARD;
    }
}
/*
GoStateStruct::GoStateStruct( void** pointers ){
    action = *((int*) pointers[0]);
    num_open = *((int*) pointers[1]);
    for( int i=0; i<BOARDSIZE; i++ ){
        board[i] = ((char*) pointers[2])[i];
    }
    player = *((char*) pointers[3]);
    for( int i=0; i<PAST_STATE_SIZE; i++ ){
        past_boards[i] = ((char*) pointers[4])[i];
    }
    for( int i=0; i<NUM_PAST_STATES; i++ ){
        past_players[i] = ((char*) pointers[5])[i];
    }
    for( int i=0; i<NUM_PAST_STATES; i++ ){
        past_actions[i] = ((int*) pointers[6])[i];
    }
    for( int i=0; i<BOARDSIZE; i++ ){
        frozen_board[i] = ((char*) pointers[7])[i];
    }
    frozen_num_open = *((int*) pointers[8]);
}*/

int GoStateStruct::numElementsToCopy(){
    return 9;
}

/*
void GoStateStruct::cudaAllocateAndCopy( void** pointers ){
    int* dev_action;
    int* dev_num_open;
    char* dev_board;
    char* dev_player;
    char* dev_past_boards;
    char* dev_past_players;
    int* dev_past_actions;
    char* dev_frozen_board;
    int* dev_frozen_num_open;

    cudaMalloc( (void**)&dev_action, sizeof(int) );
    cudaMemcpy( dev_action, &(action), sizeof(int), cudaMemcpyHostToDevice );
    pointers[0] = (void*) dev_action;

    cudaMalloc( (void**)&dev_num_open, sizeof(int) );
    cudaMemcpy( dev_num_open, &(num_open), sizeof(int), cudaMemcpyHostToDevice );
    pointers[1] = (void*) dev_num_open;

    cudaMalloc( (void**)&dev_board, BOARDSIZE*sizeof(char) );
    cudaMemcpy( dev_board, board, BOARDSIZE*sizeof(char), cudaMemcpyHostToDevice );
    pointers[2] = (void*) dev_board;

    cudaMalloc( (void**)&dev_player, sizeof(char) );
    cudaMemcpy( dev_player, &(player), sizeof(char), cudaMemcpyHostToDevice );
    pointers[3] = (void*) dev_player;

    cudaMalloc( (void**)&dev_past_boards, sizeof(char)*PAST_STATE_SIZE );
    cudaMemcpy( dev_past_boards, past_boards, PAST_STATE_SIZE*sizeof(char), cudaMemcpyHostToDevice );
    pointers[4] = (void*) dev_past_boards;

    cudaMalloc( (void**)&dev_past_players, sizeof(int)*NUM_PAST_STATES );
    cudaMemcpy( dev_past_players, past_players, sizeof(int)*NUM_PAST_STATES, cudaMemcpyHostToDevice );
    pointers[5] = (void*) dev_past_players;
    
    cudaMalloc( (void**)&dev_past_actions, sizeof(int)*NUM_PAST_STATES );
    cudaMemcpy( dev_past_actions, past_actions, sizeof(int)*NUM_PAST_STATES, cudaMemcpyHostToDevice );
    pointers[6] = (void*) dev_past_actions;
    
    cudaMalloc( (void**)&dev_frozen_board, sizeof(char)*BOARDSIZE );
    cudaMemcpy( dev_frozen_board, frozen_board, BOARDSIZE*sizeof(char), cudaMemcpyHostToDevice );
    pointers[7] = (void*) dev_frozen_board;
    
    cudaMalloc( (void**)&dev_frozen_num_open, sizeof(int) );
    cudaMemcpy( dev_frozen_num_open, &frozen_num_open, sizeof(int), cudaMemcpyHostToDevice );
    pointers[8] = (void*) dev_frozen_num_open;
}
*/

void GoStateStruct::freezeBoard(){
    for( int i=0; i<BOARDSIZE; i++ ){
        frozen_board[i] = board[i];
    }
    frozen_num_open = num_open;
}

void GoStateStruct::thawBoard(){
    for( int i=0; i<BOARDSIZE; i++ ){
        //setBoard(i, frozen_board[i]);
        board[i] = frozen_board[i];
    }
    num_open = frozen_num_open;
}

void GoStateStruct::copyInto( GoStateStruct* target ){
    target->player = player;
    target->action = action;
    target->num_open = num_open;
    target->frozen_num_open = frozen_num_open;
    for( int i=0; i<BOARDSIZE; i++ ){
        target->board[i] = board[i];
        target->frozen_board[i] = frozen_board[i];
    }
    for( int i=0; i<PAST_STATE_SIZE; i++ ){
        target->past_boards[i] = past_boards[i];
    }
    for( int i=0; i<NUM_PAST_STATES; i++ ){
        target->past_players[i] = past_players[i];
        target->past_actions[i] = past_actions[i];
    }
}

void* GoStateStruct::copy(){
    GoStateStruct* s = (GoStateStruct*) malloc(sizeof(GoStateStruct));
    for( int i=0; i<BOARDSIZE; i++ ){
        s->board[i] = board[i];
        s->frozen_board[i] = frozen_board[i];
    }

    s->num_open = num_open;
    s->player = player;
    s->action = action;

    for( int i=0; i<PAST_STATE_SIZE; i++){
        s->past_boards[i] = past_boards[i];
    }
    for( int i=0; i<NUM_PAST_STATES; i++ ){
        s->past_players[i] = past_players[i];
        s->past_actions[i] = past_actions[i];
    }
    return (void*) s; 
};

char GoStateStruct::flipColor( char c ){
    //assert( c == WHITE || c == BLACK );
    return (c == WHITE) ? BLACK : WHITE;
}

bool GoStateStruct::sameAs( char* other_board, char player ){
    if( player != player ){
        return false;
    }
    else{
        for( int i=0; i < BOARDSIZE; i++ ){
            if( board[i] != other_board[i] ){
                return false;
            }
        }
        return true;
    }
}

//bool GoStateStruct::sameAs( GoStateStruct* gss2 ){
//return sameAs( gss2->board, gss2->player );
//}

void GoStateStruct::togglePlayer() {
    (player == WHITE) ? player = BLACK : player = WHITE;
}

string GoStateStruct::boardToString( char* board ){
    string out;
    for( int i=0; i<BOARDSIZE; i++ ){
        if(      board[i] == BLACK    ){ out += "x"; }
        else if( board[i] == WHITE    ){ out += "o"; }
        else if( board[i] == OFFBOARD ){ out += "_"; }
        else if( board[i] == EMPTY    ){ out += "."; }
        else{   
            printf( "offending pos is %d, %c\n", i, board[i] );
            assert(false); }

        out += " ";
        if( i % BIGDIM == BIGDIM-1 ){
            out += "\n";
        }
    }
    return out;
}

string GoStateStruct::toString(){
    string out;

    stringstream ss;
    ss << "Player: " << player << endl;
    out += ss.str();
    
    out += boardToString( board );
    
    ss << "Action : " << action << endl;
    out += ss.str();
    return out;
}

int GoStateStruct::neighbor( int ix, DIRECTION dir){
    if( board[ix] == OFFBOARD ){
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

int GoStateStruct::ix2action( int ix, char player ){
    int parity;
    if( player == WHITE ){
        parity = 1;
    }
    else{
        parity = -1;
    }
    return ix * parity;
}

int GoStateStruct::action2ix( int action ){
    return abs(action);
}

char GoStateStruct::action2color( int action ){
    //assert( action != 0 );
    return (action > 0) ? WHITE : BLACK;
}

int GoStateStruct::ix2color( int ix ){
    return (ix == OFFBOARD) ? OFFBOARD : board[ix];
}

int GoStateStruct::coord2ix( int i, int j ){
    return BIGDIM*i + j;
}

int GoStateStruct::ixColor2Action( int ix, char color ){
    //assert( color==WHITE || color==BLACK );
    int mod = (color == WHITE) ? 1 : -1;
    return ix*mod;
}

int GoStateStruct::coordColor2Action( int i, int j, char color ){
    int ix = coord2ix(i,j);
    return ixColor2Action(ix, color);
}

bool GoStateStruct::isPass( int action ){
    return action == PASS;
}

void GoStateStruct::setBoard( int ix, char color ){ 
    if( ix >= BOARDSIZE || board[ix] == OFFBOARD ){ return; } 
 
    if( color == EMPTY ){ 
        num_open++; 
    } 
    else{ 
        //assert( board[ix] == EMPTY ); 
        num_open--; 
    } 
    board[ix] = color; 
}

void GoStateStruct::setBoard( int* ixs, int len, char color ){
    for( int i=0; i<len; i++ ){
        int ix = ixs[i];
        setBoard( ix, color );
    }
}    

void GoStateStruct::neighborsOf( int* to_fill, int ix, int adjacency ){
    //assert( adjacency==4 || adjacency==8 );
    for( int dir=0; dir<adjacency; dir++ ){
        to_fill[dir] = neighbor( ix, (DIRECTION) dir);
    }                        
}

void GoStateStruct::filterByColor( 
                    int* to_fill, 
                    int* to_fill_len,
                    int* neighbs,
                    int adjacency,
                    char* color_array,
                    int filter_len ){

    *to_fill_len = 0;
    //int fillix = 0;  
    char ncolor;       
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

bool GoStateStruct::floodFill(  
                int* to_fill,
                int* to_fill_len,
                int epicenter_ix,
                int adjacency,
                char* flood_color_array,
                int filter_len,
                char* stop_color_array,
                int stop_len ){

    marked.clear(); //initBitMask();
    on_queue.clear(); //initBitMask();
    queue.clear(); //initQueue();

    queue.push( epicenter_ix );
    
    //int neighbs[adjacency];
    bool stop_color_not_encountered = true;

    while( !queue.isEmpty() ){
        int ix = queue.pop();
        marked.set( ix, true);
 
        neighborsOf( internal_neighbor_array, 
                     ix, 
                     adjacency );

        //find if there are neighbors that cause flood fill to stop
        int filtered_len = 0;
        filterByColor( internal_filtered_array,
                       &filtered_len,
                       internal_neighbor_array,
                       adjacency,
                       stop_color_array,
                       stop_len);

        bool stop_color_is_in_neighbs = filtered_len > 0;
        if( stop_color_is_in_neighbs ){
            stop_color_not_encountered = false;
            break;
        }
        else {
            //find connected neighbors
            filterByColor( internal_filtered_array,
                           &filtered_len,
                           internal_neighbor_array,
                           adjacency,
                           flood_color_array,
                           filter_len );
            //see if connector neighbors are already in marked
            //if not, add them
            //assert( filtered_len <= 4 );
            for( int faix=0; faix < filtered_len; faix++ ){
                int ix = internal_filtered_array[faix];
                if( !marked.get( ix) && !on_queue.get( ix) ){
                    queue.push(ix);
                    on_queue.set( ix,true);
                }
            }
        }
    }

    //populate to_fill; kinda of unnecassay just did to keep use pattern same
    int i = 0;
    for( int j=0; j<BOARDSIZE; j++ ){
        if( marked.get( j ) ){
            to_fill[i++] = j;
        }
    }
    *to_fill_len = i;

    return stop_color_not_encountered;
}

bool GoStateStruct::isSuicide( int action ){
    char color = action2color( action );
    int ix = action2ix( action );
    //cout << "ix: " << ix << endl;
    int adjacency = 4;

    neighborsOf( neighbor_array, ix, adjacency );

    //same colored neighbors
    char colors[1] = {color};
    int filtered[ADJ_PLUS_ONE];
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
    //set<int> marked;
    BitMask marked;

    char stop_array[1] = {EMPTY};

    for( int i=0; i < filtered_len; i++ ){
        int nix = filtered[i];
        //cout << "nix: " << nix << endl;
        if( marked.get( nix ) ){ continue; }

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
                //marked.insert( floodfill_array[j] );
                marked.set( floodfill_array[j], true );
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

bool GoStateStruct::isDuplicatedByPastState(){
    for( int i=NUM_PAST_STATES-1; i>=0; i-- ){
        int offset = i*BOARDSIZE;
        //int end = begin+BOARDSIZE-1;
        if( sameAs( &past_boards[offset], past_players[i] ) ){
            return true;
        }
    }
    return false;
}

//void GoStateStruct::advancePastStates( GoStateStruct* newest_past_state ){
void GoStateStruct::advancePastStates( char* past_board, 
                                       char past_player,
                                       int past_action ){
    //memcpy?
    int c = PAST_STATE_SIZE-BOARDSIZE;
    //shift last i-1 boards over one
    for( int i=0; i<c; i++ ){
        past_boards[i] = past_boards[i+BOARDSIZE];
    }
    //fill in vacant end spots with the latest board (past_board)
    for( int i=c; i<PAST_STATE_SIZE; i++ ){
        //printf("%d : %c\n", i-c, past_board[i-c]);
        past_boards[i] = past_board[i-c];
    }

    for( int i=0; i<NUM_PAST_STATES-1; i++ ){
        past_players[i] = past_players[i+1];
        past_actions[i] = past_actions[i+1];
    }
    past_players[NUM_PAST_STATES-1] = past_player;
    past_actions[NUM_PAST_STATES-1] = past_action;
    
}

bool GoStateStruct::applyAction( int action,
                                 bool side_effects ){

    //cout << "inside applyAction" << endl;
    //assert( action >= 0 );
    //cout << state->toString() << endl;
    //cout << "ix: " << action << " state->player: " << state->player << endl;

    bool legal = true;
    freezeBoard();
    //GoStateStruct* frozen = (GoStateStruct*) state->copy();
    //cout << "froxqne toString: " << frozen->toString() << endl;

    //The action parameter is really the index of the action to be taken
    //need to convert to signed action i.e BLACK or WHITE ie. *-1 or *1
    int ix = action;
    char color = player;
    action = ix2action( action, color);

    if( ! isPass(action) ){
        //assert( state->action2color(action) == state->player );
        //char color = state->action2color(action);
        setBoard( ix, color );
        //resolve captures
        int adjacency = 4;
        //int neighbs[adjacency];
        neighborsOf( neighbor_array, ix, adjacency );
        char opp_color = flipColor(color); 

        //int opp_neighbs[adjacency];
        int opp_len = 0;
        char filter_array[1] = {opp_color};
        filterByColor( filtered_array, &opp_len,
                       neighbor_array, adjacency,
                       filter_array, 1 );

        for( int onix=0; onix < opp_len; onix++ ){
            int floodfill_len = 0;
            char stop_color_array[1] = {EMPTY};
            bool fill_completed =
            floodFill( floodfill_array, &floodfill_len,
                              filtered_array[onix],
                              adjacency,
                              filter_array, 1,
                              stop_color_array, 1 );
            if( fill_completed ){
                setBoard(
                          floodfill_array,
                          floodfill_len, 
                          EMPTY );
            }
        }
        if( isSuicide( action ) ){
            legal = false;
        }

        //TODO
        //change this for new abstraction
        //check past states for duplicates
        legal &= !isDuplicatedByPastState();
        /*
        for( int i=0; i < NUM_PAST_STATES; i++ ){
            GoStateStruct* past_state = state->past_states[i];
            if( state->sameAs( past_state->board,
                        state->flipColor( past_state->player ) ) ){
                legal = false;
                break;
            }
        }*/
    }
    
    //cout << "testing legality" << endl;
    if( legal ){
        if( side_effects ){
            //cout << "legal and side effects" <<endl;
            advancePastStates( frozen_board, 
                                      player,
                                      action );

            action = action;
            togglePlayer();
        }
        else{
            //cout << "legal, no se" << endl;
            thawBoard();
        }
        return true;
    }
    else{
        if( side_effects ){
            //cout << "action: " << action << endl;
            //assert(false);
        }
        else{
            thawBoard();
        }
        return false;
    }  
}

//return an unsigned action, i.e an ix in the board
__device__
int GoStateStruct::randomAction( curandState* globalState,
                  int tid,
                  BitMask* to_exclude ){
    //cout << "inside randomAction" << endl;
    //GoStateStruct* state = (GoStateStruct*) uncast_state;
    int size = num_open; //state->open_positions.size();
    int empty_ixs[ BOARDSIZE /*size*/  ];
    //cout << "size: " << size << endl;

    int i = 0;
    int j;
    //can shuffle randomly as we insert...
    for( int ix=0; ix<BOARDSIZE; ix++ ){
        //cout << "random shuffle i: " << i << endl;
        if( board[ix] == EMPTY ){
            if( i == 0 ){
                empty_ixs[0] = ix;
            }
            else{
                //j = rand() % i;
                j = (int) ( generate(globalState,tid) * i );
                empty_ixs[i] = empty_ixs[j];
                empty_ixs[j] = ix;
            }
            i++;
        }
    }
    //cout << "after shuffled" << endl;

    //try each one to see if legal
    bool legal_moves_available = false;
    int candidate;
    for( int j=0; j<size; j++ ){
        candidate = empty_ixs[j];
        bool is_legal = applyAction( candidate, false );
        if( is_legal ){
            legal_moves_available = true;
            if( !to_exclude->get( candidate ) ){
                return candidate;
            }
        }
    }

    if( legal_moves_available ){ //but all were excluded...
        return EXCLUDED_ACTION;
    }
    else {
        return PASS;
    }
}

bool GoStateStruct::isTerminal(){
    bool r = action == PASS && 
             past_actions[NUM_PAST_STATES-1] == PASS; //last_state->action == PASS;
    return r;
}

 void GoStateStruct::getRewards( int* to_fill ){
    //TODO use state's bitmask
    BitMask marked; //( state->boardsize );

    int white_score = 0;
    int black_score = 0;
    for( int ix=0; ix < BOARDSIZE; ix++ ){
        if( board[ix] == OFFBOARD ||
            marked.get( ix ) ){
            continue;
        }
        if( board[ix] == WHITE ){
            white_score++;
            continue;
        }
        if( board[ix] == BLACK ){
            black_score++;
            continue;
        }

        //find if ix has a neighbors of {WHITE,EMPTY} or {BLACK,EMPTY}
        //if so, set ncolor to be WHITE or BLACK
        //       set nix to be the ix of one such stone
        //else, the ix is not anybody's territory
        char ncolor;
        int nix;

        int adjacency = 4;
        //int neighbs[adjacency];
        neighborsOf( neighbor_array,
                     ix,
                     adjacency );
        int* white_neighbs = filtered_array;
        //int white_neighbs[adjacency];
        int num_white_neighbs = 0;
        char filter_colors[1] = {WHITE};
        filterByColor( white_neighbs, &num_white_neighbs,
                       neighbor_array, adjacency, 
                       filter_colors, 1 );

        //int black_neighbs[adjacency];
        int* black_neighbs = internal_filtered_array;
        int num_black_neighbs = 0;
        filter_colors[0] = BLACK;
        filterByColor( black_neighbs, &num_black_neighbs,
                       neighbor_array, adjacency,
                       filter_colors, 1 );

        bool has_white = num_white_neighbs > 0;
        bool has_black = num_black_neighbs > 0;
        if(      has_white && ! has_black ) { ncolor = WHITE; }
        else if( has_black && ! has_white ) { ncolor = BLACK; }
        else                                { ncolor = EMPTY; }

        //set nix to the first neighbor of the char ncolor
        for( int j=0; j<adjacency; j++ ){
            nix = neighbor_array[j];
            if( ix2color( nix ) == ncolor ){
                break;
            }
        }

        if( ncolor == BLACK || ncolor == WHITE ){
            //this is overkill given how we are moving
            //is enough to just see a color adjacent to an empty
            //assuming the rest bug free, it will be that colors territory
            int floodfill_len = 0;
            char flood_colors[1] = {EMPTY};
            char stop_colors[1] = {flipColor(ncolor)};
            bool are_territories = 
                floodFill( floodfill_array, &floodfill_len,
                           ix, 
                           adjacency,
                           flood_colors, 1,
                           stop_colors, 1 );

            //mark these empty positions regardless of their territory 
            //status
            for( int i=0; i<floodfill_len; i++ ){
                marked.set( floodfill_array[i], true );
                //marked.insert( state->floodfill_array[i] );
                if( are_territories ){
                    if( ncolor == WHITE ){
                        white_score++;
                    }
                    else if( ncolor == BLACK ){
                        black_score++;
                    }
                    //else{ assert(false); }
                }
            }
        }
    }
    white_score *= 2;
    black_score *= 2;
    white_score += 11; //5.5*2

    to_fill[0] = white_score > black_score ? 1 : 0;
    to_fill[1] = black_score > white_score ? 1 : 0;
    //to_fill[0] = white_score;
    //to_fill[1] = black_score;
    //cout << "end getRewards" << endl;
    return;
}

/////////////////////////////////////////////////////////////////////////////
//                     BitMask.cu
/////////////////////////////////////////////////////////////////////////////

BitMask::BitMask(){
    clear();
}

void BitMask::clear(){
    for( int i=0; i<BITMASK_SIZE; i++ ){
        masks[i] = 0;
    }
}

void BitMask::set(int bit, bool value ) {
    if( bit >= BOARDSIZE ){ return; }
    int row = bit / MOD;
    int col = bit % MOD;
    if( value == 1){
        masks[row] |= (int) value << col;
    }
    else if( value == 0 ){
        masks[row] &= (int) value << col;
    }
    else{
        return;
    }
}

bool BitMask::get( int bit ){
    int row = bit / MOD;
    int col = bit % MOD;
    return (bool) ((masks[row] >> col) & 1);
}


/////////////////////////////////////////////////////////////////////////////
//                     Queue.cu
/////////////////////////////////////////////////////////////////////////////

int Queue::ringInc( int i ){
    if( i == BOARDSIZE-1 ){ return 0; }
    else { return i+1; }
}

Queue::Queue(){
    clear();
}

void Queue::clear(){
    for( int i=0; i<BOARDSIZE; i++){
        array[i] = -1;
    }
    begin = 0;
    end = 0;
}

void Queue::push( int a ){
    array[end] = a;
    end = ringInc( end );
    return;
}

int Queue::pop(){
    int r = array[begin];
    array[begin] = -1;
    begin = ringInc(begin);
    return r;
}

bool Queue::isEmpty(){
    return begin == end && array[end] == -1;
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////


__global__ void stochasticSwapper( int* a, 
                                   int* dev_results, 
                                   curandState* globalState  ){
    int tid = blockIdx.x;    // handle the data at this index
    __shared__ int scratch_space[num_elems];

    //init the shared memory
    for( int i=0; i<num_elems; i++ ){
        scratch_space[i] = a[i];
    }

    //do stuff
    int times = (int) (generate( globalState, tid ) * num_elems);
    int rnd1, rnd2, temp;
    for( int i=0; i<times; i++ ){
        rnd1 = (int) (generate( globalState, tid ) * num_elems);
        rnd2 = (int) (generate( globalState, tid ) * num_elems);
        temp = scratch_space[rnd1];
        scratch_space[rnd1] = scratch_space[rnd2];
        scratch_space[rnd2] = temp;
    }
    //scratch_space[rnd] = 42;
    

    //fill the correct row of results
    for( int i=0; i<num_elems; i++ ){
        int ix = tid*num_elems + i;
        dev_results[ix] = scratch_space[i];
    }

}

__global__ void setup_kernel( curandState* state, unsigned long seed ){
    int tid = blockIdx.x;
    curand_init( seed+tid, tid, 0, &state[tid] );
}

__global__ void checkShit( int* member_values, int size, int* results ){
    for( int i=0; i<size; i++ ){
        results[i] = member_values[i] + 42;
    }
}

/*
__global__ void reconstruct( void** pointers, int* results ){
    int tid = blockIdx.x;
    __shared__ simplestruct ss;
    SimpleClass sc( pointers );
    results[tid] = sc.number;
    //do something with sc and put it int results
}
*/

__global__ void reconstructGoState( GoStateStruct* gss,
                                    curandState* globalState, 
                                    int* inputs, 
                                    char* results,
                                    int* winners ){
    int tid = blockIdx.x;
    //int pos = inputs[tid];

    //will want
    __shared__ GoStateStruct gss_local;
    gss->copyInto( &gss_local );
    BitMask to_exclude;
    int count = 0;
    while( count < 1000 && !gss_local.isTerminal() ){
        int action = gss_local.randomAction( globalState, tid, &to_exclude );
    //bool is_legal = gss_local.applyAction( 48, true );
        bool is_legal = gss_local.applyAction( action, true );
        count++;
    }

    for( int i=0; i<BOARDSIZE; i++ ){
        results[tid*BOARDSIZE+i] = gss_local.board[i];
    }

    gss_local.getRewards( &(winners[tid*2]) );

}


int main(void){
    srand( time(NULL) );
    int a[num_elems];
    int* dev_a;
    for( int i=0; i<num_elems; i++ ){
        a[i] = i;
    }


    int results[num_blocks*num_elems];
    int* dev_results;

    //setup rand generators on kernel
    curandState* devStates;
    checkCudaErrors( cudaMalloc( &devStates, num_blocks*sizeof(curandState) ) );
    setup_kernel<<<num_blocks,1>>>( devStates, time(NULL) );

///////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////
    GoStateStruct gss;
    gss.setBoard( 49, WHITE );
    //int num_copy2 = gss.numElementsToCopy();
    //void* pointers2[num_copy2];
    //gss.cudaAllocateAndCopy( pointers2 );

    //void** dev_pointers2;
    //cudaMalloc( (void***)&dev_pointers2, num_copy2*sizeof(void*) );
    //cudaMemcpy( dev_pointers2, pointers2, num_copy2*sizeof(void*), cudaMemcpyHostToDevice);

    //void* pboard;
    //cudaMalloc( (void**)&pboard, BOARDSIZE*sizeof(char) );
    //cudaMemcpy( pboard, gss.board, BOARDSIZE*sizeof(char), cudaMemcpyHostToDevice );

    GoStateStruct* dev_gss;
    //gss = (GoStateStruct*) malloc(sizeof(GoStateStruct));
    cudaMalloc( (void**)&dev_gss, sizeof(GoStateStruct));
    cudaMemcpy( dev_gss, &gss, sizeof(GoStateStruct), cudaMemcpyHostToDevice );
    
    int inputs[num_blocks];
    for( int i=0; i<num_blocks; i++ ){
        int r = rand();
        inputs[i] = r % BOARDSIZE;
    }
    
    int* dev_inputs;
    cudaMalloc( (void**)&dev_inputs, num_blocks*sizeof(int) );
    cudaMemcpy( dev_inputs, inputs, num_blocks*sizeof(int), cudaMemcpyHostToDevice );

    char results2[BOARDSIZE*num_blocks];
    char* dev_results2;
    checkCudaErrors( cudaMalloc( (void**)&dev_results2, num_blocks*BOARDSIZE*sizeof(char) ) );

    int results3[num_blocks*2];
    int* dev_results3;
    cudaMalloc( (void**)&dev_results3, num_blocks*2*sizeof(int) );

    printf("calling kernel\n");
    reconstructGoState<<<num_blocks, 1>>>(dev_gss, 
                                          devStates, 
                                          dev_inputs, 
                                          dev_results2,
                                          dev_results3 );

    int t = clock();
    printf("kernel return, %f secs\n", ((float)t)/CLOCKS_PER_SEC) ;

    cudaMemcpy( results2, dev_results2, num_blocks*BOARDSIZE*sizeof(char), cudaMemcpyDeviceToHost );
    cudaMemcpy( results3, dev_results3, num_blocks*2*sizeof(int), cudaMemcpyDeviceToHost );
    
    t = clock();
    printf("mem transfer, %f secs\n", ((float)t)/CLOCKS_PER_SEC) ;
/*
    for( int i=0; i<num_blocks; i++ ){
        char* p = &(results2[i*BOARDSIZE]);
        const char* s = gss.boardToString( p ).c_str();
        printf("%s\n\n", s );
        printf("%d to %d\n\n", results3[i*2], results3[i*2+1] );

        //printf("%d : %c\n", i, results2[i]);
    }*/
    
    //for( int i=0; i<num_copy2; i++ ){
    //cudaFree( pointers2[i] );
    //}
    
    
    ////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////
    /*
    SimpleClass sc;
    int num_copy = sc.numElementsToCopy();
    //char* types = new char[num_copy];
    //void** pointers = new void*[num_copy];
    void* pointers[num_copy];
    void** dev_pointers;
    //int* sizes = new int[num_copy];
    sc.cudaAllocateAndCopy(pointers);
    
    cudaMalloc( (void***)&dev_pointers, num_copy*sizeof(void*) );
    cudaMemcpy( dev_pointers, pointers, num_copy*sizeof(void*), cudaMemcpyHostToDevice);

    int results2[num_blocks];
    int* dev_results2;
    checkCudaErrors( cudaMalloc( (void**)&dev_results2, num_blocks*sizeof(int) ) );
    reconstruct<<<num_blocks,1>>>( dev_pointers, dev_results2);

    checkCudaErrors( cudaMemcpy( results2, dev_results2, num_blocks*sizeof(int), cudaMemcpyDeviceToHost ) );

    for( int i=0; i<num_blocks; i++){
        printf("%d, ", results2[i]);
    }
    printf("\n");

    for( int i=0; i<num_copy; i++ ){
        cudaFree( pointers[i] );
    }
*/
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////

    checkCudaErrors( cudaMalloc( (void**)&dev_a, num_elems * sizeof(int) ) );
    checkCudaErrors( cudaMalloc( (void**)&dev_results, num_elems * num_blocks * sizeof(int) ) );

    checkCudaErrors( cudaMemcpy( dev_a, a, num_elems * sizeof(int),
                              cudaMemcpyHostToDevice ) );

    stochasticSwapper<<<num_blocks,1>>>(dev_a, dev_results, devStates);

    //copy back
    checkCudaErrors( cudaMemcpy( results, 
                                 dev_results, 
                                 num_elems*num_blocks*sizeof(int), 
                                 cudaMemcpyDeviceToHost ) );

    for( int i=0; i<num_blocks; i++ ){
        for( int j=0; j<num_elems; j++ ){
            int ix = i*num_elems+j;
            //printf("ix: %d", ix );
            printf("%d,", results[ix] );
        }
        printf( "\n" );
    }

    cudaFree( dev_a );
    cudaFree( dev_results );
    //cudaFree( dev_results2 );
    //TODO
    //cudaFree( everything allocated by sc.cudaAllocate )

    return 0;
}

