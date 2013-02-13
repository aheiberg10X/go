#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>

#include "gostate_struct.h"
#include "kernel.h"
#include "godomain.cpp"
#include "zobrist.h"

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
void GoStateStruct::ctor(ZobristHash* zh){
    zhasher = zh;
    player = BLACK;
    action = 12345678;
    zhash = 0;
    frozen_zhash = 0;
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
    //for( int i=0; i<PAST_STATE_SIZE; i++ ){
    //past_boards[i] = OFFBOARD;
    //}
    for( int i=0; i<NUM_PAST_STATES; i++ ){
        past_actions[i] = -1;
        past_zhashes[i] = 0;
        //past_players[i] = OFFBOARD;
    }

    //marked.clear();
    //connected_to_lib.clear();
    //queue.clear();
    open_intersections.clear();
}

void GoStateStruct::freezeBoard(){
    //for( int i=0; i<BOARDSIZE; i++ ){
    //frozen_board[i] = board[i];
    //}
    memcpy( frozen_board, board, BOARDSIZE*sizeof(char) );
    frozen_num_open = num_open;
    frozen_zhash = zhash;
    //open_intersections
}

void GoStateStruct::thawBoard(){
    //for( int i=0; i<BOARDSIZE; i++ ){
    ////setBoard(i, frozen_board[i]);
    //board[i] = frozen_board[i];
    //}
    memcpy( board, frozen_board, BOARDSIZE*sizeof(char) );
    num_open = frozen_num_open;
    zhash = frozen_zhash;
}

void GoStateStruct::copyInto( GoStateStruct* target ){
    target->zhash = zhash;
    target->player = player;
    target->action = action;
    target->num_open = num_open;
    target->frozen_num_open = frozen_num_open;
    target->zhasher = zhasher;
    memcpy( target->board, board, BOARDSIZE*sizeof(char) );
    memcpy( target->frozen_board, frozen_board, BOARDSIZE*sizeof(char) );
    /*
    for( int i=0; i<BOARDSIZE; i++ ){
        target->board[i] = board[i];
        target->frozen_board[i] = frozen_board[i];
    }
    */
    //for( int i=0; i<PAST_STATE_SIZE; i++ ){
    //target->past_boards[i] = past_boards[i];
    //}
    //TODO: do we every use the past actions besides the most recent?
    memcpy( target->past_actions, past_actions, NUM_PAST_STATES*sizeof(int) );
    memcpy( target->past_zhashes, past_zhashes, NUM_PAST_STATES*sizeof(int) );
    /*
    for( int i=0; i<NUM_PAST_STATES; i++ ){
        //target->past_players[i] = past_players[i];
        target->past_actions[i] = past_actions[i];
        target->past_zhashes[i] = past_zhashes[i];
    }
    */

    open_intersections.copyInto( &(target->open_intersections) );
}

//no need for copyInto and copy
void* GoStateStruct::copy(){
    GoStateStruct* s = (GoStateStruct*) malloc(sizeof(GoStateStruct));
    this->copyInto(s);
    return (void*) s;
    /*
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
    return (void*) s;*/ 
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
    ss << "Num open: " << num_open << endl;
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

    char incumbant_color = board[ix]; 
    if( color == EMPTY ){ 
        zhash = zhasher->updateHash( zhash, ix, incumbant_color );
        num_open++; 
        //open_intersections.set( ix, true );
    } 
    else{ 
        //assert( board[ix] == EMPTY ); 
        zhash = zhasher->updateHash( zhash, ix, color );
        num_open--; 
        //open_intersections.set( ix, false );
    } 
    board[ix] = color; 
}

void GoStateStruct::setBoard( int* ixs, int len, char color ){
    for( int i=0; i<len; i++ ){
        int ix = ixs[i];
        setBoard( ix, color );
    }
}    

void GoStateStruct::capture( BitMask* bm ){
    int nSet = 0;
    int ix = 0;
    while( nSet < bm->count ){
        if( bm->get(ix) ){ 
            setBoard( ix, EMPTY );
            nSet++;
        }
        ix++;
    }
}

void GoStateStruct::neighborsOf2( int* to_fill, int* to_fill_len,
                                  int ix, int adjacency,
                                  char filter_color ){
    if( board[ix] == OFFBOARD ){ return; }
    int fillix = 0;
    if( adjacency == 4 ){
        //N
        if( board[ix-BIGDIM] == filter_color  ){ 
            to_fill[fillix++] = ix-BIGDIM;
        }
        //S
        if( board[ix+BIGDIM] == filter_color ){
            to_fill[fillix++] = ix+BIGDIM;
        }
        //E
        if( board[ix+1] == filter_color ){
            to_fill[fillix++] = ix+1;
        }
        //W
        if( board[ix-1] == filter_color ){
            to_fill[fillix++] = ix-1;
        }
    }
    else{
    }
    *to_fill_len = fillix;
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

    //*to_fill_len = 0;
    int fillix = 0;  
    char ncolor;       
    for(int naix=0; naix<adjacency; naix++){
        int nix = neighbs[naix];
        ncolor = board[nix];
        for( int caix=0; caix<filter_len; caix++ ){
            if( ncolor == color_array[caix] ){ 
                to_fill[ fillix++ ] = nix;
                //to_fill[(*to_fill_len)] = nix;
                //*to_fill_len = *to_fill_len + 1;
            }
        }
    }
    *to_fill_len = fillix;
}

bool GoStateStruct::floodFill2(  
        //BitMask* connected_to_lib
        //BitMask* marked, 
                int epicenter_ix,
                int adjacency,
                char flood_color,
                char stop_color ){

    //on_queue.clear(); //initBitMask();
    queue.clear(); //initQueue();
    marked.clear();

    //cout << "epicenter: " << epicenter_ix << endl;
    queue.push( epicenter_ix );
    //connected_to_lib.set( epicenter_ix, true );
    marked.set( epicenter_ix, true );
    
    //int neighbs[adjacency];
    bool stop_color_not_encountered = true;

    while( !queue.isEmpty() ){
        int ix = queue.pop();
        //marked->set( ix, true);
 
        //find if there are neighbors that cause flood fill to stop
        int filtered_len = 0;

        neighborsOf2( internal_filtered_array,
                      &filtered_len,
                      ix, adjacency, stop_color );

        bool stop_color_is_in_neighbs = filtered_len > 0;
        if( stop_color_is_in_neighbs ){
            //cout << "stop color in neighbs" << endl;
            stop_color_not_encountered = false;
            break;
        }
        else if( connected_to_lib.get(ix) ){
            //cout << "ix: " << ix << " connected to lib" << endl;
            stop_color_not_encountered = false;
            continue;
        }
        else {
            neighborsOf2( internal_filtered_array,
                          &filtered_len,
                          ix, adjacency, flood_color );
            
            //see if connector neighbors are already in marked
            //if not, add them
            //assert( filtered_len <= 4 );
            for( int faix=0; faix < filtered_len; faix++ ){
                int nix = internal_filtered_array[faix];
                //cout << "same color neighb of: " << ix << " : " << nix << endl;
                if( /*!marked->get( ix) &&*/ !marked.get( nix) ){
                    //cout << "pushing: " << nix << " on queue" << endl;
                    queue.push(nix);
                    marked.set( nix,true);
                    //will clear this if turns out no libs
                    //connected_to_lib.set( ix, true );
                }
            }
        }
    }

    return stop_color_not_encountered;
}

/*
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
}*/

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
    //(Remember space making opponent captures have already been applied)
    bool left_with_no_liberties = false;
    //BitMask marked;
    char stop_array[1] = {EMPTY};
    for( int i=0; i < filtered_len; i++ ){
        int nix = filtered[i];
        //cout << "nix: " << nix << endl;
        //if( marked.get( nix ) ){ continue; }
        //if( connected_to_lib.get( nix ) ){ continue; }

        //int flood_len = 0;
        bool fill_completed = 
            floodFill2( nix, adjacency, color, EMPTY );
        //cout << "nix: " << nix << " fill compl: " << fill_completed << endl;
        //floodFill( floodfill_array, &flood_len,
        //nix,
        //adjacency,
        //colors, 1,
        //stop_array, 1 );

        /*
        if( fill_completed ){
            for( int j=0; j < flood_len; j++ ){
                //cout << "marking: " << floodfill_array[j] << endl;
                //marked.insert( floodfill_array[j] );
                marked.set( floodfill_array[j], true );
            }
        }*/
        //cout << "nix: " << nix << "no_liber: " << (flood_len > 0) << endl;
        left_with_no_liberties |= fill_completed;
    }
                    
    return left_with_no_liberties;
}

bool GoStateStruct::isDuplicatedByPastState(){
    for( int i=NUM_PAST_STATES-1; i>=0; i-- ){
        if( zhash == past_zhashes[i] ){ return true; }
    }
    /*
    for( int i=NUM_PAST_STATES-1; i>=0; i-- ){
        int past_player = ((NUM_PAST_STATES-1)-i % 2 == 0) ? WHITE : BLACK;
        int offset = i*BOARDSIZE;
        //int end = begin+BOARDSIZE-1;
        if( sameAs( &past_boards[offset], past_players[i] ) ){
            return true;
        }
    }*/
    return false;
}

//void GoStateStruct::advancePastStates( GoStateStruct* newest_past_state ){
void GoStateStruct::advancePastStates( int past_zhash, //char* past_board, 
                                        //char past_player,
                                       int past_action ){
    /*
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
    */

    for( int i=0; i<NUM_PAST_STATES-1; i++ ){
        past_zhashes[i] = past_zhashes[i+1];
        //past_players[i] = past_players[i+1];
        past_actions[i] = past_actions[i+1];
    }
    past_zhashes[NUM_PAST_STATES-1] = past_zhash;
    //past_players[NUM_PAST_STATES-1] = past_player;
    past_actions[NUM_PAST_STATES-1] = past_action;
    
}

bool GoStateStruct::applyAction( int action,
                                 bool side_effects ){


    //cout << "ix: " <<  action << endl;
    //assert( action >= 0 );
    //cout << state->toString() << endl;

    bool legal = true;
    bool board_frozen = false;

    //The action parameter is really the index of the action to be taken
    //need to convert to signed action i.e BLACK or WHITE ie. *-1 or *1
    int ix = action;
    char color = player;
    char opp_color = flipColor(color); 
    action = ix2action( ix, color);

    if( ! isPass(action) ){
        //assert( state->action2color(action) == state->player );
        //char color = state->action2color(action);
        //cout << "setting: " << ix << endl;

        int adjacency = 4;
        neighborsOf( neighbor_array, ix, adjacency );
        bool surrounded_by_kin = true;
        bool no_opp_color_neighbors = true;
        for( int i=0; i<adjacency; i++ ){
            int ncolor = ix2color( neighbor_array[i] );
            surrounded_by_kin &= ncolor == color || ncolor == OFFBOARD;
            no_opp_color_neighbors &= ncolor != opp_color;
        }
        if( no_opp_color_neighbors && !surrounded_by_kin ){
            setBoard( ix, color );
            legal = true;
        }
        else if( surrounded_by_kin ){
            legal = false;
        }
        else{  //must check for captures
            freezeBoard();
            board_frozen = true;

            setBoard( ix, color );
            int opp_len = 0;
            neighborsOf2( filtered_array, &opp_len, 
                          ix, adjacency, opp_color );

            connected_to_lib.clear();
            for( int onix=0; onix < opp_len; onix++ ){
                marked.clear();
                bool fill_completed =
                    floodFill2( 
                        filtered_array[onix],
                        adjacency,
                        opp_color,
                        EMPTY );
                if( fill_completed ){
                    connected_to_lib.clear();
                    capture( &marked );
                }
                else {
                    connected_to_lib.Or( marked );
                }
            }
            if( isSuicide( action ) ){
                legal = false;
            }
        }
        legal &= !isDuplicatedByPastState();
    }
    else{
        //PASS played
    }
    
    //cout << "legal" << legal << endl;
    if( legal ){
        if( side_effects ){
            //cout << "legal and side effects" <<endl;
            advancePastStates( frozen_zhash, //frozen_board, 
                               this->action ); //not the new 'action'?);

            this->action = action;
            togglePlayer();
        }
        else{
            //cout << "legal, no se" << endl;
            if( board_frozen ){
                thawBoard();
            }
        }
        return true;
    }
    else{
        if( board_frozen ){
            thawBoard();
        }
        return false;
    }  
}

//return an unsigned action, i.e an ix in the board
__device__
int GoStateStruct::randomAction( curandState* globalState,
                  int tid,  //tid really blockId, we are changing code
                  BitMask* to_exclude,
                  bool side_effects ){
    //cout << "inside randomAction" << endl;
    //GoStateStruct* state = (GoStateStruct*) uncast_state;
    //int size = num_open; //state->open_positions.size();
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
    //
    return randomActionBase( to_exclude, side_effects, empty_ixs );

    /*
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
    }*/
}

__host__ __device__
int GoStateStruct::randomActionBase( BitMask* to_exclude,
                                     bool side_effects,
                                     int* empty_ixs
                                    ){
    //try each one to see if legal
    //bool legal_moves_available = false;
    bool legal_but_excluded_move_available = false;
    for( int j=0; j<num_open; j++ ){
        int ix = empty_ixs[j];
        //cout << "candidatae: "<< ix << endl;
        bool ix_is_excluded = to_exclude->get(ix);
        if( legal_but_excluded_move_available ){
            if( ! ix_is_excluded ){
                bool is_legal = applyAction( ix, side_effects );
                if( is_legal ){
                    //cout << "num tried until legal1: " << j << endl;
                    return ix;
                }
            }
        }
        else{
            bool is_legal = applyAction( ix, side_effects );
            if( is_legal ){
                if( ix_is_excluded ){
                    legal_but_excluded_move_available = true;
                }
                else{
                    //cout << "num tried until legal: " << j << endl;
                    return ix;
                }
            }
        }
    }
    
    if( legal_but_excluded_move_available ){ return EXCLUDED_ACTION; }
    else { 
        applyAction( PASS, side_effects );
        return PASS;
    }

}

//pick a random position, and iterate until the first empty found
//if legal, 
//  if non-excluded : apply/return
//  else : note that there is a legal move left
//else, keep iterating
//  this time check exclusion before legality
//if you reach original start, stop
//  if legal move was excluded : return EXLCUDED
//  else : return pass
//
//  ah shit this isn't actually random, consider:
//  xxxxxxxxxxxxxxxx__xxxxxxxxxxxxxxx
//  the left empty spot will be chosen almost all the time
//  TODO: take the trick with the legal_but_excluded and apply to truly random version
int GoStateStruct::smarterRandomAction( BitMask* to_exclude,
                                        bool side_effects ){
   
    bool legal_but_excluded_move_available = false;
    int ix = rand() % BOARDSIZE;
    cout << "random ix: " << ix << endl;
    for( int iter=0; iter<BOARDSIZE; iter++ ){
        if( board[ix] == EMPTY ){
            bool ix_is_excluded = to_exclude->get(ix);
            if( legal_but_excluded_move_available ){
                if( ! ix_is_excluded ){
                    bool is_legal = applyAction( ix, side_effects );
                    if( is_legal ){
                        return ix;
                    }
                }
            }
            else{
                bool is_legal = applyAction( ix, side_effects );
                if( is_legal ){
                    if( ix_is_excluded ){
                        legal_but_excluded_move_available = true;
                    }
                    else{
                        return ix;
                    }
                }
            }
        }
        //ring increment
        if( ++ix >= BOARDSIZE ){ ix = 0; }
    }
    if( legal_but_excluded_move_available ){ return EXCLUDED_ACTION; }
    else { return PASS; }
}

__host__
int GoStateStruct::randomAction( BitMask* to_exclude, 
                                 bool side_effects ){
    //stores the empty positions we come across in board
    //filled from right to left
    int empty_intersections[num_open];
    int end = num_open-1;
    int begin = num_open;
    int r;

    bool legal_but_excluded_move_available = false;

    //board_ix tracks our progress through board looking for empty intersections
    int board_ix, num_needed, num_found;
    while( end >= 0 ){
        r = rand() % (end+1);
        //we want to swap the rth and end_th empty intersections.
        //but we are lazy, so don't the rth position might not be filled yet
        //if not, keep searching through board until more are found
        if( r < begin ){
            num_needed = begin-r;
            num_found = 0;
            while( num_found < num_needed ){
                if( board[board_ix] == EMPTY ){
                    begin--;
                    empty_intersections[begin] = board_ix;
                    num_found++;
                }
                board_ix++;
                assert( board_ix <= BOARDSIZE );
            }
        }

        //swap empty[end] and empty[r]
        int temp = empty_intersections[end];
        empty_intersections[end] = empty_intersections[r];
        empty_intersections[r] = temp;

        //test whether the move legal and not excluded
        //return intersection if so
        int candidate = empty_intersections[end];
        //cout << "candidate: " << candidate << endl;
        bool ix_is_excluded = to_exclude->get(candidate);
        if( legal_but_excluded_move_available ){
            if( ! ix_is_excluded ){
                bool is_legal = applyAction( candidate, side_effects );
                if( is_legal ){
                    //cout << "num tried until legal1: " << j << endl;
                    return candidate;
                }
            }
        }
        else{
            bool is_legal = applyAction( candidate, side_effects );
            if( is_legal ){
                if( ix_is_excluded ){
                    legal_but_excluded_move_available = true;
                }
                else{
                    //cout << "num tried until legal: " << j << endl;
                    return candidate;
                }
            }
        }

        //if did not return a legal move, keep going
        end--;
    }


    if( legal_but_excluded_move_available ){ return EXCLUDED_ACTION; }
    else { 
        applyAction( PASS, side_effects );
        return PASS;
    }

    /*

    //can shuffle randomly as we insert...
    for( int ix=0; ix<BOARDSIZE; ix++ ){
        //cout << "random shuffle i: " << i << endl;
        if( board[ix] == EMPTY ){
            nEmpty++;
            if( i == 0 ){
                empty_ixs[0] = ix;
            }
            else{
                j = rand() % i;
                empty_ixs[i] = empty_ixs[j];
                empty_ixs[j] = ix;
            }
            i++;
        }
    }
    assert( nEmpty == num_open );
    return randomActionBase( to_exclude, side_effects, empty_ixs );

    */
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
        if( board[ix] == OFFBOARD || marked.get( ix ) ){
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
        //char ncolor;
        //int nix;

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
        if(      has_white && ! has_black ) { white_score++; /*ncolor = WHITE;*/ }
        else if( has_black && ! has_white ) { black_score++; /*ncolor = BLACK;*/ }
        //else                                { ncolor = EMPTY; }

        /*
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
        }*/
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
// OMFG just just use a bool[] ?
BitMask::BitMask(){
    clear();
}

void BitMask::copyInto( BitMask* target ){
    memcpy( target->masks, masks, BITMASK_SIZE*sizeof(int) );
    //for( int i=0; i<BITMASK_SIZE; i++ ){
    //bm->masks[i] = masks[i];;
    //}
    target->count = count;
}


void BitMask::clear(){
    for( int i=0; i<BITMASK_SIZE; i++ ){
        masks[i] = 0;
    }
    count = 0;
}

void BitMask::set(int bit, bool value ) {
    if( bit >= BOARDSIZE ){ return; }
    int row = bit / MOD;
    int col = bit % MOD;
    if( value == 1){
        masks[row] |= (int) value << col;
        count++;
    }
    else if( value == 0 ){
        masks[row] &= (int) value << col;
        count--;
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

void BitMask::Or( BitMask bm ){
    for( int i=0; i<BITMASK_SIZE; i++ ){
        masks[i] |= bm.masks[i];
    }
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
    //for( int i=0; i<BOARDSIZE; i++){
    //array[i] = -1;
    //}
    begin = 0;
    end = 0;
    nElems = 0;
}

void Queue::push( int a ){
    array[end] = a;
    end = ringInc( end );
    nElems++;
    return;
}

int Queue::pop(){
    int r = array[begin];
    //array[begin] = -1;
    begin = ringInc(begin);
    nElems--;
    return r;
}

bool Queue::isEmpty(){
    return nElems == 0; //begin == end && array[end] == -1;
}


////////////////////////////////////////////////////////////////////////////
//             Zobrist
////////////////////////////////////////////////////////////////////////////

__host__
void ZobristHash::ctor(){
    //B,W,E
    for( int ix=0; ix<NUM_ZOBRIST_VALUES; ix++ ){
        values[ix] = rand();
    }
}

//__device__
//void ZobristHash::ctor( curandState* states, int block_id ){
//for( int ix=0; ix<NUM_ZOBRIST_VALUES; ix++ ){
//values[ix] = generate( states, block_id ) * INT_MAX
//}

void ZobristHash::copyInto( ZobristHash* target ){
    memcpy( target->values, values, NUM_ZOBRIST_VALUES*sizeof(int) );
    //for( int ix=0; ix<NUM_ZOBRIST_VALUES; ix++ ){
    //target->values[ix] = values[ix];
    //}
}

int ZobristHash::getValue( char color, int ix ){
    int i;
    if(      color == BLACK ){ i = ix; }
    else if( color == WHITE ){ i = BOARDSIZE+ix; }
    return values[i];
}

//empty is the default.
//things go B/W->empty, or B/W->empty
int ZobristHash::updateHash( int hash, int position, char color ){
    return hash ^ getValue( color, position );
}

////////////////////////////////////////////////////////////////////////////
//             Kernels
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

__global__ void setupRandomGenerators( curandState* state, unsigned long seed ){
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

__global__ void leafSim( GoStateStruct* gss,
                         ZobristHash* zh,
                         curandState* globalState, 
                         int* winners,
                         int* iterations ){
    int tid = blockIdx.x + threadIdx.x;

    //TODO: still giving __shared__ non-empty constructor warning
    //because of mem variables like BitMask.  Fix eventually
    __shared__ GoStateStruct gss_local;
    __shared__ ZobristHash zhasher_local;

    __syncthreads();
    if( tid == 0 ){
        //TODO modify BitMask to fit this paradig,
        gss->copyInto( &gss_local ); 
        zh->copyInto( &zhasher_local );
        gss_local.zhasher = &zhasher_local;
    }
    __syncthreads();

    /*
    winners[tid] = tid+42;
    if( tid % 2 == 0 ){
        int num = zhasher_local.updateHash( gss_local.zhash, tid, BLACK );
        iterations[tid/2] = num; 
    }

    __syncthreads();*/

    //do if tid == 0 here, race conditions?
    BitMask to_exclude;
    int count = 0;
    while( count < MAX_MOVES && !gss_local.isTerminal() ){
        int action = gss_local.randomAction( globalState, tid, &to_exclude, true );
        //bool is_legal = gss_local.applyAction( action, true );
        count++;
    }

    iterations[tid] = count;
    gss_local.getRewards( &(winners[tid*2]) );
    

}

int launchSimulationKernel( GoStateStruct* gss, int* rewards ){
    int white_win = 0;
    int black_win = 0;
    if( USE_GPU ){
        //setup rand generators on kernel
        curandState* devStates;
        cudaMalloc( &devStates, NUM_SIMULATIONS*sizeof(curandState) );
        setupRandomGenerators<<<NUM_SIMULATIONS,1>>>( devStates, time(NULL) );

        GoStateStruct* dev_gss;
        cudaMalloc( (void**)&dev_gss, sizeof(GoStateStruct));
        cudaMemcpy( dev_gss, gss, sizeof(GoStateStruct), cudaMemcpyHostToDevice );

        printf( "hash test: %d\n", gss->zhasher->updateHash( gss->zhash, 1, BLACK ) );
        ZobristHash* dev_zh;
        cudaMalloc( (void**) &dev_zh, sizeof(ZobristHash) );
        cudaMemcpy( dev_zh, gss->zhasher, sizeof(ZobristHash), cudaMemcpyHostToDevice );
        
        int ta = clock();
        //printf("time to do memcpys to device: %f s\n", ((float) t)/CLOCKS_PER_SEC);

        int winner_elems = NUM_SIMULATIONS*2;
        int winners[winner_elems];
        int* dev_winners;
        int sizeof_winners = winner_elems*sizeof(int); 
        cudaMalloc( (void**)&dev_winners, sizeof_winners );

        int iteration_elems = NUM_SIMULATIONS;
        int iterations[iteration_elems];
        int* dev_iterations;
        int sizeof_iterations = iteration_elems*sizeof(int);
        cudaMalloc( (void**)&dev_iterations, sizeof_iterations );

        //printf("calling kernel\n");
        
        //this is the default, setting it to PreferL1 slowed down 3x
        cudaFuncSetCacheConfig(leafSim, cudaFuncCachePreferShared );

        leafSim<<<NUM_SIMULATIONS, NUM_THREADS>>>
                                  (dev_gss, 
                                   dev_zh,                                    
                                   devStates, 
                                   dev_winners,
                                   dev_iterations );
        cudaThreadSynchronize();

        int t = clock();
        //printf("kernel return, %f secs\n", ((float)t)/CLOCKS_PER_SEC) ;

        cudaMemcpy( winners, dev_winners, sizeof_winners, 
                    cudaMemcpyDeviceToHost );
        cudaMemcpy( iterations, dev_iterations, sizeof_iterations, 
                    cudaMemcpyDeviceToHost );
        
        t = clock();
        //printf("mem transfer, %f secs\n", ((float)t)/CLOCKS_PER_SEC) ;

        //For histogram of game lengths
        int iteration_counts[MAX_MOVES];
        for( int i=0; i<MAX_MOVES; i++ ){
            iteration_counts[i] = 0;
        }

        //count the number of wins for each side
        for( int i=0; i<NUM_SIMULATIONS; i++ ){
            if( winners[i*2] == 1 ){
               white_win++;
            }
            else if( winners[i*2+1] == 1 ){
                black_win++;
            }
            printf( "White: %d v. Black: %d\n", winners[i*2], winners[i*2+1] );
            printf( "Took: %d steps\n\n", iterations[i] );
            printf("%d - %d\n", winners[i*2], winners[i*2+1] );
            printf("%d\n\n", iterations[i] );
            //iteration_counts[iterations[i]]++;

        }

        /*
        int cdf_sum = 0;
        for( int i=0; i<MAX_MOVES; i++ ){
            if( iteration_counts[i] > 0 ){
                printf( "%d steps : %d\n", i, iteration_counts[i] );
                cdf_sum += iteration_counts[i];
                printf( "cdf: %f\n\n", cdf_sum / ((float)NUM_SIMULATIONS) );
            }
        }
        */
        cudaFree( dev_gss );
        cudaFree( dev_zh );
        cudaFree( dev_winners );
        cudaFree( dev_iterations );
        cudaFree( devStates );

        int tb = clock();

        printf("time taken is: %f\n", ((float) tb-ta)/CLOCKS_PER_SEC);
    
    }
    else{
        int ta = clock();
        int t1,t1b, tcopy_avg;
        int t2,t2b, trand_avg;
        int t3,t3b, tappl_avg;
        int trewd_avg;
        tcopy_avg = 0;
        trand_avg = 0;
        tappl_avg = 0;
        trewd_avg = 0;
        int total_move_count = 0;
        for( int i=0; i<NUM_SIMULATIONS; i++ ){
            t1 = clock();
            GoStateStruct* linear = (GoStateStruct*) gss->copy();
             
            t1b = clock();
            tcopy_avg += t1b-t1;
            BitMask to_exclude;
            int move_count = 0;
            while( move_count < MAX_MOVES && !linear->isTerminal() ){
                t2 = clock();
                int action = linear->randomAction( &to_exclude, true );

                t2b = clock();
                trand_avg += t2b-t2;

                //printf( "%s\n\n", linear->toString().c_str() );
                //cout << "hit any key..." << endl;
                //cin.ignore();
                
                t3b = clock();
                tappl_avg += t3b-t3;
                move_count++;
                total_move_count++;
            }

            int rewards[2];
            int t4 = clock();
            linear->getRewards( rewards );
            int t4b = clock();
            trewd_avg += t4b - t4;
            if( rewards[0] == 1 ){
                white_win++;
            }
            else if( rewards[1] == 1 ){
                black_win++;
            }
            //printf("rewards[0]: %d, rewards[1]: %d\n", rewards[0], rewards[1] );
        }
        int tb = clock();
        printf("time taken is: %f\n", ((float) tb-ta)/CLOCKS_PER_SEC);
        printf("copy time: %f\n", ((float) tcopy_avg)/CLOCKS_PER_SEC);
        printf("rand time: %f\n", ((float) trand_avg)/CLOCKS_PER_SEC);
        //printf("avg appl time: %f\n", ((float) tappl_avg)/CLOCKS_PER_SEC);  // /div by total_move_count to get avg
        printf("avg rewd time: %f\n", ((float) trewd_avg)/CLOCKS_PER_SEC);
        
    }

    //printf( "dev white win count: %d\n", white_win_dev );
    //printf( "host white win count: %d\n", white_win_host );
    
    //assert( white_win+black_win == NUM_SIMULATIONS );
    rewards[0] = white_win;
    rewards[1] = black_win;
    
    return 0;
}

