#include "gostate_struct.h"
#include "kernel.h"

uint32_t m_z = rand();
uint32_t m_w = rand();
uint32_t nextRandom;
static uint32_t getRandom(uint32_t * m_z, uint32_t * m_w)
{
    *m_z = 36969 * (*m_z & 65535) + (*m_z >> 16);
    *m_w = 18000 * (*m_w & 65535) + (*m_w >> 16);
    return (*m_z << 16) + *m_w;
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
    //cout << "gss ctor called:" << endl;
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
            empty_intersections[num_open] = i;
            frozen_empty_intersections[num_open++] = i;
        }
    }

    past_action = -1;
    for( int i=0; i<NUM_PAST_STATES; i++ ){
        //past_actions[i] = -1;
        past_zhashes[i] = 0;
    }

}

//GoStateStruct::~GoStateStruct(){
//cout << "called from where?" << endl;
    //don't delete zhasher, it is shared by everyone

    //setBoard(11,'w');
    //delete[] frozen_board;
    //delete[] past_zhashes;
    //delete[] neighbor_array;
    //delete[] internal_neighbor_array;
    //delete[] filtered_array; 
    //delete[] internal_filtered_array;
    //delete[] color_array;
    //delete[] empty_intersections;
    //delete[] frozen_empty_intersections;
    //delete marked;
    //delete    
    //}

void GoStateStruct::freezeBoard(){
    memcpy( frozen_board, board, BOARDSIZE*sizeof(char) );
    memcpy( frozen_empty_intersections, empty_intersections, MAX_EMPTY*sizeof(uint16_t) );
    frozen_num_open = num_open;
    frozen_zhash = zhash;
}

void GoStateStruct::thawBoard(){
    memcpy( board, frozen_board, BOARDSIZE*sizeof(char) );
    memcpy( empty_intersections, frozen_empty_intersections, MAX_EMPTY*sizeof(uint16_t) );
    num_open = frozen_num_open;
    zhash = frozen_zhash;
}

void GoStateStruct::copyInto( GoStateStruct* target ){
    target->zhash = zhash;
    target->player = player;
    target->action = action;
    target->past_action = past_action;
    target->num_open = num_open;
    target->frozen_num_open = frozen_num_open;
    target->zhasher = zhasher;

    memcpy( target->board, board, BOARDSIZE*sizeof(char) );
    memcpy( target->frozen_board, frozen_board, BOARDSIZE*sizeof(char) );
    memcpy( target->empty_intersections, empty_intersections, MAX_EMPTY*sizeof(uint16_t) );
    memcpy( target->frozen_empty_intersections, frozen_empty_intersections, MAX_EMPTY*sizeof(uint16_t) );

    memcpy( target->past_zhashes, past_zhashes, NUM_PAST_STATES*sizeof(int) );
}

GoStateStruct* GoStateStruct::copy(){
    GoStateStruct* s = (GoStateStruct*) malloc(sizeof(GoStateStruct));
    this->copyInto(s);
    return s;
};

char GoStateStruct::flipColor( char c ){
    //assert( c == WHITE || c == BLACK );
    return (c == WHITE) ? BLACK : WHITE;
}

void GoStateStruct::togglePlayer() {
    (player == WHITE) ? player = BLACK : player = WHITE;
}

string GoStateStruct::boardToString( char* board ){
    stringstream out;
    out << "   ";
    for( int i=0; i<BIGDIM; i++){
        out << i+1;
        if( i+1 >= 10 ){
           out << " ";
        }
        else{
            out << "  ";
        }
    }
    out << "\n 0 ";
    //out += ss.str();
    for( int i=0; i<BOARDSIZE; i++ ){
        //cout << "i: " << i << endl;
        if(      board[i] == BLACK    ){ out << "x"; }
        else if( board[i] == WHITE    ){ out << "o"; }
        else if( board[i] == OFFBOARD ){ out << "_"; }
        else if( board[i] == EMPTY    ){ out << "."; }
        else{   
            printf( "offending pos is %d, %c\n", i, board[i] );
            assert(false); }

        out << "  ";
        if( i % BIGDIM == BIGDIM-1 ){
            out << "\n";
            int row = i/BIGDIM+1;
            if( row >= 10 ){
                out << "";
            }
            else{
                out << " ";
            }
            out << row << " ";
        }
    }
    return out.str();
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

int GoStateStruct::ix2action( int ix, char player ){
    int parity = player == WHITE ? 1 : -1;
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
        empty_intersections[num_open++] = ix;
            //num_open++; 
    } 
    else{ 
        //assert( board[ix] == EMPTY ); 
        zhash = zhasher->updateHash( zhash, ix, color );
        //look through and find ix in empty_intersections. Swap with last
        //empty, then decrement num_open, effectively removing this ix
        //from the list of empties
        //cout << "setboard " << ix << endl;
        //cout << "last in empyt: " << empty_intersections[num_open-1] << endl;
        if( empty_intersections[num_open-1] != ix ){
            //cout << "looking to remove ix: " << ix << endl;
            for( int i=num_open-2; i >= 0; i-- ){
                if( empty_intersections[i] == ix ){
                    //cout << "found swap out in: " << num_open-i << " moves" << endl;
                    uint16_t temp = empty_intersections[num_open-1];
                    empty_intersections[num_open-1] = empty_intersections[i];
                    empty_intersections[i] = temp;
                }
            }
        }
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

//if space is no object, why not just have an array with marked values
//this way don't need to iterate all the way though the 0's
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

bool GoStateStruct::isBorder( int ix ){
    if( BIGDIM < ix && ix < 2*BIGDIM-1 ){ return true; }
    else if( ix % BIGDIM == 1 ){return true; }
    else if( ix % BIGDIM == BIGDIM-2 ){ return true; }
    else if( BIGDIM*(BIGDIM-2) < ix && ix < BIGDIM*(BIGDIM-1)-1 ){return true;}
    else{ return false; }
}

void GoStateStruct::neighborsOf2( int* to_fill, int* to_fill_len,
                                  int ix, int adjacency,
                                  char filter_color ){
    if( board[ix] == OFFBOARD ){ return; }
    int fillix = 0;
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
    if( board[ix-1] == filter_color ){  //W
        to_fill[fillix++] = ix-1;
    }
    if( adjacency == 8 ){
        //NW
        if( board[ix-BIGDIM-1] == filter_color  ){ 
            to_fill[fillix++] = ix-BIGDIM-1;
        }
        //NE
        if( board[ix-BIGDIM+1] == filter_color ){
            to_fill[fillix++] = ix-BIGDIM+1;
        }
        //SW
        if( board[ix+BIGDIM-1] == filter_color ){
            to_fill[fillix++] = ix+BIGDIM-1;
        }
        //SE
        if( board[ix+BIGDIM+1] == filter_color ){
            to_fill[fillix++] = ix+BIGDIM+1;
        }
    }
    *to_fill_len = fillix;
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
/* deprecated because slow 
void GoStateStruct::neighborsOf( int* to_fill, int ix, int adjacency ){
    //assert( adjacency==4 || adjacency==8 );
    for( int dir=0; dir<adjacency; dir++ ){
        to_fill[dir] = neighbor( ix, (DIRECTION) dir);
    }                        
}
*/

void GoStateStruct::neighborsOf( int* to_fill, int ix, int adjacency ){
    if( board[ix] == OFFBOARD ){
        memset( to_fill, OFFBOARD, adjacency );
    }
    to_fill[0] = ix - BIGDIM;
    to_fill[1] = ix + BIGDIM;
    to_fill[2] = ix + 1;
    to_fill[3] = ix - 1;
    if( adjacency == 8 ){
        to_fill[4] = ix - BIGDIM - 1;
        to_fill[5] = ix - BIGDIM + 1;
        to_fill[6] = ix + BIGDIM - 1;
        to_fill[7] = ix + BIGDIM + 1;
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

//returns true if the stop color wasn't found, when flooding 'flood_color' 
//starting from 'epicenter_ix'
//fills this->marked BitMask with the stones marked
bool GoStateStruct::floodFill(  
                int epicenter_ix,
                int adjacency,
                char flood_color,
                char stop_color ){

    queue.clear(); 
    marked.clear();

    //cout << "epicenter: " << epicenter_ix << endl;
    queue.push( epicenter_ix );
    marked.set( epicenter_ix, true );
    
    bool stop_color_not_encountered = true;
    while( !queue.isEmpty() ){
        int ix = queue.pop();
 
        if( false ){ //connected_to_lib.get(ix) ){
            //cout << "ix: " << ix << " connected to lib" << endl;
            stop_color_not_encountered = false;
            continue;
        }
        else {
            //N
            int nix;
            if( board[ix-BIGDIM] == stop_color ||
                board[ix+BIGDIM] == stop_color || 
                board[ix+1] == stop_color ||
                board[ix-1] == stop_color ){ 
                stop_color_not_encountered = false;
                break;
            }

            nix = ix-BIGDIM;
            if( board[nix] == flood_color  ){ 
                if( !marked.get( nix) ){
                    queue.push(nix);
                    marked.set( nix,true);
                }
            }
            //S
            nix = ix+BIGDIM;
            if( board[nix] == flood_color ){
                if( !marked.get( nix) ){
                    queue.push(nix);
                    marked.set( nix,true);
                }
            }
            //E
            nix = ix+1;
            if( board[nix] == flood_color ){
                if( !marked.get( nix) ){
                    queue.push(nix);
                    marked.set( nix,true);
                }
            }
            //W
            nix = ix-1;
            if( board[nix] == flood_color ){  //W
                if( !marked.get( nix) ){
                    queue.push(nix);
                    marked.set( nix,true);
                }
            }
        }
    }

    return stop_color_not_encountered;
}

//TODO much faster if we have access to a hash_map O(n) -> O(1)
bool GoStateStruct::isDuplicatedByPastState(){
    for( int i=NUM_PAST_STATES-1; i>=0; i-- ){
        if( zhash == past_zhashes[i] ){ return true; break; }
    }
    return false;
}

void GoStateStruct::advancePastStates( int past_zhash,  
                                       int past_action ){
    //TODO: how to faster?  can't use memcpy on overlapping regions, 
    //cant use memmove on cuda
    //memmove( past_zhashes, &past_zhashes[1], 
    //this->zhasher->sizeOf()*(NUM_PAST_STATES-1) ); 
    for( int i=0; i<NUM_PAST_STATES-1; i++ ){
        past_zhashes[i] = past_zhashes[i+1];
    }
    past_zhashes[NUM_PAST_STATES-1] = past_zhash;
    this->past_action = past_action;
    
}

bool GoStateStruct::applyAction( int action,
                                 bool side_effects ){
    bool legal = true;
    bool board_frozen = false;

    //The action parameter is really the index of the action to be taken
    //need to convert to signed action i.e BLACK or WHITE ie. *-1 or *1
    int ix = action;
    char color = player;
    char opp_color = flipColor(color); 
    action = ix2action( ix, color);
    if( ! isPass(action) ){
        int adjacency = 8;
        neighborsOf( neighbor_array, ix, adjacency );

        //will go through all neighbors and set these booleans appropriately
        bool orthog_all_kin = true;
        bool no_orthog_opps = true;
        bool lt2_diag_opps, no_diag_opps;
        bool ix_is_border = isBorder(ix);
        bool surrounded_by_enemy = true;
        bool has_lib = false;

        int ncolor;
        //TODO unroll these loops
        //NSEW (orthog) neighbs
        for( int i=0; i<4; i++ ){
            ncolor = ix2color( neighbor_array[i] );
            has_lib |= ncolor == EMPTY;
            orthog_all_kin &= ncolor == color || ncolor == OFFBOARD;
            no_orthog_opps &= ncolor != opp_color;
            surrounded_by_enemy &= ncolor == opp_color || ncolor == OFFBOARD;
        }

        //diagonal neighbs
        int n_diagonal_opps = 0;
        for( int i=4; i<adjacency; i++ ){
            ncolor = ix2color( neighbor_array[i] );
            if( ncolor == opp_color ){
                n_diagonal_opps++;
            }
        }
        lt2_diag_opps = n_diagonal_opps < 2;
        no_diag_opps = n_diagonal_opps == 0;

        bool is_eye = orthog_all_kin && ( (ix_is_border && no_diag_opps)  || 
                                         (!ix_is_border && lt2_diag_opps)  );
        bool no_orthog_opps_and_liberty = no_orthog_opps && has_lib;
        if( is_eye ){
            //cout << "is eye" << endl;
            legal = false;
        }
        else if( no_orthog_opps_and_liberty ){
            legal = true;
            //cout << "easy place, zhash: " << zhash << endl;
            if( legal ){
                //rather than do full fledged freeze/thaw,  
                //maintain the old color in case this placement causes ko
                char old_color = ix2color(ix);
                setBoard( ix, color );
                //cout << "intermediate zhash: " << zhash << endl;
                legal = !isDuplicatedByPastState();
                //cout << "duplicated: " << legal << endl;
                if( !legal || !side_effects ){
                    setBoard( ix, old_color );
                }
            }
            //cout << "easy place end, legal: " << legal << " zhash: " << zhash << endl;
        }
        else{  //must check for captures
            freezeBoard();
            board_frozen = true;

            setBoard( ix, color );
            int len;
            adjacency = 4;
            neighborsOf2( filtered_array, &len, 
                          ix, adjacency, opp_color );
            //deprecated, seems overhead didn't make the short-circuit worth it
            //TODO: test more thoroughly on 19x19 though
            //as we explored the opp_color neighbors of the recently placed
            //piece, we will mark which stones have liberties.
            //That way, if we are investigating the second neighbor and it
            //runs into stones of the first that have a lib, we can short
            //circuit
            //connected_to_lib.clear();
            bool capture_made = false;
            for( int onix=0; onix < len; onix++ ){
                marked.clear();
                bool fill_completed = floodFill( filtered_array[onix],
                                                  adjacency,  
                                                  opp_color, 
                                                  EMPTY );
                if( fill_completed ){
                    //connected_to_lib.clear();
                    capture( &marked );
                    capture_made = true;
                }
                else {
                    //add the marked stones into the ones with liberties
                    //marked.copyInto( &connected_to_lib );
                    //connected_to_lib.Or( marked );
                }
            }
            
            //check suicide
            if( !capture_made ){
                if( surrounded_by_enemy ){ 
                    legal = false;
                    //cout << "surround by enemy" << endl;
                } 
                else {
                    if( !has_lib ) {
                        //floodFill each same color neighb
                        //if one reaches the end without finding liberty
                        //return legal=false
                        neighborsOf2( filtered_array, &len, 
                                      ix, adjacency, color );
                        for( int i=0; i < len; i++ ){
                            int nix = filtered_array[i];
                            bool fill_completed = floodFill( nix, 
                                                             adjacency, 
                                                             color, 
                                                             EMPTY );
                            if( fill_completed ) { 
                                legal = false;
                                //cout << "no lib left" << endl;
                                break; 
                            }
                        }
                    }
                }
            }
            else{
                //if a capture is made, the newly placed stone and all friends 
                //it is connected to has one liberty by def
                //else we must check that 
            }
            //after done looking for and applying captures, check superko
            if( legal ){
                //cout << "cur zhash: " << zhash << endl;
                legal &= !isDuplicatedByPastState();
                //cout << "dup by past state" << legal << endl;
            }
        }
        
    }
    else{
        //PASS played
    }
    
    //do we apply, or rollback?
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

int GoStateStruct::randomAction( BitMask* to_exclude,
                                 bool side_effects ){
    int end = num_open-1;
    uint32_t r;

    bool legal_but_excluded_move_available = false;
    while( end >= 0 ){
        //TODO: rand can generate a 0 move here, PASS when we don't want it
        //NOTE: can't use rand() because it causes kernel blocking
        //      not good when trying to parallelize
        //r = rand() % (end+1);
        r = getRandom(&m_z,&m_w) % (end+1);
        
        //swap empty[end] and empty[r]
        int temp = empty_intersections[end];
        empty_intersections[end] = empty_intersections[r];
        empty_intersections[r] = temp;

        //swap empty[end] to empty[num_open-1]
        temp = empty_intersections[num_open-1];
        empty_intersections[num_open-1] = empty_intersections[end];
        empty_intersections[end] = temp;
        
        //test whether the move legal and not excluded
        //return intersection if so
        int candidate = empty_intersections[num_open-1];
        bool ix_is_excluded = to_exclude->get(candidate);
        if( legal_but_excluded_move_available ){
            if( ! ix_is_excluded ){
                bool is_legal = applyAction( candidate, side_effects );
                if( is_legal ){
                    return candidate;
                }
            }
        }
        else{
            bool is_legal = applyAction( candidate, 
                                         side_effects && !ix_is_excluded);
            if( is_legal ){
                if( ix_is_excluded ){
                    legal_but_excluded_move_available = true;
                }
                else{
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
}


bool GoStateStruct::isTerminal(){
    bool r = action == PASS && past_action == PASS;
    return r;
}

void GoStateStruct::getRewards( int* to_fill ){
    int white_score = 0;
    int black_score = 0;
    for( int ix=0; ix < BOARDSIZE; ix++ ){
        if( board[ix] == OFFBOARD ){
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
        //

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
        if(      has_white && ! has_black ) { white_score++; }
        else if( has_black && ! has_white ) { black_score++; }

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

//will need to rework this if ever use playouts where it is not the case that 
//only single empty positions are left
/*
 void GoStateStruct::getRewardsComplete( int* to_fill ){
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
        //
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
*/

/////////////////////////////////////////////////////////////////////////////
//                     BitMask.cu
/////////////////////////////////////////////////////////////////////////////
// OMFG just just use a bool[] ?
// Have deprecated the old,smaller but slower BitMask into bitmask_small.h
// This is it's impl, just not renamed yet
/*
BitMask::BitMask(){
    clear();
}

void BitMask::copyInto( BitMask* target ){
    memcpy( target->masks, masks, BITMASK_SIZE*sizeof(int) );
    target->count = count;
}

void BitMask::clear(){
    memset( masks, 0, sizeof masks );
    //for( int i=0; i<BITMASK_SIZE; i++ ){
    //masks[i] = 0;
    //}
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
*/

////////////////////

BitMask::BitMask(){
    clear();
}

void BitMask::copyInto( BitMask* target ){
    memcpy( target->mask, mask, BOARDSIZE*sizeof(bool) );
    target->count = count;
}

void BitMask::clear(){
    memset( mask, false, BOARDSIZE );
    count = 0;
}

void BitMask::set(int bit, bool value ) {
    mask[bit] = value;
    if( value ){ count++;}
    else{        count--;}

}

bool BitMask::get( int bit ){
    return mask[bit];
}


//void BitMask::Or( BitMask bm ){
//for( int i=0; i < BOARDSIZE; i++ ){ 
//this->mask[i] |= bm.mask[i];
//}
//}
    

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
    begin = ringInc(begin);
    nElems--;
    return r;
}

bool Queue::isEmpty(){
    return nElems == 0; 
}


////////////////////////////////////////////////////////////////////////////
//             Zobrist
////////////////////////////////////////////////////////////////////////////

//TODO 64 bit impl
void ZobristHash::ctor(){
    int_bits = 32;
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

int ZobristHash::sizeOf(){ return int_bits; }

int launchSimulationKernel( GoStateStruct* gss, int* rewards ){
    int white_win = 0;
    int black_win = 0;
    //int ta = clock();
    //int t1,t1b, tcopy_avg;
    //int t2,t2b, trand_avg;
    //int t3,t3b, tappl_avg;
    //int trewd_avg;
    //tcopy_avg = 0;
    //trand_avg = 0;
    //tappl_avg = 0;
    //trewd_avg = 0;
    //int total_move_count = 0;
    for( int i=0; i<NUM_SIMULATIONS; i++ ){
        //t1 = clock();
        GoStateStruct* linear = (GoStateStruct*) gss->copy();

        //NOTE: Turn timing off when OpenMP used, clock() synchronizes
        //      in the kernel and slows everything down
        //t1b = clock();
        //tcopy_avg += t1b-t1;
        BitMask to_exclude;
        int move_count = 0;
        while( move_count < MAX_MOVES && !linear->isTerminal() ){
            //t2 = clock();
            int action = linear->randomAction( &to_exclude, true );

            //t2b = clock();
            //trand_avg += t2b-t2;

            //printf( "%s\n\n", linear->toString().c_str() );
            //cout << "hit any key..." << endl;
            //cin.ignore();

            //t3b = clock();
            //tappl_avg += t3b-t3;
            move_count++;
        }
        //cout << "after sim: " << linear->toString() << endl;

        int rewards[2];
        //int t4 = clock();
        linear->getRewards( rewards );
        delete linear;
        //int t4b = clock();
        //trewd_avg += t4b - t4;
        if( rewards[0] == 1 ){
            white_win++;
        }
        else if( rewards[1] == 1 ){
            black_win++;
        }
        //printf("rewards[0]: %d, rewards[1]: %d\n", rewards[0], rewards[1] );
    }
    //int tb = clock();
    //printf("time taken is: %f\n", ((float) tb-ta)/CLOCKS_PER_SEC);
    //printf("copy time: %f\n", ((float) tcopy_avg)/CLOCKS_PER_SEC);
    //printf("rand time: %f\n", ((float) trand_avg)/CLOCKS_PER_SEC);
    //printf("avg rewd time: %f\n", ((float) trewd_avg)/CLOCKS_PER_SEC);

    //printf( "dev white win count: %d\n", white_win_dev );
    //printf( "host white win count: %d\n", white_win_host );

    //assert( white_win+black_win == NUM_SIMULATIONS );
    rewards[0] = white_win;
    rewards[1] = black_win;

    return 0;
}

//DEPRECATED
//pre the empty_intersection days
//a nice lazy approach to Fischer-Yates shuffle and empty ix iteration
//could be useful if we go back to using GPU
/*
__host__
int GoStateStruct::randomAction2( BitMask* to_exclude, 
                                 bool side_effects ){
    //stores the empty positions we come across in board
    //filled from right to left
    //int empty_intersections[num_open];
    int end = num_open-1;
    int begin = num_open;
    int r;

    bool legal_but_excluded_move_available = false;

    //board_ix tracks our progress through board looking for empty intersections
    int board_ix, num_needed, num_found;
    board_ix = 0;
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
            bool is_legal = applyAction( candidate, 
                                         side_effects && !ix_is_excluded);
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
}
*/

