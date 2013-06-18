#include "gostate.h"

//turns out rand() does some blocking in the OS kernel.  When using OpenMP
//to parallelize, all the threads would step on each other's toes.
//Got this simple pseudo-rand funtion from stackoverflow that doesn't do 
//this
uint32_t m_z = rand();
uint32_t m_w = rand();
uint32_t nextRandom;
static uint32_t getRandom(uint32_t * m_z, uint32_t * m_w)
{
    *m_z = 36969 * (*m_z & 65535) + (*m_z >> 16);
    *m_w = 18000 * (*m_w & 65535) + (*m_w >> 16);
    return (*m_z << 16) + *m_w;
}

const int GoState::getNumPlayers(){ return 2; }

int GoState::getNumActions(){ return BOARDSIZE ; }

int GoState::getPlayerIx(){ return player == WHITE ? 0 : 1; }

int GoState::movesMade(){ return MAX_EMPTY - num_open; }

void GoState::deleteState(){ delete this; }

bool GoState::fullyExpanded( int action ){
    return action == EXCLUDED_ACTION;
}

bool GoState::isChanceAction(){
    return false;
}

GoState::GoState(ZobristHash* zh){
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

void GoState::freezeBoard(){
    memcpy( frozen_board, 
            board, 
            BOARDSIZE*sizeof(char) );

    memcpy( frozen_empty_intersections, 
            empty_intersections, 
            MAX_EMPTY*sizeof(uint16_t) );

    frozen_num_open = num_open;
    frozen_zhash = zhash;
}

void GoState::thawBoard(){
    memcpy( board, 
            frozen_board, 
            BOARDSIZE*sizeof(char) );

    memcpy( empty_intersections, 
            frozen_empty_intersections, 
            MAX_EMPTY*sizeof(uint16_t) );

    num_open = frozen_num_open;
    zhash = frozen_zhash;
}

void GoState::copyInto( MCTS_State* target ){
    this->copyInto( (GoState*) target );
}

void GoState::copyInto( GoState* target ){
    target->zhash = zhash;
    target->player = player;
    target->action = action;
    target->past_action = past_action;
    target->num_open = num_open;
    target->frozen_num_open = frozen_num_open;
    target->zhasher = zhasher;

    //black_known_illegal.copyInto( &(target->black_known_illegal) );
    //white_known_illegal.copyInto( &(target->white_known_illegal) );

    memcpy( target->board, board, BOARDSIZE*sizeof(char) );
    memcpy( target->frozen_board, frozen_board, BOARDSIZE*sizeof(char) );
    memcpy( target->empty_intersections, empty_intersections, MAX_EMPTY*sizeof(uint16_t) );
    memcpy( target->frozen_empty_intersections, frozen_empty_intersections, MAX_EMPTY*sizeof(uint16_t) );

    //target->zhash = zhash;
    memcpy( target->past_zhashes, past_zhashes, NUM_PAST_STATES*sizeof(int) );
}

MCTS_State* GoState::copy(){
    GoState* s = new GoState( this->zhasher );
    this->copyInto(s);
    return (MCTS_State*) s;
};

char GoState::flipColor( char c ){
    //assert( c == WHITE || c == BLACK );
    return (c == WHITE) ? BLACK : WHITE;
}

void GoState::togglePlayer() {
    (player == WHITE) ? player = BLACK : player = WHITE;
}

void GoState::board2MATLAB( double* matlab_board ){
    for( int i=0; i<BOARDSIZE; i++ ){
        int nobufferix = GoState::bufferix2nobufferix( i );
        if( board[i] == OFFBOARD ){
            assert( nobufferix == -1 );
        }
        else if( board[i] == WHITE ){
            matlab_board[nobufferix] = -1;
        }
        else if( board[i] == EMPTY ){
            matlab_board[nobufferix] = 0;
        }
        else if( board[i] == BLACK ){
            matlab_board[nobufferix] = 1;
        }
    }
}

void GoState::MATLAB2board( double* matlab_board ){
    for( int ix=0; ix<MAX_EMPTY; ix++ ){
        int bufferix = GoState::nobufferix2bufferix(ix);
        if( matlab_board[ix] == -1 ){
            setBoard( bufferix, WHITE );
            //board[bufferix] = WHITE;
        }
        else if( matlab_board[ix] == 0 ){
            setBoard( bufferix, EMPTY );
            //board[bufferix] = EMPTY;
        }
        else if( matlab_board[ix] == 1 ){
            setBoard( bufferix, BLACK );
            //board[bufferix] = BLACK;
        }
        else{
            assert(false);
        }
    }
}

string GoState::boardToString( char* board ){
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

string GoState::prettyBoard( string* board, int gap ){
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
        out << board[i]; 
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

string GoState::featuresToString( int* features, int nfeatures ){
    string out;
    stringstream ss;

    string descriptors[31] = {"off-limit",
                              "empty",
                              "isolated black",
                              "isolated white",
                              "all black",
                              "liberty for size 1 black group",
                              "liberty for size 2 black group",
                              "liberty for size 3 black group",
                              "liberty for size 4 black group",
                              "liberty for size 5-6 black group",
                              "liberty for size 7-9 black group",
                              "liberty for size 10 black group",
                              "all white",
                              "liberty for size 1 white group",
                              "liberty for size 2 white group",
                              "liberty for size 3 white group",
                              "liberty for size 4 white group",
                              "liberty for size 5-6 white group",
                              "liberty for size 7-9 white group",
                              "liberty for size 10 white group",
                              "abs(friend_dist-foe_dist) == 0", 
                              "abs(friend_dist-foe_dist) == -1", 
                              "abs(friend_dist-foe_dist) == -2,-3", 
                              "abs(friend_dist-foe_dist) == -4,-5", 
                              "abs(friend_dist-foe_dist) == -6,-7,-8", 
                              "abs(friend_dist-foe_dist) == -9...", 
                              "abs(friend_dist-foe_dist) == 1", 
                              "abs(friend_dist-foe_dist) == 2,3", 
                              "abs(friend_dist-foe_dist) == 4,5", 
                              "abs(friend_dist-foe_dist) == 6,7,8", 
                              "abs(friend_dist-foe_dist) == 9..."
                             };
    for( int f=0; f<nfeatures; ++f ){
        ss << "Feature " << f << " ( " << descriptors[f] << " )" << endl;
        for( int ix=0; ix < MAX_EMPTY; ++ix ){
            if( ix % DIMENSION == 0 ){
                ss << endl;
            }
            int fix = MAX_EMPTY*f + ix;
            ss << features[fix] << " ";
        }
        ss << endl << "============================" << endl;
    }
    return ss.str();
}


string GoState::toString(){
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

//TODO this is static, make it so
int GoState::bufferix2nobufferix( int ix ){
    int row = ix / BIGDIM;
    int col = ix % BIGDIM;
    if( row == 0 || row == BIGDIM-1 || col == 0 || col == BIGDIM-1 ){ 
        return -1;
    }
    else{
        return ix - BIGDIM - 1 - (row-1) * 2;
    }
}

int GoState::nobufferix2bufferix( int ix ){
    int row = ix / DIMENSION;
    return ix + BIGDIM + 1 + row*2;
}

int GoState::ix2action( int ix, char player ){
    int parity = player == WHITE ? 1 : -1;
    return ix * parity;
}

int GoState::action2ix( int action ){
    return abs(action);
}

char GoState::action2color( int action ){
    //assert( action != 0 );
    return (action > 0) ? WHITE : BLACK;
}

int GoState::ix2color( int ix ){
    return isBorder(ix) ? OFFBOARD : board[ix];
}

int GoState::coord2ix( int i, int j ){
    return BIGDIM*i + j;
}

int GoState::ixColor2Action( int ix, char color ){
    //assert( color==WHITE || color==BLACK );
    int mod = (color == WHITE) ? 1 : -1;
    return ix*mod;
}

int GoState::coordColor2Action( int i, int j, char color ){
    int ix = coord2ix(i,j);
    return ixColor2Action(ix, color);
}

bool GoState::isPass( int action ){
    return action == PASS;
}

void GoState::setBoard( int ix, char color ){ 
    if( ix >= BOARDSIZE || board[ix] == OFFBOARD ){ return; } 
    //if tyring to set the same color as it is, do nothing
    if( color == board[ix] ){ return; }

    char incumbant_color = board[ix]; 
    if( color == EMPTY ){ 
        zhash = zhasher->updateHash( zhash, ix, incumbant_color );
        empty_intersections[num_open++] = ix;
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

void GoState::setBoard( int* ixs, int len, char color ){
    for( int i=0; i<len; i++ ){
        int ix = ixs[i];
        setBoard( ix, color );
    }
}    

void GoState::capture(){
    for( vector<int>::iterator it = marked_group.begin();
            it != marked_group.end();
            ++it ){
        int ix = *it;
        setBoard( ix, EMPTY );
    }

    ////was the way to capture using the bitmask, 
    ////before the marked_group vector was introduced
    //int nSet = 0;
    //int ix = 0;
    //while( nSet < bm->count ){
    //if( bm->get(ix) ){ 
    //setBoard( ix, EMPTY );
    //nSet++;
    //}
    //ix++;
    //}
}

inline bool GoState::isBorder( int ix ){
    if( ix <= BIGDIM-1 ){ return true; }
    else if( ix % BIGDIM == 0 ){return true; }
    else if( ix % BIGDIM == BIGDIM-1 ){ return true; }
    else if( BIGDIM*(BIGDIM-1) <= ix ){return true;}
    else{ return false; }
}


void GoState::neighborsOf2( int* to_fill, int* to_fill_len,
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

int GoState::neighbor( int ix, DIRECTION dir){
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
void GoState::neighborsOf( int* to_fill, int ix, int adjacency ){
    //assert( adjacency==4 || adjacency==8 );
    for( int dir=0; dir<adjacency; dir++ ){
        to_fill[dir] = neighbor( ix, (DIRECTION) dir);
    }                        
}
*/

void GoState::neighborsOf( int* to_fill, int ix, int adjacency ){
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

void GoState::filterByColor( 
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
bool GoState::floodFill(  
                int epicenter_ix,
                int adjacency,
                char flood_color,
                char stop_color ){

    queue.clear(); 
    marked.clear();
    marked_group.clear();

    //cout << "epicenter: " << epicenter_ix << endl;
    queue.push( epicenter_ix );
    marked.set( epicenter_ix, true );
    marked_group.push_back( epicenter_ix );
    
    bool stop_color_not_encountered = true;
    while( !queue.isEmpty() ){
        int ix = queue.pop();
 
        if( false ){ //connected_to_lib.get(ix) ){
            //cout << "ix: " << ix << " connected to lib" << endl;
            stop_color_not_encountered = false;
            continue;
        }
        else {
            if( board[ix-BIGDIM] == stop_color ||
                board[ix+BIGDIM] == stop_color || 
                board[ix+1] == stop_color ||
                board[ix-1] == stop_color ){ 
                stop_color_not_encountered = false;
                break;
            }

            int nixs[8];
            if( adjacency == 4 ){
                nixs[0] = ix-BIGDIM;
                nixs[1] = ix+BIGDIM;
                nixs[2] = ix-1;
                nixs[3] = ix+1;
            }
            else if( adjacency == 8 ){
                nixs[0] = ix-BIGDIM;
                nixs[1] = ix+BIGDIM;
                nixs[2] = ix-1;
                nixs[3] = ix+1;
                nixs[4] = ix-BIGDIM-1;
                nixs[5] = ix-BIGDIM+1;
                nixs[6] = ix+BIGDIM+1;
                nixs[7] = ix+BIGDIM-1;
            }
            else assert(false);

            for( int a=0; a<adjacency; ++a ){
                int nix = nixs[a];
                if( board[nix] == flood_color  ){ 
                    if( !marked.get( nix) ){
                        queue.push(nix);
                        marked.set( nix,true);
                        marked_group.push_back(nix);
                    }
                }
            }
            
            /*
            int nix;
            //N
            nix = ix-BIGDIM;
            if( board[nix] == flood_color  ){ 
                if( !marked.get( nix) ){
                    queue.push(nix);
                    marked.set( nix,true);
                    marked_group.push_back(nix);
                }
            }
            //S
            nix = ix+BIGDIM;
            if( board[nix] == flood_color ){
                if( !marked.get( nix) ){
                    queue.push(nix);
                    marked.set( nix,true);
                    marked_group.push_back(nix);
                }
            }
            //E
            nix = ix+1;
            if( board[nix] == flood_color ){
                if( !marked.get( nix) ){
                    queue.push(nix);
                    marked.set( nix,true);
                    marked_group.push_back(nix);
                }
            }
            //W
            nix = ix-1;
            if( board[nix] == flood_color ){  //W
                if( !marked.get( nix) ){
                    queue.push(nix);
                    marked.set( nix,true);
                    marked_group.push_back(nix);
                }
            }
            */
        }
    }

    return stop_color_not_encountered;
}

int GoState::floodFillSize(){
    return marked.count;
}

vector<int> GoState::getMarkedGroup(){
    return marked_group;
}

//TODO much faster if we have access to a hash_map O(n) -> O(1)
bool GoState::isDuplicatedByPastState(){
    for( int i=NUM_PAST_STATES-1; i>=0; i-- ){
        if( zhash == past_zhashes[i] ){ return true; break; }
    }
    return false;
}

void GoState::advancePastStates( int past_zhash,  
                                       int past_action ){
    //TODO: how to faster?  can't use memcpy on overlapping regions, 
    //cant use memmove on cuda
    memmove( past_zhashes, &past_zhashes[1], NUM_PAST_STATES-1 ); 
    //this->zhasher->sizeOf()*(NUM_PAST_STATES-1) ); 
    //for( int i=0; i<NUM_PAST_STATES-1; i++ ){
    //past_zhashes[i] = past_zhashes[i+1];
    //}
    past_zhashes[NUM_PAST_STATES-1] = past_zhash;
    this->past_action = past_action;
    
}

void GoState::setKnownIllegal( int ix ){
    if( player == BLACK ){
        black_known_illegal.set(ix,true);
    }
    else {
        white_known_illegal.set(ix,true);
    }
}

bool GoState::isKnownIllegal( int ix ){

    if( player == BLACK ){
        return black_known_illegal.get(ix);
    }
    else {
        return white_known_illegal.get(ix);
    }
}


bool GoState::applyAction( int action,
                                 bool side_effects ){
    if( !isBorder(action) && board[action] != EMPTY ){ 
        return false;
    }
    bool legal = true;
    bool board_frozen = false;

    //The action parameter is really the index of the action to be taken
    //need to convert to signed action i.e BLACK or WHITE ie. *-1 or *1
    int ix = action;
    char color = player;
    
    //if( isKnownIllegal( action ) ){
    ////cout << "known illegal, DENIED!" << endl;
    //return false;
    //}

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
            //setKnownIllegal( ix );
            //known_illegal.set( ix, true );
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
                if( !legal ){
                    //cout << "duplicated: " << endl;
                }
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
                    capture();
                    capture_made = true;
                    //black_known_illegal.clear();
                    //white_known_illegal.clear();
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
                                //setKnownIllegal( ix );
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
                bool dup = isDuplicatedByPastState();
                if( dup ){
                    //cout << "state duplicates past" << endl;
                }
                legal &= !dup; 
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

int GoState::randomAction( BitMask* to_exclude,
                                 bool side_effects ){
    int end = num_open-1;
    uint32_t r;

    bool legal_but_excluded_move_available = false;
    while( end >= 0 ){
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

bool GoState::isTerminal(){
    bool r = action == PASS && past_action == PASS;
    return r;
}

void GoState::getRewards( int* to_fill ){
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
        neighborsOf( neighbor_array,
                     ix,
                     adjacency );
        int* white_neighbs = filtered_array;
        int num_white_neighbs = 0;
        char filter_colors[1] = {WHITE};
        filterByColor( white_neighbs, &num_white_neighbs,
                       neighbor_array, adjacency, 
                       filter_colors, 1 );

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
 void GoState::getRewardsComplete( int* to_fill ){
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

//take the feature board, and the ix and compute
//the offset in the feature array
//PERF TODO: if we compute nbix once ahead of time, will save a bit
int featureIX( int feature, int ix ){
    int nbix = GoState::bufferix2nobufferix( ix ); 
    if( nbix == -1 ) assert(false);
    else {
        return MAX_EMPTY*feature + nbix;
    }
}

void GoState::setBinaryFeatures( int* features, int nfeatures ){

    //the group numbers for every intersection.  -1 for offboards and empties
    int group_id = 0;
    int group_assignments[BOARDSIZE];
    memset( group_assignments, -1, sizeof(int)*BOARDSIZE );

    //for now, just int size info
    vector<int> group_info;

    //don't need to FF intersections that have already been assigned a group
    BitMask in_group;

    //cout << "find the and build metadata about the groups" << endl;
    for( int ix=0; ix < BOARDSIZE; ++ix ){
        char ixcolor = ix2color(ix);

        if( (ixcolor == WHITE || ixcolor == BLACK) &&
             in_group.get(ix) == false ){

            char fill_color = ixcolor;
            //don't want any stopping, 'n' will never be seen
            char stop_color = 'n';
            bool fill_completed = floodFill( ix, 8, 
                                             fill_color, stop_color );
            assert( fill_completed );

            vector<int> marked_group = getMarkedGroup();
            vector<int>::iterator it;
            for( it = marked_group.begin();
                 it != marked_group.end();
                 ++it ){
                int claimed_ix = *it;
                in_group.set(claimed_ix,true);
                group_assignments[claimed_ix] = group_id;
            }
            group_info.push_back( floodFillSize() );
            //cout << "group " << group_id << " size: " << floodFillSize() << endl ;
            ++group_id;
        }
        else{
            //either an empty, offboard, or already assigned a group
            //assumes intersection can be part of only one group
            continue;
        }
    }
    
    //cout << "groups assignments" << endl;
    //for( int j=0; j<BOARDSIZE; j++ ){
    //if( j % BIGDIM == 0 ){
    //cout << endl;
    //}
    //cout << group_assignments[j] << ", ";
    //}
    //cout << endl;

    // features' vector (binary, 1x20)
    // 1 = off-limit goban
    // 2 = empty intersection 
    // 3 = isolated Black stone
    // 4 = isolated White stone
    // 5 = all Black stones
    // 6-12 = # liberties for a Black group 1, 2, 3, 4, 5-6, 7-9, 10 or more
    // 13 = all White stones 
    // 14-20 = # liberties for a White group 1, 2, 3, 4, 5-6, 7-9, 10 or more
        
    for(int ix=0; ix<BOARDSIZE; ++ix){
        
        char color = ix2color(ix);

        //the ix into the feature array
        int fix;        

        if( color == OFFBOARD ){
            continue;
        }
        if( color == EMPTY ){
            //1 is the empty/not feature board
            fix = featureIX(1,ix);
            features[ fix ] = 1;

            //examine group id's of 4 neighbors to see if this empty
            //position is a liberty of a group
            const int adj = 4;
            int nixs[adj] = {ix-BIGDIM,ix+BIGDIM,ix-1,ix+1};
            for( int a=0; a<adj; ++a ){
                int nix = nixs[a];
                char ncolor = ix2color(nix);
                if( ncolor == OFFBOARD || ncolor == EMPTY ){
                    continue;
                }
                int ngroup_id = group_assignments[nix];
                int ngroup_size = group_info[ngroup_id];

                //offset the feature index depending on the neighbor group
                //color
                int color_feature_offset;
                if( ncolor == BLACK ){
                    color_feature_offset = 0; 
                }
                else if( ncolor == WHITE ){
                    color_feature_offset = 8;
                }
                else assert(false);

                //if black:
                //feature boards 5-8 : sizes 1-4
                //               9   : sizes 5-6
                //              10   : sizes 7-9
                //              11   : sizes > 10
                //if white: each feature index+8
                int feature = color_feature_offset;
                if( 1 <= ngroup_size && ngroup_size <= 4 ){
                    feature +=  ngroup_size + 4;
                    features[ featureIX(feature,ix) ] = 1;
                }
                else if( ngroup_size == 5 || ngroup_size == 6 ){
                    feature += 9;
                    features[ featureIX(feature,ix) ] = 1;
                }
                else if( 7 <= ngroup_size && ngroup_size <= 9 ){
                    feature += 10;
                    features[ featureIX(feature,ix) ] = 1;
                }
                else{
                    feature += 11;
                    features[ featureIX(feature,ix) ] = 1;
                }
            }
        }
        else { //color isn't EMPTY or OFFBOARD
            int group_id = group_assignments[ix];
            int group_size = group_info[group_id];

            if( color == BLACK ){
                if( group_size == 1 ){
                    //2 : isolated
                    features[ featureIX(2,ix) ] = 1;
                }
                //4 : all black
                features[ featureIX(4,ix) ] = 1;
            }
            else if( color == WHITE ){
                if( group_size == 1 ){
                    //3 : isolated
                    features[ featureIX(3,ix) ] = 1;
                }
                //12 : all white
                features[ featureIX(12,ix) ] = 1;
            }
            else assert(false);

            // there're also 11 additional features based on difference of
            // Manhattan distance between friendly and opponent stones
            // 21 = 0 (new stone equal distance from friendly and opponent)
            // 22 = -1
            // 23 = -2 or -3
            // 24 = -4 or -5
            // 25= -6 to -8
            // 26 = < -8
            // 27 = 1
            // 28 = 2 or 3
            // 29 = 4 or 5
            // 30 6 to 8
            // 31 = >8
            // EXCEPT -1 FROM EVERY INDEX BECAUSE THIS ISN"T MATLAB

            //manhattan distance
            //do standard spiral algo first
            //will qtree's help? yes, but lengthy (fun) impl time
            pair<int,int> dists = getManhattanDistPair(ix);
            int friend_dist = dists.first;
            int foe_dist = dists.second;
            int diff = friend_dist - foe_dist;
            if( diff == 0 )       
                features[ featureIX(20,ix) ] = 1;
            else if( diff == -1 ) 
                features[ featureIX(21,ix) ] = 1;
            else if( diff == -2 || diff == -3 ) 
                features[ featureIX(22,ix) ] = 1;
            else if( diff == -4 || diff == -5 ) 
                features[ featureIX(23,ix) ] = 1;
            else if( -8 <= diff && diff <= -6 ) 
                features[ featureIX(24,ix) ] = 1;
            else if( diff <= -8 ) 
                features[ featureIX(25,ix) ] = 1;
            else if( diff == 1 ) 
                features[ featureIX(26,ix) ] = 1;
            else if( diff == 2 || diff == 3 ) 
                features[ featureIX(27,ix) ] = 1;
            else if( diff == 4 || diff == 5 ) 
                features[ featureIX(28,ix) ] = 1;
            else if( 6 <= diff && diff <= 8 ) 
                features[ featureIX(29,ix) ] = 1;
            else if( diff >= 8 ) 
                features[ featureIX(30,ix) ] = 1;

        }
    }
}


//  4 3 2 3 4
//  3 2 1 2 3
//  2 1 0 1 2
//  3 2 1 2 3
//  4 3 2 3 4
pair<int,int> GoState::getManhattanDistPair( int ix ){
    char frend_color = ix2color(ix);
    assert( frend_color != EMPTY && frend_color != OFFBOARD );
    char foe_color = flipColor( frend_color );

    int frend_dist = 0;
    int foe_dist = 0;
    bool frend_found = false;
    bool foe_found = false;

    //spiral around neighbors at manhat dist 1,2,3,etc until
    int radius = 1;
    //start at the NW neighbor
    int spiral_head = ix-BIGDIM-1;
    while( !(frend_found && foe_found) && radius < DIMENSION ){
        
        //we start at the corners, so need to take radius*2 steps
        //to reach the other corner
        int nsteps = radius*2;

        //after radius steps inwards
        //we are directly above/below,left/right of the starting ix
        //and therefore min distance
        int min_dist_step = radius;

        for( int dir=0; dir < 4; ++dir ){

            for( int step=0; step < nsteps; ++step ){
                
                //check if friend or foe
                char spiral_color = ix2color(spiral_head);
                if( spiral_color == frend_color ){
                    frend_dist = radius + abs(step - min_dist_step);
                    //cout << "frend_dist: " << frend_dist << endl;
                    frend_found = true;
                    if( foe_found ) break;

                }
                else if( spiral_color == foe_color ){
                    foe_dist = radius + abs(step - min_dist_step);
                    //cout << "foe_dist: " << foe_dist << endl;
                    foe_found = true;
                    if( frend_found ) break;
                }
                else{
                    //nothing
                }

                //update the spiral_head
                if( dir == 0 )       ++spiral_head;
                else if( dir == 1 )  spiral_head += BIGDIM;
                else if( dir == 2 )  --spiral_head;
                else                 spiral_head -= BIGDIM;
                
            }
            if( frend_found && foe_found ) break;
        }

        //now spiral_head is back where it started, top-left corner of
        //the given radius box
        //so inc the radius and move spiral_head up and left one
        ++radius;
        spiral_head -= (BIGDIM+1);
    }

    return pair<int,int> (frend_dist,foe_dist);

}
