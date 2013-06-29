#include "feature_funcs.h"


//take the feature board, and the ix and compute
//the offset in the feature array
//PERF TODO: if we compute nbix once ahead of time, will save a bit
int FeatureFuncs::featureIX( int feature, int ix ){
    int nbix = GoState::bufferix2nobufferix( ix ); 
    if( nbix == -1 ) assert(false);
    else {
        return MAX_EMPTY*feature + nbix;
    }
}

void FeatureFuncs::setBinaryFeatures( GoState* gs, int* features, int nfeatures ){
    //the group numbers for every intersection.  -1 for offboards and empties
    int group_id = 0;
    int group_assignments[BOARDSIZE];
    memset( group_assignments, -1, sizeof(int)*BOARDSIZE );

    //for now, just int size info
    vector<int> group_info;

    //don't need to FF intersections that have already been assigned a group
    BitMask in_group;

    for( int ix=0; ix < BOARDSIZE; ++ix ){
        char ixcolor = gs->ix2color(ix);

        if( (ixcolor == WHITE || ixcolor == BLACK) &&
             in_group.get(ix) == false ){

            char fill_color = ixcolor;
            //don't want any stopping, 'n' will never be seen
            char stop_color = 'n';
            bool fill_completed = gs->floodFill( ix, 8, 
                                                 fill_color, 
                                                 stop_color );
            assert( fill_completed );

            vector<int> marked_group = gs->getMarkedGroup();
            vector<int>::iterator it;
            for( it = marked_group.begin();
                 it != marked_group.end();
                 ++it ){
                int claimed_ix = *it;
                in_group.set(claimed_ix,true);
                group_assignments[claimed_ix] = group_id;
            }
            group_info.push_back( gs->floodFillSize() );
            //cout << "group " << group_id << " size: " << floodFillSize() << endl ;
            ++group_id;
        }
        else{
            //either an empty, offboard, or already assigned a group
            //assumes intersection can be part of only one group
            continue;
        }
    }
    
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
        
        char color = gs->ix2color(ix);

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
                char ncolor = gs->ix2color(nix);
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
            pair<int,int> dists = FeatureFuncs::getManhattanDistPair(gs,ix);
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

void FeatureFuncs::board2csv( float* board, int size, int width, string filename ){
    ofstream myfile;
    myfile.open ( filename.c_str() );
    for (int i=0; i<size; i++)
    {
        if( i % width == 0 and i != 0 ){
            myfile << endl;
        }
        myfile << board[i] << ",";
    }
    myfile.close();
}

//no OFFBOARD buffer here
//this is absolutely horrendous code
void FeatureFuncs::neighborValues( int* to_fill, int* board, int ix ){
    int side = getSide( ix );
    //neighborsOf( neighbor_array, ix, ADJACENCY );
    int neighbor_array[ADJACENCY] = {ix-DIMENSION, ix+DIMENSION, ix+1, ix-1,
                      ix-DIMENSION-1, ix-DIMENSION+1,
                      ix+DIMENSION-1, ix+DIMENSION+1};
    //4 0 5
    //3 - 2
    //6 1 7
    if( side == 0 ){
        int onboard[8] = {0,1,2,3,4,5,6,7};
        for( int ob=0; ob<8; ++ob ){
            to_fill[onboard[ob]] = board[neighbor_array[onboard[ob]]];
        }
    }

    if( side < 10 ){
        int onboard[5];
        if( side == 1 ){
            onboard[0] = 1;
            onboard[1] = 2;
            onboard[2] = 3;
            onboard[3] = 6;
            onboard[4] = 7;

            to_fill[4] = -1;
            to_fill[0] = -1;
            to_fill[5] = -1;
        }
        if( side == 4 ){
            onboard[0] = 0;
            onboard[1] = 2;
            onboard[2] = 3;
            onboard[3] = 4;
            onboard[4] = 5;

            to_fill[6] = -1;
            to_fill[1] = -1;
            to_fill[7] = -1;
        }
        if( side == 2 ){
            onboard[0] = 0;
            onboard[1] = 1;
            onboard[2] = 2;
            onboard[3] = 5;
            onboard[4] = 7;

            to_fill[4] = -1;
            to_fill[3] = -1;
            to_fill[6] = -1;
        }
        if( side == 3 ){
            onboard[0] = 0;
            onboard[1] = 1;
            onboard[2] = 3;
            onboard[3] = 4;
            onboard[4] = 6;

            to_fill[5] = -1;
            to_fill[2] = -1;
            to_fill[7] = -1;
        }
        for( int ob=0; ob<5; ++ob ){
            to_fill[onboard[ob]] = board[neighbor_array[onboard[ob]]];
        }
    }
    //side 1
    //4 0 5
    //3 - 2 side 3
    //6 1 7
    //side 4
    else {
        int onboard[3];
        if( side == 12 ){
            onboard[0] = 2;
            onboard[1] = 7;
            onboard[2] = 1;
            to_fill[6] = -1;
            to_fill[3] = -1;
            to_fill[4] = -1;
            to_fill[0] = -1;
            to_fill[5] = -1;
        }
        if( side == 13 ){
            onboard[0] = 3;
            onboard[1] = 6;
            onboard[2] = 1;
            to_fill[4] = -1;
            to_fill[0] = -1;
            to_fill[5] = -1;
            to_fill[2] = -1;
            to_fill[7] = -1;
        }
        if( side == 42 ){
            onboard[0] = 0;
            onboard[1] = 5;
            onboard[2] = 2;
            to_fill[4] = -1;
            to_fill[3] = -1;
            to_fill[6] = -1;
            to_fill[1] = -1;
            to_fill[7] = -1;
        }
        if( side == 43 ){
            onboard[0] = 3;
            onboard[1] = 4;
            onboard[2] = 0;
            to_fill[5] = -1;
            to_fill[2] = -1;
            to_fill[7] = -1;
            to_fill[1] = -1;
            to_fill[6] = -1;
        }
        for( int ob=0; ob<3; ++ob ){
            to_fill[onboard[ob]] = board[neighbor_array[onboard[ob]]];
        }
    }

}

//42 matches anything
//0 can only match a 0 or a -1 (offboard from neighborValues)
//1 can only match 1
bool FeatureFuncs::matchesPattern( int* neighbors, int* pattern ){
    int n_zero_matches = 0;
    int n_one_matches = 0;
    int n_pattern_zeros = 0;
    for( int i=0; i<ADJACENCY; i++ ){
        int p = pattern[i];
        int n = neighbors[i];
        if( p == 42 ){ continue; }
        else if( p == 0 ){
            ++n_pattern_zeros;
            if( n == 0 ){
                ++n_zero_matches;
            }
        }
        else if( p == 1 ){
            if( n == 1){
                ++n_one_matches;
            }
        }
        else{ assert(false); }
    }
    return n_one_matches == 2 && n_zero_matches == n_pattern_zeros;
}

void FeatureFuncs::setEdges( int* input_board, int* output_board ){
    for( int ix=0; ix<MAX_EMPTY; ++ix ){
        output_board[ix] = 0;
        if( input_board[ix] == 1 ){ //&& ! isBorderGabor(ix) ){
            //cout << "ix: " << ix << endl;
            int nvalues[ADJACENCY];
            neighborValues( nvalues, input_board, ix );
            
            //4 0 5
            //3 - 2
            //6 1 7
            const int npatterns = 24;
            int patterns[npatterns][ADJACENCY] = 
                //vert, lside 0
                {{1,1,42,0,0,42,0,42}, 
                //vert, rside 0
                {1,1,0,42,42,0,42,0},
                //horz, uside 0
                {0,42,1,1,0,0,42,42},
                //horz, bside 0
                {42,0,1,1,42,42,0,0},
                {42,1,0,42,42,1,42,0}, //updog r, "inside 0"
                {0,1,42,0,0,1,0,42},    //updog r, "outside 0"
                {42,1,42,0,1,42,0,42}, //updog l, i 0
                {0,1,0,42,1,0,42,0},   //updog l, o 0
                {1,42,0,42,42,0,42,1}, //downdog r, i 0
                {1,0,42,0,0,42,0,1},   //downdog r, o 0
                {1,42,42,0,0,42,1,42}, //downdog l, i 0
                {1,0,0,42,42,0,1,0},   //downdog l, o 0
                {0,42,42,1,0,1,42,42}, //rightdog up, i 0
                {42,0,0,1,42,1,0,0},   //rightdog up, o 0
                {42,0,42,1,42,42,0,1}, //rightdog down, i 0
                {0,42,0,1,0,0,42,1},   //rightdog down, o 0
                {42,0,1,42,42,42,1,0}, //leftdog u, i 0
                {0,42,1,0,0,0,1,42},   //leftdog u, o 0 
                {0,42,1,42,1,0,42,42},  //leftdog d, i 0 
                {42,0,1,0,1,42,0,0},   //leftdog d, o 0
                {42,0,0,42,42,1,1,0},  //rdiag, 0 down
                {0,42,42,0,0,1,1,42},  //rdiag, 0 up
                {0,42,0,42,1,0,42,1},  //ldiag, 0 up
                {42,0,42,0,1,42,0,1}   //ldiag, 0 down
                };
        
            for( int i=0; i<npatterns; ++i ){
                bool match = matchesPattern( nvalues, patterns[i] );
                if( match ){
                    output_board[ix] = 1;
                    break;
                }
            }


            //for( int i=0; i<ADJACENCY; ++i ){
            //cout << nvalues[i] << ",";
            //}
            //cout << endl;

            //neighborsOf(ix)
            //map this array to their colors
            //iterate through all the neighbor patterns
            //1 * *
            //0 1 *  ->  [42 0 42 0 1 42 0 1]
            //0 0 1a
            //neighborsOf returns neighbs in this order:
            //N,S,E,W,NW,NE,SW,SE
            //*'s mean don't care, 1's and 0's must match between
            //template and neighbors
            //for each neighbor pattern, sum the # of matching 1,0's
            //if num matching 1's = total # of 1's
            //and num matching 0's >= total # of 0's - 1
            //then we say ix is a edge point.
            //Can break from template iteration matching


            //if( inVerticalEdge( input_board, ix ) ){
            //output_board[ix] = 1;
            //continue;
            //}
            //else if( inHorizontalEdge( input_board, ix ) ){
            //output_board[ix] = 1;
            //continue;
            //}
            //
            //}
            //output_board[ix] = 0;
        }
    }
}

//  4 3 2 3 4
//  3 2 1 2 3
//  2 1 0 1 2
//  3 2 1 2 3
//  4 3 2 3 4
pair<int,int> FeatureFuncs::getManhattanDistPair( GoState* gs, int ix ){
    char frend_color = gs->ix2color(ix);
    assert( frend_color != EMPTY && frend_color != OFFBOARD );
    char foe_color = gs->flipColor( frend_color );

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
                char spiral_color = gs->ix2color(spiral_head);
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

int FeatureFuncs::getSide( int ix ){
    if( ix <= DIMENSION-1 ){ 
        if( ix % DIMENSION == 0 ){
            return 12;
        }
        if( ix % DIMENSION == DIMENSION-1 ){
            return 13;
        }
        return 1;
    }
    else if( DIMENSION*(DIMENSION-1) <= ix ){
        if( ix % DIMENSION == 0 ){
            return 42;
        }
        if( ix % DIMENSION == DIMENSION-1 ){
            return 43;
        }
        return 4;
    }
    else if( ix % DIMENSION == 0 ){return 2; }
    else if( ix % DIMENSION == DIMENSION-1 ){ return 3; }
    else{ return 0; }
}



/*
bool FeatureFuncs::inVerticalEdge( int* input_board, int ix ){
    int side = getSide(ix);
    if( side == 1 || side == 4 ){ 
        //if stone on top or bottom, can't have its top and bottom 
        //neighbs both be one
        return false; }

    int north = input_board[ix-DIMENSION];
    int south = input_board[ix+DIMENSION];
    if( north==1 && south==1 ){
        cout << "up down == 1" << endl;
        int sum = 0;
        if( side == 2 ){
            //left side is off board
        }
        else {
            //left side empty
            int nw = input_board[ix-DIMENSION-1] == 0;
            int w = input_board[ix-1] == 0;
            int sw = input_board[ix+DIMENSION-1] == 0;
            sum = nw+w+sw;
            cout << "left sum: " << sum << endl;
        }

        if( sum >= 2 ){
            return true;
        }
        else{
            if( side == 3 ){
                //right side is off board
                return false;
            }
            else {
                int ne = input_board[ix-DIMENSION+1] == 0;
                int e = input_board[ix+1] == 0;
                int se = input_board[ix+DIMENSION+1] == 0;
                sum = ne+e+se;
                cout << "right sum: " << sum << endl;
                if( sum >= 2 ){
                    return true;
                }
            }
        }
    }
    cout << "ret false: " << endl;
    return false;
}

bool FeatureFuncs::inHorizontalEdge( int* input_board, int ix ){
    int side = getSide(ix);
    cout << "side : " << side << endl;
    if( side == 2 || side == 3 ){ 
        return false; }

    int east = input_board[ix+1];
    int west = input_board[ix-1];
    if( east==1 && west==1 ){
        cout << "east west 1 1" << endl;
        int sum = 0;

        if( side == 1 ){
        }
        else {
            //north side empty
            int nw = input_board[ix-DIMENSION-1] == 0;
            int n = input_board[ix-DIMENSION] == 0;
            int ne = input_board[ix-DIMENSION+1] == 0;
            sum = nw+n+ne;
        }
        cout << "north sum: " << sum << endl;
        if( sum >= 2 ){
            return true;
        }
        else{
            if( side == 4 ){
                return false;
            }
            else {
                int sw = input_board[ix+DIMENSION-1] == 0;
                int s = input_board[ix+DIMENSION] == 0;
                int se = input_board[ix+DIMENSION+1] == 0;
                sum = sw+s+se;
                cout << "south sum: " << sum << endl;
                if( sum >= 2 ){
                    return true;
                }
            }
        }
    }
    return false;
}

*/
