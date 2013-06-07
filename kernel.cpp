#include "gostate_struct.h"
#include "kernel.h"


void launchSimulationKernel( GoStateStruct* gss, int* rewards ){
    int white_win = 0;
    int black_win = 0;

    //a decent guess at the number of actions taken thus far
    int n_moves_made = MAX_EMPTY - gss->num_open;
    GoStateStruct* linear = (GoStateStruct*) gss->copy();
    for( int i=0; i<NUM_SIMULATIONS; i++ ){
        gss->copyInto( linear );

        //NOTE: Turn timing off when OpenMP used, clock() synchronizes
        //      in the kernel and slows everything down
        BitMask to_exclude;
        int move_count = 0;
        while( move_count < MAX_MOVES-n_moves_made && 
               !linear->isTerminal() ){
            int action = linear->randomAction( &to_exclude, true );

            //printf( "%s\n\n", linear->toString().c_str() );
            //cout << "hit any key..." << endl;
            //cin.ignore();
            ++move_count;
        }

        int rewards[2];
        linear->getRewards( rewards );
        if( rewards[0] == 1 ){
            white_win++;
        }
        else if( rewards[1] == 1 ){
            black_win++;
        }
    }
    delete linear;
    assert( white_win+black_win == NUM_SIMULATIONS );
    rewards[0] = white_win;
    rewards[1] = black_win;
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

