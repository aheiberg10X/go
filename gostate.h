#ifndef GOSTATE_H 
#define GOSTATE_H

#include <set>
#include <vector>
#include <string>

using namespace std;

enum COLOR {
    BLACK = -1,
    WHITE = 1,
    EMPTY = 0,
    OFFBOARD = -7
};

enum DIRECTION {
    N = 0,
    S = 1,
    E = 2,
    W = 3,
    NW = 4,
    NE = 5,
    SW = 6,
    SE = 7
};

const int PASS = 0;

const int NUM_PAST_STATES = 10;

class GoState {
    public :
        std::string name;
        int dim;
        int bigdim;
        int boardsize;
        COLOR* board;
        int action;
        bool shallow;

        //straightforward to remove if need memory
        set<int> open_positions;
        COLOR player;
        GoState* past_states[NUM_PAST_STATES];
        //TODO: past states
        
        //ctor
        GoState ( string, int, bool );
        ~GoState();
        
        GoState* copy(bool shallow);
        void copyInto( GoState* state );

        string toString();

        COLOR flipColor( COLOR c );
        bool sameAs( COLOR* board, COLOR player );
        bool sameAs( GoState s );

        void togglePlayer();

        int    action2ix(int);
        int    ix2action(int, COLOR);
        COLOR  action2color(int);
        int    ix2color(int);
        int    coord2ix( int i, int j );
        string ix2coord( int ix );
        bool isPass( int action );
        int coordColor2Action( int i, int j, COLOR color );
        int ixColor2Action( int ix, COLOR color );

        void setBoard( int ix, COLOR color );
        void setBoard( int* ixs, int len, COLOR );

        int neighbor(int ix, DIRECTION dir);
        void* neighborsOf( int* to_fill,
                           int ix, 
                           int adjacency );

        void filterByColor( int* to_fill,
                            int* to_fill_len,
                            int* neighbors_array, 
                            int adjacency,  
                            COLOR* color_array, 
                            int filter_len );

         bool floodFill( int* to_fill,
                        int* to_fill_len,
                        int epicenter_ix,
                        /*int* neighbor_array, */
                        int adjacency, 
                        COLOR* color_array,
                        int color_len,
                        COLOR* stop_colors, 
                        int stop_len );


        bool isSuicide( int action );

        //TODO
        //figure out if these static or member
        //might not be a good idea to uses these at all
        //functions call each other and sometimes use the same array
        int* floodfill_array;
        int neighbor_array[8];
        int filtered_array[8];
        COLOR color_array[3];

        /*private :*/



};

#endif

