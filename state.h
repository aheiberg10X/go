#ifndef STATE_H 
#define STATE_H

#include <set>
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

class State {
    public :
        int dim;
        int bigdim;
        int boardsize;
        COLOR* board;

        //straightforward to remove if need memory
        set<int> open_positions;
        COLOR player;
        //TODO: past states
        
        //ctor
        State ( int, bool );
        
        State copy();
        string toString();

        int neighbor(int ix, DIRECTION dir);

        int action2ix(int);
        int action2color(int);
        int ix2color(int);

        void setBoard( int* ixs, int len, COLOR );

        void* neighborsOf( int* to_fill,
                           int ix, 
                           int adjacency );

        void filterByColor( int* to_fill,
                            int* to_fill_len,
                            int* neighbors_array, 
                            int adjacency,  
                            COLOR* color_array, 
                            int filter_len );

        void floodFill( int* to_fill,
                        int* to_fill_len,
                        int epicenter_ix,
                        int* neighbor_array, 
                        int adjacency, 
                        COLOR* color_array,
                        int color_len,
                        COLOR* stop_colors, 
                        int stop_len );

        //TODO
        //figure out if these static or member
        int* floodfill_array;
        int* floodfill_len;

        int neighbor_array[8];

        int filtered_array[8];
        int* filtered_len;

        COLOR color_array[3];

        /*private :*/



};

#endif

