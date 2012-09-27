#ifndef STATE_H 
#define STATE_H

#include <set>

using namespace std;

enum COLORS {
    BLACK = -1,
    WHITE = 1,
    EMPTY = 0,
    OFFBOARD = -7
};

class State {
    public :
        int dim;
        int boardsize;
        int* board;

        //straightforward to remove if need memory
        set<int> open_positions;
        COLORS player;
        //TODO: past states
        
        //ctor
        State ( int, bool );
        
        State copy();
        int getNorth(int);
        int getSouth(int);
        int getEast(int);
        int getWest(int);
        int getNorthWest(int);
        int getNorthEast(int);
        int getSouthWest(int);
        int getSouthEast(int);

        int action2ix(int);
        int action2color(int);
        int ix2color(int);

        void setBoard( int* ixs, int len, COLORS );

        //Left off here
        int* neighborsOf( int ix, int adjacency );
        int* filterByColor( int ixs, COLORS* colors );
        int* floodFill( int ix, 
                        COLORS* filter_colors,
                        COLORS* stop_colors, 
                        int adjacency );
                                 


};

#endif

