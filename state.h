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
        /*int getNorth(int);*/
        /*int getSouth(int);*/
        /*int getEast(int);*/
        /*int getWest(int);*/
        /*int getNorthWest(int);*/
        /*int getNorthEast(int);*/
        /*int getSouthWest(int);*/
        /*int getSouthEast(int);*/

        int action2ix(int);
        int action2color(int);
        int ix2color(int);

        void setBoard( int* ixs, int len, COLOR );

        //Left off here
        void neighborsOf( int ix, int adjacency );
        void filterByColor( int* neighbors_array, 
                            int adjacency,  
                            COLOR* color_array, 
                            int filter_len );
        int* floodFill( int ix, 
                        COLOR* color_array,
                        COLOR* stop_colors, 
                        int adjacency );

        int neighbor_array[8];
        int filtered_array[8];
        int filtered_len;
        COLOR color_array[3];

        /*private :*/



};

#endif

