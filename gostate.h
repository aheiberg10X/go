#ifndef GOSTATE_H 
#define GOSTATE_H

/*#include <functional>*/
/*#include <set>*/
/*#include <vector>*/
#include <string>
/*#include <unordered_set>*/
/*#include <hash_map>*/

/*using namespace std;*/
//TODO
//mem gains to be had by getting rid of ENUM type and replacing 'board' with 
//char array.  ENUMs are converted to int, which take 4 bytes, char's take 1
/*enum COLOR {*/
/*BLACK = -1,*/
/*WHITE = 1,*/
/*EMPTY = 0,*/
/*OFFBOARD = -7*/
/*};*/

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
/*const int DIM = 9;*/
/*const big*/
#define DIM 9
#define BIGDIM 11
#define BOARDSIZE 121

#define BLACK 'b'
#define WHITE 'w'
#define EMPTY 'e'
#define OFFBOARD 'o'

class GoState {
    public :
        /*std::string name;*/
        /*int dim;*/
        int bigdim;
        int boardsize;
        /*char* board;*/
        char board[BOARDSIZE];
        int action;
        bool shallow;
        /*std::unordered_set<int> test;*/

        //straightforward to remove if need memory
        /*std::set<int> open_positions;*/
        /*int open_positions[boardsize];*/
        /*int num_open;*/
        int num_open;

        char player;
        //TODO 
        //hard allocate this
        GoState* past_states[NUM_PAST_STATES];
        //TODO: past states
        
        //ctor
        GoState ( /*std::string, int,*/ bool shallow);
        __device__ __host__ GoState( void** pointers );
        __device__ __host__ ~GoState();

        //cuda specific
        int numElementsToCopy();
        void cudaAllocateAndCopy( void** pointers );
        
        GoState* copy(bool shallow);
        void copyInto( GoState* state );

        std::string toString();

        char flipColor( char c );
        bool sameAs( char* board, char player );
        bool sameAs( GoState s );

        void togglePlayer();

        int    action2ix(int);
        int    ix2action(int, char);
        char  action2color(int);
        int    ix2color(int);
        int    coord2ix( int i, int j );
        /*std::string ix2coord( int ix );*/
        bool isPass( int action );
        int coordColor2Action( int i, int j, char color );
        int ixColor2Action( int ix, char color );

        void setBoard( int ix, char color );
        void setBoard( int* ixs, int len, char );

        __device__ __host__ int neighbor(int ix, DIRECTION dir);
        void neighborsOf( int* to_fill,
                           int ix, 
                           int adjacency );

        void filterByColor( int* to_fill,
                            int* to_fill_len,
                            int* neighbors_array, 
                            int adjacency,  
                            char* color_array, 
                            int filter_len );

         bool floodFill( int* to_fill,
                        int* to_fill_len,
                        int epicenter_ix,
                        /*int* neighbor_array, */
                        int adjacency, 
                        char* color_array,
                        int color_len,
                        char* stop_colors, 
                        int stop_len );


        bool isSuicide( int action );
        
        /*int* floodfill_array;*/
        int floodfill_array[BOARDSIZE];

        int neighbor_array[8];
        int filtered_array[8];
        char color_array[3];

        /*private :*/



};

#endif

