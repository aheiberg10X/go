#ifndef QUEUE
#define QUEUE

#include "constants.h"

struct Queue{
    //int length;
    int array[BOARDSIZE];
    int begin, end;
    int nElems;

    Queue();
    /*~Queue();*/

    int ringInc( int i );

    void clear();

    void push( int a );

    int pop();

    bool isEmpty();

};

#endif
