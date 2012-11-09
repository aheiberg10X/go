#ifndef QUEUE
#define QUEUE

#include "constants.h"

struct Queue{
    //int length;
    int array[BOARDSIZE];
    int begin, end;

    Queue();

    int ringInc( int i );

    /*void initQueue();*/

    void clear();

    void push( int a );

    int pop();

    bool isEmpty();

};

#endif
