#include "queue.h"

int Queue::ringInc( int i ){
    if( i == BOARDSIZE-1 ){ return 0; }
    else { return i+1; }
}

Queue::Queue(){
    clear();
}

void Queue::clear(){
    for( int i=0; i<BOARDSIZE; i++){
        array[i] = -1;
    }
    begin = 0;
    end = 0;
}


void Queue::push( int a ){
    array[end] = a;
    end = ringInc( end );
    return;
}

int Queue::pop(){
    int r = array[begin];
    array[begin] = -1;
    begin = ringInc(begin);
    return r;
}

bool Queue::isEmpty(){
    return begin == end && array[end] == -1;
}
