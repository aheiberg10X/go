#include "queue.h"

int Queue::ringInc( int i ){
    if( i == BOARDSIZE-1 ){ return 0; }
    else { return i+1; }
}

Queue::Queue(){
    clear();
}

void Queue::clear(){
    begin = 0;
    end = 0;
    nElems = 0;
}

void Queue::push( int a ){
    array[end] = a;
    end = ringInc( end );
    nElems++;
    return;
}

int Queue::pop(){
    int r = array[begin];
    begin = ringInc(begin);
    nElems--;
    return r;
}

bool Queue::isEmpty(){
    return nElems == 0; 
}

