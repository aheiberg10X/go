class Queue {
public :
    int length;
    int* array;
    int begin, end;

    Queue( int l ){
        length = l;
        array = new int[length];
        for( int i=0; i<length; i++){
            array[i] = -1;
        }
        begin = 0;
        end = 0;
    }
    ~Queue(){
        delete array;
    }
    void push( int a ){
        array[end] = a;
        end = ringInc( end );
        return;
    }
    int pop(){
        int r = array[begin];
        array[begin] = -1;
        begin = ringInc(begin);
        return r;
    }
    bool isEmpty(){
        return begin == end && array[end] == -1;
    }

private :
    int ringInc( int i ){
        if( i == length-1 ){ return 0; }
        else { return i+1; }
    }
};

