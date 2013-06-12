#include <omp.h>
#include <assert.h>
#include <iostream>
#include <time.h>
#include <stdint.h>

//#include "gostate_struct.h"

using namespace std;

int foo( int a ){
    return a+5;
}

struct TestStruct {
    int tid;
    int linkid;
    TestStruct* next;
};

TestStruct* ctor( int t, int l ){
    TestStruct* n = new TestStruct;
    n->tid = t;
    n->linkid = l;
    return n;
}

int main ()  {

    int size_per_thread = 12;
    int nthreads = 2;

    cout << "sizeof TestStruct: " << sizeof(TestStruct) << endl;

    TestStruct** states = new TestStruct*[nthreads];
    for( int i=0; i<nthreads; i++ ){
        states[i] = ctor(i,0); //new TestStruct;
    }

    int tid;
    int nlinks = 10;

/* Fork a team of threads with each thread having a private tid variable */
    #pragma omp parallel shared(states) private(tid) num_threads(nthreads)
    {

        /* Obtain and print thread id */
        tid = omp_get_thread_num();
        TestStruct* orig = states[tid];
        TestStruct* ts = orig;
        for( int i = 1; i < nlinks; i++ ){
            TestStruct* next = ctor(tid, i);
            ts->next = next;
            ts = next;
        }
        ts = orig;
        for( int i=0; i < nlinks; i++ ){
            cout << "tid: " << ts->tid << " link: " << ts->linkid << endl;
            ts = ts->next;
        }

        /* Only master thread does this */
        if (tid == 0) 
        {
            cout << "I am da MASTA" << endl;
        }   

    }  /* All threads join master thread and terminate */

}
