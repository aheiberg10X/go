#include <string>
#include "value_functions/value2.h"
#include "matrix.h"
#include "weights.h"

using namespace std;

int main(){
    //feautresPointsInitialize();
    mclInitializeApplication(NULL,0);
    value2Initialize();

    //output value
    mxArray* V = mxCreateDoubleScalar(0);


    const int size = 361;
    double board[size];// = {-1,-1,1,-1,0,1,0,0,-1,-1,1,-1,-1,0,-1,1,0,-1,0,-1,0,1,-1,-1,0,0,0,0,0,1,0,1,0,1,1,-1,0,0,-1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,-1,0,0,0,0,0,0,0,-1,-1,0,1,1,0,-1,0,-1,0,-1,0,1,0,0,0,0,-1,0,0,-1,0,0,0,0,0,1,1,0,0,-1,1,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,1,0,-1,1,-1,0,-1,1,0,1,-1,1,0,-1,-1,1,0,0,-1,0,-1,1,0,1,1,-1,0,0,1,-1,0,0,0,0,1,0,1,1,0,0,1,-1,-1,0,0,0,-1,0,1,0,0,0,0,1,1,-1,-1,0,1,0,0,-1,1,1,1,0,0,0,0,0,0,0,1,0,1,0,0,0,1,-1,0,0,0,0,0,0,0,-1,0,0,0,-1,0,0,-1,-1,0,0,0,0,0,0,0,-1,1,0,1,0,0,0,-1,0,1,-1,-1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,-1,1,-1,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,-1,1,-1,-1,0,1,0,0,0,1,0,1,0,0,0,0,-1,-1,0,0,1,1,0,0,0,-1,0,0,-1,-1,0,1,0,-1,1,0,1,-1,0,0,0,1,0,1,1,0,1,-1,-1,0,1,0,0,-1,0,-1,0,0,-1,1,1,1,0,-1,-1,1,0,0,0,0,0,0,0,1,-1,1,1,0,1,0,0,-1,-1,1,0,0,0,0,-1,0,-1,0,0,0,0,0,0,1,1,-1,1,1};
    for( int i=0; i<size; i++ ){
        board[i] = 0;
    }
    //board[45] = 0;
    mxArray *x = mxCreateNumericMatrix(size, 1, mxDOUBLE_CLASS, mxREAL);
    memcpy( (double*) mxGetPr(x), board, size*sizeof(double) );

    cout << WEIGHTS[0] << endl;
    
    mxArray *w = mxCreateNumericMatrix(WEIGHTS_SIZE, 1, mxDOUBLE_CLASS, mxREAL);
    memcpy( (double*) mxGetPr(w), WEIGHTS, WEIGHTS_SIZE*sizeof(double) );

    cout << "start call" << endl;
    bool success = mlfValue2(1, &V, x, w);
    cout << "success: " << success << endl;

    double* r = mxGetPr(V);
    cout << " Value: " << r[0] << endl;

    value2Terminate();
    mclTerminateApplication();
    ////feautresPointsTerminate();
    return 0;
}
