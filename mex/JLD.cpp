#include <iostream>
#include </usr/local/include/armadillo>
#include "mex.h"

using namespace std;
using namespace arma;

void matlab2arma(mat& A, const mxArray *mxdata){
    // delete [] A.mem; // don't do this!
    access::rw(A.mem)=mxGetPr(mxdata);
    access::rw(A.n_rows)=mxGetM(mxdata); // transposed!
    access::rw(A.n_cols)=mxGetN(mxdata);
    access::rw(A.n_elem)=A.n_rows*A.n_cols;
};

void freeVar(mat& A, const double *ptr){
    access::rw(A.mem)=ptr;
    access::rw(A.n_rows)=1; // transposed!
    access::rw(A.n_cols)=1;
    access::rw(A.n_elem)=1;
};

mxArray *process(const mxArray *mxA, const mxArray *mxB)
{
    if (mxGetNumberOfDimensions(mxA) != 2 || mxGetNumberOfDimensions(mxB) != 2)
        mexErrMsgTxt("Input dimension must be 2.\n");
    
    const int *dimsA = mxGetDimensions(mxA);
    const int *dimsB = mxGetDimensions(mxB);
    
    if (dimsA[0] != dimsA[1] || dimsB[0] != dimsB[1])
        mexErrMsgTxt("Input matrix must be square.\n");
    if (dimsA[0] != dimsB[0])
        mexErrMsgTxt("Input matrix dimensions do not match.\n");
    
    mat A(1,1);
    const double* A_mem = access::rw(A.mem);
//    access::rw(A.mem)=mxGetPr(mxA);
    matlab2arma(A,mxA);
    
    mat B(1,1);
    const double* B_mem = access::rw(B.mem);
//    access::rw(B.mem)=mxGetPr(mxB);
    matlab2arma(B,mxB);
    
//    mat E = 1e-6 * eye(dimsA[0],dimsA[1]);
//    A += E;
//    B += E;

    double d;
    double val1,val2, val3;
    double sign = 1.0;
    log_det(val1, sign, 0.5*A+0.5*B);
    log_det(val2, sign, A);
    log_det(val3, sign, B);
    d = val1 - 0.5 * val2 - 0.5 * val3;
    mxArray *output = mxCreateDoubleScalar(d);
    
    freeVar(A,A_mem); // Change back the pointers!!
    freeVar(B,B_mem);
    return output;
}

// matlab entry point
// distance = mexJLD(A,B)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2)
        mexErrMsgTxt("Wrong number of inputs\n");
    if (nlhs != 1)
        mexErrMsgTxt("Wrong number of outputs\n");
    plhs[0] = process(prhs[0],prhs[1]);
}