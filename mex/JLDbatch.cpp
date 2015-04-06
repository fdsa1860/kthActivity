//
//  JLDbatch.cpp
//  
//
//  Created by Xikang Zhang on 4/5/15.
//
//

#include <iostream>
#include </usr/local/include/armadillo>
#include "mex.h"

using namespace std;
using namespace arma;

mxArray *process(const mxArray *mxHH1, const mxArray *mxHH2)
{
    if (mxGetNumberOfDimensions(mxHH1) != 2 || mxGetNumberOfDimensions(mxHH2) != 2)
        mexErrMsgTxt("Input dimension must be 2.\n");
    
    const int *dimsHH1 = mxGetDimensions(mxHH1);
    const int *dimsHH2 = mxGetDimensions(mxHH2);
    
    if (dimsHH1[1] == 0 || dimsHH2[1] == 0)
        mexErrMsgTxt("Inputs must be nonempty.\n");
    if (dimsHH1[0] != 1 || dimsHH2[0] != 1)
        mexErrMsgTxt("Inputs must be 1 by n cell array.\n");
    
    mxArray *mxA = mxGetCell(mxHH1, 0);
    mxArray *mxB = mxGetCell(mxHH2, 0);
    const int *dimsA = mxGetDimensions(mxA);
    const int *dimsB = mxGetDimensions(mxB);
    if (dimsA[0] != dimsA[1] || dimsB[0] != dimsB[1])
        mexErrMsgTxt("Input matrix must be square.\n");
    if (dimsA[0] != dimsB[0])
        mexErrMsgTxt("Input matrix dimensions do not match.\n");
    
    int k = dimsA[0];
    mat A(k,k);
    mat B(k,k);
    const double* A_mem;
    const double* B_mem;
    A_mem = access::rw(A.mem);
    B_mem = access::rw(B.mem);
    
    mxArray *output = mxCreateDoubleMatrix(dimsHH1[1], dimsHH2[1],mxREAL);
    double *D = mxGetPr(output);
    
    int count = 0;
    for(unsigned int j = 0; j < dimsHH2[1]; j++)
    {
        for(unsigned int i = 0; i < dimsHH1[1]; i++)
        {
            mxArray *mxA = mxGetCell(mxHH1, i);
            mxArray *mxB = mxGetCell(mxHH2, j);
            const int *dimsA = mxGetDimensions(mxA);
            const int *dimsB = mxGetDimensions(mxB);
            if (dimsA[0] != dimsA[1] || dimsB[0] != dimsB[1]
                || dimsA[0] != dimsB[0] || dimsB[0] != k)
                mexErrMsgTxt("Input matrices must be of same square dimension.\n");

            access::rw(A.mem)=mxGetPr(mxA);
            access::rw(B.mem)=mxGetPr(mxB);

            //    mat E = 1e-6 * eye(dimsA[0],dimsA[1]);
            //    A += E;
            //    B += E;
    
            double val1,val2, val3;
            double sign = 1.0;
            log_det(val1, sign, 0.5*A+0.5*B);
            log_det(val2, sign, A);
            log_det(val3, sign, B);
            D[j*dimsHH1[1]+i] = val1 - 0.5 * val2 - 0.5 * val3;
        }
    }

    access::rw(A.mem)=A_mem;
    access::rw(B.mem)=B_mem;
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
