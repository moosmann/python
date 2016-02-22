#include "math.h"
#include "matrix.h"
#include "limits.h"
#include "mex.h"
#include "wavelet_transform.h"
#include "mexutils.h"

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    //Validate input
    if (nrhs != 1) {
        mexErrMsgTxt("Wrong number of input arguments (expected 1)");
    }
    if (nlhs != 1) {
        mexErrMsgTxt("Wrong number of output arguments (expected 1)");
    }
    if (!mxIsClass(prhs[0], MEX_INPUT_CLASS_NAME)) {
        mexErrMsgTxt("Input array of wrong type, you may need to recompile with HIGH_PRECISION on/off");
    }
    if (mxIsComplex(prhs[0])) {
        mexErrMsgTxt("Input array is complex.");
    }

    //Create a copy of indata since it is modified which is not normal matlab behaviour
    mxArray* inArray = mxDuplicateArray(prhs[0]);

    //Get number of dimensions. Note: matlab treats 1d arrays as 2d.
    mwSize ndim = mxGetNumberOfDimensions(inArray);
    const mwSize* dims = mxGetDimensions(inArray);
    if (ndim > 3) {
        mexErrMsgTxt("Input array size not supported.");
    }
    bool is1D = (ndim == 2 && (dims[0] == 1 || dims[1] == 1)); //Classify as 1D if either dimension is 1

    //Validate data size
    mwSize numel = mxGetNumberOfElements(inArray);
    if (numel > INT_MAX) {
        mexErrMsgTxt("Number of elements is to large, has to fit in INT_MAX");
    }

    //Allocate output data
    plhs[0] = mxCreateNumericArray(ndim, dims, MEX_INPUT_CLASS_TYPE, 0);

    //Get the data pointers
    FLOAT* out = mxGetPr(plhs[0]);
    FLOAT* in = mxGetPr(inArray);

    //Perform the wavelet transform
    if (is1D) {
        wavelet_transform1D(in, (int)(dims[0] * dims[1]), out);
    } else if (ndim == 2) {
        wavelet_transform2D(in, (int)(dims[0]), (int)(dims[1]), out);
    } else if (ndim == 3) {
        wavelet_transform3D(in, (int)(dims[0]), (int)(dims[1]), (int)(dims[2]), out);
    }
}