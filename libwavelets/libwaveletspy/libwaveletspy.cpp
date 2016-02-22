#include <boost/python.hpp>
#include "boost/python/extract.hpp"
#include "boost/python/numeric.hpp"
#include <iostream>
#include <vector>

extern "C" {
#include <libwavelets/wavelet_transform.h>
}

using namespace boost::python;

// 1D, to be removed use wavelet_decompose3Py
void wavelet_transform1DPy(uintptr_t invector,
                           int xlength,
                           uintptr_t waveletcoefficients) {
    wavelet_transform1D(reinterpret_cast<FLOAT*>(invector),
                        xlength,
                        reinterpret_cast<FLOAT*>(waveletcoefficients));
}
//to be removed, use wavelet_reconstruct3Py
void invwavelet_transform1DPy(uintptr_t waveletcoefficients,
                              int xlength,
                              uintptr_t outvector) {
    invwavelet_transform1D(reinterpret_cast<FLOAT*>(waveletcoefficients),
                           xlength,
                           reinterpret_cast<FLOAT*>(outvector));
}

// 2D, to be removed use wavelet_decompose3Py
void wavelet_transform2DPy(uintptr_t invector,
                           int xlength,
			               int ylength,
                           uintptr_t waveletcoefficients) {
    wavelet_transform2D(reinterpret_cast<FLOAT*>(invector),
                        xlength,
                        ylength,
                        reinterpret_cast<FLOAT*>(waveletcoefficients));
}
//to be removed, use wavelet_reconstruct3Py
void invwavelet_transform2DPy(uintptr_t waveletcoefficients,
                              int xlength,
                              int ylength,
                              uintptr_t outvector) {
    invwavelet_transform2D(reinterpret_cast<FLOAT*>(waveletcoefficients),
                           xlength,
                           ylength,
                           reinterpret_cast<FLOAT*>(outvector));
}

// 3D, to be removed use wavelet_decompose3Py
void wavelet_transform3DPy(uintptr_t invector,
                           int xlength,
			               int ylength,
			               int zlength,
                           uintptr_t waveletcoefficients) {
    wavelet_transform3D(reinterpret_cast<FLOAT*>(invector),
                        xlength,
                        ylength,
                        zlength,
                        reinterpret_cast<FLOAT*>(waveletcoefficients));
}
//to be removed, use wavelet_reconstruct3Py
void invwavelet_transform3DPy(uintptr_t waveletcoefficients,
                              int xlength,
                              int ylength,
                              int zlength,
                              uintptr_t outvector) {
    invwavelet_transform3D(reinterpret_cast<FLOAT*>(waveletcoefficients),
                           xlength,
                           ylength,
                           zlength,
                           reinterpret_cast<FLOAT*>(outvector));
}

// Main method
// See wavelet_dec3.c
int wavelet_decompose3Py(uintptr_t inspacevector,
                       int Xlength,
                       int Ylength,
                       int Zlength,
                       int Filterlength,
                       int Levels,
                       int minZLevels,
                       int MaxZLevels,
                       int minXYLevels,
                       int MaxXYLevels,
                       int Skip,
                       uintptr_t covector,
                       int ifnotSilent) {
   int colength;

    wavelet_decompose3(reinterpret_cast<FLOAT*>(inspacevector),
                       Xlength,
                       Ylength,
                       Zlength,
                       Filterlength,
                       Levels,
                       minZLevels,
                       MaxZLevels,
                       minXYLevels,
                       MaxXYLevels,
                       Skip,
                       reinterpret_cast<FLOAT*>(covector),
                       &colength,
                       ifnotSilent);
}

// See wavelet_rec3.c
void wavelet_reconstruct3Py(uintptr_t reccovector,
                         int colength,
                         uintptr_t outvector,
                         int Xlength,
                         int Ylength,
                         int Zlength,
                         int Filterlength,
                         int Levels,
                         int minZLevels,
                         int maxZLevels,
                         int minXYLevels,
                         int maxXYLevels,
                         int Skip,
                         int ifnotSilent) {
   wavelet_reconstruct3(reinterpret_cast<FLOAT*>(reccovector),
                         colength,
                         reinterpret_cast<FLOAT*>(outvector),
                         Xlength,
                         Ylength,
                         Zlength,
                         Filterlength,
                         Levels,
                         minZLevels,
                         maxZLevels,
                         minXYLevels,
                         maxXYLevels,
                         Skip,
                         ifnotSilent);
}

// Expose classes and methods to Python
BOOST_PYTHON_MODULE(libwaveletspy) {
    // 1D
    def("wavelet_transform1D", wavelet_transform1DPy);
    def("invwavelet_transform1D", invwavelet_transform1DPy);
    // 2D
    def("wavelet_transform2D", wavelet_transform2DPy);
    def("invwavelet_transform2D", invwavelet_transform2DPy);
    // 3D
    def("wavelet_transform3D", wavelet_transform3DPy);
    def("invwavelet_transform3D", invwavelet_transform3DPy);
    // Generic
    def("wavelet_decompose3", wavelet_decompose3Py);
    def("wavelet_reconstruct3", wavelet_reconstruct3Py);
}
