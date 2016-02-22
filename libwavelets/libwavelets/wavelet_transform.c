#include "wavelet_transform.h"

int wavelet_transform3D(FLOAT* inspacevector,
                        int xlength,
                        int ylength,
                        int zlength,
                        FLOAT* waveletcoefficients) {
    int co_length;

    return wavelet_decompose3(inspacevector,
                              xlength,
                              ylength,
                              zlength,
                              9,  /* char Filterlength */
                              10, /* char Levels */
                              0,  /* char minZLevels */
                              10, /* char MaxZLevels */
                              0,  /* char minXYLevels */
                              10, /*char MaxXYLevels */
                              0,  /*char Skip */
                              waveletcoefficients,
                              &co_length,
                              0 /* char ifnotSilent: 1 for printing */
                              );
}

int invwavelet_transform3D(FLOAT* waveletcoefficients,
                           int xlength,
                           int ylength,
                           int zlength,
                           FLOAT* outvector) {

    return wavelet_reconstruct3(waveletcoefficients,
                                xlength * ylength * zlength, /*coeffcient_length*/
                                outvector,
                                xlength,
                                ylength,
                                zlength,
                                9,  /*Filterlength*/
                                10, /*Levels of transforms*/
                                0,  /* min ZLevel (changes coeff_length) */
                                10, /* maxZLevels */
                                0,  /* minXYLevels (changes coeff_length)*/
                                10, /* maxXYLevels */
                                0,  /* Skip: general skiplevel
				    (changes coeff_length)*/
                                0   /*ifnotSilent  set 1 for printing */
                                );
}

int wavelet_transform2D(FLOAT* inspacevector,
                        int xlength,
                        int ylength,
                        FLOAT* waveletcoefficients) {
    int co_length;

    return wavelet_decompose3(inspacevector,
                              xlength,
                              ylength,
                              1,  /*zlength,*/
                              9,  /* char Filterlength */
                              10, /* char Levels */
                              0,  /* char minZLevels */
                              0,  /* char MaxZLevels */
                              0,  /* char minXYLevels */
                              10, /*char MaxXYLevels */
                              0,  /*char Skip */
                              waveletcoefficients,
                              &co_length,
                              0 /* char ifnotSilent: 1 for printing */
                              );
}

int invwavelet_transform2D(FLOAT* waveletcoefficients,
                           int xlength,
                           int ylength,
                           FLOAT* outvector) {

    return wavelet_reconstruct3(waveletcoefficients,
                                xlength * ylength, /*coeffcient_length*/
                                outvector,
                                xlength,
                                ylength,
                                1,  /* zlength*/
                                9,  /*Filterlength*/
                                10, /*Levels of transforms*/
                                0,  /* min ZLevel (changes coeff_length) */
                                0,  /* maxZLevels */
                                0,  /* minXYLevels (changes coeff_length)*/
                                10, /* maxXYLevels */
                                0,  /* Skip: general skiplevel
				    (changes coeff_length)*/
                                0   /*ifnotSilent  set 1 for printing */
                                );
}

int wavelet_transform1D(FLOAT* inspacevector,
                        int xlength,
                        FLOAT* waveletcoefficients) {
    int co_length;

    return wavelet_decompose3(inspacevector,
                              xlength,
                              1,  /*ylength*/
                              1,  /*zlength,*/
                              9,  /* char Filterlength */
                              10, /* char Levels */
                              0,  /* char minZLevels */
                              0,  /* char MaxZLevels */
                              0,  /* char minXYLevels */
                              10, /*char MaxXYLevels */
                              0,  /*char Skip */
                              waveletcoefficients,
                              &co_length,
                              0 /* char ifnotSilent: 1 for printing */
                              );
}

int invwavelet_transform1D(FLOAT* waveletcoefficients,
                           int xlength,
                           FLOAT* outvector) {

    return wavelet_reconstruct3(waveletcoefficients,
                                xlength, /*coeffcient_length*/
                                outvector,
                                xlength,
                                1,  /*ylength*/
                                1,  /* zlength*/
                                9,  /*Filterlength*/
                                10, /*Levels of transforms*/
                                0,  /* min ZLevel (changes coeff_length) */
                                0,  /* maxZLevels */
                                0,  /* minXYLevels (changes coeff_length)*/
                                10, /* maxXYLevels */
                                0,  /* Skip: general skiplevel
				    (changes coeff_length)*/
                                0   /*ifnotSilent  set 1 for printing */
                                );
}
