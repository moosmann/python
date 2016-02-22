#include "bio.h"
#include "libwavelets_export.h"
#include "bio_parameters.h"

int LIBWAVELETS_EXPORT wavelet_transform3D(FLOAT* invector,
                                           int xlength,
                                           int ylength,
                                           int zlength,
                                           FLOAT* waveletcoefficients);

int LIBWAVELETS_EXPORT invwavelet_transform3D(FLOAT* waveletcoefficients,
                                              int xlength,
                                              int ylength,
                                              int zlength,
                                              FLOAT* outvector);

int LIBWAVELETS_EXPORT wavelet_transform2D(FLOAT* invector,
                                           int xlength,
                                           int ylength,
                                           FLOAT* waveletcoefficients);

int LIBWAVELETS_EXPORT invwavelet_transform2D(FLOAT* waveletcoefficients,
                                              int xlength,
                                              int ylength,
                                              FLOAT* outector);

int LIBWAVELETS_EXPORT wavelet_transform1D(FLOAT* invector,
                                           int xlength,
                                           FLOAT* waveletcoefficients);

int LIBWAVELETS_EXPORT invwavelet_transform1D(FLOAT* waveletcoefficients,
                                              int xlength,
                                              FLOAT* outvector);
