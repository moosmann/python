

/* file bio1.3d.c */

/*       Fast algorithm for a biorthorgonal Low and High pass filter
 of length 1 in dimension  3,
 that is:  ordering only  of coefficients into LLL,LHL,LLH, and LHH                
HLL,HHL,HLH, and HHH                

This file is used together with bio_3d.c which works in space,
not ordering the result in subbands.
  

                                 Algorithm and code developped 
                                               by
                                      Jan-Olov Stromberg


				      KTH, Stockholm, Sweden
                     Fast Mathematical Algorithms&Hardware, Hamden, CT, USA

                             Preliminary  version by October 18 , 1997

        
*/

#ifndef NOINCLUDES
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#endif

#include "bio_parameters.h"
#include "bio.h"
#ifndef FLOAT
#define FLOAT float
#endif
#ifndef INTYPE
#define INTYPE FLOAT
#endif
#ifndef OUTTYPE
#define OUTTYPE FLOAT
#endif

void bioD1__3d(invector, HHH, HHL, HLH, HLL,
               LHH, LHL, LLH, LLL, xlength, ylength, zlength, X, norm)
    INTYPE* invector;
FLOAT* HLL, *HLH, *HHL, *HHH;
FLOAT* LLL, *LLH, *LHL, *LHH;
int xlength, ylength, zlength;
FLOAT X, norm;

{

    register int j, s;
    OUTTYPE* vptr, *vptrend_1;
    FLOAT* HLLptr, *HLHptr, *HHLptr, *HHHptr;
    FLOAT* LLLptr, *LLHptr, *LHLptr, *LHHptr;
    FLOAT x1, x0, x_1, x_2;
    int dim;

    vptr = invector;
    HHHptr = HHH;
    HHLptr = HHL;
    HLHptr = HLH;
    HLLptr = HLL;
    LHHptr = LHH;
    LHLptr = LHL;
    LLHptr = LLH;
    LLLptr = LLL;

    dim = (xlength > 1) + (ylength > 1) + (zlength > 1);
    if (dim == 3) {
        x1 = X * norm;
        x0 = norm;
        x_1 = norm / X;
        x_2 = x_1 / X;
    }
    if (dim == 2) {
        x1 = 0;
        x0 = X * norm;
        x_1 = norm;
        x_2 = norm / X;
    }
    if (dim == 1) {
        x_1 = X * norm;
        x_2 = norm;
        x1 = 0;
        x0 = 0;
    }
    if (dim == 0) {
        x_2 = X * norm;
        x1 = 0;
        x0 = 0;
        x_1 = 0;
    }

    s = 0;

    while (s < zlength - 1) {

        j = 0;
        while (j < ylength - 1) {
            vptrend_1 = vptr + xlength - 1;
            while (vptr < vptrend_1) {
                *(LLLptr++) = x_2 * *(vptr++);
                *(LLHptr++) = x_1 * *(vptr++);
            }

            if (vptr == vptrend_1) {
                *(LLLptr++) = x_2 * *(vptr++);
            }
            vptrend_1 = vptr + xlength - 1;
            while (vptr < vptrend_1) {
                *(LHLptr++) = x_1 * *(vptr++);
                *(LHHptr++) = x0 * *(vptr++);
            }
            if (vptr == vptrend_1) {
                *(LHLptr++) = x_1 * *(vptr++);
            }
            j += 2;
        }

        if (j == ylength - 1) {
            vptrend_1 = vptr + xlength - 1;
            while (vptr < vptrend_1) {
                *(LLLptr++) = x_2 * *(vptr++);
                *(LLHptr++) = x_1 * *(vptr++);
            }
            if (vptr == vptrend_1) {
                *(LLLptr++) = x_2 * *(vptr++);
            }
            j++;
        }

        j = 0;
        while (j < ylength - 1) {
            vptrend_1 = vptr + xlength - 1;
            while (vptr < vptrend_1) {
                *(HLLptr++) = x_1 * *(vptr++);
                *(HLHptr++) = x0 * *(vptr++);
            }

            if (vptr == vptrend_1) {
                *(HLLptr++) = x_1 * *(vptr++);
            }
            vptrend_1 = vptr + xlength - 1;
            while (vptr < vptrend_1) {
                *(HHLptr++) = x0 * *(vptr++);
                *(HHHptr++) = x1 * *(vptr++);
            }
            if (vptr == vptrend_1) {
                *(HHLptr++) = x0 * *(vptr++);
            }
            j += 2;
        }

        if (j == ylength - 1) {
            vptrend_1 = vptr + xlength - 1;
            while (vptr < vptrend_1) {
                *(HLLptr++) = x_1 * *(vptr++);
                *(HLHptr++) = x0 * *(vptr++);
            }
            if (vptr == vptrend_1) {
                *(HLLptr++) = x_1 * *(vptr++);
            }
            j++;
        }
        s += 2;
    }
    if (s == zlength - 1) {
        j = 0;
        while (j < ylength - 1) {
            vptrend_1 = vptr + xlength - 1;
            while (vptr < vptrend_1) {
                *(LLLptr++) = x_2 * *(vptr++);
                *(LLHptr++) = x_1 * *(vptr++);
            }

            if (vptr == vptrend_1) {
                *(LLLptr++) = x_2 * *(vptr++);
            }
            vptrend_1 = vptr + xlength - 1;
            while (vptr < vptrend_1) {
                *(LHLptr++) = x_1 * *(vptr++);
                *(LHHptr++) = x0 * *(vptr++);
            }
            if (vptr == vptrend_1) {
                *(LHLptr++) = x_1 * *(vptr++);
            }
            j += 2;
        }

        if (j == ylength - 1) {
            vptrend_1 = vptr + xlength - 1;
            while (vptr < vptrend_1) {
                *(LLLptr++) = x_2 * *(vptr++);
                *(LLHptr++) = x_1 * *(vptr++);
            }
            if (vptr == vptrend_1) {
                *(LLLptr++) = x_2 * *(vptr++);
            }
            j++;
        }
        s++;
    }
}

/************************************/

void bioR1__3d(
    HHH, HHL, HLH, HLL, LHH, LHL, LLH, LLL,
    outvector, xlength, ylength, zlength, X, norm)

    FLOAT* HLL,
    *HLH, *HHL, *HHH;
FLOAT* LLL, *LLH, *LHL, *LHH;
OUTTYPE* outvector;
int xlength, ylength, zlength;
FLOAT X, norm;

{

    OUTTYPE* vptr, *vptrx0, *vptry0;
    FLOAT* HLLptr, *HLHptr, *HHLptr, *HHHptr;
    FLOAT* LLLptr, *LLHptr, *LHLptr, *LHHptr;
    FLOAT x1, x0, x_1, x_2;
    int dim;

    dim = (xlength > 1) + (ylength > 1) + (zlength > 1);

    if (dim == 3) {
        x1 = X * norm;
        x0 = norm;
        x_1 = norm / X;
        x_2 = x_1 / X;
    }
    if (dim == 2) {
        x1 = 0;
        x0 = X * norm;
        x_1 = norm;
        x_2 = norm / X;
    }
    if (dim == 1) {
        x_1 = X * norm;
        x_2 = norm;
        x1 = 0;
        x0 = 0;
    }
    if (dim == 0) {
        x_2 = X * norm;
        x1 = 0;
        x0 = 0;
        x_1 = 0;
    }

    HHHptr = HHH + (zlength >> 1) * (ylength >> 1) * (xlength >> 1) - 1;
    LHHptr = LHH + ((zlength + 1) >> 1) * (ylength >> 1) * (xlength >> 1) - 1;
    HLHptr = HLH + (zlength >> 1) * ((ylength + 1) >> 1) * (xlength >> 1) - 1;
    LLHptr = LLH + ((zlength + 1) >> 1) * ((ylength + 1) >> 1) * (xlength >> 1) - 1;
    HHLptr = HHL + (zlength >> 1) * (ylength >> 1) * ((xlength + 1) >> 1) - 1;
    LHLptr = LHL + ((zlength + 1) >> 1) * (ylength >> 1) * ((xlength + 1) >> 1) - 1;
    HLLptr = HLL + (zlength >> 1) * ((ylength + 1) >> 1) * ((xlength + 1) >> 1) - 1;
    LLLptr = LLL + ((zlength + 1) >> 1) * ((ylength + 1) >> 1) * ((xlength + 1) >> 1) - 1;

    /*************/

    vptr = outvector + zlength * ylength * xlength - 1;
    vptry0 = vptr;
    vptrx0 = vptr;

    if (1 & zlength) {
        vptry0 -= xlength * ylength;
        if (1 & ylength) {
            vptrx0 -= xlength;
            if (1 & xlength) *(vptr--) = x_2 * *(LLLptr--);
            while (vptr > vptrx0) {
                *(vptr--) = x_1 * *(LLHptr--);
                *(vptr--) = x_2 * *(LLLptr--);
            }
        }
        while (vptrx0 > vptry0) {
            vptrx0 -= xlength;
            if (1 & xlength) *(vptr--) = x_1 * *(LHLptr--);
            while (vptr > vptrx0) {
                *(vptr--) = x0 * *(LHHptr--);
                *(vptr--) = x_1 * *(LHLptr--);
            }
            vptrx0 -= xlength;
            if (1 & xlength) *(vptr--) = x_2 * *(LLLptr--);
            while (vptr > vptrx0) {
                *(vptr--) = x_1 * *(LLHptr--);
                *(vptr--) = x_2 * *(LLLptr--);
            }
        }
    }

    while (outvector <= vptry0) {
        vptry0 -= xlength * ylength;
        if (1 & ylength) {
            vptrx0 -= xlength;
            if (1 & xlength) *(vptr--) = x_1 * *(HLLptr--);
            while (vptr > vptrx0) {
                *(vptr--) = x0 * *(HLHptr--);
                *(vptr--) = x_1 * *(HLLptr--);
            }
        }
        while (vptrx0 > vptry0) {
            vptrx0 -= xlength;
            if (1 & xlength) *(vptr--) = x0 * *(HHLptr--);
            while (vptr > vptrx0) {
                *(vptr--) = x1 * *(HHHptr--);
                *(vptr--) = x0 * *(HHLptr--);
            }
            vptrx0 -= xlength;
            if (1 & xlength) *(vptr--) = x_1 * *(HLLptr--);
            while (vptr > vptrx0) {
                *(vptr--) = x0 * *(HLHptr--);
                *(vptr--) = x_1 * *(HLLptr--);
            }
        }
        vptry0 -= xlength * ylength;
        if (1 & ylength) {
            vptrx0 -= xlength;
            if (1 & xlength) *(vptr--) = x_2 * *(LLLptr--);
            while (vptr > vptrx0) {
                *(vptr--) = x_1 * *(LLHptr--);
                *(vptr--) = x_2 * *(LLLptr--);
            }
        }
        while (vptrx0 > vptry0) {
            vptrx0 -= xlength;
            if (1 & xlength) *(vptr--) = x_1 * *(LHLptr--);
            while (vptr > vptrx0) {
                *(vptr--) = x0 * *(LHHptr--);
                *(vptr--) = x_1 * *(LHLptr--);
            }
            vptrx0 -= xlength;
            if (1 & xlength) *(vptr--) = x_2 * *(LLLptr--);
            while (vptr > vptrx0) {
                *(vptr--) = x_1 * *(LLHptr--);
                *(vptr--) = x_2 * *(LLLptr--);
            }
        }
    }
}

/**************/

/***************************************/

void bioD1_3dskip(FLOAT* invector,
                  FLOAT* LLL,
                  int xlength,
                  int ylength,
                  int zlength,
                  FLOAT X,
                  FLOAT norm)

{

    register int j, s;
    OUTTYPE* vptr, *vptrend_1;
    FLOAT* LLLptr;
    FLOAT x_2;
    int dim;
    vptr = invector;
    LLLptr = LLL;

    s = 0;

    dim = (xlength > 1) + (ylength > 1) + (zlength > 1);

    if (dim == 3) {
        x_2 = norm / (X * X);
    }
    if (dim == 2) {
        x_2 = norm / X;
    }
    if (dim == 1) {
        x_2 = norm;
    }
    if (dim == 0) {
        x_2 = X * norm;
    }

    while (s < zlength - 1) {
        j = 0;
        while (j < ylength - 1) {
            vptrend_1 = vptr + xlength - 1;
            while (vptr < vptrend_1) {
                *(LLLptr++) = x_2 * *(vptr++);
                vptr++;
            }
            if (vptr == vptrend_1) {
                *(LLLptr++) = x_2 * *(vptr++);
            }
            vptr += xlength;
            j += 2;
        }
        if (j == ylength - 1) {
            vptrend_1 = vptr + xlength - 1;
            while (vptr < vptrend_1) {
                *(LLLptr++) = x_2 * *(vptr++);
                vptr++;
            }
            if (vptr == vptrend_1) {
                *(LLLptr++) = x_2 * *(vptr++);
            }
        }
        vptr += xlength * ylength;
        s += 2;
    }
    if (s == zlength - 1) {
        j = 0;
        while (j < ylength - 1) {
            vptrend_1 = vptr + xlength - 1;
            while (vptr < vptrend_1) {
                *(LLLptr++) = x_2 * *(vptr++);
                vptr++;
            }
            if (vptr == vptrend_1) {
                *(LLLptr++) = x_2 * *(vptr++);
            }
            vptr += xlength;
            j += 2;
        }
        if (j == ylength - 1) {
            vptrend_1 = vptr + xlength - 1;
            while (vptr < vptrend_1) {
                *(LLLptr++) = x_2 * *(vptr++);
                vptr++;
            }
            if (vptr == vptrend_1) {
                *(LLLptr++) = x_2 * *(vptr++);
            }
        }
    }
}

/*************************************************************/
void bioR1_3dskip(FLOAT* LLL,
                  FLOAT* vector,
                  int xlength,
                  int ylength,
                  int zlength,
                  FLOAT X,
                  FLOAT norm)

{
    FLOAT* LLLptr;
    OUTTYPE* vptr, *vptrx0, *vptry0, *vptrz0, *vptrmem;
    FLOAT x_2;
    int dim;

    LLLptr = LLL + ((zlength + 1) >> 1) * ((ylength + 1) >> 1) * ((xlength + 1) >> 1) - 1;
    vptr = vector + zlength * ylength * xlength - 1;

    dim = (xlength > 1) + (ylength > 1) + (zlength > 1);

    if (dim == 3) {
        x_2 = norm / (X * X);
    }
    if (dim == 2) {
        x_2 = norm / X;
    }
    if (dim == 1) {
        x_2 = norm;
    }
    if (dim == 0) {
        x_2 = X * norm;
    }

    /***************/

    vptrz0 = vector - 1;
    vptr = vptrz0 + zlength * ylength * xlength;
    vptry0 = vptr;
    vptrx0 = vptr;

    if (1 & zlength) {
        vptry0 -= xlength * ylength;
        if (1 & ylength) {
            vptrx0 -= xlength;
            vptrmem = vptrx0 + ((xlength + 1) >> 1);
            memset(vptrmem + 1, 0, (vptr - vptrmem) * sizeof(FLOAT));
            if (1 & xlength) *(vptr--) = x_2 * *(LLLptr--);
            while (vptr > vptrmem) {
                vptr--;
                *(vptr--) = x_2 * *(LLLptr--);
            }
            while (vptr > vptrx0) {
                *vptr-- = 0;
                *(vptr--) = x_2 * *(LLLptr--);
            }
        }
        vptrmem = vptry0 + ((xlength + 1) >> 1) * (ylength >> 1);
        memset(vptrmem + 1, 0, (vptr - vptrmem) * sizeof(FLOAT));
        while (vptrx0 > vptrmem) {
            vptrx0 -= 2 * xlength;
            vptr -= xlength;
            if (1 & xlength) *(vptr--) = x_2 * *(LLLptr--);
            while (vptr > vptrx0) {
                vptr--;
                *(vptr--) = x_2 * *(LLLptr--);
            }
        }
        while (vptrx0 > vptry0) {
            vptrx0 -= 2 * xlength;
            vptr -= xlength;
            memset(vptr, 0, xlength * sizeof(FLOAT));
            if (1 & xlength) *(vptr--) = x_2 * *(LLLptr--);
            while (vptr > vptrx0) {
                *vptr-- = 0;
                *(vptr--) = x_2 * *(LLLptr--);
            }
        }
    }
    vptrmem = vector + ((xlength + 1) >> 1) * ((ylength + 1) >> 1) * (zlength >> 1) - 1;
    memset(vptrmem + 1, 0, (vptr - vptrmem) * sizeof(FLOAT));
    vptrmem += 2 * xlength * ylength;
    while (vptry0 > vptrmem) {
        vptry0 -= 2 * xlength * ylength;
        vptrx0 -= xlength * ylength;
        vptr -= xlength * ylength;
        if (1 & ylength) {
            vptrx0 -= xlength;
            if (1 & xlength) *(vptr--) = x_2 * *(LLLptr--);
            while (vptr > vptrx0) {
                vptr--;
                *(vptr--) = x_2 * *(LLLptr--);
            }
        }
        while (vptrx0 > vptry0) {
            vptrx0 -= 2 * xlength;
            vptr -= xlength;
            if (1 & xlength) *(vptr--) = x_2 * *(LLLptr--);
            while (vptr > vptrx0) {
                vptr--;
                *(vptr--) = x_2 * *(LLLptr--);
            }
        }
    }

    while (vptry0 > vptrz0) {
        vptry0 -= 2 * xlength * ylength;
        vptrx0 -= xlength * ylength;
        vptr -= xlength * ylength;
        memset(vptr, 0, xlength * ylength * sizeof(FLOAT));
        if (1 & ylength) {
            vptrx0 -= xlength;
            if (1 & xlength) *(vptr--) = x_2 * *(LLLptr--);
            while (vptr > vptrx0) {
                *vptr-- = 0;
                *(vptr--) = x_2 * *(LLLptr--);
            }
        }

        vptrmem = vptry0 + ((xlength + 1) >> 1) * (ylength >> 1);
        memset(vptrmem + 1, 0, (vptr - vptrmem) * sizeof(FLOAT));

        vptrmem += 2 * xlength;
        while (vptrx0 > vptrmem) {
            vptrx0 -= 2 * xlength;
            vptr -= xlength;
            if (1 & xlength) *(vptr--) = x_2 * *(LLLptr--);
            while (vptr > vptrx0) {
                vptr--;
                *(vptr--) = x_2 * *(LLLptr--);
            }
        }

        while (vptrx0 > vptry0) {
            vptrx0 -= 2 * xlength;
            vptr -= xlength;
            memset(vptr, 0, xlength * sizeof(FLOAT));
            if (1 & xlength) *(vptr--) = x_2 * *(LLLptr--);
            while (vptr > vptrx0) {
                *vptr-- = 0;
                *(vptr--) = x_2 * *(LLLptr--);
            }
        }
    }
}
