

/* file bio1.2d.c */

/*       Fast algorithm for a biorthorgonal Low and High pass filter
 of length 1 in dimension  2,
 that is:  ordering only  of coefficients into LL,HL,LH, and HH                

This file is used together with bio_2d.c which works in space,
not ordering the result in subbands.
  

                                 Algorithm and code developped 
                                               by
                                      Jan-Olov Stromberg


                                Univiversity of Tromso, Norway,
                     Fast Mathematical Algorithms&Hardware, Hamden, CT, USA


                             Last updated version by May 11, 1997
*/

#ifndef NOINCLUDES
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "bio_parameters.h"
#include "bio.h"
#endif
#ifndef FLOAT
#define FLOAT float
#endif

#ifndef INTYPE
#define INTYPE FLOAT
#endif
#ifndef OUTTYPE
#define OUTTYPE FLOAT
#endif

void bioD1__2d_y(invector, H, L, xlength, ylength, X, norm)
    INTYPE* invector;
FLOAT* L, *H;
int xlength, ylength;
FLOAT X, norm;

{

    register int j, k;
    OUTTYPE* vptr;
    FLOAT* Lptr, *Hptr;
	FLOAT x0, x1;
    vptr = invector;
    Hptr = H;
    Lptr = L;
    j = 0;
    x1 = norm * X;
    x0 = norm;
    if (ylength == 1) x0 *= X;

    while (j < ylength - 1) {
        k = 0;
        while (k++ < xlength) {
            *(Lptr++) = *(vptr++) * x0;
        }
        k = 0;
        while (k++ < xlength) {
            *(Hptr++) = *(vptr++) * x1;
        }
        j += 2;
    }

    if (j == ylength - 1) {
        k = 0;
        while (k++ < xlength) {
            *(Lptr++) = *(vptr++) * x0;
        }
    }
}

/************************************/

void bioR1__2d_y(H, L, outvector, xlength, ylength, X, norm)

    FLOAT* L,
    *H;
OUTTYPE* outvector;
int xlength, ylength;
float X;
float norm;

{

    OUTTYPE* vptr, *vptr0;
    FLOAT* Lptr, *Hptr;
	FLOAT x0, x1;
    vptr = outvector;
    Hptr = H;
    Lptr = L;

    x1 = norm * X;
    x0 = norm;
    if (ylength == 1) x0 *= X;

    vptr = outvector + xlength * ylength - 1;
    vptr0 = vptr;
    Lptr = L + xlength * ((ylength + 1) >> 1) - 1;
    Hptr = H + xlength * (ylength >> 1) - 1;
    x0 = norm;
    if (ylength == 1) x0 *= X;

    if (ylength & 1) {
        vptr0 -= xlength;
        while (vptr > vptr0)
            *(vptr--) = *(Lptr--) * x0;
    }
    while (outvector <= vptr0) {
        vptr0 -= xlength;
        while (vptr > vptr0)
            *(vptr--) = *(Hptr--) * x1;
        vptr0 -= xlength;
        while (vptr > vptr0)
            *(vptr--) = *(Lptr--) * x0;
    }
}

/***************************************/

void bioD1_skip_2d_y(invector, L, xlength, ylength, X, norm)
    INTYPE* invector;
FLOAT* L;
int xlength, ylength;
FLOAT X;
FLOAT norm;

{

    register int j, k;
    OUTTYPE* vptr;
    FLOAT* Lptr;
	FLOAT x0;
    vptr = invector;
    Lptr = L;

    x0 = norm;
    if (ylength == 1) x0 *= X;

    j = 0;
    while (j < ylength - 1) {
        k = 0;
        while (k++ < xlength) {
            *(Lptr++) = *(vptr++) * x0;
        }
        vptr += xlength;
        j += 2;
    }

    if (j == ylength - 1) {
        k = 0;
        while (k++ < xlength) {
            *(Lptr++) = *(vptr++) * x0;
        }
    }
}

/*************************************************************/
void bioR1_skip_2d_y(L, vector, xlength, ylength, X, norm)
    FLOAT* L;
OUTTYPE* vector;
int xlength, ylength;
FLOAT X;
FLOAT norm;

{
    FLOAT* Lptr;
    OUTTYPE* vptr, *vptr0;
	FLOAT x0;
    vptr = vector + xlength * ylength - 1;
    vptr0 = vptr;
    Lptr = L + xlength * ((ylength + 1) >> 1) - 1;
    x0 = norm;
    if (ylength == 1) x0 *= X;

    if (ylength & 1) {
        vptr0 -= xlength;
        while (vptr > vptr0)
            *(vptr--) = *(Lptr--) * x0;
    }
    while (vector <= vptr0) {
        vptr0 -= (xlength << 1);
        vptr -= xlength;
        memset(vptr + 1, 0, xlength * sizeof(float));
        while (vptr > vptr0)
            *(vptr--) = *(Lptr--) * x0;
    }
}
