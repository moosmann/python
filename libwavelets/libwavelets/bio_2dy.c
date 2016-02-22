

/* file bio_2d.c */
/*       Fast algorithm for a biorthorgonal Low and High pass filter
 main local fiteroperation in space 

             
 
                                   Algorithm and code developped 
                                               by
                                      Jan-Olov Stromberg


                                Univiversity of Tromso, Norway,
                     Fast Mathematical Algorithms&Hardware, Hamden, CT, USA


                             Last updated version by May 24, 1998
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
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

void bio_2d_y_premult(vector, xlength, ylength, X, pairity)
    FLOAT* vector;
int xlength, ylength;
FLOAT X;
int pairity;
{

    register int j, k;
    FLOAT* vptr0, *vptr1, *vptr2;
    FLOAT a = 0, f = 0, g;
    FLOAT x1;

    x1 = X;
    j = 0;
    if ((pairity == 0) && (ylength > 1)) {
        vptr1 = vector;
        vptr0 = vector + xlength;
        k = 0;
        while (k < xlength) {
            f = vptr1[k];
            a = vptr0[k];
            a *= x1;
            vptr1[k] = 2 * a + f;
            vptr0[k++] = a;
        }
        j = 2;
    }
    if ((pairity == 1) || ((pairity == 0) && (ylength == 1))) {
        vptr0 = vector;
        k = 0;
        while (k < xlength)
            vptr0[k++] *= x1;
        j = 1;
    }

    while (j < ylength - 1) {/* starts main outer loop doing 
				  two lines each time  */
        vptr1 = vector + j * xlength;
        vptr0 = vector + (j + 1) * xlength;
        vptr2 = vector + (j - 1) * xlength;
        k = 0;
        while (k < xlength) {
            f = vptr1[k];
            a = vptr0[k];
            g = vptr2[k];
            a *= x1;
            vptr1[k] = a + f + g;
            vptr0[k++] = a;
        }
        j += 2;
    }
    if (j == ylength - 1) {/* in case the last line remins to be done
                          it is done alone  */
        vptr1 = vector + j * xlength;
        vptr2 = vector + (j - 1) * xlength;
        k = 0;
        while (k < xlength) {
            f = vptr1[k];
            g = vptr2[k];
            vptr1[k++] = f + 2 * g;
        }
    }
}

void bio_2d_y_postmult(vector, xlength, ylength, X, pairity)
    FLOAT* vector;
int xlength, ylength;
FLOAT X;
int pairity;
{

    register int j, k;
    FLOAT* vptr0, *vptr1, *vptr2;
    FLOAT a = 0, f = 0, g;
	FLOAT x1;

    x1 = X;
    j = 0;
    if ((pairity == 0) && (ylength > 2)) {
        vptr1 = vector;
        vptr0 = vector + xlength;
        k = 0;
        while (k < xlength) {
            f = vptr1[k];
            a = vptr0[k];
            vptr1[k++] = 2 * a + f;
        }
        j = 2;
    }
    if ((pairity == 0) && (ylength == 2)) {
        vptr1 = vector;
        vptr0 = vector + xlength;
        k = 0;
        while (k < xlength) {
            f = vptr1[k];
            a = vptr0[k];
            vptr1[k] = 2 * a + f;
            vptr0[k++] = a * x1;
        }
        j = 2;
    }
    if (ylength == 1) {
        vptr0 = vector;
        k = 0;
        while (k < xlength)
            vptr0[k++] *= x1;
        j = 1;
    }

    if ((pairity == 1) && (ylength > 1)) {
        j = 1;
    }

    while (j < ylength - 2) {/* starts main outer loop doing 
				  two lines each time  */
        vptr1 = vector + j * xlength;
        vptr0 = vector + (j + 1) * xlength;
        vptr2 = vector + (j - 1) * xlength;
        k = 0;
        while (k < xlength) {
            f = vptr1[k];
            a = vptr0[k];
            g = vptr2[k];
            vptr1[k] = a + f + g;
            vptr2[k++] = x1 * g;
        }
        j += 2;
    }

    while (j == ylength - 2) {/* starts main outer loop doing 
				  two lines each time  */
        vptr1 = vector + j * xlength;
        vptr0 = vector + (j + 1) * xlength;
        vptr2 = vector + (j - 1) * xlength;
        k = 0;
        while (k < xlength) {
            f = vptr1[k];
            a = vptr0[k];
            g = vptr2[k];
            vptr1[k] = a + f + g;
            vptr0[k] = x1 * a;
            vptr2[k++] = x1 * g;
        }
        j += 2;
    }

    if (j == ylength - 1) {/* in case the last line remins to be done
                          it is done alone  */
        vptr1 = vector + j * xlength;
        vptr2 = vector + (j - 1) * xlength;
        k = 0;
        while (k < xlength) {
            f = vptr1[k];
            g = vptr2[k];
            vptr1[k] = f + 2 * g;
            vptr2[k++] = x1 * g;
        }
    }
}
