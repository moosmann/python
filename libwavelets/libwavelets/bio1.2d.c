/* new file Febr 13, 1999 with bugs corrected  */

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


                             Last updated version by Febr 13, 1999
*/

#ifndef NOINCLUDES
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#endif
#include "bio_parameters.h"
#ifndef FLOAT
#define FLOAT float
#endif

#ifndef INTYPE
#define INTYPE FLOAT
#endif
#ifndef OUTTYPE
#define OUTTYPE FLOAT
#endif

void bioD1__2d_mult(invector, HH, HL, LH, LL, xlength, ylength, x0,
                    vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc)
    FLOAT* invector;
FLOAT* LL, *LH, *HL, *HH;
int xlength, ylength;
FLOAT x0;
int vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc;

{

    register int j;
    FLOAT* vptr, *vptrend_1, *vptr0;
    FLOAT* Lptr, *Hptr;
    FLOAT* LLptr0, *LHptr0, *HLptr0, *HHptr0;
    FLOAT ix0;

    if (vrowinc < xlength) vrowinc = xlength;
    if (HHrowinc < (xlength >> 1)) HHrowinc = xlength >> 1;
    if (HLrowinc < ((xlength + 1) >> 1)) HLrowinc = (xlength + 1) >> 1;
    if (LHrowinc < (xlength >> 1)) LHrowinc = xlength >> 1;
    if (LLrowinc < ((xlength + 1) >> 1)) LLrowinc = (xlength + 1) >> 1;

    ix0 = 1.0 / x0;

    vptr0 = invector;
    HHptr0 = HH;
    HLptr0 = HL;
    LHptr0 = LH;
    LLptr0 = LL;
    j = 0;
    while (j < ylength - 1) {

        vptr = vptr0;

        Hptr = LHptr0;
        Lptr = LLptr0;

        vptrend_1 = vptr0 + xlength - 1;
        vptr++;
        while (vptr <= vptrend_1) {

            *(Hptr++) = *vptr;
            vptr += 2;
        }
        vptr = vptr0;
        while (vptr < vptrend_1) {
            *(Lptr++) = ix0 * (*vptr);
            vptr += 2;
        }
        if (vptr == vptrend_1) {
            *(Lptr++) = *(vptr++) * ix0;
        }
        vptr0 += vrowinc;
        Hptr = HHptr0;
        Lptr = HLptr0;
        vptr = vptr0;
        vptrend_1 = vptr0 + xlength - 1;
        vptr++;
        while (vptr <= vptrend_1) {
            *(Hptr++) = (*vptr) * x0;
            vptr += 2;
        }
        vptr = vptr0;
        while (vptr < vptrend_1) {
            *(Lptr++) = *vptr;
            vptr += 2;
        }
        if (vptr == vptrend_1) {
            *(Lptr++) = *(vptr++);
        }
        vptr0 += vrowinc;
        HHptr0 += HHrowinc;
        HLptr0 += HLrowinc;
        LHptr0 += LHrowinc;
        LLptr0 += LLrowinc;

        j += 2;
    }

    if (j == ylength - 1) {
        vptr = vptr0;
        Hptr = LHptr0;
        Lptr = LLptr0;
        vptrend_1 = vptr0 + xlength - 1;
        vptr++;
        while (vptr <= vptrend_1) {
            *(Hptr++) = *vptr;
            vptr += 2;
        }
        vptr = vptr0;
        while (vptr < vptrend_1) {
            *(Lptr++) = (*vptr) * ix0;
            vptr += 2;
        }
        if (vptr == vptrend_1) {
            *(Lptr++) = *(vptr++) * ix0;
        }
    }
}

void bioR1__2d_mult(HH, HL, LH, LL, outvector, xlength, ylength, x0,
                    vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc)
    FLOAT* LL,
    *LH, *HL, *HH;
FLOAT* outvector;
int vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc;
FLOAT x0;
int xlength, ylength;

{

    FLOAT* vptr, *vptr0;
    FLOAT* ptr;
    FLOAT* LLptr0, *LHptr0, *HLptr0, *HHptr0;
    FLOAT ix0;
    int Lxlength, Hxlength;
    vptr0 = outvector;

    if (vrowinc < xlength) vrowinc = xlength;
    if (HHrowinc < (xlength >> 1)) HHrowinc = xlength >> 1;
    if (HLrowinc < ((xlength + 1) >> 1)) HLrowinc = (xlength + 1) >> 1;
    if (LHrowinc < (xlength >> 1)) LHrowinc = xlength >> 1;
    if (LLrowinc < ((xlength + 1) >> 1)) LLrowinc = (xlength + 1) >> 1;
    ix0 = 1.0 / x0;

    Lxlength = (xlength + 1) >> 1;
    Hxlength = (xlength) >> 1;

    vptr0 = outvector + vrowinc * (ylength - 1) - 1;
    HHptr0 = HH + HHrowinc * ((ylength - 2) >> 1) - 1;
    HLptr0 = HL + HLrowinc * ((ylength - 2) >> 1) - 1;
    ;
    LHptr0 = LH + LHrowinc * ((ylength - 1) >> 1) - 1;
    LLptr0 = LL + LLrowinc * ((ylength - 1) >> 1) - 1;

    vptr = vptr0 + xlength;

    if (ylength & 1) {

        vptr = vptr0 + xlength;
        ptr = LLptr0 + Lxlength;
        if (xlength & 1) {
            *vptr = *(ptr--) * ix0;
            vptr -= 2;
        } else
            vptr--;
        while (vptr > vptr0) {
            *vptr = *(ptr--) * ix0;
            vptr -= 2;
        }
        LLptr0 -= LLrowinc;

        vptr = vptr0 + xlength;
        ptr = LHptr0 + Hxlength;
        if (xlength & 1) vptr--;
        while (vptr > vptr0) {
            *vptr = *(ptr--);
            vptr -= 2;
        }
        LHptr0 -= LHrowinc;

        vptr0 -= vrowinc;
    }
    /*************************/

    while (vptr > outvector - 1) {
        vptr = vptr0 + xlength;
        ptr = HHptr0 + Hxlength;
        if (xlength & 1) vptr--;
        while (vptr > vptr0) {
            *vptr = *(ptr--) * x0;
            vptr -= 2;
        }
        HHptr0 -= HHrowinc;
        vptr = vptr0 + xlength;
        ptr = HLptr0 + Lxlength;
        if (xlength & 1) {
            *vptr = *(ptr--);
            vptr -= 2;
        } else
            vptr--;
        while (vptr > vptr0) {
            *vptr = *(ptr--);
            vptr -= 2;
        }
        vptr0 -= vrowinc;
        HLptr0 -= HLrowinc;

        /**********************/
        vptr = vptr0 + xlength;
        ptr = LLptr0 + Lxlength;

        if (xlength & 1) {
            *vptr = *(ptr--) * ix0;
            vptr -= 2;
        } else
            vptr--;
        while (vptr > vptr0) {
            *vptr = *(ptr--) * ix0;
            vptr -= 2;
        }

        LLptr0 -= LLrowinc;
        /**************/
        vptr = vptr0 + xlength;
        ptr = LHptr0 + Hxlength;
        if (xlength & 1) vptr--;
        while (vptr > vptr0) {
            *(vptr) = *(ptr--);
            vptr -= 2;
        }
        LHptr0 -= LHrowinc;
        vptr0 -= vrowinc;
        /**********************/
    }
}

/***************************************/

void bioD1_2dskip_mult(FLOAT* invector,
                       FLOAT* LL,
                       int xlength,
                       int ylength,
                       FLOAT x0,
                       int vrowinc,
                       int LLrowinc)

{

    register int j;
    FLOAT* vptr, *vptrend_1;
    FLOAT* vptr0;
    FLOAT* LLptr0;
    FLOAT* LLptr;
    FLOAT ix0;
    if (vrowinc == 0) vrowinc = xlength;
    if (LLrowinc == 0) LLrowinc = (xlength + 1) >> 1;
    ix0 = 1.0 / x0;

    vptr0 = invector;
    LLptr0 = LL;
    j = 0;
    while (j < ylength - 1) {
        vptr = vptr0;
        LLptr = LLptr0;
        vptrend_1 = vptr0 + xlength - 1;
        while (vptr < vptrend_1) {
            *(LLptr++) = *(vptr++) * ix0;
            vptr++;
        }
        if (vptr == vptrend_1) {
            *(LLptr++) = *(vptr++) * ix0;
        }
        vptr0 += vrowinc << 1;
        LLptr0 += LLrowinc;
        j += 2;
    }
    if (j == ylength - 1) {
        vptrend_1 = vptr0 + xlength - 1;
        vptr = vptr0;
        LLptr = LLptr0;
        while (vptr < vptrend_1) {
            *(LLptr++) = *(vptr++) * ix0;
            vptr++;
        }
        if (vptr == vptrend_1) {
            *(LLptr++) = *(vptr++) * ix0;
        }
    }
}

/*************************************************************/
void bioR1_2dskip_mult(FLOAT* LL,
                       FLOAT* vector,
                       int xlength,
                       int ylength,
                       FLOAT x0,
                       int vrowinc,
                       int LLrowinc) {
    FLOAT* LLptr;
    FLOAT* LLptr0;
    FLOAT* vptr;
    FLOAT* vptr0, *vptrmem;
    FLOAT ix0;
    int Lxlength;

    if (vrowinc == 0) vrowinc = xlength;
    if (LLrowinc == 0) LLrowinc = (xlength + 1) >> 1;

    ix0 = 1.0 / x0;

    if (vrowinc < xlength) vrowinc = xlength;
    if (LLrowinc < ((xlength + 1) >> 1)) LLrowinc = (xlength + 1) >> 1;
    ix0 = 1.0 / x0;

    Lxlength = (xlength + 1) >> 1;
    vptr0 = vector + vrowinc * (ylength - 1) - 1;
    LLptr0 = LL + LLrowinc * ((ylength - 1) >> 1) - 1;
    vptr = vptr0 + xlength;

    if (ylength & 1) {
        vptr = vptr0 + xlength;
        LLptr = LLptr0 + Lxlength;
        vptrmem = vptr0 + ((xlength + 1) >> 1);
        memset(vptrmem + 1, 0, (vptr - vptrmem) * sizeof(FLOAT));
        if (xlength & 1) *(vptr--) = *(LLptr--) * ix0;
        while (vptr > vptrmem) {
            vptr--;
            *(vptr--) = *(LLptr--);
        }
        while (vptr > vptr0) {
            *vptr-- = 0;
            *(vptr--) = *(LLptr--);
        }
        vptr0 -= vrowinc;
        LLptr0 -= LLrowinc;
    }

    vptrmem = vector + LLrowinc * (ylength >> 1) - 1;
    memset(vptrmem + 1, 0, (vptr - vptrmem) * sizeof(FLOAT));
    vptrmem += vrowinc;
    while (vptr0 > vptrmem) {
        vptr0 -= vrowinc;
        vptr = vptr0 + xlength;
        LLptr = LLptr0 + Lxlength;

        if (xlength & 1) *(vptr--) = *(LLptr--) * ix0;
        while (vptr > vptr0) {
            vptr--;
            *(vptr--) = *(LLptr--) * ix0;
        }
        vptr0 -= vrowinc;
        LLptr0 -= LLrowinc;
    }
    while (vptr0 > vector - 1) {
        vptr0 -= vrowinc;
        vptr = vptr0 + xlength;
        LLptr = LLptr0 + Lxlength;

        if (xlength & 1) *(vptr--) = *(LLptr--) * ix0;
        vptrmem = vptr0 + (xlength >> 1);
        memset(vptrmem + 1, 0, (vptr - vptrmem) * sizeof(FLOAT));
        while (vptr > vptrmem) {
            vptr--;
            *(vptr--) = *(LLptr--) * ix0;
        }
        while (vptr > vptr0) {
            *vptr-- = 0;
            *(vptr--) = *(LLptr--) * ix0;
        }
        vptr0 -= vrowinc;
        LLptr0 -= LLrowinc;
    }
}
