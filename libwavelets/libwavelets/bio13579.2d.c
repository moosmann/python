/* file bio13579.2d.c */

/*       Fast algorithm for a symmetric biorthorgonal Low and High
          pass filter  of length  1,3,5,7 and 9
         i dimension two.
         This is a control file. The used procedures are defined
         in the files bio_2.c  (the local operations)
                      bio1.2d.c ordering in HH.LH.HL and LL subbands

                                 Algorithm and code developped 
                                               by
                                      Jan-Olov Stromberg


                                Univiversity of Tromso, Norway,
                     Fast Mathematical Algorithms&Hardware, Hamden, CT, USA


                             Last updated version by May 11, 1997
*/

#include <time.h>
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
#define PRINT 0

#define NORMALIZATION9 (0.763)
#define NORMALIZATION7 (0.98)
#define NORMALIZATION5 (0.5)

/**********************************************/
void bioD9_2d(invector, HH, HL, LH, LL, xlength, ylength,
              vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc,
              ifnotAllskip)
    FLOAT* invector;
FLOAT* LL, *LH, *HL, *HH;
int xlength, ylength;
int vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc;
int ifnotAllskip;
{

    FLOAT x0 = -1.586134342059;
    FLOAT x1 = -0.052980118573;
    FLOAT x2 = 0.882911075529;
    FLOAT x3 = 0.443506852045;
    int Rotlevels = 4;
    FLOAT X[6];

    X[0] = x0;
    X[1] = x0 * x1;
    X[2] = x1 * x2;
    X[3] = x2 * x3;
    X[4] = NORMALIZATION9 / x3;

    bio_2d_premult(invector, HH, HL, LH, LL, xlength, ylength,
                   vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc,
                   Rotlevels, X, ifnotAllskip);
}

/*******************************************/

void bioR9_2d(HH, HL, LH, LL, outvector, xlength, ylength,
              vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc,
              ifnotAllskip)

    FLOAT* LL,
    *LH, *HL, *HH;
FLOAT* outvector;
int xlength, ylength;
int vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc;
int ifnotAllskip;
{
    FLOAT ix0 = 1.586134342059;
    FLOAT ix1 = 0.052980118573;
    FLOAT ix2 = -0.882911075529;
    FLOAT ix3 = -0.443506852045;

    int Rotlevels;
    FLOAT X[4];
    int pairity;

    if (ifnotAllskip) {
        Rotlevels = 4;
        X[0] = 1.0 / (ix3 * ix2);
        X[1] = 1.0 / (ix2 * ix1);
        X[2] = 1.0 / (ix1 * ix0);
        X[3] = 1.0 / ix0;
#if (PRINT)
        time0 = clock();
#endif
        bioR1__2d_mult(HH, HL, LH, LL, outvector, xlength, ylength, ix3 / NORMALIZATION9,
                       vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc);
#if (PRINT)
        time1 = clock();
        printf(" HighLowpass-ordering  : %4.2f s\n", ((FLOAT)(time1 - time0) / CLOCKS_PER_SEC));
#endif
        pairity = 0;
#if (PRINT)
        time0 = clock();
#endif

        bio_2d_postmult(outvector, xlength, ylength, vrowinc, Rotlevels, X, pairity, 1);
#if (PRINT)
        time1 = clock();
        printf(" filterrotations : %4.2f s\n", ((FLOAT)(time1 - time0) / CLOCKS_PER_SEC));
#endif

    } else

        {
        Rotlevels = 3;
        X[0] = 1.0 / (ix2 * ix1);
        X[1] = 1.0 / (ix1 * ix0);
        X[2] = 1.0 / ix0;
#if (PRINT)
        time0 = clock();
#endif
        bioR1_2dskip_mult(LL, outvector, xlength, ylength, (1.0 / (ix2 * NORMALIZATION9)),
                          vrowinc, LLrowinc);
#if (PRINT)
        time1 = clock();
        printf(" HighLowpass-ordering  : %4.2f s\n", ((FLOAT)(time1 - time0) / CLOCKS_PER_SEC));
#endif

        pairity = 1;
#if (PRINT)
        time0 = clock();
#endif
        bio_2d_postmult(outvector, xlength, ylength, vrowinc, Rotlevels, X, pairity, 0);
#if (PRINT)
        time1 = clock();
        printf(" skiprotations : %4.2f s\n", ((FLOAT)(time1 - time0) / CLOCKS_PER_SEC));
#endif
    }
}

/*********************************************/

void bioD7_2d(invector, HH, HL, LH, LL, xlength, ylength,
              vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc,
              ifnotAllskip)

    FLOAT* invector;
FLOAT* LL, *LH, *HL, *HH;
int xlength, ylength;
int vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc;
int ifnotAllskip;
{

    FLOAT x0 = 0.2;
    FLOAT x1 = -0.357142857136;
    FLOAT x2 = 0.21;

    int Rotlevels = 3;
    FLOAT X[5];

    X[0] = x0;
    X[1] = x0 * x1;
    X[2] = x1 * x2;
    X[3] = NORMALIZATION7 / x2;

    bio_2d_premult(invector, HH, HL, LH, LL, xlength, ylength,
                   vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc,
                   Rotlevels, X, ifnotAllskip);
}

/**********************************************/
void bioR7_2d(HH, HL, LH, LL, outvector, xlength, ylength,
              vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc,
              ifnotAllskip)

    FLOAT* LL,
    *LH, *HL, *HH;
FLOAT* outvector;
int xlength, ylength;
int vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc;
int ifnotAllskip;
{

    FLOAT ix0 = -0.2;
    FLOAT ix1 = 0.357142857136;
    FLOAT ix2 = -0.21;

    int Rotlevels;
    int pairity;
    FLOAT X[3];

#if (PRINT)
    long time0, time1;
#endif

    if (ifnotAllskip) {
        Rotlevels = 3;
        X[0] = 1.0 / (ix2 * ix1);
        X[1] = 1.0 / (ix1 * ix0);
        X[2] = 1.0 / ix0;

#if (PRINT)
        time0 = clock();
#endif
        bioR1__2d_mult(HH, HL, LH, LL, outvector, xlength, ylength, ix2 / NORMALIZATION7,
                       vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc);
#if (PRINT)
        time1 = clock();
        printf(" HighLowpass-ordering  : %4.2f s\n", ((FLOAT)(time1 - time0) / CLOCKS_PER_SEC));
#endif
        pairity = 0;
#if (PRINT)
        time0 = clock();
#endif
        bio_2d_postmult(outvector, xlength, ylength, vrowinc, Rotlevels, X, pairity, 1);
#if (PRINT)
        time1 = clock();
        printf(" filterrotations : %4.2f s\n", ((FLOAT)(time1 - time0) / CLOCKS_PER_SEC));
#endif

    } else {
        Rotlevels = 2;

        X[0] = 1.0 / (ix1 * ix0);
        X[1] = 1.0 / ix0;
        bioR1_2dskip_mult(LL, outvector, xlength, ylength, (1.0 / (ix1 * NORMALIZATION7)),
                          vrowinc, LLrowinc);
        pairity = 1;
        bio_2d_postmult(outvector, xlength, ylength, vrowinc, Rotlevels, X, pairity, 0);
    }
}

/*****************************************************/

void bioD5_2d(invector, HH, HL, LH, LL, xlength, ylength,
              vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc,
              ifnotAllskip)

    FLOAT* invector;
FLOAT* LL, *LH, *HL, *HH;
int xlength, ylength;
int vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc;
int ifnotAllskip;
{
    FLOAT x0 = -0.5;
    FLOAT x1 = 0.25;
    int Rotlevels = 2;
    FLOAT X[4];

    X[0] = x0;
    X[1] = x0 * x1;
    X[2] = NORMALIZATION5 / x1;
    bio_2d_premult(invector, HH, HL, LH, LL, xlength, ylength,
                   vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc,
                   Rotlevels, X, ifnotAllskip);
}
/******************************************************/
void bioR5_2d(HH, HL, LH, LL, outvector, xlength, ylength,
              vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc,
              ifnotAllskip)

    FLOAT* LL,
    *LH, *HL, *HH;
FLOAT* outvector;
int xlength, ylength;
int vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc;
int ifnotAllskip;
{

    FLOAT ix0 = 0.5;
    FLOAT ix1 = -0.25;
    int Rotlevels;
    int pairity;
    FLOAT X[2];

    if (ifnotAllskip) {

        Rotlevels = 2;
        X[0] = 1.0 / (ix1 * ix0);
        X[1] = 1.0 / ix0;
#if (PRINT)
        time0 = clock();
#endif
        bioR1__2d_mult(HH, HL, LH, LL, outvector, xlength, ylength, ix1 / NORMALIZATION5,
                       vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc);
#if (PRINT)
        time1 = clock();
        printf(" HighLowpass-ordering  : %4.2f s\n", ((FLOAT)(time1 - time0) / CLOCKS_PER_SEC));
#endif
        pairity = 0;
#if (PRINT)
        time0 = clock();
#endif
        bio_2d_postmult(outvector, xlength, ylength, vrowinc, Rotlevels, X, pairity, 1);
#if (PRINT)
        time1 = clock();
        printf(" filterrotations : %4.2f s\n", ((FLOAT)(time1 - time0) / CLOCKS_PER_SEC));
#endif
    } else {
        Rotlevels = 1;
        X[0] = 1.0 / ix0;

        bioR1_2dskip_mult(LL, outvector, xlength, ylength, (1.0 / (ix0 * NORMALIZATION5)),
                          vrowinc, LLrowinc);
        pairity = 1;
        bio_2d_postmult(outvector, xlength, ylength, vrowinc, Rotlevels, X, pairity, 0);
    }
}
/********************************/

void bioD3_2d(invector, HH, HL, LH, LL, xlength, ylength,
              vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc,
              ifnotAllskip)

    FLOAT* invector;
FLOAT* LL, *LH, *HL, *HH;
int xlength, ylength;
int vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc;
int ifnotAllskip;
{
    FLOAT x0 = -0.5;
    int Rotlevels = 1;
    FLOAT X[3];
    X[0] = x0;
    X[1] = 1.0 / x0;

    bio_2d_premult(invector, HH, HL, LH, LL, xlength, ylength,
                   vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc,
                   Rotlevels, X, ifnotAllskip);
}
/**********************************************/

void bioR3_2d(HH, HL, LH, LL, outvector, xlength, ylength,
              vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc,
              ifnotAllskip)

    FLOAT* LL,
    *LH, *HL, *HH;
FLOAT* outvector;
int xlength, ylength;
int vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc;
int ifnotAllskip;
{
    FLOAT ix0 = 0.5;
    int Rotlevels;
	int pairity;
#if (PRINT)
    long time0, time1;
#endif
    FLOAT X[1];
    FLOAT x0;

    if (ifnotAllskip) {
        Rotlevels = 1;
        X[0] = 1.0 / ix0;
        x0 = X[0];
#if (PRINT)
        time0 = clock();
#endif
        bioR1__2d_mult(HH, HL, LH, LL, outvector, xlength, ylength,
                       /*ix0*/ x0, vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc);
#if (PRINT)
        time1 = clock();
        printf(" HighLowpass-ordering  : %4.2f s\n", ((FLOAT)(time1 - time0) / CLOCKS_PER_SEC));
#endif
        /*
pairity=0;
*/
        pairity = 1;
#if (PRINT)
        time0 = clock();
#endif

        bio_2d_postmult(outvector, xlength, ylength, vrowinc, Rotlevels, X, pairity, 1);
#if (PRINT)
        time1 = clock();
        printf(" filterrotations : %4.2f s\n", ((FLOAT)(time1 - time0) / CLOCKS_PER_SEC));
#endif
    } else {
        Rotlevels = 0;
#if (PRINT)
        time0 = clock();
#endif
        bioR1_2dskip_mult(LL, outvector, xlength, ylength, (1.0),
                          vrowinc, LLrowinc);
#if (PRINT)
        time1 = clock();
        printf(" skipordering : %4.2f s\n", ((FLOAT)(time1 - time0) / CLOCKS_PER_SEC));
#endif
    }
}
/*******************************/

void bioD1_2d(invector, HH, HL, LH, LL, xlength, ylength,
              vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc,
              ifnotAllskip)

    FLOAT* invector;
FLOAT* LL, *LH, *HL, *HH;
int vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc;
int xlength, ylength;
int ifnotAllskip;

{
    if (ifnotAllskip)
        bioD1__2d_mult(invector, HH, HL, LH, LL, xlength, ylength, 1.0,
                       vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc);
    else
        bioD1_2dskip_mult(invector, LL, xlength, ylength, 1.0,
                          vrowinc, LLrowinc);
}

/********************************/

void bioR1_2d(HH, HL, LH, LL, outvector, xlength, ylength,
              vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc,
              ifnotAllskip)

    FLOAT* LL,
    *LH, *HL, *HH;
FLOAT* outvector;
int xlength, ylength;
int vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc;
int ifnotAllskip;
{
    if (ifnotAllskip)
        bioR1__2d_mult(HH, HL, LH, LL, outvector, xlength, ylength, 1.0,
                       vrowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc);
    else
        bioR1_2dskip_mult(LL, outvector, xlength, ylength, 1.0,
                          vrowinc, LLrowinc);
}

/*****************************************/
