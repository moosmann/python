/* file bio13579.2dy.c */

/*       Fast algorithm for a symmetric biorthorgonal Low and High
          pass filter  of length  1,3,5,7 and 9
         i dimension two.
         This is a control file. The used procedures are defined
         in the files bio_3.c  (the local operations)
                      bio1.3d.c ordering in HH.LH.HL and LL subbands

                                 Algorithm and code developped 
                                               by
                                      Jan-Olov Stromberg

				      KTH, Stockholm, Sweden
                     Fast Mathematical Algorithms&Hardware, Hamden, CT, USA


                             Preliminary  version by October 18 , 1997

*/

#include "bio.h"
#include "bio_parameters.h"

#define NORMALIZATION9 (0.763)
#define NORMALIZATION7 (0.98)
#define NORMALIZATION5 (0.5)
#define NORMALIZATION3 (1.0)
int k;

/**********************************************/

void bioD9_2d_y(invector, H, L, xlength, ylength,
                ifnotAllskip)
    INTYPE* invector;
FLOAT* H;
FLOAT* L;
int xlength, ylength;
int ifnotAllskip;

{

    FLOAT x0 = -1.586134342059;
    FLOAT x1 = -0.052980118573;
    FLOAT x2 = 0.882911075529;
    FLOAT x3 = 0.443506852045;

    /* Rotlevels=4; */
    FLOAT X0, X1, X2, X3, X4;
    int pairity;
    FLOAT Norm;

    if (0 || ifnotAllskip) {
        X0 = x0;
        X1 = x0 * x1;
        X2 = x1 * x2;
        X3 = x2 * x3;
        X4 = NORMALIZATION9 / x3;
    } else {
        X0 = x0;
        X1 = x0 * x1;
        X2 = x1 * x2;
        X3 = 1.0;
        X4 = NORMALIZATION9 / x2;
    }

    Norm = 1.0 / (X0 * X1 * X2 * X3 * X4);
    if (ylength > 1) Norm *= X2 * X0;

    /*warning invector will be overwritten */
    pairity = 1;
    bio_2d_y_premult(invector, xlength, ylength, X0, pairity);
    pairity = 0;
    bio_2d_y_premult(invector, xlength, ylength, X1, pairity);
    pairity = 1;
    bio_2d_y_premult(invector, xlength, ylength, X2, pairity);
    pairity = 0;
    bio_2d_y_premult(invector, xlength, ylength, X3, pairity);
    if (ifnotAllskip) {
        bioD1__2d_y(invector, H, L, xlength, ylength, X4, Norm);
    } else
        bioD1_skip_2d_y(invector, L, xlength, ylength, X4, Norm);
}

/*********************************************/
void bioR9_2d_y(H, L, outvector, xlength, ylength,
                ifnotAllskip)
    FLOAT* H;
FLOAT* L;
OUTTYPE* outvector;
int xlength, ylength;
int ifnotAllskip;
{
    FLOAT ix0 = 1.586134342059;
    FLOAT ix1 = 0.052980118573;
    FLOAT ix2 = -0.882911075529;
    FLOAT ix3 = -0.443506852045;

    FLOAT iX0, iX1, iX2, iX3, iX4;
    int pairity;
    FLOAT Norm;

    if (0 || ifnotAllskip) {
        iX4 = ix3 / NORMALIZATION9;
        iX3 = 1.0 / (ix3 * ix2);
        iX2 = 1.0 / (ix2 * ix1);
        iX1 = 1.0 / (ix1 * ix0);
        iX0 = 1.0 / ix0;
    } else {
        iX4 = ix2 / NORMALIZATION9;
        iX3 = 1.0;
        iX2 = 1.0 / (ix2 * ix1);
        iX1 = 1.0 / (ix1 * ix0);
        iX0 = 1.0 / ix0;
    }

    Norm = 1.0 / (iX0 * iX1 * iX2 * iX3 * iX4);
    if (ylength > 1) Norm *= iX2 * iX0;

    if (ifnotAllskip) {
        bioR1__2d_y(H, L, outvector, xlength, ylength, iX4, Norm);
        pairity = 0;
        bio_2d_y_postmult(outvector, xlength, ylength, iX3, pairity);
    } else
        bioR1_skip_2d_y(L, outvector, xlength, ylength, iX4, Norm);
    pairity = 1;
    bio_2d_y_postmult(outvector, xlength, ylength, iX2, pairity);
    pairity = 0;
    bio_2d_y_postmult(outvector, xlength, ylength, iX1, pairity);
    pairity = 1;
    bio_2d_y_postmult(outvector, xlength, ylength, iX0, pairity);
}

/*********************************************/
void bioD7_2d_y(invector, H, L, xlength, ylength,
                ifnotAllskip)
    INTYPE* invector;
FLOAT* H;
FLOAT* L;
int xlength, ylength;
int ifnotAllskip;
{

    FLOAT x0 = 0.2;
    FLOAT x1 = -0.357142857136;
    FLOAT x2 = 0.21;

    FLOAT X0, X1, X2, X3;
    int pairity;
    FLOAT Norm;

    if (0 || ifnotAllskip) {
        X0 = x0;
        X1 = x0 * x1;
        X2 = x1 * x2;
        X3 = NORMALIZATION7 / x2;
    } else {
        X0 = x0;
        X1 = x0 * x1;
        X2 = 1;
        X3 = NORMALIZATION7 / x1;
    }

    Norm = 1.0 / (X0 * X1 * X2 * X3);
    if (ylength > 1) Norm *= X1;

    /*warning invector will be overwritten */
    pairity = 0;
    bio_2d_y_premult(invector, xlength, ylength, X0, pairity);
    pairity = 1;
    bio_2d_y_premult(invector, xlength, ylength, X1, pairity);
    pairity = 0;
    bio_2d_y_premult(invector, xlength, ylength, X2, pairity);
    if (ifnotAllskip)
        bioD1__2d_y(invector, H, L, xlength, ylength, X3, Norm);
    else
        bioD1_skip_2d_y(invector, L, xlength, ylength, X3, Norm);
}
/**********************************************/
void bioR7_2d_y(H, L, outvector, xlength, ylength,
                ifnotAllskip)
    FLOAT* H;
FLOAT* L;
OUTTYPE* outvector;
int xlength, ylength;
int ifnotAllskip;
{

    FLOAT ix0 = -0.2;
    FLOAT ix1 = 0.357142857136;
    FLOAT ix2 = -0.21;

    FLOAT iX0, iX1, iX2, iX3;
    int pairity;
    FLOAT Norm;

    if (0 || ifnotAllskip) {
        iX3 = ix2 / NORMALIZATION7;
        iX2 = 1.0 / (ix2 * ix1);
        iX1 = 1.0 / (ix1 * ix0);
        iX0 = 1.0 / ix0;
    } else {
        iX3 = ix1 / NORMALIZATION7;
        iX2 = 1.0;
        iX1 = 1.0 / (ix1 * ix0);
        iX0 = 1.0 / ix0;
    }

    Norm = 1.0 / (iX0 * iX1 * iX2 * iX3);
    if (ylength > 1) Norm *= iX1;

    if (ifnotAllskip) {
        bioR1__2d_y(H, L, outvector, xlength, ylength, iX3, Norm);
        pairity = 0;
        bio_2d_y_postmult(outvector, xlength, ylength, iX2, pairity);
    } else
        bioR1_skip_2d_y(L, outvector, xlength, ylength, iX3, Norm);
    pairity = 1;
    bio_2d_y_postmult(outvector, xlength, ylength, iX1, pairity);
    pairity = 0;
    bio_2d_y_postmult(outvector, xlength, ylength, iX0, pairity);
}

/*****************************************************/
void bioD5_2d_y(invector, H, L, xlength, ylength,
                ifnotAllskip)
    INTYPE* invector;
FLOAT* H;
FLOAT* L;
int xlength, ylength;
int ifnotAllskip;

{

    FLOAT x0 = -0.50;
    FLOAT x1 = 0.25;

    FLOAT X0, X1, X2;
    int pairity;
    FLOAT Norm;

    if (0 || ifnotAllskip) {
        X0 = x0;
        X1 = x0 * x1;
        X2 = NORMALIZATION5 / x1;
    } else {
        X0 = x0;
        X1 = 1.0;
        X2 = NORMALIZATION5 / x0;
    }

    Norm = 1.0 / (X0 * X1 * X2);
    if (ylength > 1) Norm *= X0;

    /*warning invector will be overwritten */
    pairity = 1;
    bio_2d_y_premult(invector, xlength, ylength, X0, pairity);
    pairity = 0;
    bio_2d_y_premult(invector, xlength, ylength, X1, pairity);
    if (ifnotAllskip)
        bioD1__2d_y(invector, H, L, xlength, ylength, X2, Norm);
    else {
        bioD1_skip_2d_y(invector, L, xlength, ylength, X2, Norm);
    }
}
/******************************************************/
void bioR5_2d_y(H, L, outvector, xlength, ylength,
                ifnotAllskip)
    FLOAT* H;
FLOAT* L;
OUTTYPE* outvector;
int xlength, ylength;
int ifnotAllskip;
{

    FLOAT ix0 = 0.50;
    FLOAT ix1 = -0.25;

    FLOAT iX0, iX1, iX2;
    int pairity;
    FLOAT Norm;

    if (ifnotAllskip) {
        iX2 = ix1 / NORMALIZATION5;
        iX1 = 1.0 / (ix1 * ix0);
        iX0 = 1.0 / ix0;
    } else {
        iX2 = ix0 / NORMALIZATION5;
        iX1 = 1.0;
        iX0 = 1.0 / ix0;
    }

    Norm = 1.0 / (iX0 * iX1 * iX2);
    if (ylength > 1) Norm *= iX0;

    if (ifnotAllskip) {
        bioR1__2d_y(H, L, outvector, xlength, ylength, iX2, Norm);
        pairity = 0;
        bio_2d_y_postmult(outvector, xlength, ylength, iX1, pairity);
    } else
        bioR1_skip_2d_y(L, outvector, xlength, ylength, iX2, Norm);
    pairity = 1;
    bio_2d_y_postmult(outvector, xlength, ylength, iX0, pairity);
}

/********************************/
void bioD3_2d_y(invector, H, L, xlength, ylength,
                ifnotAllskip)
    INTYPE* invector;
FLOAT* H;
FLOAT* L;
int xlength, ylength;
int ifnotAllskip;
{
    FLOAT x0 = -0.5;

    FLOAT X0 = x0;
    FLOAT X1 = NORMALIZATION3 / x0;

    int pairity;

    FLOAT Norm;
    Norm = 1.0 / (X0 * X1);
    /* norm = X0 / X1 ==1;*/

    /*warning invector will be overwritten */
    pairity = 0;
    bio_2d_y_premult(invector, xlength, ylength, X0, pairity);
    /* bio_2d_y(invector,xlength,ylength,tempvector,x0,pairity);*/

    if (ifnotAllskip)
        bioD1__2d_y(invector, H, L, xlength, ylength, X1, Norm);
    else
        bioD1_skip_2d_y(invector, L, xlength, ylength, X1, Norm);
}
/**********************************************/
void bioR3_2d_y(H, L, outvector, xlength, ylength,
                ifnotAllskip)
    FLOAT* H;
FLOAT* L;
OUTTYPE* outvector;
int xlength, ylength;
int ifnotAllskip;
{
    FLOAT ix0 = 0.5;

    FLOAT iX1;
    FLOAT iX0;

    int pairity;
    FLOAT Norm;

    if (ifnotAllskip) {
        iX1 = ix0 / NORMALIZATION3;
        iX0 = 1.0 / ix0;
    } else {
        iX1 = 1.0 / NORMALIZATION3;
        iX0 = 1.0;
    }

    Norm = 1.0 / (iX0 * iX1);

    if (ifnotAllskip) {
        bioR1__2d_y(H, L, outvector, xlength, ylength, iX1, Norm);
        pairity = 0;
        bio_2d_y_postmult(outvector, xlength, ylength, iX0, pairity);
    } else
        bioR1_skip_2d_y(L, outvector, xlength, ylength, iX1, Norm);
}

/***************************************/
void bioD1_2d_y(invector, H, L, xlength, ylength,
                ifnotAllskip)
    INTYPE* invector;
FLOAT* H;
FLOAT* L;
int xlength, ylength;
int ifnotAllskip;
{
    float X0 = 1.0;
    float Norm = 1.0;

    if (ifnotAllskip)
        bioD1__2d_y(invector, H, L, xlength, ylength, X0, Norm);
    else
        bioD1_skip_2d_y(invector, L, xlength, ylength, X0, Norm);
}
/*************************************************/
void bioR1_2d_y(H, L, outvector, xlength, ylength,
                ifnotAllskip)
    FLOAT* H;
FLOAT* L;
OUTTYPE* outvector;
int xlength, ylength;
int ifnotAllskip;
{
    float iX0 = 1.0;

    float Norm = 1.0;

    if (ifnotAllskip)
        bioR1__2d_y(H, L, outvector, xlength, ylength, iX0, Norm);
    else
        bioR1_skip_2d_y(L, outvector, xlength, ylength, iX0, Norm);
}
