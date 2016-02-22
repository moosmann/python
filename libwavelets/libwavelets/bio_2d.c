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
#define ALT1

extern void bio_2d_premult(vector, HH, HL, LH, LL, xlength, ylength,
                           rowinc, HHrowinc, HLrowinc, LHrowinc, LLrowinc,
                           Rotlevels, X, ifnotAllskip)
    FLOAT* vector;
FLOAT* HH, *HL, *LH, *LL;
int xlength, ylength;
int rowinc;
int HHrowinc, HLrowinc, LHrowinc, LLrowinc;
int Rotlevels;
FLOAT* X;
int ifnotAllskip;
{

    register int j, k, r;
    FLOAT* vptr0, *vptr1, *vptr2;
    FLOAT a = 0, e = 0, f = 0;
    FLOAT c0 = 0, d0 = 0, b0 = 0;
    FLOAT x0, ix0;
    int Skip;
    FLOAT X1backup;
    FLOAT* vptrend_1;
    FLOAT* LLptr0, *LHptr0, *HLptr0, *HHptr0;

    Skip = 1 - ifnotAllskip;
    if (HHrowinc < (xlength >> 1)) HHrowinc = xlength >> 1;
    if (HLrowinc < ((xlength + 1) >> 1)) HLrowinc = (xlength + 1) >> 1;
    if (LHrowinc < (xlength >> 1)) LHrowinc = xlength >> 1;
    if (LLrowinc < ((xlength + 1) >> 1)) LLrowinc = (xlength + 1) >> 1;

    if (rowinc < xlength) rowinc = xlength;

    HHptr0 = HH;
    HLptr0 = HL;
    LHptr0 = LH;
    LLptr0 = LL;

    /*******************************/
    /****the row selector***************/

    r = 0;
    j = (Rotlevels & 1) - 3;

    /*   20011003          Changing pairity for filter of length 3:   */
    if (Rotlevels == 1) {
        X1backup = X[1];
        X[1] = 1.0 / X1backup;
        j = -3;
    }

    while (!((j >= ylength - 2) && (r >= Rotlevels))) {/****the big outer loop***/

        if ((j >= 0) && (r < Rotlevels - 1)) {
            /*ongoing filtering
doing next level */
            j--;
            r++;

        } else if ((j >= 0) && (r == Rotlevels - 1)) {
            /* doing an ongoin ordering loop */
            r++;
            if (Rotlevels == 1) j--;
        } else {
            /* jumping up few rows in order*/
            j += r + 2;

            /* if(r==Rotlevels)*/
            if ((r == Rotlevels) && (Rotlevels > 1)) j--;
            /* to start
       first level again */
            if (j < ylength)
                r = 0;
            else {
                r = j - ylength + 1;
                j -= r;
                /*     if(r==Rotlevels){
       */
                if ((r == Rotlevels) && (Rotlevels > 1)) {

                    j++;
                }
            }
        }

        x0 = X[r];

        /*************end row selector *****************/

        if ((r < Rotlevels - 1) || ((r == Rotlevels - 1) && !Skip)) {
            /***********************************/
            ix0 = 1.0 / x0;

            if ((j == 0) && (ylength > 1)) {
                vptr1 = vector;
                vptr0 = vector + rowinc;
                if (xlength > 1) {
                    a = vptr0[1]; /* virtuall vptr0[-1]  */
                    f = vptr1[1]; /* vrtually  vptr1[-1]  */
                    a *= x0;
                } else {
                    a = 0;
                    f = 0;
                }
                k = 0;

                while (k < xlength - 1) {
                    b0 = vptr1[k];
                    d0 = f;
                    c0 = a + vptr0[k];
                    f = vptr1[k + 1];
                    a = vptr0[k + 1];
                    a *= x0;
                    d0 += f;
                    c0 += a;
                    vptr0[k] = c0;
                    d0 += 2 * c0;
                    vptr1[k + 1] = 2 * a + f;
                    vptr0[k + 1] = a;
                    vptr1[k] = ix0 * b0 + d0;
                    k += 2;
                }

                if (k == xlength - 1) {
                    b0 = vptr1[k];
                    d0 = f;
                    c0 = 2 * a + vptr0[k];
                    d0 += f;
                    vptr0[k] = c0;
                    d0 += 2 * c0;
                    vptr0[k] = c0;
                    vptr1[k] = ix0 * b0 + d0;
                }
            }

            if (j == -1) {/*y_pairity: 1  doing line 0 alone  */

                vptr0 = vector;
                a = vptr0[0];
                a *= x0;
                vptr0[0] = a;
                k = 1;

                while (k < xlength - 1) {
                    c0 = a + vptr0[k];
                    a = vptr0[k + 1];
                    a *= x0;
                    c0 += a;
                    vptr0[k] = c0;
                    vptr0[k + 1] = a;
                    k += 2;
                }
                if (k == xlength - 1) {
                    c0 = 2 * a + vptr0[k];
                    vptr0[k] = c0;
                }
            }

            if ((j > 0) && (j < ylength - 1)) {/* starts main outer loop doing */
                vptr2 = vector + (j - 1) * rowinc;
                vptr1 = vector + j * rowinc;
                vptr0 = vector + (j + 1) * rowinc;

                if ((j & 1) == 0) {/* left boundary reflecting one point */
                    if (xlength > 1) {
                        a = vptr0[1]; /* virtuall vptr0[-1]  */
                        f = vptr1[1]; /* vrtually  vptr1[-1]  */
                        a *= x0;
                    } else {
                        a = 0;
                        f = 0;
                    }
                    k = 0;
                }
                if ((j & 1) == 1) {/* left boundary: first single point at 0 */
                    a = vptr0[0];  /* no reflection is needed  */
                    a *= x0;
                    f = vptr1[0];
                    vptr1[0] = a + f + vptr2[0];
                    vptr0[0] = a;
                    k = 1;
                }

                /******************************************/

                /***************HERE IS THE MAIN INNER LOOP TO OPTIMIZE : :  ***********/
                /*  it is running  about  m x N x M /4   times   for one level 
 biorthogonal filter  2m+1    on  N  x   M  images           */

                while (k < xlength - 1) {
                    e = vptr2[k];
                    b0 = vptr1[k];
                    d0 = f;
                    c0 = a + vptr0[k];
                    f = vptr1[k + 1];
                    a = vptr0[k + 1];
                    d0 += e;
                    a *= x0;
                    d0 += f;
                    c0 += a;
                    vptr0[k] = c0;
                    d0 += c0;
#ifndef ALT
                    vptr1[k + 1] = a + f + vptr2[k + 1];
                    vptr0[k + 1] = a;
                    vptr1[k] = ix0 * b0 + d0;

#else
                    vptr1[k + 1] = a + f + x0 * vptr2[k + 1];
                    vptr1[k] = b0 + x0 * d0;
#endif
                    k += 2;
                }

                /**************END MAIN INNER LOOP***************************/
                /**************************************************/
                if (k == xlength - 1) {/* rigth boundary  if one point at end */
                    e = vptr2[k];      /* remains we reflect to get an extra point*/
                    b0 = vptr1[k];
                    d0 = f;
                    c0 = 2 * a + vptr0[k]; /* thus the x0*a from last time will be the*/
                    d0 += e;               /* same so 2*a is a + last a   */
                    d0 += f;               /* we use same f as last time */
                    vptr0[k] = c0;
                    d0 += c0;
                    vptr1[k] = ix0 * b0 + d0;
                }
            }

            if ((j == ylength - 1) && (j > 0)) {/* in case the last line remins to be done
                          it is done alone  */

                vptr1 = vector + j * rowinc;
                vptr2 = vector + (j - 1) * rowinc;

                if ((j & 1) == 0) {
                    if (xlength > 1) {
                        f = vptr1[1]; /* vrtually  vptr1[-1]  */
                    } else {
                        f = 0;
                    }
                    k = 0;
                }
                if ((j & 1) == 1) {
                    f = vptr1[0];
                    vptr1[0] = f + 2 * vptr2[0];
                    k = 1;
                }

                while (k < xlength - 1) {
                    e = vptr2[k];
                    b0 = vptr1[k];
                    d0 = f;
                    f = vptr1[k + 1];
                    d0 += 2 * e;
                    d0 += f;
                    vptr1[k + 1] = f + 2 * vptr2[k + 1];
                    vptr1[k] = ix0 * b0 + d0;
                    k += 2;
                }
                if (k == xlength - 1) {
                    e = vptr2[k];
                    b0 = vptr1[k];
                    d0 = 2 * (e + f);
                    vptr1[k] = b0 * ix0 + d0;
                }
            }

            if ((ylength == 1) && (j == 0)) {/* in case the last line remins to be done
                          it is done alone  */
                vptr1 = vector;

                if (xlength > 1) {
                    f = vptr1[1]; /* vrtually  vptr1[-1]  */
                } else {
                    f = 0;
                }
                k = 0;

                while (k < xlength - 1) {
                    b0 = vptr1[k];
                    d0 = f;
                    f = vptr1[k + 1];
                    d0 += f;
                    vptr1[k] = ix0 * b0 + d0;
                    k += 2;
                }
                if (k == xlength - 1) {
                    b0 = vptr1[k];
                    d0 = 2 * f;
                    vptr1[k] = b0 * ix0 + d0;
                }
            }

        } else

            if (Skip && r == Rotlevels - 1) {
            ix0 = 1.0 / x0;

            if ((j == 0) && (ylength > 1)) {
                vptr1 = vector;
                vptr0 = vector + rowinc;
                if (xlength > 1) {
                    a = vptr0[1]; /* virtuall vptr0[-1]  */
                    f = vptr1[1]; /* vrtually  vptr1[-1]  */
                    a *= x0;
                } else {
                    a = 0;
                    f = 0;
                }
                k = 0;

                while (k < xlength - 1) {
                    b0 = vptr1[k];
                    d0 = f;
                    c0 = a + vptr0[k];
                    f = vptr1[k + 1];
                    a = vptr0[k + 1];
                    a *= x0;
                    d0 += f;
                    c0 += a;
                    vptr0[k] = c0;
                    d0 += 2 * c0;

                    vptr1[k] = ix0 * b0 + d0;
                    k += 2;
                }

                if (k == xlength - 1) {
                    b0 = vptr1[k];
                    d0 = f;
                    c0 = 2 * a + vptr0[k];
                    d0 += f;
                    vptr0[k] = c0;
                    d0 += 2 * c0;
                    vptr0[k] = c0;
                    vptr1[k] = ix0 * b0 + d0;
                }
            }

            if (j == -1) {/*y_pairity: 1  doing line 0 alone  */

                vptr0 = vector;
                a = vptr0[0];
                a *= x0;
                vptr0[0] = a;
                k = 1;

                while (k < xlength - 1) {
                    c0 = a + vptr0[k];
                    a = vptr0[k + 1];
                    a *= x0;
                    c0 += a;
                    vptr0[k] = c0;

                    k += 2;
                }
                if (k == xlength - 1) {
                    c0 = 2 * a + vptr0[k];
                    vptr0[k] = c0;
                }
            }

            if ((j > 0) && (j < ylength - 1)) {/* starts main outer loop doing */
                vptr2 = vector + (j - 1) * rowinc;
                vptr1 = vector + j * rowinc;
                vptr0 = vector + (j + 1) * rowinc;

                if ((j & 1) == 0) {/* left boundary reflecting one point */
                    if (xlength > 1) {
                        a = vptr0[1]; /* virtuall vptr0[-1]  */
                        f = vptr1[1]; /* vrtually  vptr1[-1]  */
                        a *= x0;
                    } else {
                        a = 0;
                        f = 0;
                    }
                    k = 0;
                }

                /******************************************/

                /***************HERE IS THE MAIN INNER LOOP TO OPTIMIZE : :  ***********/
                /*  it is running  about  m x N x M /4   times   for one level 
 biorthogonal filter  2m+1    on  N  x   M  images           */

                while (k < xlength - 1) {
                    /********main loop 1*********/
                    e = vptr2[k];
                    b0 = vptr1[k];
                    d0 = f;
                    c0 = a + vptr0[k];
                    f = vptr1[k + 1];
                    a = vptr0[k + 1];
                    d0 += e;
                    a *= x0;
                    d0 += f;
                    c0 += a;
                    vptr0[k] = c0;
                    d0 += c0;
                    vptr1[k] = ix0 * b0 + d0;
                    k += 2;
                    /******************/
                }

                /**************END MAIN INNER LOOP***************************/
                /**************************************************/
                if (k == xlength - 1) {/* rigth boundary  if one point at end */
                    e = vptr2[k];      /* remains we reflect to get an extra point*/
                    b0 = vptr1[k];
                    d0 = f;
                    c0 = 2 * a + vptr0[k]; /* thus the x0*a from last time will be the*/
                    d0 += e;               /* same so 2*a is a + last a   */
                    d0 += f;               /* we use same f as last time */
                    vptr0[k] = c0;
                    d0 += c0;
                    vptr1[k] = ix0 * b0 + d0;
                }
            }
            if ((j > 0) && (j == ylength - 1)) {/* in case the last line remins to be done
                          it is done alone  */

                vptr1 = vector + j * rowinc;
                vptr2 = vector + (j - 1) * rowinc;

                if ((j & 1) == 0) {
                    if (xlength > 1) {
                        f = vptr1[1]; /* vrtually  vptr1[-1]  */
                    } else {
                        f = 0;
                    }
                    k = 0;
                }

                while (k < xlength - 1) {
                    e = vptr2[k];
                    b0 = vptr1[k];
                    d0 = f;
                    f = vptr1[k + 1];
                    d0 += 2 * e;
                    d0 += f;
                    vptr1[k] = ix0 * b0 + d0;
                    k += 2;
                }
                if (k == xlength - 1) {
                    e = vptr2[k];
                    b0 = vptr1[k];
                    d0 = 2 * (e + f);
                    vptr1[k] = b0 * ix0 + d0;
                }
            }
            if ((j == 0) && (ylength == 1)) {/* in case the last line remins to be done
                          it is done alone  */

                vptr1 = vector;

                if ((j & 1) == 0) {
                    if (xlength > 1) {
                        f = vptr1[1]; /* vrtually  vptr1[-1]  */
                    } else {
                        f = 0;
                    }
                    k = 0;
                }

                while (k < xlength - 1) {
                    b0 = vptr1[k];
                    d0 = f;
                    f = vptr1[k + 1];
                    d0 += f;
                    vptr1[k] = ix0 * b0 + d0;
                    k += 2;
                }
                if (k == xlength - 1) {
                    b0 = vptr1[k];
                    d0 = 2 * f;
                    vptr1[k] = b0 * ix0 + d0;
                }
            }

        } else

            /**************begin ordering loops*************************************/

            if (Skip) {
            if (j < ylength - 1) {
                ix0 = 1.0 / x0;

                vptr1 = vector + j * rowinc;
                vptrend_1 = vptr1 + xlength - 1;
                vptr0 = LLptr0;
                while (vptr1 < vptrend_1) {
                    *(vptr0++) = ix0 * (*vptr1);
                    vptr1 += 2;
                }
                if (vptr1 == vptrend_1) {
                    *(vptr0++) = *(vptr1++) * ix0;
                }
                LLptr0 += LLrowinc;
            }

            if (j == ylength - 1) {
                ix0 = 1.0 / x0;

                vptr1 = vector + j * rowinc;
                vptrend_1 = vptr1 + xlength - 1;
                vptr0 = LLptr0;
                while (vptr1 < vptrend_1) {
                    *(vptr0++) = ix0 * (*vptr1);
                    vptr1 += 2;
                }
                if (vptr1 == vptrend_1) {
                    *(vptr0++) = *(vptr1++) * ix0;
                }
            }
        } else {
            if (j < ylength - 1) {
                ix0 = 1.0 / x0;

                vptr1 = vector + j * rowinc + 1;
                vptr2 = LHptr0;
                vptrend_1 = vptr1 + xlength - 1;
                /**************************/
                while (vptr1 < vptrend_1) {
                    *(vptr2++) = *vptr1;
                    vptr1 += 2;
                }
                /***********************/
                vptr1 = vector + j * rowinc;
                vptrend_1 = vptr1 + xlength - 1;
                vptr0 = LLptr0;
                while (vptr1 < vptrend_1) {
#ifndef ALT
                    *(vptr0++) = ix0 * (*vptr1);
#else
                    *(vptr0++) = (*vptr1);
#endif
                    vptr1 += 2;
                }
                if (vptr1 == vptrend_1) {
                    *(vptr0++) = *(vptr1++) * ix0;
                }

                vptr2 = HHptr0;
                vptr0 = HLptr0;
                vptr1 = vector + (j + 1) * rowinc + 1;
                vptrend_1 = vptr1 + xlength - 1;
                /*****************************/
                while (vptr1 < vptrend_1) {
#ifndef ALT
                    *(vptr2++) = x0 * (*vptr1);
                    ;

#else
                    *(vptr2++) = (*vptr1);
#endif
                    vptr1 += 2;
                }
                /************************************/
                vptr1 = vector + (j + 1) * rowinc;
                vptrend_1 = vptr1 + xlength - 1;

                while (vptr1 < vptrend_1) {

                    *(vptr0++) = *vptr1;
                    vptr1 += 2;
                }
                if (vptr1 == vptrend_1) {
                    *(vptr0++) = *(vptr1++);
                }

                /**************************************/
                HHptr0 += HHrowinc;
                HLptr0 += HLrowinc;
                LHptr0 += LHrowinc;
                LLptr0 += LLrowinc;
            }

            if (j == ylength - 1) {
                ix0 = 1.0 / x0;

                vptr1 = vector + j * rowinc + 1;
                vptr2 = LHptr0;
                vptrend_1 = vptr1 + xlength - 1;
                /**************************/
                while (vptr1 < vptrend_1) {
                    *(vptr2++) = *vptr1;
                    vptr1 += 2;
                }
                /***********************/
                vptr1 = vector + j * rowinc;
                vptrend_1 = vptr1 + xlength - 1;
                vptr0 = LLptr0;
                while (vptr1 < vptrend_1) {
                    *(vptr0++) = ix0 * (*vptr1);
                    vptr1 += 2;
                }
                if (vptr1 == vptrend_1) {
                    *(vptr0++) = *(vptr1++) * ix0;
                }
            }
        }

    } /*end big j loop */
    if (Rotlevels == 1) {
        X[1] = X1backup;
    }
}

void bio_2d_postmult(vector, xlength, ylength, rowinc, Rotlevels, X, pairity, ifnotAllskip)
    FLOAT* vector;
int xlength, ylength;
int rowinc;
int Rotlevels;
FLOAT* X;
int ifnotAllskip;

int pairity;

{

    register int j, k, r;
    FLOAT* vptr0, *vptr1, *vptr2;
    FLOAT a = 0, e = 0, f = 0;
    FLOAT c0 = 0, d0 = 0, b0 = 0;
    FLOAT x0, ix0;
    int Skip;

    Skip = 1 - ifnotAllskip;
    if (rowinc < xlength) rowinc = xlength;

    /*******************************/
    /****the row selector***************/

    r = 0;
    j = -pairity - 2;

    while (!((j >= ylength - 2) && (r >= Rotlevels - 1))) {

        if ((j >= 0) && (r < Rotlevels - 1)) {
            j--;
            r++;
        } else {
            j += r + 2;
            if (j < ylength)
                r = 0;
            else {
                r = j - ylength + 1;
                j -= r;
            }
        }

        x0 = X[r];

        /*printf("r=%d, j=%d\n",r,j);*/
        /*************end row selector *****************/

        if ((r < Rotlevels) && ((r > 0) || (!Skip))) {

            /***** below is the body of the big j-loop  ******************/
            ix0 = 1.0 / x0;

            if ((j == 0) && (ylength > 2)) {
                vptr1 = vector;
                vptr0 = vector + rowinc;

                if (xlength > 1) {
                    a = vptr0[1];
                    f = vptr1[1];
                } else {
                    a = 0;
                    f = 0;
                }
                k = 0;

                while (k < xlength - 1) {
                    b0 = vptr1[k];
                    d0 = f;
                    c0 = a + vptr0[k];
                    f = vptr1[k + 1];
                    a = vptr0[k + 1];
                    d0 += f;
                    c0 += a;
                    vptr0[k] = c0;
                    d0 += 2 * c0;

                    vptr1[k + 1] = 2 * a + f;
                    vptr1[k] = ix0 * (b0 + d0);
                    k += 2;
                }

                if (k == xlength - 1) {/* rigth boundary  if o ne point at end */
                    b0 = vptr1[k];
                    d0 = f;
                    c0 = 2 * a + vptr0[k]; /* thus the x0*a from last time will be the*/
                    d0 += f;               /* we use same f as last time */
                    vptr0[k] = c0;
                    d0 += 2 * c0;
                    vptr1[k] = ix0 * (b0 + d0);
                }
            }

            if ((j == 0) && (ylength == 2)) {
                vptr1 = vector;
                vptr0 = vector + rowinc;

                if (xlength > 1) {
                    a = vptr0[1];
                    f = vptr1[1];
                } else {
                    a = 0;
                    f = 0;
                }
                k = 0;

                while (k < xlength - 1) {
                    b0 = vptr1[k];
                    d0 = f;
                    c0 = a + vptr0[k];
                    f = vptr1[k + 1];
                    a = vptr0[k + 1];
                    d0 += f;
                    c0 += a;
                    vptr0[k] = c0;
                    d0 += 2 * c0;

                    vptr1[k + 1] = 2 * a + f;
                    vptr0[k + 1] = x0 * a;
                    vptr1[k] = ix0 * (b0 + d0);
                    k += 2;
                }

                if (k == xlength - 1) {/* rigth boundary  if o ne point at end */
                    b0 = vptr1[k];
                    d0 = f;
                    c0 = 2 * a + vptr0[k]; /* thus the x0*a from last time will be the*/
                    d0 += f;               /* we use same f as last time */
                    vptr0[k] = c0;
                    d0 += 2 * c0;
                    vptr1[k] = ix0 * (b0 + d0);
                }
            }

            if ((j == -1) && (ylength > 1)) {/*y_pairity: 1  doing line 0 alone  */

                vptr0 = vector;
                a = vptr0[0];
                k = 1;

                while (k < xlength - 1) {
                    c0 = a + vptr0[k];
                    a = vptr0[k + 1];
                    c0 += a;
                    vptr0[k] = c0;
                    k += 2;
                }
                if (k == xlength - 1) {
                    c0 = 2 * a + vptr0[k];
                    vptr0[k] = c0;
                }
            }

            if ((j == -1) && (ylength == 1)) {/*y_pairity: 1  doing line 0 alone  */

                vptr0 = vector;
                a = vptr0[0];
                vptr0[0] = x0 * a;
                k = 1;

                while (k < xlength - 1) {
                    c0 = a + vptr0[k];
                    a = vptr0[k + 1];
                    c0 += a;
                    vptr0[k] = c0;
                    vptr0[k + 1] = x0 * a;
                    k += 2;
                }
                if (k == xlength - 1) {
                    c0 = 2 * a + vptr0[k];
                    vptr0[k] = c0;
                }
            }

            /*000000000000000000000000000000000000000000000000000000000000000000000000*/
            if ((j > 0) && (j < ylength - 2)) {/* starts main outer loop doing */
                vptr2 = vector + (j - 1) * rowinc;
                vptr1 = vector + j * rowinc;
                vptr0 = vector + (j + 1) * rowinc;
                if ((j & 1) == 0) {/* left boundary reflecting one point */
                    if (xlength > 1) {
                        a = vptr0[1]; /* virtuall vptr0[-1]  */
                        f = vptr1[1]; /* vrtually  vptr1[-1]  */
                    } else {
                        a = 0;
                        f = 0;
                    }
                    k = 0;
                }
                if ((j & 1) == 1) {/* left boundary: first single point at 0 */
                    a = vptr0[0];  /* no reflection is needed  */
                    f = vptr1[0];

                    e = vptr2[0];
                    vptr1[0] = a + f + e;
                    vptr2[0] = x0 * e;

                    vptr0[0] = a;
                    k = 1;
                }

                /******************************************/

                /***************HERE IS THE MAIN INNER LOOP TO OPTIMIZE : :  ***********/
                /*  it is running  about  m x N x M /4   times   for one level 
 biorthogonal filter  2m+1    on  N  x   M  images           */

                while (k < xlength - 1) {
                    e = vptr2[k];
                    b0 = vptr1[k];
                    d0 = f;
                    c0 = a + vptr0[k];
                    f = vptr1[k + 1];
                    a = vptr0[k + 1];
                    d0 += e;
                    d0 += f;
                    c0 += a;
                    vptr0[k] = c0;
                    d0 += c0;
                    e = vptr2[k + 1];
                    vptr1[k + 1] = a + f + e;
                    vptr2[k + 1] = x0 * e;
                    vptr1[k] = ix0 * (b0 + d0);
                    k += 2;
                }

                /**************END MAIN INNER LOOP***************************/
                /**************************************************/
                if (k == xlength - 1) {/* rigth boundary  if one point at end */
                    e = vptr2[k];      /* remains we reflect to get an extra point*/
                    b0 = vptr1[k];
                    d0 = f;
                    c0 = 2 * a + vptr0[k]; /* thus the x0*a from last time will be the*/
                    d0 += e;               /* same so 2*a is a + last a   */
                    d0 += f;               /* we use same f as last time */
                    vptr0[k] = c0;
                    d0 += c0;
                    vptr1[k] = ix0 * (b0 + d0);
                }
            }
            /*00000000000000000000000000000000000000000000000000000*/

            if ((j > 0) && (j == ylength - 2)) {/* starts main outer loop doing */
                vptr2 = vector + (j - 1) * rowinc;
                vptr1 = vector + j * rowinc;
                vptr0 = vector + (j + 1) * rowinc;
                if ((j & 1) == 0) {/* left boundary reflecting one point */
                    if (xlength > 1) {
                        a = vptr0[1]; /* virtuall vptr0[-1]  */
                        f = vptr1[1]; /* vrtually  vptr1[-1]  */
                    } else {
                        a = 0;
                        f = 0;
                    }
                    k = 0;
                }
                if ((j & 1) == 1) {/* left boundary: first single point at 0 */
                    a = vptr0[0];  /* no reflection is needed  */
                    f = vptr1[0];

                    e = vptr2[0];
                    vptr1[0] = a + f + e;
                    vptr2[0] = x0 * e;
                    vptr0[0] = x0 * a;
                    k = 1;
                }

                /******************************************/

                /***************HERE IS THE MAIN INNER LOOP TO OPTIMIZE : :  ***********/
                /*  it is running  about  m x N x M /4   times   for one level 
 biorthogonal filter  2m+1    on  N  x   M  images           */

                while (k < xlength - 1) {
                    e = vptr2[k];
                    b0 = vptr1[k];
                    d0 = f;
                    c0 = a + vptr0[k];
                    f = vptr1[k + 1];
                    a = vptr0[k + 1];
                    d0 += e;
                    d0 += f;
                    c0 += a;
                    vptr0[k] = c0;
                    d0 += c0;
                    e = vptr2[k + 1];
                    vptr1[k + 1] = a + f + e;
                    vptr2[k + 1] = x0 * e;
                    vptr0[k + 1] = x0 * a;
                    vptr1[k] = ix0 * (b0 + d0);
                    k += 2;
                }

                /**************END MAIN INNER LOOP***************************/
                /**************************************************/
                if (k == xlength - 1) {/* rigth boundary  if one point at end */
                    e = vptr2[k];      /* remains we reflect to get an extra point*/
                    b0 = vptr1[k];
                    d0 = f;
                    c0 = 2 * a + vptr0[k]; /* thus the x0*a from last time will be the*/
                    d0 += e;               /* same so 2*a is a + last a   */
                    d0 += f;               /* we use same f as last time */
                    d0 += c0;
                    vptr0[k] = c0;
                    vptr1[k] = ix0 * (b0 + d0);
                }
            }

            if ((j > 0) && (j == ylength - 1)) {/* in case the last line remins to be done
                          it is done alone  */

                vptr1 = vector + j * rowinc;
                vptr2 = vector + (j - 1) * rowinc;

                if ((j & 1) == 0) {
                    if (xlength > 1) {
                        f = vptr1[1]; /* vrtually  vptr1[-1]  */
                    } else {
                        f = 0;
                    }
                    k = 0;
                }
                if ((j & 1) == 1) {
                    f = vptr1[0];

                    e = vptr2[0];
                    vptr1[0] = f + 2 * e;
                    vptr2[0] = x0 * e;

                    k = 1;
                }

                while (k < xlength - 1) {
                    e = vptr2[k];
                    b0 = vptr1[k];
                    d0 = f;
                    f = vptr1[k + 1];
                    d0 += 2 * e;
                    d0 += f;

                    e = vptr2[k + 1];
                    vptr1[k + 1] = f + 2 * e;
                    vptr2[k + 1] = x0 * e;

                    vptr1[k] = ix0 * (b0 + d0);
                    k += 2;
                }
                if (k == xlength - 1) {
                    e = vptr2[k];
                    b0 = vptr1[k];
                    d0 = 2 * (e + f);
                    vptr1[k] = ix0 * (b0 + d0);
                }
            }
            if ((j == 0) && (ylength == 1)) {/* in case the last line remins to be done
                          it is done alone  */

                vptr1 = vector;

                if ((j & 1) == 0) {
                    if (xlength > 1) {
                        f = vptr1[1]; /* vrtually  vptr1[-1]  */
                    } else {
                        f = 0;
                    }
                    k = 0;
                }
                if ((j & 1) == 1) {
                    f = vptr1[0];
                    k = 1;
                }

                while (k < xlength - 1) {
                    b0 = vptr1[k];
                    d0 = f;
                    f = vptr1[k + 1];
                    d0 += f;
                    vptr1[k] = ix0 * (b0 + d0);
                    k += 2;
                }
                if (k == xlength - 1) {
                    b0 = vptr1[k];
                    d0 = 2 * f;
                    vptr1[k] = ix0 * (b0 + d0);
                }
            }

        } else if ((j + 2) & 1) {/*if(Skip)) */

            /***** below is the body of the big j-loop  ******************/
            ix0 = 1.0 / x0;

            /*
if(j==0){
  vptr1=vector;    
  vptr0=vector+ rowinc;


      if(xlength>1){   
      a=vptr0[1];
    f=0; 
  }else{
    a=0;
    f=0;
  }
  k=0;  
  
    while(k< xlength-1){
      c0 =a; 
      a=vptr0[k+1];
      c0  += a;
      vptr0[k]= a;
      d0 =2*c0;
      vptr1[k+1]=2*a;
      vptr0[k+1]=a;
      vptr1[k]=ix0*d0;
      k +=2;
  }




  if(k==xlength-1){       
    c0 =2*a+vptr0[k];     
    vptr0[k]=c0;
    d0 =2*c0;
    vptr1[k]=ix0*d0;
  }
*/

            if (j == -1) {/*y_pairity: 1  doing line 0 alone  */

                vptr0 = vector;
                a = vptr0[0];
                k = 1;

                while (k < xlength - 1) {
                    c0 = a;
                    a = vptr0[k + 1];
                    c0 += a;
                    vptr0[k] = c0;
                    vptr0[k + 1] = a;
                    k += 2;
                }
                if (k == xlength - 1) {
                    c0 = 2 * a;
                    vptr0[k] = c0;
                }
                /*next lines will be line 1 and 2 */
            }

            if ((j > 0) && (j < ylength - 2)) {/* starts main outer loop doing */
                vptr2 = vector + (j - 1) * rowinc;
                vptr1 = vector + j * rowinc;
                vptr0 = vector + (j + 1) * rowinc;
                /* j should always be odd here   */
                /*  if((j&1)==0){      
    if(xlength>1){   
      a=vptr0[1]; 

  }else  {
    a=0;
  }
  k=0;  
  }
*/

                /*  if((j&1)==1){  */
                a = vptr0[0];
                e = vptr2[0];
                vptr1[0] = a + e;
                vptr2[0] = x0 * e;

                k = 1;
                /*  } */

                /******************************************/

                /***************HERE IS THE MAIN INNER LOOP TO OPTIMIZE : :  ***********/
                /*  it is running  about  m x N x M /4   times   for one level 
 biorthogonal filter  2m+1    on  N  x   M  images           */

                while (k < xlength - 1) {
                    d0 = vptr2[k];
                    c0 = a;
                    a = vptr0[k + 1];
                    c0 += a;
                    vptr0[k] = c0;
                    d0 += c0;
                    e = vptr2[k + 1];
                    vptr1[k + 1] = a + e;
                    vptr2[k + 1] = x0 * e;

                    vptr1[k] = ix0 * d0;
                    k += 2;
                }

                /**************END MAIN INNER LOOP***************************/
                /**************************************************/
                if (k == xlength - 1) {/* rigth boundary  if one point at end */
                    e = vptr2[k];      /* remains we reflect to get an extra point*/
                    c0 = 2 * a;        /* thus the x0*a from last time will be the*/
                    d0 = e;            /* same so 2*a is a + last a   */
                    vptr0[k] = c0;
                    d0 += c0;
                    vptr1[k] = ix0 * d0;
                }
            }

            if ((j > 0) && (j == ylength - 2)) {/* starts main outer loop doing */
                vptr2 = vector + (j - 1) * rowinc;
                vptr1 = vector + j * rowinc;
                vptr0 = vector + (j + 1) * rowinc;

                /* j is always odd here */
                /* if((j&1)==0){      

   if(xlength>1){   
      a=vptr0[1]; 
   }else{
    a=0;
   }
  k=0;  
  }

  if((j&1)==1){
*/
                a = vptr0[0];

                e = vptr2[0];
                vptr1[0] = a + e;
                vptr2[0] = x0 * e;
                vptr0[0] = x0 * a;
                k = 1;
                /*}   */
            }

            if ((j == ylength - 1) && (j > 0)) {/* in case the last line remins to be done
                          it is done alone  */

                vptr1 = vector + j * rowinc;
                vptr2 = vector + (j - 1) * rowinc;

                e = vptr2[0];
                vptr1[0] = 2 * e;
                vptr2[0] = x0 * e;

                k = 1;

                while (k < xlength - 1) {
                    e = vptr2[k];
                    d0 = 2 * e;
                    e = vptr2[k + 1];
                    vptr1[k + 1] = 2 * e;
                    vptr2[k + 1] = x0 * e;

                    vptr1[k] = ix0 * (b0 + d0);
                    k += 2;
                }
                if (k == xlength - 1) {
                    e = vptr2[k];
                    vptr1[k] = ix0 * b0;
                }
            }
        }

    } /*end big j loop */
}
