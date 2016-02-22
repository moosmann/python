/* file bio_#d.c */
/*       Fast algorithm for a biorthorgonal Low and High pass filter
   main local fiteroperation in space 


que
   Algorithm and code developped 
   by
   Jan-Olov Stromberg


   KTH, Stockholm, Sweden
   Fast Mathematical Algorithms&Hardware, Hamden, CT, USA


   Preliminary version  by October 18, 1998
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
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

void bio_3d_premult(FLOAT* Vector,
                    int xlength,
                    int ylength,
                    int zlength,
                    FLOAT x1,
                    int pairity) { /* start of function */

    register int j, k, s;
    FLOAT* vptr00, *vptr01, *vptr02;
    FLOAT* vptr10, *vptr11, *vptr12;
    FLOAT* vptr20, *vptr21, *vptr22;
    FLOAT a = 0, e = 0, f = 0, g = 0;
    FLOAT c = 0, d = 0, b = 0;
    FLOAT a1 = 0, f1 = 0;
    FLOAT c1 = 0;
    FLOAT B, E, G, H, F;
    FLOAT /* x0, */ x_1, x_2;

    int FrameSize;
    FLOAT* vector;
    FrameSize = xlength * ylength;

    /* x0  = 1.0; */
    x_1 = 1 / x1;
    x_2 = x_1 * x_1;

    s = 0;
    vptr10 = NULL; /*compiler gives warning otherwise */
    vptr11 = NULL;
    H = 0.0;
    if ((pairity == 0) && (zlength > 1)) { /* starts main z- loop doing  two lines each time  */
        vector = Vector + s * FrameSize;

        if ((ylength > 2)) {
            j = 0;
            vptr11 = vector;
            vptr10 = vector + xlength;

            vptr01 = vptr11 + FrameSize;
            ;
            vptr00 = vptr10 + FrameSize;
            ;

            if (xlength > 1) {
                b = x_1 * vptr01[0];
                B = /* x0* */ vptr00[0];
                f = /* x0* */ vptr01[1];
                a = x1 * vptr00[1];
                c = 2 * a + B;
                vptr00[0] = c;
                d = 2 * (c + f);
                F = 2 * a + f;
                vptr01[1] = F;
                vptr00[1] = a;
                H = d + b;
                vptr01[0] = H;
                b = x_2 * vptr11[0];
                f1 = x_1 * vptr11[1];
                a1 = /* x0* */ vptr10[1];
                vptr10[1] = a1;
                c1 = 2 * a1 + x_1 * vptr10[0];
                d = 2 * (c1 + f1);
                B = 2 * F;
                vptr11[1] = 2 * a1 + f1 + B;
                d += 2 * H;
                vptr10[0] = c1;
                vptr11[0] = b + d;
                k = 2;
            }

            else {
                f = /* x0* */ vptr01[0];
                a = x1 * vptr00[0];
                F = 2 * a + f;
                vptr01[0] = F;
                vptr00[0] = a;
                f1 = x_1 * vptr11[0];
                a1 = /* x0* */ vptr10[0];
                vptr10[0] = a1;
                B = 2 * F;
                vptr11[0] = 2 * a1 + f1 + B;
                k = 1;
            }

            while (k < xlength - 1) {

                b = x_1 * vptr01[k];
                d = f;
                B = /* x0* */ vptr00[k];
                c = a + B;
                f = /* x0* */ vptr01[k + 1];
                a = x1 * vptr00[k + 1];
                d += f;
                c += a;
                vptr00[k] = c;
                d += 2 * c;
                F = 2 * a + f;
                vptr01[k + 1] = F;
                vptr00[k + 1] = a;
                H = d + b;
                vptr01[k] = H;
                b = x_2 * vptr11[k];
                d = f1;
                c1 = a1 + x_1 * vptr10[k];
                f1 = x_1 * vptr11[k + 1];
                a1 = /* x0* */ vptr10[k + 1];
                vptr10[k + 1] = a1;
                d += f1;
                c1 += a1;
                d += 2 * c1;
                B = 2 * F;
                vptr11[k + 1] = 2 * a1 + f1 + B;
                d += 2 * H;
                vptr10[k] = c1;
                vptr11[k] = b + d;
                k += 2;
            }
            if (k == xlength - 1) {
                d = 2 * f;
                b = x_1 * vptr01[k];
                B = /* x0* */ vptr00[k];
                c = 2 * a + B;
                vptr00[k] = c;
                d += 2 * c;
                H = d + b;
                vptr01[k] = H;
                b = x_2 * vptr11[k];
                d = 2 * f1;
                c1 = 2 * a1 + x_1 * vptr10[k];
                d += 2 * c1;
                d += 2 * H;
                vptr10[k] = c1;
                vptr11[k] = b + d;
            }

            j = 2; /*next lines will be line 0 and 1 */
        } else if (ylength == 2) {
            j = 0;
            vptr11 = vector;
            vptr10 = vector + xlength;

            vptr01 = vptr11 + FrameSize;
            ;
            vptr00 = vptr10 + FrameSize;
            ;

            if (xlength > 1) {

                b = x_1 * vptr01[0];
                B = /* x0* */ vptr00[0];
                f = /* x0* */ vptr01[1];
                a = x1 * vptr00[1];
                c = 2 * a + B;
                vptr00[0] = c;
                d = 2 * (c + f);
                F = 2 * a + f;
                vptr01[1] = F;
                vptr00[1] = a;
                H = d + b;
                vptr01[0] = H;
                b = x_2 * vptr11[0];
                f1 = x_1 * vptr11[1];
                a1 = /* x0* */ vptr10[1];
                vptr10[1] = a1 + 2 * a;
                c1 = 2 * a1 + x_1 * vptr10[0];
                d = 2 * (c1 + f1);
                c1 += 2 * c;
                B = 2 * F;
                vptr11[1] = 2 * a1 + f1 + B;
                d += 2 * H;
                vptr10[0] = c1;
                vptr11[0] = b + d;

                k = 2;

            }

            else {
                f = /* x0* */ vptr01[0];
                a = x1 * vptr00[0];
                F = 2 * a + f;
                vptr01[0] = F;
                vptr00[0] = a;
                f1 = x_1 * vptr11[0];
                a1 = /* x0* */ vptr10[0];
                vptr10[0] = a1 + 2 * a;
                B = 2 * F;
                vptr11[0] = 2 * a1 + f1 + B;
                k = 1;
            }

            while (k < xlength - 1) {

                b = x_1 * vptr01[k];
                d = f;
                B = /* x0* */ vptr00[k];
                c = a + B;
                f = /* x0* */ vptr01[k + 1];
                a = x1 * vptr00[k + 1];
                d += f;
                c += a;
                vptr00[k] = c;
                d += 2 * c;
                F = 2 * a + f;
                vptr01[k + 1] = F;
                vptr00[k + 1] = a;
                H = d + b;
                vptr01[k] = H;
                b = x_2 * vptr11[k];
                d = f1;
                c1 = a1 + x_1 * vptr10[k];
                f1 = x_1 * vptr11[k + 1];
                a1 = /* x0* */ vptr10[k + 1];
                vptr10[k + 1] = a1 + 2 * a;
                d += f1;
                c1 += a1;
                d += 2 * c1;
                c1 += 2 * c;
                B = 2 * F;
                vptr11[k + 1] = 2 * a1 + f1 + B;
                d += 2 * H;
                vptr10[k] = c1;
                vptr11[k] = b + d;
                k += 2;
            }
            if (k == xlength - 1) {
                b = x_1 * vptr01[k];
                d = 2 * f;
                B = /* x0* */ vptr00[k];
                c = 2 * a + B;
                vptr00[k] = c;
                d += 2 * c;
                F = 2 * a + f;
                H = d + b;
                vptr01[k] = H;
                b = x_2 * vptr11[k];
                d = 2 * f1;
                c1 = 2 * a1 + x_1 * vptr10[k];
                d += 2 * c1;
                c1 += 2 * c;
                d += 2 * H;
                vptr10[k] = c1;
                vptr11[k] = b + d;
            }

            j = 2; /*next lines will be line 0 and 1 */

        }

        else /* if (ylength==1))*/
        {

            vptr10 = vector;
            vptr00 = vptr10 + FrameSize;

            if (xlength > 1) {
                B = /* x0* */ vptr00[0];
                a = x1 * vptr00[1];
                c = 2 * a + B;
                vptr00[0] = c;
                vptr00[1] = a;
                a1 = /* x0* */ vptr10[1];
                vptr10[1] = a1 + 2 * a;
                c1 = 2 * a1 + x_1 * vptr10[0];
                c1 += 2 * c;
                d += 2 * H;
                vptr10[0] = c1;
                k = 2;
            } else { /*xlength ==1 */
                a = x1 * vptr00[0];
                vptr00[0] = a;
                a1 = /* x0* */ vptr10[0];
                vptr10[0] = a1 + 2 * a;
                k = 1;
            }

            while (k < xlength - 1) {
                B = /* x0* */ vptr00[k];
                c = a + B;
                a = x1 * vptr00[k + 1];
                c += a;
                vptr00[k] = c;
                vptr00[k + 1] = a;
                c1 = a1 + x_1 * vptr10[k];
                a1 = /* x0* */ vptr10[k + 1];
                vptr10[k + 1] = a1 + 2 * a;
                c1 += a1;
                c1 += 2 * c;
                vptr10[k] = c1;
                k += 2;
            }

            if (k == xlength - 1) {
                B = /* x0* */ vptr00[k];
                c = 2 * a + B;
                vptr00[k] = c;
                c1 = 2 * a1 + x_1 * vptr10[k];
                c1 += 2 * c;
                vptr10[k] = c1;
            }
            j = 1; /*next lines will be line 1 and 2 */
        }
        while (j < ylength - 2) { /* starts main outer loop doing 
				   two lines each time  */

            vptr12 = vector + (j - 1) * xlength;
            vptr11 = vector + j * xlength;
            vptr10 = vector + (j + 1) * xlength;

            vptr02 = vptr12 + FrameSize;
            ;
            vptr01 = vptr11 + FrameSize;
            ;
            vptr00 = vptr10 + FrameSize;
            ;

            if (xlength > 1) {

                E = vptr02[0];
                b = x_1 * vptr01[0];
                B = /* x0* */ vptr00[0];
                f = /* x0* */ vptr01[1];
                a = x1 * vptr00[1];
                d = E + 2 * f;
                c = B + 2 * a;
                vptr00[0] = c;
                d += c;
                G = vptr02[1];
                F = a + f + G;
                vptr01[1] = F;
                vptr00[1] = a;
                H = d + b;
                vptr01[0] = H;
                e = vptr12[0];
                b = x_2 * vptr11[0];
                f1 = x_1 * vptr11[1];
                a1 = /* x0* */ vptr10[1];
                vptr10[1] = a1;
                d = e + 2 * f1;
                c1 = x_1 * vptr10[0] + 2 * a1;
                d += c1;
                B = 2 * F;
                g = vptr12[1];
                vptr11[1] = a1 + f1 + g + B;
                d += 2 * H;
                vptr10[0] = c1;
                vptr11[0] = b + d;
                vptr12[0] = e + 2 * E;
                vptr12[1] = g + 2 * G;
                k = 2;
            }

            else /* if (xlength==1)  pairity =1 loop */
            {
                d = f;
                f = /* x0* */ vptr01[0];
                a = x1 * vptr00[0];
                G = vptr02[0];
                F = a + f + G;
                vptr01[0] = F;
                vptr00[0] = a;
                f1 = x_1 * vptr11[0];
                a1 = /* x0* */ vptr10[0];
                vptr10[0] = a1;
                ;
                B = 2 * F;
                g = vptr12[0];
                vptr11[0] = a1 + f1 + g + B;
                vptr12[0] = g + 2 * G;
                k = 1;
            }

            /**********HERE IS THE MAIN INNER LOOP TO OPTIMIZE : :  ***********/
            /*  it is running  about  m x N x M /4   times   for one level 
			biorthogonal filter  2m+1    on  N  x   M  images           */
            while (k < xlength - 1) {
                E = vptr02[k];
                b = x_1 * vptr01[k];
                d = f;
                B = /* x0* */ vptr00[k];
                c = a + B;
                f = /* x0* */ vptr01[k + 1];
                a = x1 * vptr00[k + 1];
                d += E;
                d += f;
                c += a;
                vptr00[k] = c;
                d += c;
                G = vptr02[k + 1];
                F = a + f + G;
                vptr01[k + 1] = F;
                vptr00[k + 1] = a;
                H = d + b;
                vptr01[k] = H;
                e = vptr12[k];
                b = x_2 * vptr11[k];
                d = f1;
                c1 = a1 + x_1 * vptr10[k];
                f1 = x_1 * vptr11[k + 1];
                a1 = /* x0* */ vptr10[k + 1];
                d += e;
                vptr10[k + 1] = a1;
                d += f1;
                c1 += a1;
                d += c1;
                B = 2 * F;
                g = vptr12[k + 1];
                vptr11[k + 1] = a1 + f1 + +g + B;
                d += 2 * H;
                vptr10[k] = c1;
                vptr11[k] = b + d;
                vptr12[k] = e + 2 * E;
                vptr12[k + 1] = g + 2 * G;
                k += 2;
            }
            /**************END MAIN INNER LOOP***************************/
            if (k == xlength - 1) { /* rigth boundary  if one point at end */
                                    /* remains we reflect to get an extra point */

                E = vptr02[k];
                b = x_1 * vptr01[k];
                d = 2 * f;
                B = /* x0* */ vptr00[k];
                c = 2 * a + B;
                d += E;
                vptr00[k] = c;
                d += c;
                H = d + b;
                vptr01[k] = H;
                e = vptr12[k];
                b = x_2 * vptr11[k];
                d = 2 * f1;
                c1 = 2 * a1 + x_1 * vptr10[k];
                d += e;
                d += c1;
                d += 2 * H;
                vptr10[k] = c1;
                vptr11[k] = b + d;
                vptr12[k] = e + 2 * E;
            }

            /************************/

            j += 2;
        }
        if (j == ylength - 2) { /* starts main outer loop doing 
				   two lines each time  */

            vptr12 = vector + (j - 1) * xlength;
            vptr11 = vector + j * xlength;
            vptr10 = vector + (j + 1) * xlength;

            vptr02 = vptr12 + FrameSize;
            ;
            vptr01 = vptr11 + FrameSize;
            ;
            vptr00 = vptr10 + FrameSize;
            ;

            if (xlength > 1) {

                E = vptr02[0];
                b = x_1 * vptr01[0];
                B = /* x0* */ vptr00[0];
                f = /* x0* */ vptr01[1];
                a = x1 * vptr00[1];
                d = E + 2 * f;
                c = B + 2 * a;
                vptr00[0] = c;
                d += c;
                G = vptr02[1];
                F = a + f + G;
                vptr01[1] = F;
                vptr00[1] = a;
                H = d + b;
                vptr01[0] = H;
                e = vptr12[0];
                b = x_2 * vptr11[0];
                f1 = x_1 * vptr11[1];
                a1 = /* x0* */ vptr10[1];
                vptr10[1] = a1 + 2 * a;
                d = e + 2 * f1;
                c1 = x_1 * vptr10[0] + 2 * a1;
                d += c1;
                c1 += 2 * c;
                B = 2 * F;
                g = vptr12[1];
                vptr11[1] = a1 + f1 + +g + B;
                d += 2 * H;
                vptr10[0] = c1;
                vptr11[0] = b + d;
                vptr12[0] = e + 2 * E;
                vptr12[1] = g + 2 * G;
                k = 2;
            }

            else /* if (xlength==1)  pairity =1 loop */
            {
                d = f;
                f = /* x0* */ vptr01[0];
                a = x1 * vptr00[0];
                G = vptr02[0];
                F = a + f + G;
                vptr01[0] = F;
                vptr00[0] = a;
                f1 = x_1 * vptr11[0];
                a1 = /* x0* */ vptr10[0];
                vptr10[0] = a1 + 2 * a;
                B = 2 * F;
                g = vptr12[0];
                vptr11[0] = a1 + f1 + +g + B;
                vptr12[0] = g + 2 * G;
                k = 1;
            }

            /**********HERE IS THE MAIN INNER LOOP TO OPTIMIZE : :  ***********/
            /*  it is running  about  m x N x M /4   times   for one level 
			biorthogonal filter  2m+1    on  N  x   M  images           */
            while (k < xlength - 1) {
                E = vptr02[k];
                b = x_1 * vptr01[k];
                d = f;
                B = /* x0* */ vptr00[k];
                c = a + B;
                f = /* x0* */ vptr01[k + 1];
                a = x1 * vptr00[k + 1];
                d += E;
                d += f;
                c += a;
                vptr00[k] = c;
                d += c;
                G = vptr02[k + 1];
                F = a + f + G;
                vptr01[k + 1] = F;
                vptr00[k + 1] = a;
                H = d + b;
                vptr01[k] = H;
                e = vptr12[k];
                b = x_2 * vptr11[k];
                d = f1;
                c1 = a1 + x_1 * vptr10[k];
                f1 = x_1 * vptr11[k + 1];
                a1 = /* x0* */ vptr10[k + 1];
                d += e;
                vptr10[k + 1] = a1 + 2 * a;
                d += f1;
                c1 += a1;
                d += c1;
                c1 += 2 * c;
                B = 2 * F;
                g = vptr12[k + 1];
                vptr11[k + 1] = a1 + f1 + +g + B;
                d += 2 * H;
                vptr10[k] = c1;
                vptr11[k] = b + d;
                vptr12[k] = e + 2 * E;
                vptr12[k + 1] = g + 2 * G;
                k += 2;
            }
            /**************END MAIN INNER LOOP***************************/
            if (k == xlength - 1) { /* rigth boundary  if one point at end */
                                    /* remains we reflect to get an extra point */

                E = vptr02[k];
                b = x_1 * vptr01[k];
                d = 2 * f;
                B = /* x0* */ vptr00[k];
                c = 2 * a + B;
                d += E;
                vptr00[k] = c;
                d += c;
                H = d + b;
                vptr01[k] = H;
                e = vptr12[k];
                b = x_2 * vptr11[k];
                d = 2 * f1;
                c1 = 2 * a1 + x_1 * vptr10[k];
                d += e;
                d += c1;
                c1 += 2 * c;
                d += 2 * H;
                vptr10[k] = c1;
                vptr11[k] = b + d;
                vptr12[k] = e + 2 * E;
            }

            /************************/

            j += 2;
        }

        if (j == ylength - 1) { /* in case the last line remins to be done
				   it is done alone  */

            vptr12 = vector + (j - 1) * xlength;
            vptr11 = vector + j * xlength;

            vptr02 = vptr12 + FrameSize;
            ;
            vptr01 = vptr11 + FrameSize;
            ;

            if (xlength > 1) {

                E = vptr02[0];
                b = x_1 * vptr01[0];
                f = /* x0* */ vptr01[1];
                d = 2 * (E + f);
                G = vptr02[1];
                F = f + 2 * G;
                vptr01[1] = F;
                H = d + b;
                vptr01[0] = H;
                e = vptr12[0];
                b = x_2 * vptr11[0];
                f1 = x_1 * vptr11[1];
                d = 2 * (e + f1);
                B = 2 * F;
                g = vptr12[1];
                vptr11[1] = f1 + 2 * g + B;
                d += 2 * H;
                vptr11[0] = b + d;
                vptr12[1] = g + 2 * G;
                vptr12[0] = e + 2 * E;
                k = 2;
            } else {
                f = /* x0* */ vptr01[0];
                G = vptr02[0];
                F = f + 2 * G;
                vptr01[0] = F;
                f1 = x_1 * vptr11[0];
                B = 2 * F;
                g = vptr12[0];
                vptr11[0] = f1 + 2 * g + B;
                vptr12[0] = g + 2 * G;
                k = 1;
            }

            while (k < xlength - 1)

            {

                E = vptr02[k];
                b = x_1 * vptr01[k];
                d = f;
                f = /* x0* */ vptr01[k + 1];
                d += 2 * E;
                d += f;
                G = vptr02[k + 1];
                F = f + 2 * G;
                vptr01[k + 1] = F;
                H = d + b;
                vptr01[k] = H;
                e = vptr12[k];
                b = x_2 * vptr11[k];
                d = f1;
                f1 = x_1 * vptr11[k + 1];
                d += 2 * e;
                d += f1;
                B = 2 * F;
                g = vptr12[k + 1];
                vptr11[k + 1] = f1 + 2 * g + B;
                d += 2 * H;
                vptr11[k] = b + d;
                vptr12[k + 1] = g + 2 * G;
                vptr12[k] = e + 2 * E;
                k += 2;
            }

            if (k == xlength - 1) {

                E = vptr02[k];
                b = x_1 * vptr01[k];
                d = 2 * f;
                d += 2 * E;
                H = d + b;
                vptr01[k] = H;
                e = vptr12[k];
                b = x_2 * vptr11[k];
                d = 2 * f1;
                d += 2 * e;
                d += 2 * H;
                vptr11[k] = b + d;
                vptr12[k] = e + 2 * E;
            }

        } /* end last line */
        s = 2;
    } /*************end z-loop*************/

    else /*if ((pairity == 1)||Zlength==1))*/
    {    /* starts main z- loop doing  two lines each time  */
        vector = Vector;
        if ((ylength > 1) && (pairity == 0))
        /*if(pairity == 0)   thus Zlength ==1 */
        {

            vptr01 = Vector;
            vptr00 = Vector + xlength;

            if (xlength > 1) {
                b = x_1 * vptr01[0];
                B = /* x0* */ vptr00[0];
                f = /* x0* */ vptr01[1];
                a = x1 * vptr00[1];
                c = 2 * a + B;
                vptr00[0] = c;
                d = 2 * (c + f);
                F = 2 * a + f;
                vptr01[1] = F;
                vptr00[1] = a;
                H = d + b;
                vptr01[0] = H;
                k = 2;
            } else /* xlength =1 */
            {
                b = x_1 * vptr01[0];
                f = /* x0* */ vptr01[0];
                a = x1 * vptr00[0];
                F = 2 * a + f;
                vptr01[0] = F;
                vptr00[0] = a;
                k = 1;
            }

            while (k < xlength - 1) {

                b = x_1 * vptr01[k];
                d = f;
                B = /* x0* */ vptr00[k];
                c = a + B;
                f = /* x0* */ vptr01[k + 1];
                a = x1 * vptr00[k + 1];
                d += f;
                c += a;
                vptr00[k] = c;
                d += 2 * c;
                F = 2 * a + f;
                vptr01[k + 1] = F;
                vptr00[k + 1] = a;
                H = d + b;
                vptr01[k] = H;
                k += 2;
            }
            if (k == xlength - 1) {
                b = x_1 * vptr01[k];
                d = 2 * f;
                B = /* x0* */ vptr00[k];
                c = 2 * a + B;
                vptr00[k] = c;
                d += 2 * c;
                F = 2 * a + f;
                H = d + b;
                vptr01[k] = H;
            }
            j = 2; /*next lines will be line 0 and 1 */
        } else     /*if (pairity == 1)||(ylength ==1) */
        {

            vptr00 = Vector;

            if ((pairity == 0) && (xlength > 1)) {
                B = /* x0* */ vptr00[0];
                a = x1 * vptr00[1];
                c = 2 * a + B;
                vptr00[0] = c;
                vptr00[1] = a;
                k = 2;
            } else {
                a = x1 * vptr00[0];
                vptr00[0] = a;
                k = 1;
            }
            while (k < xlength - 1) {
                B = /* x0* */ vptr00[k];
                c = a + B;
                a = x1 * vptr00[k + 1];
                c += a;
                vptr00[k] = c;
                vptr00[k + 1] = a;
                k += 2;
            }
            if (k == xlength - 1) {
                B = /* x0* */ vptr00[k];
                c = 2 * a + B;
                vptr00[k] = c;
            }

            j = 1; /*next lines will be line 1 and 2 */
        }

        while (j < ylength - 2) {

            /* starts main outer j  loop doing 
				   two lines each time  */

            vptr02 = Vector + (j - 1) * xlength;
            vptr01 = Vector + j * xlength;
            vptr00 = Vector + (j + 1) * xlength;

            if ((pairity == 0) && (xlength > 1)) {

                E = vptr02[0];
                b = x_1 * vptr01[0];
                B = /* x0* */ vptr00[0];
                f = /* x0* */ vptr01[1];
                a = x1 * vptr00[1];
                d = E + 2 * f;
                c = B + 2 * a;
                vptr00[0] = c;
                d += c;
                G = vptr02[1];
                F = a + f + G;
                vptr01[1] = F;
                vptr00[1] = a;
                H = d + b;
                vptr01[0] = H;
                k = 2;
            }

            else { /* left boundary: first single point at 0 */
                /* no reflection is needed  */
                d = f;
                f = /* x0* */ vptr01[0];
                a = x1 * vptr00[0];
                G = vptr02[0];
                F = a + f + G;
                vptr01[0] = F;
                vptr00[0] = a;

                k = 1;
            }

            /**********HERE IS THE MAIN INNER LOOP TO OPTIMIZE : :  ***********/
            /*  it is running  about  m x N x M /4   times   for one level 
		  biorthogonal filter  2m+1    on  N  x  M  images           */
            while (k < xlength - 1) {

                E = vptr02[k];
                b = x_1 * vptr01[k];
                d = f;
                B = /* x0* */ vptr00[k];
                c = a + B;
                f = /* x0* */ vptr01[k + 1];
                a = x1 * vptr00[k + 1];
                d += E;
                d += f;
                c += a;
                vptr00[k] = c;
                d += c;
                G = vptr02[k + 1];
                F = a + f + G;
                vptr01[k + 1] = F;
                vptr00[k + 1] = a;
                H = d + b;
                vptr01[k] = H;
                k += 2;
            }
            /**************END MAIN INNER LOOP***************************/
            if (k == xlength - 1) { /* rigth boundary  if one point at end */
                /* remains we reflect to get an extra point */

                E = vptr02[k];
                b = x_1 * vptr01[k];
                d = 2 * f;
                B = /* x0* */ vptr00[k];
                c = 2 * a + B;
                d += E;
                vptr00[k] = c;
                d += c;
                H = d + b;
                vptr01[k] = H;
            }

            /************************/

            j += 2;
        }
        if (j == ylength - 2) {

            /* starts main outer j  loop doing 
				   two lines each time  */

            vptr02 = Vector + (j - 1) * xlength;
            vptr01 = Vector + j * xlength;
            vptr00 = Vector + (j + 1) * xlength;

            if ((pairity == 0) && (xlength > 1)) {

                E = vptr02[0];
                b = x_1 * vptr01[0];
                B = /* x0* */ vptr00[0];
                f = /* x0* */ vptr01[1];
                a = x1 * vptr00[1];
                d = E + 2 * f;
                c = B + 2 * a;
                vptr00[0] = c;
                d += c;
                G = vptr02[1];
                F = a + f + G;
                vptr01[1] = F;
                vptr00[1] = a;
                H = d + b;
                vptr01[0] = H;
                k = 2;
            }

            else { /* left boundary: first single point at 0 */
                /* no reflection is needed  */
                d = f;
                f = /* x0* */ vptr01[0];
                a = x1 * vptr00[0];
                G = vptr02[0];
                F = a + f + G;
                vptr01[0] = F;
                vptr00[0] = a;
                k = 1;
            }

            /**********HERE IS THE MAIN INNER LOOP TO OPTIMIZE : :  ***********/
            /*  it is running  about  m x N x M /4   times   for one level 
		  biorthogonal filter  2m+1    on  N  x   M  images           */
            while (k < xlength - 1) {

                E = vptr02[k];
                b = x_1 * vptr01[k];
                d = f;
                B = /* x0* */ vptr00[k];
                c = a + B;
                f = /* x0* */ vptr01[k + 1];
                a = x1 * vptr00[k + 1];
                d += E;
                d += f;
                c += a;
                vptr00[k] = c;
                d += c;
                G = vptr02[k + 1];
                F = a + f + G;
                vptr01[k + 1] = F;
                vptr00[k + 1] = a;
                H = d + b;
                vptr01[k] = H;
                k += 2;
            }
            /**************END MAIN INNER LOOP***************************/
            if (k == xlength - 1) { /* rigth boundary  if one point at end */
                /* remains we reflect to get an extra point */

                E = vptr02[k];
                b = x_1 * vptr01[k];
                d = 2 * f;
                B = /* x0* */ vptr00[k];
                c = 2 * a + B;
                d += E;
                vptr00[k] = c;
                d += c;
                H = d + b;
                vptr01[k] = H;
            }

            /************************/

            j += 2;
        }

        if (j == ylength - 1) {

            vptr02 = vector + (j - 1) * xlength;
            vptr01 = vector + j * xlength;

            /* in case the last line remins to be done
		 it is done alone  */
            if ((xlength > 1) && (pairity == 0)) {
                E = vptr02[0];
                b = x_1 * vptr01[0];
                f = /* x0* */ vptr01[1];
                d = 2 * (E + f);
                G = vptr02[1];
                F = f + 2 * G;
                vptr01[1] = F;
                H = d + b;
                vptr01[0] = H;
                k = 2;
            } else /* if (xlength==1)*/
            {
                f = /* x0* */ vptr01[0];
                G = vptr02[0];
                F = f + 2 * G;
                vptr01[0] = F;
                k = 1;
            }

            while (k < xlength - 1)

            {

                E = vptr02[k];
                b = x_1 * vptr01[k];
                d = f;
                f = /* x0* */ vptr01[k + 1];
                d += 2 * E;
                d += f;
                G = vptr02[k + 1];
                F = f + 2 * G;
                vptr01[k + 1] = F;
                H = d + b;
                vptr01[k] = H;
                k += 2;
            }
            if (k == xlength - 1) {
                E = vptr02[k];
                b = x_1 * vptr01[k];
                d = 2 * f;
                d += 2 * E;
                H = d + b;
                vptr01[k] = H;
            }
        }
        s = 1;
    } /*************end z-loop*************/

    while (s < zlength - 1) { /* starts main z- loop doing  two lines each time  */
        vector = Vector + s * FrameSize;
        j = 0;
        if ((ylength > 2) && (pairity == 0)) {

            vptr11 = vector;
            vptr10 = vector + xlength;

            vptr01 = vptr11 + FrameSize;
            ;
            vptr00 = vptr10 + FrameSize;
            ;

            vptr21 = vptr11 - FrameSize;
            ;
            vptr20 = vptr10 - FrameSize;
            ;

            if (xlength > 1) {
                b = x_1 * vptr01[0];
                B = /* x0* */ vptr00[0];
                f = /* x0* */ vptr01[1];
                a = x1 * vptr00[1];
                c = 2 * a + B;
                vptr00[0] = c;
                d = 2 * (c + f);
                F = 2 * a + f;
                vptr01[1] = F;
                vptr00[1] = a;
                H = d + b;
                vptr01[0] = H;
                b = x_2 * vptr11[0];
                f1 = x_1 * vptr11[1];
                a1 = /* x0* */ vptr10[1];
                vptr10[1] = a1;
                c1 = 2 * a1 + x_1 * vptr10[0];
                d = 2 * (c1 + f1);
                B = F + vptr21[1];
                vptr11[1] = 2 * a1 + f1 + B;
                d += vptr21[0] + H;
                vptr10[0] = c1;
                vptr11[0] = b + d;
                k = 2;

            } else {
                f = /* x0* */ vptr01[0];
                a = x1 * vptr00[0];
                F = 2 * a + f;
                vptr01[0] = F;
                vptr00[0] = a;
                f1 = x_1 * vptr11[0];
                a1 = /* x0* */ vptr10[0];
                vptr10[0] = a1;
                B = F + vptr21[0];
                vptr11[0] = 2 * a1 + f1 + B;
                k = 1;
            }

            while (k < xlength - 1) {

                b = x_1 * vptr01[k];
                d = f;
                B = /* x0* */ vptr00[k];
                c = a + B;
                f = /* x0* */ vptr01[k + 1];
                a = x1 * vptr00[k + 1];
                d += f;
                c += a;
                vptr00[k] = c;
                d += 2 * c;
                F = 2 * a + f;
                vptr01[k + 1] = F;
                vptr00[k + 1] = a;
                H = d + b;
                vptr01[k] = H;
                b = x_2 * vptr11[k];
                d = f1;
                c1 = a1 + x_1 * vptr10[k];
                f1 = x_1 * vptr11[k + 1];
                a1 = /* x0* */ vptr10[k + 1];
                vptr10[k + 1] = a1;
                d += f1;
                c1 += a1;
                d += 2 * c1;
                B = F + vptr21[k + 1];
                vptr11[k + 1] = 2 * a1 + f1 + B;
                d += vptr21[k] + H;
                vptr10[k] = c1;
                vptr11[k] = b + d;
                k += 2;
            }
            if (k == xlength - 1) {

                b = x_1 * vptr01[k];
                d = 2 * f;
                B = /* x0* */ vptr00[k];
                c = 2 * a + B;
                vptr00[k] = c;
                d += 2 * c;
                F = 2 * a + f;
                H = d + b;
                vptr01[k] = H;
                b = x_2 * vptr11[k];
                d = 2 * f1;
                c1 = 2 * a1 + x_1 * vptr10[k];
                d += 2 * c1;
                d += vptr21[k] + H;
                vptr10[k] = c1;
                vptr11[k] = b + d;
            }

            j = 2; /*next lines will be line 0 and 1 */
        } else if ((ylength == 2) && (pairity == 0)) {
            vptr11 = vector;
            vptr10 = vector + xlength;

            vptr01 = vptr11 + FrameSize;
            ;
            vptr00 = vptr10 + FrameSize;
            ;

            vptr21 = vptr11 - FrameSize;
            ;
            vptr20 = vptr10 - FrameSize;
            ;

            if (xlength > 1) {
                b = x_1 * vptr01[0];
                B = /* x0* */ vptr00[0];
                f = /* x0* */ vptr01[1];
                a = x1 * vptr00[1];
                c = 2 * a + B;
                vptr00[0] = c;
                d = 2 * (c + f);
                F = 2 * a + f;
                vptr01[1] = F;
                vptr00[1] = a;
                H = d + b;
                vptr01[0] = H;
                b = x_2 * vptr11[0];
                f1 = x_1 * vptr11[1];
                a1 = /* x0* */ vptr10[1];
                vptr10[1] = a1 + a + vptr20[1];
                c1 = 2 * a1 + x_1 * vptr10[0];
                d = 2 * (c1 + f1);
                B = F + vptr21[1];
                vptr11[1] = 2 * a1 + f1 + B;
                d += vptr21[0] + H;
                c1 += vptr20[0] + c;
                vptr10[0] = c1;
                vptr11[0] = b + d;
                k = 2;

            } else {
                f = /* x0* */ vptr01[0];
                a = x1 * vptr00[0];
                F = 2 * a + f;
                vptr01[0] = F;
                vptr00[0] = a;
                f1 = x_1 * vptr11[0];
                a1 = /* x0* */ vptr10[0];
                vptr10[0] = a1 + a + vptr20[0];
                B = F + vptr21[0];
                vptr11[0] = 2 * a1 + f1 + B;
                k = 1;
            }

            while (k < xlength - 1) {

                b = x_1 * vptr01[k];
                d = f;
                B = /* x0* */ vptr00[k];
                c = a + B;
                f = /* x0* */ vptr01[k + 1];
                a = x1 * vptr00[k + 1];
                d += f;
                c += a;
                vptr00[k] = c;
                d += 2 * c;
                F = 2 * a + f;
                vptr01[k + 1] = F;
                vptr00[k + 1] = a;
                H = d + b;
                vptr01[k] = H;
                b = x_2 * vptr11[k];
                d = f1;
                c1 = a1 + x_1 * vptr10[k];
                f1 = x_1 * vptr11[k + 1];
                a1 = /* x0* */ vptr10[k + 1];
                vptr10[k + 1] = a1 + a + vptr20[k + 1];
                d += f1;
                c1 += a1;
                d += 2 * c1;
                c1 += vptr20[k] + c;
                B = F + vptr21[k + 1];
                vptr11[k + 1] = 2 * a1 + f1 + B;
                d += vptr21[k] + H;
                vptr10[k] = c1;
                vptr11[k] = b + d;
                k += 2;
            }
            if (k == xlength - 1) {

                b = x_1 * vptr01[k];
                d = 2 * f;
                B = /* x0* */ vptr00[k];
                c = 2 * a + B;
                vptr00[k] = c;
                d += 2 * c;
                F = 2 * a + f;
                H = d + b;
                vptr01[k] = H;
                b = x_2 * vptr11[k];
                d = 2 * f1;
                c1 = 2 * a1 + x_1 * vptr10[k];
                d += 2 * c1;
                c1 += vptr20[k] + c;
                d += vptr21[k] + H;
                vptr10[k] = c1;
                vptr11[k] = b + d;
            }

            j = 2; /*next lines will be line 0 and 1 */

        } else if (ylength == 1) {
            vptr10 = vector;

            vptr00 = vptr10 + FrameSize;
            ;

            vptr20 = vptr10 - FrameSize;
            ;

            if ((pairity == 1) || (xlength == 1)) {
                a = x1 * vptr00[0];
                vptr00[0] = a;
                a1 = /* x0* */ vptr10[0];
                vptr10[0] = a1 + a + vptr20[0];
                k = 1;
            } else {
                B = /* x0* */ vptr00[0];
                a = x1 * vptr00[1];
                c = 2 * a + B;
                vptr00[0] = c;
                vptr00[1] = a;
                a1 = /* x0* */ vptr10[1];
                vptr10[1] = a1 + a + vptr20[1];
                c1 = 2 * a1 + x_1 * vptr10[0];
                vptr10[0] = c1 + c + vptr20[0];
                k = 2;
            }

            while (k < xlength - 1) {

                B = /* x0* */ vptr00[k];
                c = a + B;
                a = x1 * vptr00[k + 1];
                c += a;
                vptr00[k] = c;
                vptr00[k + 1] = a;
                c1 = a1 + x_1 * vptr10[k];
                a1 = /* x0* */ vptr10[k + 1];
                vptr10[k + 1] = a1 + a + vptr20[k + 1];
                c1 += a1;
                c1 += vptr20[k] + c;
                vptr10[k] = c1;
                k += 2;
            }
            if (k == xlength - 1) {
                B = /* x0* */ vptr00[k];
                c = 2 * a + B;
                vptr00[k] = c;
                c1 = 2 * a1 + x_1 * vptr10[k];
                c1 += vptr20[k] + c;
                vptr10[k] = c1;
            }
            j = 1; /*next lines will be line 1 and 2 */

        } else /*if ((pairity == 1)&&(ylength>2))*/
        {
            vptr10 = vector;

            vptr00 = vptr10 + FrameSize;
            ;

            vptr20 = vptr10 - FrameSize;
            ;

            if ((pairity == 1) || (xlength == 1)) {
                a = x1 * vptr00[0];
                vptr00[0] = a;
                a1 = /* x0* */ vptr10[0];
                vptr10[0] = a1;
                k = 1;
            } else {
                b = x_1 * vptr01[0];
                B = /* x0* */ vptr00[0];
                a = x1 * vptr00[1];
                c = 2 * a + B;
                vptr00[0] = c;
                vptr00[1] = a;
                a1 = /* x0* */ vptr10[1];
                vptr10[1] = a1;
                c1 = 2 * a1 + x_1 * vptr10[0];
                vptr10[0] = c1;
                k = 1;
            }

            while (k < xlength - 1) {

                B = /* x0* */ vptr00[k];
                c = a + B;
                a = x1 * vptr00[k + 1];
                c += a;
                vptr00[k] = c;
                vptr00[k + 1] = a;
                c1 = a1 + x_1 * vptr10[k];
                a1 = /* x0* */ vptr10[k + 1];
                vptr10[k + 1] = a1;
                c1 += a1;
                vptr10[k] = c1;
                k += 2;
            }
            if (k == xlength - 1) {
                B = /* x0* */ vptr00[k];
                c = 2 * a + B;
                vptr00[k] = c;
                c1 = 2 * a1 + x_1 * vptr10[k];
                vptr10[k] = c1;
            }
            j = 1; /*next lines will be line 1 and 2 */
        }
        while (j < ylength - 2) { /* starts main outer loop doing 
					   two lines each time  */

            vptr12 = vector + (j - 1) * xlength;
            vptr11 = vector + j * xlength;
            vptr10 = vector + (j + 1) * xlength;

            vptr02 = vptr12 + FrameSize;
            ;
            vptr01 = vptr11 + FrameSize;
            ;
            vptr00 = vptr10 + FrameSize;
            ;

            vptr22 = vptr12 - FrameSize;
            ;
            vptr21 = vptr11 - FrameSize;
            ;
            vptr20 = vptr10 - FrameSize;
            ;

            if ((pairity == 0) && (xlength > 1)) {

                E = vptr02[0];
                b = x_1 * vptr01[0];
                B = /* x0* */ vptr00[0];
                f = /* x0* */ vptr01[1];
                a = x1 * vptr00[1];
                d = E + 2 * f;
                c = B + 2 * a;
                vptr00[0] = c;
                d += c;
                G = vptr02[1];
                F = a + f + G;
                vptr01[1] = F;
                vptr00[1] = a;
                H = d + b;
                vptr01[0] = H;
                e = vptr12[0];
                b = x_2 * vptr11[0];
                f1 = x_1 * vptr11[1];
                a1 = /* x0* */ vptr10[1];
                vptr10[1] = a1;
                d = e + 2 * f1;
                c1 = x_1 * vptr10[0] + 2 * a1;
                d += c1;
                B = F + vptr21[1];
                g = vptr12[1];
                vptr11[1] = a1 + f1 + +g + B;
                d += vptr21[0] + H;
                vptr10[0] = c1;
                vptr11[0] = b + d;
                vptr12[0] = e + E + vptr22[0];
                vptr12[1] = g + G + vptr22[1];
                k = 2;
            } else /*if ((pairity == 1)||(xlength==1))  */ {
                d = f;
                f = /* x0* */ vptr01[0];
                a = x1 * vptr00[0];
                G = vptr02[0];
                F = a + f + G;
                vptr01[0] = F;
                vptr00[0] = a;
                f1 = x_1 * vptr11[0];
                a1 = /* x0* */ vptr10[0];
                vptr10[0] = a1;
                B = F + vptr21[0];
                g = vptr12[0];
                vptr11[0] = a1 + f1 + +g + B;
                vptr12[0] = g + G + vptr22[0];
                k = 1;
            }

            /**********HERE IS THE MAIN INNER LOOP TO OPTIMIZE : :  ***********/
            /*  it is running  about  m x N x M /4   times   for one level 
		  biorthogonal filter  2m+1    on  N  x   M  images           */
            while (k < xlength - 1) {

                E = vptr02[k];
                b = x_1 * vptr01[k];
                d = f;
                B = /* x0* */ vptr00[k];
                c = a + B;
                f = /* x0* */ vptr01[k + 1];
                a = x1 * vptr00[k + 1];
                d += E;
                d += f;
                c += a;
                vptr00[k] = c;
                d += c;
                G = vptr02[k + 1];
                F = a + f + G;
                vptr01[k + 1] = F;
                vptr00[k + 1] = a;
                H = d + b;
                vptr01[k] = H;
                e = vptr12[k];
                b = x_2 * vptr11[k];
                d = f1;
                c1 = a1 + x_1 * vptr10[k];
                f1 = x_1 * vptr11[k + 1];
                a1 = /* x0* */ vptr10[k + 1];
                d += e;
                vptr10[k + 1] = a1;
                d += f1;
                c1 += a1;
                d += c1;
                B = F + vptr21[k + 1];
                g = vptr12[k + 1];
                vptr11[k + 1] = a1 + f1 + g + B;
                d += vptr21[k] + H;
                vptr10[k] = c1;
                vptr11[k] = b + d;
                vptr12[k] = e + E + vptr22[k];
                vptr12[k + 1] = g + G + vptr22[k + 1];
                k += 2;
            }
            /**************END MAIN INNER LOOP***************************/
            if (k == xlength - 1) { /* rigth boundary  if one point at end */
                /* remains we reflect to get an extra point */

                E = vptr02[k];
                b = x_1 * vptr01[k];
                d = 2 * f;
                B = /* x0* */ vptr00[k];
                c = 2 * a + B;
                d += E;
                vptr00[k] = c;
                d += c;
                H = d + b;
                vptr01[k] = H;
                e = vptr12[k];
                b = x_2 * vptr11[k];
                d = 2 * f1;
                c1 = 2 * a1 + x_1 * vptr10[k];
                d += e;
                d += c1;
                d += vptr21[k] + H;
                vptr10[k] = c1;
                vptr11[k] = b + d;
                vptr12[k] = e + E + vptr22[k];
            }

            /************************/

            j += 2;
        }

        if (j == ylength - 2) { /* starts main outer loop doing 
					   two lines each time  */

            vptr12 = vector + (j - 1) * xlength;
            vptr11 = vector + j * xlength;
            vptr10 = vector + (j + 1) * xlength;

            vptr02 = vptr12 + FrameSize;
            ;
            vptr01 = vptr11 + FrameSize;
            ;
            vptr00 = vptr10 + FrameSize;
            ;

            vptr22 = vptr12 - FrameSize;
            ;
            vptr21 = vptr11 - FrameSize;
            ;
            vptr20 = vptr10 - FrameSize;
            ;

            if ((pairity == 0) && (xlength > 1)) {

                E = vptr02[0];
                b = x_1 * vptr01[0];
                B = /* x0* */ vptr00[0];
                f = /* x0* */ vptr01[1];
                a = x1 * vptr00[1];
                d = E + 2 * f;
                c = B + 2 * a;
                vptr00[0] = c;
                d += c;
                G = vptr02[1];
                F = a + f + G;
                vptr01[1] = F;
                vptr00[1] = a;
                H = d + b;
                vptr01[0] = H;
                e = vptr12[0];
                b = x_2 * vptr11[0];
                f1 = x_1 * vptr11[1];
                a1 = /* x0* */ vptr10[1];
                vptr10[1] = a1 + a + vptr20[1];
                d = e + 2 * f1;
                c1 = x_1 * vptr10[0] + 2 * a1;
                d += c1;
                B = F + vptr21[1];
                g = vptr12[1];
                vptr11[1] = a1 + f1 + +g + B;
                d += vptr21[0] + H;
                vptr10[0] = c1 + c + vptr20[0];
                vptr11[0] = b + d;
                vptr12[0] = e + E + vptr22[0];
                vptr12[1] = g + G + vptr22[1];
                k = 2;
            } else /*if ((pairity == 1)||(xlength==1))  */ {
                d = f;
                f = /* x0* */ vptr01[0];
                a = x1 * vptr00[0];
                G = vptr02[0];
                F = a + f + G;
                vptr01[0] = F;
                vptr00[0] = a;
                f1 = x_1 * vptr11[0];
                a1 = /* x0* */ vptr10[0];
                vptr10[0] = a1 + vptr20[0] + a;
                ;
                B = F + vptr21[0];
                g = vptr12[0];
                vptr11[0] = a1 + f1 + g + B;
                vptr12[0] = g + G + vptr22[0];
                k = 1;
            }

            /**********HERE IS THE MAIN INNER LOOP TO OPTIMIZE : :  ***********/
            /*  it is running  about  m x N x M /4   times   for one level 
		  biorthogonal filter  2m+1    on  N  x   M  images           */
            while (k < xlength - 1) {

                E = vptr02[k];
                b = x_1 * vptr01[k];
                d = f;
                B = /* x0* */ vptr00[k];
                c = a + B;
                f = /* x0* */ vptr01[k + 1];
                a = x1 * vptr00[k + 1];
                d += E;
                d += f;
                c += a;
                vptr00[k] = c;
                d += c;
                G = vptr02[k + 1];
                F = a + f + G;
                vptr01[k + 1] = F;
                vptr00[k + 1] = a;
                H = d + b;
                vptr01[k] = H;
                e = vptr12[k];
                b = x_2 * vptr11[k];
                d = f1;
                c1 = a1 + x_1 * vptr10[k];
                f1 = x_1 * vptr11[k + 1];
                a1 = /* x0* */ vptr10[k + 1];
                d += e;
                vptr10[k + 1] = a1 + vptr20[k + 1] + a;
                d += f1;
                c1 += a1;
                d += c1;
                c1 += vptr20[k] + c;
                B = F + vptr21[k + 1];
                g = vptr12[k + 1];
                vptr11[k + 1] = a1 + f1 + +g + B;
                d += vptr21[k] + H;
                vptr10[k] = c1;
                vptr11[k] = b + d;
                vptr12[k] = e + E + vptr22[k];
                vptr12[k + 1] = g + G + vptr22[k + 1];
                k += 2;
            }
            /**************END MAIN INNER LOOP***************************/
            if (k == xlength - 1) { /* rigth boundary  if one point at end */
                /* remains we reflect to get an extra point */

                E = vptr02[k];
                b = x_1 * vptr01[k];
                d = 2 * f;
                B = /* x0* */ vptr00[k];
                c = 2 * a + B;
                d += E;
                vptr00[k] = c;
                d += c;
                H = d + b;
                vptr01[k] = H;
                e = vptr12[k];
                b = x_2 * vptr11[k];
                d = 2 * f1;
                c1 = 2 * a1 + x_1 * vptr10[k];
                d += e;
                d += c1;
                c1 += vptr20[k] + c;
                d += vptr21[k] + H;
                vptr10[k] = c1;
                vptr11[k] = b + d;
                vptr12[k] = e + E + vptr22[k];
            }

            /************************/

            j += 2;
        }

        if (j == ylength - 1) { /* in case the last line remins to be done
				   it is done alone  */

            vptr12 = vector + (j - 1) * xlength;
            vptr11 = vector + j * xlength;

            vptr02 = vptr12 + FrameSize;
            ;
            vptr01 = vptr11 + FrameSize;
            ;

            vptr22 = vptr12 - FrameSize;
            ;
            vptr21 = vptr11 - FrameSize;
            ;

            if ((xlength > 1) && (pairity == 0)) {
                E = vptr02[0];
                b = x_1 * vptr01[0];
                f = /* x0* */ vptr01[1];
                d = 2 * (E + f);
                G = vptr02[1];
                F = f + 2 * G;
                vptr01[1] = F;
                H = d + b;
                vptr01[0] = H;
                e = vptr12[0];
                b = x_2 * vptr11[0];
                f1 = x_1 * vptr11[1];
                d = 2 * (e + f1);
                B = F + vptr21[1];
                g = vptr12[1];
                vptr11[1] = f1 + 2 * +g + B;
                d += vptr21[0] + H;
                vptr11[0] = b + d;
                vptr12[0] = e + E + vptr22[0];
                vptr12[1] = g + G + vptr22[1];
                k = 2;
            } else {
                f = /* x0* */ vptr01[0];
                G = vptr02[0];
                F = f + 2 * G;
                vptr01[0] = F;
                f1 = x_1 * vptr11[0];
                B = F + vptr21[0];
                g = vptr12[0];
                vptr11[0] = f1 + 2 * g + B;
                vptr12[0] = g + G + vptr22[0];
                k = 1;
            }
            while (k < xlength - 1)

            {

                E = vptr02[k];
                b = x_1 * vptr01[k];
                d = f;
                f = /* x0* */ vptr01[k + 1];
                d += 2 * E;
                d += f;
                G = vptr02[k + 1];
                F = f + 2 * G;
                vptr01[k + 1] = F;
                H = d + b;
                vptr01[k] = H;
                e = vptr12[k];
                b = x_2 * vptr11[k];
                d = f1;
                f1 = x_1 * vptr11[k + 1];
                d += 2 * e;
                d += f1;
                B = F + vptr21[k + 1];
                g = vptr12[k + 1];
                vptr11[k + 1] = f1 + 2 * +g + B;
                d += vptr21[k] + H;
                vptr11[k] = b + d;
                vptr12[k] = e + E + vptr22[k];
                vptr12[k + 1] = g + G + vptr22[k + 1];
                k += 2;
            }

            if (k == xlength - 1) {

                E = vptr02[k];
                b = x_1 * vptr01[k];
                d = 2 * f;
                d += 2 * E;
                H = d + b;
                vptr01[k] = H;
                e = vptr12[k];
                b = x_2 * vptr11[k];
                d = 2 * f1;
                d += 2 * e;
                B = F + vptr21[k + 1];
                d += vptr21[k] + H;
                vptr11[k] = b + d;
                vptr12[k] = e + E + vptr22[k];
            }
        }
        s += 2;
    } /*************end z-loop*************/

    if (s == zlength - 1) { /* in case the last frame to be done
					   it is done alone  */
                            /*****************last frame only*****************/
        vector = Vector + s * FrameSize;

        if ((pairity == 0) && (ylength > 2)) {

            vptr11 = vector;
            vptr10 = vector + xlength;

            vptr21 = vptr11 - FrameSize;
            ;
            vptr20 = vptr10 - FrameSize;
            ;

            if (xlength > 1) {
                b = x_2 * vptr11[0];
                f1 = x_1 * vptr11[1];
                a1 = /* x0* */ vptr10[1];
                vptr10[1] = a1;
                c1 = 2 * a1 + x_1 * vptr10[0];
                d = 2 * (c1 + f1);
                B = 2 * vptr21[1];
                vptr11[1] = 2 * a1 + f1 + B;
                d += 2 * vptr21[0];
                vptr10[0] = c1;
                vptr11[0] = b + d;
                k = 2;
            } else {
                f1 = x_1 * vptr11[0];
                a1 = /* x0* */ vptr10[0];
                vptr10[0] = a1;
                B = 2 * vptr21[0];
                vptr11[0] = 2 * a1 + f1 + B;
                k = 1;
            }
            while (k < xlength - 1) {
                b = x_2 * vptr11[k];
                d = f1;
                c1 = a1 + x_1 * vptr10[k];
                f1 = x_1 * vptr11[k + 1];
                a1 = /* x0* */ vptr10[k + 1];
                vptr10[k + 1] = a1;
                d += f1;
                c1 += a1;
                d += 2 * c1;
                B = 2 * vptr21[k + 1];
                vptr11[k + 1] = 2 * a1 + f1 + B;
                d += 2 * vptr21[k];
                vptr10[k] = c1;
                vptr11[k] = b + d;
                k += 2;
            }
            if (k == xlength - 1) {

                b = x_2 * vptr11[k];
                d = 2 * f1;
                c1 = 2 * a1 + x_1 * vptr10[k];
                d += 2 * c1;
                d += 2 * vptr21[k];
                vptr10[k] = c1;
                vptr11[k] = b + d;
            }

            j = 2; /*next lines will be line 0 and 1 */
        } else if (ylength == 2) {
            if (pairity == 0) {
                vptr11 = vector;
                vptr10 = vector + xlength;
            }
            if (pairity == 1) {
                vptr10 = vector;
                vptr11 = vector + xlength;
            }
            vptr21 = vptr11 - FrameSize;
            ;
            vptr20 = vptr10 - FrameSize;
            ;

            if ((xlength > 1) && (pairity == 0)) {
                b = x_2 * vptr11[0];
                f1 = x_1 * vptr11[1];
                a1 = /* x0* */ vptr10[1];
                vptr10[1] = a1 + 2 * vptr20[1];
                c1 = 2 * a1 + x_1 * vptr10[0];
                d = 2 * (c1 + f1);
                c1 += 2 * vptr20[0];
                B = 2 * vptr21[1];
                vptr11[1] = 2 * a1 + f1 + B;
                d += 2 * vptr21[0];
                vptr10[0] = c1;
                vptr11[0] = b + d;
                k = 2;
            } else {
                f1 = x_1 * vptr11[0];
                a1 = /* x0* */ vptr10[0];
                vptr10[0] = a1 + 2 * vptr20[0];
                B = 2 * vptr21[0];
                vptr11[0] = 2 * a1 + f1 + B;
                k = 1;
            }
            while (k < xlength - 1) {
                b = x_2 * vptr11[k];
                d = f1;
                c1 = a1 + x_1 * vptr10[k];
                f1 = x_1 * vptr11[k + 1];
                a1 = /* x0* */ vptr10[k + 1];
                vptr10[k + 1] = a1 + 2 * vptr20[k + 1];
                d += f1;
                c1 += a1;
                d += 2 * c1;
                c1 += 2 * vptr20[k];
                B = 2 * vptr21[k + 1];
                vptr11[k + 1] = 2 * a1 + f1 + B;
                d += 2 * vptr21[k];
                vptr10[k] = c1;
                vptr11[k] = b + d;
                k += 2;
            }
            if (k == xlength - 1) {

                b = x_2 * vptr11[k];
                d = 2 * f1;
                c1 = 2 * a1 + x_1 * vptr10[k];
                d += 2 * c1;
                c1 += 2 * vptr20[k];
                d += 2 * vptr21[k];
                vptr10[k] = c1;
                vptr11[k] = b + d;
            }
            j = 2; /*next lines will be line 0 and 1 */
        } else if (ylength == 1) {

            vptr10 = vector;

            vptr20 = vptr10 - FrameSize;
            ;

            if ((xlength > 1) && (pairity == 0)) {
                a1 = /* x0* */ vptr10[1];
                vptr10[1] = a1 + 2 * vptr20[1];
                c1 = 2 * a1 + x_1 * vptr10[0];
                c1 += 2 * vptr20[0];
                vptr10[0] = c1;
                k = 2;
            } else {
                a1 = /* x0* */ vptr10[0];
                vptr10[0] = a1 + 2 * vptr20[0];
                k = 1;
            }

            while (k < xlength - 1)

            {

                c1 = a1 + x_1 * vptr10[k];
                a1 = /* x0* */ vptr10[k + 1];
                vptr10[k + 1] = a1 + 2 * vptr20[k + 1];
                c1 += a1;
                c1 += 2 * vptr20[k];
                vptr10[k] = c1;
                k += 2;
            }

            if (k == xlength - 1) {
                c1 = 2 * a1 + x_1 * vptr10[k];
                c1 += 2 * vptr20[k];
                vptr10[k] = c1;
            }
            j = 1;
        } else { /* if *pairity == 1 and ylength > 2 )   */

            vptr10 = vector;

            vptr20 = vptr10 - FrameSize;
            ;

            if ((xlength > 1) && (pairity == 0)) {
                a1 = /* x0* */ vptr10[1];
                vptr10[1] = a1;
                c1 = 2 * a1 + x_1 * vptr10[0];
                vptr10[0] = c1;
                k = 2;
            } else {
                a1 = /* x0* */ vptr10[0];
                vptr10[0] = a1;
                k = 1;
            }

            while (k < xlength - 1)

            {

                c1 = a1 + x_1 * vptr10[k];
                a1 = /* x0* */ vptr10[k + 1];
                vptr10[k + 1] = a1;
                c1 += a1;
                vptr10[k] = c1;
                k += 2;
            }

            if (k == xlength - 1) {
                c1 = 2 * a1 + x_1 * vptr10[k];
                vptr10[k] = c1;
            }
            j = 1;
        }

        while (j < ylength - 2) { /* starts main outer loop doing 
					   two lines each time  */

            vptr12 = vector + (j - 1) * xlength;
            vptr11 = vector + j * xlength;
            vptr10 = vector + (j + 1) * xlength;

            vptr22 = vptr12 - FrameSize;
            ;
            vptr21 = vptr11 - FrameSize;
            ;
            vptr20 = vptr10 - FrameSize;
            ;

            if ((pairity == 0) && (xlength > 1)) {
                e = vptr12[0];
                b = x_2 * vptr11[0];
                f1 = x_1 * vptr11[1];
                a1 = /* x0* */ vptr10[1];
                vptr10[1] = a1;
                d = e + 2 * f1;
                c1 = x_1 * vptr10[0] + 2 * a1;
                d += c1;
                B = 2 * vptr21[1];
                g = vptr12[1];
                vptr11[1] = a1 + f1 + +g + B;
                d += 2 * vptr21[0];
                vptr10[0] = c1;
                vptr11[0] = b + d;
                vptr12[0] = e + 2 * vptr22[0];
                vptr12[1] = g + 2 * vptr22[1];
                k = 2;
            }

            else { /* left boundary: first single point at 0 */
                /* no reflection is needed  */
                d = f;
                f1 = x_1 * vptr11[0];
                a1 = /* x0* */ vptr10[0];
                vptr10[0] = a1;
                B = 2 * vptr21[0];
                g = vptr12[0];
                vptr11[0] = a1 + f1 + g + B;
                vptr12[0] = g + 2 * vptr22[0];
                k = 1;
            }

            /**********HERE IS THE MAIN INNER LOOP TO OPTIMIZE : :  ***********/
            /*  it is running  about  m x N x M /4   times   for one level 
			  biorthogonal filter  2m+1    on  N  x   M  images           */
            while (k < xlength - 1) {

                e = vptr12[k];
                b = x_2 * vptr11[k];
                d = f1;
                c1 = a1 + x_1 * vptr10[k];
                f1 = x_1 * vptr11[k + 1];
                a1 = /* x0* */ vptr10[k + 1];
                d += e;
                vptr10[k + 1] = a1;
                d += f1;
                c1 += a1;
                d += c1;
                B = 2 * vptr21[k + 1];
                g = vptr12[k + 1];
                vptr11[k + 1] = a1 + f1 + +g + B;
                d += 2 * vptr21[k];
                vptr10[k] = c1;
                vptr11[k] = b + d;
                vptr12[k] = e + 2 * vptr22[k];
                vptr12[k + 1] = g + 2 * vptr22[k + 1];
                k += 2;
            }
            /**************END MAIN INNER LOOP***************************/
            if (k == xlength - 1) { /* rigth boundary  if one point at end */
                /* remains we reflect to get an extra point */

                e = vptr12[k];
                b = x_2 * vptr11[k];
                d = 2 * f1;
                c1 = 2 * a1 + x_1 * vptr10[k];
                d += e;
                d += c1;
                d += 2 * vptr21[k];
                vptr10[k] = c1;
                vptr11[k] = b + d;
                vptr12[k] = e + 2 * vptr22[k];
            }

            /************************/

            j += 2;
        }

        if (j == ylength - 2) { /* starts main outer loop doing 
					   two lines each time  */

            vptr12 = vector + (j - 1) * xlength;
            vptr11 = vector + j * xlength;
            vptr10 = vector + (j + 1) * xlength;

            vptr22 = vptr12 - FrameSize;
            ;
            vptr21 = vptr11 - FrameSize;
            ;
            vptr20 = vptr10 - FrameSize;
            ;

            if ((pairity == 0) && (xlength > 1)) {
                e = vptr12[0];
                b = x_2 * vptr11[0];
                f1 = x_1 * vptr11[1];
                a1 = /* x0* */ vptr10[1];
                vptr10[1] = a1 + 2 * vptr20[1];
                d = e + 2 * f1;
                c1 = x_1 * vptr10[0] + 2 * a1;
                d += c1;
                c1 += 2 * vptr20[0];
                B = 2 * vptr21[1];
                g = vptr12[1];
                vptr11[1] = a1 + f1 + +g + B;
                d += 2 * vptr21[0];
                vptr10[0] = c1;
                vptr11[0] = b + d;
                vptr12[0] = e + 2 * vptr22[0];
                vptr12[1] = g + 2 * vptr22[1];
                k = 2;
            }

            else { /* left boundary: first single point at 0 */
                /* no reflection is needed  */
                d = f;
                f1 = x_1 * vptr11[0];
                a1 = /* x0* */ vptr10[0];
                vptr10[0] = a1 + 2 * vptr20[0];
                B = 2 * vptr21[0];
                g = vptr12[0];
                vptr11[0] = a1 + f1 + +g + B;
                vptr12[0] = g + 2 * vptr22[0];
                k = 1;
            }

            /**********HERE IS THE MAIN INNER LOOP TO OPTIMIZE : :  ***********/
            /*  it is running  about  m x N x M /4   times   for one level 
			  biorthogonal filter  2m+1    on  N  x   M  images           */
            while (k < xlength - 1) {

                e = vptr12[k];
                b = x_2 * vptr11[k];
                d = f1;
                c1 = a1 + x_1 * vptr10[k];
                f1 = x_1 * vptr11[k + 1];
                a1 = /* x0* */ vptr10[k + 1];
                d += e;
                vptr10[k + 1] = a1 + 2 * vptr20[k + 1];
                d += f1;
                c1 += a1;
                d += c1;
                c1 += 2 * vptr20[k];
                B = 2 * vptr21[k + 1];
                g = vptr12[k + 1];
                vptr11[k + 1] = a1 + f1 + +g + B;
                d += 2 * vptr21[k];
                vptr10[k] = c1;
                vptr11[k] = b + d;
                vptr12[k] = e + 2 * vptr22[k];
                vptr12[k + 1] = g + 2 * vptr22[k + 1];
                k += 2;
            }
            /**************END MAIN INNER LOOP***************************/
            if (k == xlength - 1) { /* rigth boundary  if one point at end */
                /* remains we reflect to get an extra point */

                e = vptr12[k];
                b = x_2 * vptr11[k];
                d = 2 * f1;
                c1 = 2 * a1 + x_1 * vptr10[k];
                d += e;
                d += c1;
                c1 += 2 * vptr20[k];
                d += 2 * vptr21[k];
                vptr10[k] = c1;
                vptr11[k] = b + d;
                vptr12[k] = e + 2 * vptr22[k];
            }

            /************************/

            j += 2;
        }

        if (j == ylength - 1) { /* in case the last line remins to be done
					   it is done alone  */

            vptr12 = vector + (j - 1) * xlength;
            vptr11 = vector + j * xlength;

            vptr22 = vptr12 - FrameSize;
            ;
            vptr21 = vptr11 - FrameSize;
            ;

            if ((pairity == 0) && (xlength > 1)) {

                e = vptr12[0];
                b = x_2 * vptr11[0];
                f1 = x_1 * vptr11[1];
                d = 2 * (e + f1);
                B = 2 * vptr21[1];
                g = vptr12[1];
                vptr11[1] = f1 + 2 * +g + B;
                d += 2 * vptr21[0];
                vptr11[0] = b + d;
                vptr12[0] = e + 2 * vptr22[0];
                vptr12[1] = g + 2 * vptr22[1];
                k = 2;
            } else

            {
                f1 = x_1 * vptr11[0];
                B = 2 * vptr21[0];
                g = vptr12[0];
                vptr11[0] = f1 + 2 * g + B;
                vptr12[0] = g + 2 * vptr22[0];
                k = 1;
            }
            while (k < xlength - 1)

            {

                e = vptr12[k];
                b = x_2 * vptr11[k];
                d = f1;
                f1 = x_1 * vptr11[k + 1];
                d += 2 * e;
                d += f1;
                B = 2 * vptr21[k + 1];
                g = vptr12[k + 1];
                vptr11[k + 1] = f1 + 2 * +g + B;
                d += 2 * vptr21[k];
                vptr11[k] = b + d;
                vptr12[k] = e + 2 * vptr22[k];
                vptr12[k + 1] = g + 2 * vptr22[k + 1];
                k += 2;
            }

            if (k == xlength - 1) {
                e = vptr12[k];
                b = x_2 * vptr11[k];
                d = 2 * f1;
                d += 2 * e;
                B = 2 * vptr21[k + 1];
                d += 2 * vptr21[k];
                vptr11[k] = b + d;
                vptr12[k] = e + 2 * vptr22[k];
            }
        }
        s += 2;
    } /*************end z-loop*************/
    return;
}

void bio_3d_postmult(FLOAT* Vector,
                     int xlength,
                     int ylength,
                     int zlength,
                     FLOAT x1,
                     int pairity) { /* start of function */

    register int j, k, s;
    FLOAT* vptr00, *vptr01, *vptr02;
    FLOAT* vptr10, *vptr11, *vptr12;
    FLOAT* vptr20, *vptr21;
    FLOAT a = 0, e = 0, f = 0, g = 0;
    FLOAT c = 0, d = 0, b = 0;
    FLOAT a1 = 0, f1 = 0;
    FLOAT c1 = 0;
    FLOAT B, E, G, H, F;
    FLOAT /*  x0, */ x_1, x_2;

    int FrameSize, framesize;
    FLOAT* vector;
    FrameSize = xlength * ylength;

    /* x0  = 1.0; */
    x_1 = 1 / x1;
    x_2 = x_1 * x_1;

    vptr00 = NULL; /*compiler gives warning otherwise */
    vptr01 = NULL;

    s = pairity - 1;
    /*************************44444444444444start ****************/
    /*  if (s == zlength - 1)*/
    if ((zlength > 2) && (pairity == 0)) { /* in case the last frame to be done
					   it is done alone  */
                                           /*****************last frame only*****************/
        vector = Vector;

        if ((pairity == 0) && (ylength > 2)) {

            vptr11 = vector;
            vptr10 = vector + xlength;

            vptr21 = vptr11 + FrameSize;
            ;
            vptr20 = vptr10 + FrameSize;
            ;
            k = 0;
            if (xlength > 1) {

                b = vptr11[0] + 2 * vptr21[0];
                f1 = vptr11[1] + 2 * vptr21[1];
                a1 = vptr10[1] + 2 * vptr20[1];
                c1 = vptr10[0] + 2 * (a1 + vptr20[0]);
                vptr10[1] = a1;
                d = 2 * (c1 + f1);
                vptr11[1] = x_1 * (2 * a1 + f1);
                vptr10[0] = c1;
                vptr11[0] = x_2 * (b + d);
                k = 2;
            } else {
                f1 = vptr11[0] + 2 * vptr21[0];
                a1 = vptr10[0] + 2 * vptr20[0];
                vptr10[0] = a1;
                vptr11[0] = x_1 * (2 * a1 + f1);
                k = 1;
            }
            while (k < xlength - 1) {
                d = f1;
                b = vptr11[k] + 2 * vptr21[k];
                c1 = a1 + vptr10[k] + 2 * vptr20[k];
                f1 = vptr11[k + 1] + 2 * vptr21[k + 1];
                a1 = vptr10[k + 1] + 2 * vptr20[k + 1];
                vptr10[k + 1] = a1;
                d += f1;
                c1 += a1;
                d += 2 * c1;
                vptr11[k + 1] = x_1 * (2 * a1 + f1);
                vptr10[k] = c1;
                vptr11[k] = x_2 * (b + d);
                k += 2;
            }
            if (k == xlength - 1) {

                d = 2 * f1;
                b = vptr11[k] + 2 * vptr21[k];
                c1 = 2 * a1 + vptr10[k] + 2 * vptr20[k];
                d += 2 * c1;
                vptr10[k] = c1;
                vptr11[k] = x_2 * (b + d);
            }

            j = 2; /*next lines will be line 0 and 1 */
        }

        else if ((pairity == 0) && (ylength == 2)) {
            vptr11 = vector;
            vptr10 = vector + xlength;

            vptr21 = vptr11 + FrameSize;
            ;
            vptr20 = vptr10 + FrameSize;
            ;
            k = 0;
            if (xlength > 1) {

                b = vptr11[0] + 2 * vptr21[0];
                f1 = vptr11[1] + 2 * vptr21[1];
                a1 = vptr10[1] + 2 * vptr20[1];
                c1 = 2 * a1 + vptr10[0] + 2 * vptr20[0];

                vptr10[1] = /* x0* */ a1;
                d = 2 * (c1 + f1);
                vptr11[1] = x_1 * (2 * a1 + f1);
                vptr10[0] = x_1 * c1;
                vptr11[0] = x_2 * (b + d);
                k = 2;
            } else {
                f1 = vptr11[0];
                a1 = vptr10[0];
                f1 = vptr11[0] + 2 * vptr21[0];
                a1 = vptr10[0] + 2 * vptr20[0];
                vptr10[0] = /* x0* */ a1;
                vptr11[0] = x_1 * (2 * a1 + f1);
                k = 1;
            }
            while (k < xlength - 1) {

                d = f1;
                b = vptr11[k] + 2 * vptr21[k];
                c1 = a1 + vptr10[k] + 2 * vptr20[k];
                f1 = vptr11[k + 1] + 2 * vptr21[k + 1];
                a1 = vptr10[k + 1] + 2 * vptr20[k + 1];

                vptr10[k + 1] = /* x0* */ a1;
                d += f1;
                c1 += a1;
                d += 2 * c1;
                vptr11[k + 1] = x_1 * (2 * a1 + f1);
                vptr10[k] = x_1 * c1;
                vptr11[k] = x_2 * (b + d);
                k += 2;
            }
            if (k == xlength - 1) {

                d = 2 * f1;
                b = vptr11[k] + 2 * vptr21[k];
                c1 = 2 * a1 + vptr10[k] + 2 * vptr20[k];
                d += 2 * c1;
                vptr10[k] = x_1 * c1;
                vptr11[k] = x_2 * (b + d);
            }
            j = 2; /*next lines will be line 0 and 1 */
        } else {   /* if pairity == 1  or ylength = 1 */

            vptr10 = vector;

            vptr20 = vptr10 + FrameSize;
            ;
            k = 0;

            if ((xlength > 1) && (pairity == 0)) {

                a1 = vptr10[1] + 2 * vptr20[1];
                c1 = 2 * a1 + vptr10[0] + 2 * vptr20[0];
                vptr10[1] = /* x0* */ a1;
                vptr10[0] = x_1 * c1;
                k = 2;
            } else {
                a1 = vptr10[0];

                a1 = vptr10[0] + 2 * vptr20[0];
                vptr10[0] = /* x0* */ a1;
                k = 1;
            }

            while (k < xlength - 1)

            {

                c1 = a1 + vptr10[k] + 2 * vptr20[k];
                a1 = vptr10[k + 1] + 2 * vptr20[k + 1];

                vptr10[k + 1] = /* x0* */ a1;
                c1 += a1;
                vptr10[k] = x_1 * c1;
                k += 2;
            }

            if (k == xlength - 1) {
                c1 = vptr10[k] + 2 * (a1 + vptr20[k]);
                vptr10[k] = x_1 * c1;
            }
            j = 1; /*next lines will be line 1 and 2 */
        }

        while (j < ylength - 2) { /* starts main outer loop doing 
					   two lines each time  */

            vptr12 = vector + (j - 1) * xlength;
            vptr11 = vector + j * xlength;
            vptr10 = vector + (j + 1) * xlength;

            vptr21 = vptr11 + FrameSize;
            ;
            vptr20 = vptr10 + FrameSize;
            ;
            k = 0;
            if ((pairity == 0) && (xlength > 1)) {
                e = vptr12[0];

                b = vptr11[0] + 2 * vptr21[0];
                f1 = vptr11[1] + 2 * vptr21[1];
                a1 = vptr10[1] + 2 * vptr20[1];
                c1 = vptr10[0] + 2 * (a1 + vptr20[0]);
                vptr10[1] = a1;
                vptr12[0] = x_1 * e;
                d = e + 2 * f1;
                d += c1;
                g = vptr12[1];
                vptr12[1] = /* x0* */ g;
                vptr11[1] = x_1 * (a1 + f1 + g);
                vptr10[0] = c1;
                vptr11[0] = x_2 * (b + d);
                vptr12[0] = x_1 * e;
                vptr12[1] = /* x0* */ g;
                k = 2;
            }

            else { /* left boundary: first single point at 0 */
                   /* no reflection is needed  */

                f1 = vptr11[0] + 2 * vptr21[0];
                a1 = vptr10[0] + 2 * vptr20[0];
                vptr10[0] = a1;
                g = vptr12[0];
                vptr11[0] = x_1 * (a1 + f1 + g);
                vptr12[0] = /* x0* */ g;
                k = 1;
            }

            /**********HERE IS THE MAIN INNER LOOP TO OPTIMIZE : :  ***********/
            /*  it is running  about  m N M /4   times   for one level 
			  biorthogonal filter  2m+1    on  N    M  images           */
            while (k < xlength - 1) {

                e = vptr12[k];
                d = f1;
                b = vptr11[k] + 2 * vptr21[k];
                c1 = a1 + vptr10[k] + 2 * vptr20[k];
                f1 = vptr11[k + 1] + 2 * vptr21[k + 1];
                a1 = vptr10[k + 1] + 2 * vptr20[k + 1];
                d += e;
                vptr12[k] = x_1 * e;
                vptr10[k + 1] = a1;
                d += f1;
                c1 += a1;
                d += c1;
                g = vptr12[k + 1];
                vptr12[k + 1] = /* x0* */ g;
                vptr11[k + 1] = x_1 * (a1 + f1 + g);
                vptr10[k] = c1;
                vptr11[k] = x_2 * (b + d);
                k += 2;
            }
            /**************END MAIN INNER LOOP***************************/
            if (k == xlength - 1) { /* rigth boundary  if one point at end */
                /* remains we reflect to get an extra point */

                e = vptr12[k];
                d = 2 * f1;
                b = vptr11[k] + 2 * vptr21[k];
                c1 = vptr10[k] + 2 * (a1 + vptr20[k]);
                d += e;
                vptr12[k] = x_1 * e;
                d += c1;
                vptr10[k] = c1;
                vptr11[k] = x_2 * (b + d);
            }

            /************************/

            j += 2;
        }

        if (j == ylength - 2) { /* starts main outer loop doing 
					   two lines each time  */

            vptr12 = vector + (j - 1) * xlength;
            vptr11 = vector + j * xlength;
            vptr10 = vector + (j + 1) * xlength;

            vptr21 = vptr11 + FrameSize;
            ;
            vptr20 = vptr10 + FrameSize;
            ;
            k = 0;
            if ((pairity == 0) && (xlength > 1)) {
                e = vptr12[0];

                b = vptr11[0] + 2 * vptr21[0];
                f1 = vptr11[1] + 2 * vptr21[1];
                a1 = vptr10[1] + 2 * vptr20[1];
                c1 = 2 * a1 + vptr10[0] + 2 * vptr20[0];
                vptr12[0] = x_1 * e;
                vptr10[1] = /* x0* */ a1;
                d = e + 2 * f1;
                d += c1;
                g = vptr12[1];
                vptr12[1] = /* x0* */ g;
                vptr11[1] = x_1 * (a1 + f1 + +g);
                vptr10[0] = x_1 * c1;
                vptr11[0] = x_2 * (b + d);
                k = 2;
            }

            else { /* left boundary: first single point at 0 */
                   /* no reflection is needed  */
                f1 = vptr11[0] + 2 * vptr21[0];
                a1 = vptr10[0] + 2 * vptr20[0];
                vptr10[0] = /* x0* */ a1;
                g = vptr12[0];
                vptr12[0] = /* x0* */ g;
                vptr11[0] = x_1 * (a1 + f1 + g);
                k = 1;
            }

            /**********HERE IS THE MAIN INNER LOOP TO OPTIMIZE : :  ***********/
            /*  it is running  about  m N M /4   times   for one level 
			  biorthogonal filter  2m+1    on  N    M  images           */
            while (k < xlength - 1) {

                e = vptr12[k];
                d = f1;

                b = vptr11[k] + 2 * vptr21[k];
                c1 = a1 + vptr10[k] + 2 * vptr20[k];
                f1 = vptr11[k + 1] + 2 * vptr21[k + 1];
                a1 = vptr10[k + 1] + 2 * vptr20[k + 1];
                vptr12[k] = x_1 * e;
                d += e;
                vptr10[k + 1] = /* x0* */ a1;
                d += f1;
                c1 += a1;
                d += c1;
                g = vptr12[k + 1];
                vptr12[k + 1] = /* x0* */ g;
                vptr11[k + 1] = x_1 * (a1 + f1 + g);
                vptr10[k] = x_1 * c1;
                vptr11[k] = x_2 * (b + d);
                k += 2;
            }
            /**************END MAIN INNER LOOP***************************/
            if (k == xlength - 1) { /* rigth boundary  if one point at end */
                /* remains we reflect to get an extra point */

                e = vptr12[k];
                b = vptr11[k];

                b = vptr11[k] + 2 * vptr21[k];
                c1 = 2 * a1 + vptr10[k] + 2 * vptr20[k];
                d = 2 * f1;
                vptr12[k] = x_1 * e;
                d += e;
                d += c1;
                vptr10[k] = x_1 * c1;
                vptr11[k] = x_2 * (b + d);
            }

            /************************/

            j += 2;
        }

        if (j == ylength - 1) { /* in case the last line remins to be done
					   it is done alone  */

            vptr12 = vector + (j - 1) * xlength;
            vptr11 = vector + j * xlength;

            vptr21 = vptr11 + FrameSize;
            ;
            k = 0;

            if ((pairity == 0) && (xlength > 1)) {

                e = vptr12[0];
                b = vptr11[0];
                f1 = vptr11[1];

                b = vptr11[0] + 2 * vptr21[0];
                f1 = vptr11[1] + 2 * vptr21[1];
                ;
                d = 2 * (e + f1);
                vptr12[0] = x_1 * e;
                g = vptr12[1];
                vptr11[1] = x_1 * (f1 + 2 * g);
                vptr12[1] = /* x0* */ g;
                vptr11[0] = x_2 * (b + d);
                k = 2;
            } else

            {
                f1 = vptr11[0] + 2 * vptr21[0];

                g = vptr12[0];
                vptr11[0] = x_1 * (f1 + 2 * g);
                vptr12[0] = /* x0* */ g;

                k = 1;
            }
            while (k < xlength - 1)

            {

                e = vptr12[k];
                b = vptr11[k];
                d = f1;
                f1 = vptr11[k + 1];
                b = vptr11[k] + 2 * vptr21[k];
                f1 = vptr11[k + 1] + 2 * vptr21[k + 1];

                d += 2 * e;
                vptr12[k] = x_1 * e;
                d += f1;
                g = vptr12[k + 1];
                vptr11[k + 1] = x_1 * (f1 + 2 * g);
                vptr12[k + 1] = /* x0* */ g;
                vptr11[k] = x_2 * (b + d);
                k += 2;
            }

            if (k == xlength - 1) {
                e = vptr12[k];
                b = vptr11[k] + 2 * vptr21[k];
                d = 2 * f1;
                d += 2 * e;
                vptr12[k] = e;
                vptr12[k] = x_1 * e;
                vptr11[k] = x_2 * (b + d);
            }
        }
        s += 2;
    } /*************end z-loop*************/
    else
        s = 1 - pairity;
    /********************444444444444444end*******/

    /**********************3333333333333start*************/
    while (s < zlength - 2) { /* starts main z- loop doing  two lines each time  */
        vector = Vector + s * FrameSize;
        if ((ylength > 2) && (pairity == 0)) {

            vptr01 = vector;
            vptr00 = vector + xlength;

            vptr11 = vptr01 + FrameSize;
            ;
            vptr10 = vptr00 + FrameSize;
            ;

            vptr21 = vptr11 + FrameSize;
            ;
            vptr20 = vptr10 + FrameSize;
            ;
            k = 0;

            if (xlength > 1) {
                b = vptr01[0];
                B = vptr00[0];
                f = vptr01[1];
                a = vptr00[1];
                c = 2 * a + B;
                vptr00[0] = c;
                d = 2 * (c + f);
                F = 2 * a + f;
                vptr01[1] = /* x0* */ F;
                H = d + b;
                vptr01[0] = x_1 * H;

                b = vptr11[0] + b + vptr21[0];
                f1 = vptr11[1] + f + vptr21[1];
                a1 = vptr10[1] + a + vptr20[1];
                c1 = 2 * a1 + vptr10[0] + B + vptr20[0];

                vptr10[1] = a1;

                d = 2 * (c1 + f1);
                vptr11[1] = x_1 * (2 * a1 + f1);
                vptr10[0] = c1;
                vptr11[0] = x_2 * (b + d);
                k = 2;

            } else {
                f = vptr01[0];
                a = vptr00[0];
                F = 2 * a + f;
                vptr01[0] = /* x0* */ F;
                f1 = vptr11[0] + f + vptr21[0];
                a1 = vptr10[0] + a + vptr20[0];
                vptr10[0] = a1;
                vptr11[0] = x_1 * (2 * a1 + f1);
                k = 2;
            }

            while (k < xlength - 1) {

                b = vptr01[k];
                d = f;
                B = vptr00[k];
                c = a + B;
                f = vptr01[k + 1];
                a = vptr00[k + 1];
                d += f;
                c += a;
                vptr00[k] = c;
                d += 2 * c;
                F = 2 * a + f;
                vptr01[k + 1] = /* x0* */ F;
                H = d + b;
                vptr01[k] = x_1 * H;
                d = f1;

                b = vptr11[k] + b + vptr21[k];
                c1 = a1 + vptr10[k] + B + vptr20[k];
                f1 = vptr11[k + 1] + f + vptr21[k + 1];
                a1 = vptr10[k + 1] + a + vptr20[k + 1];

                vptr10[k + 1] = a1;
                d += f1;
                c1 += a1;
                d += 2 * c1;
                vptr11[k + 1] = x_1 * (2 * a1 + f1);
                vptr10[k] = c1;
                vptr11[k] = x_2 * (b + d);
                k += 2;
            }
            if (k == xlength - 1) {

                b = vptr01[k];
                d = 2 * f;
                B = vptr00[k];
                c = 2 * a + B;
                vptr00[k] = c;
                d += 2 * c;
                F = 2 * a + f;
                H = d + b;
                vptr01[k] = x_1 * H;
                d = 2 * f1;

                b = vptr11[k] + b + vptr21[k];
                c1 = 2 * a1 + vptr10[k] + B + vptr20[k];

                d += 2 * c1;
                vptr10[k] = c1;
                vptr11[k] = x_2 * (b + d);
            }

            j = 2; /*next lines will be line 0 and 1 */
        } else if (ylength == 2) {
            if (pairity == 0) {
                vptr01 = vector;
                vptr00 = vector + xlength;
            }
            if (pairity == 1) {
                vptr00 = vector;
                vptr01 = vector + xlength;
            }

            vptr11 = vptr01 + FrameSize;
            ;
            vptr10 = vptr00 + FrameSize;
            ;

            vptr21 = vptr11 + FrameSize;
            ;
            vptr20 = vptr10 + FrameSize;
            ;
            k = 0;

            if ((xlength > 1) && (pairity == 0)) {
                b = vptr01[0];
                B = vptr00[0];
                f = vptr01[1];
                a = vptr00[1];
                c = 2 * a + B;
                vptr00[0] = /* x0* */ c;
                d = 2 * (c + f);
                F = 2 * a + f;
                vptr01[1] = /* x0* */ F;
                vptr00[1] = x1 * a;
                H = d + b;
                vptr01[0] = x_1 * H;

                b = vptr11[0] + b + vptr21[0];
                f1 = vptr11[1] + f + vptr21[1];
                a1 = vptr10[1] + a + vptr20[1];
                c1 = 2 * a1 + vptr10[0] + B + vptr20[0];

                vptr10[1] = /* x0* */ a1;

                d = 2 * (c1 + f1);
                vptr11[1] = x_1 * (2 * a1 + f1);
                vptr10[0] = x_1 * c1;
                vptr11[0] = x_2 * (b + d);
                k = 2;

            } else

            {
                f = vptr01[0];
                a = vptr00[0];
                F = 2 * a + f;
                vptr01[0] = /* x0* */ F;
                vptr00[0] = x1 * a;
                f1 = vptr11[0];
                a1 = vptr10[0];

                f1 = vptr11[0] + f + vptr21[0];
                a1 = vptr10[0] + a + vptr20[0];

                vptr10[0] = /* x0* */ a1;
                vptr11[0] = x_1 * (2 * a1 + f1);
                k = 1;
            }

            while (k < xlength - 1) {

                b = vptr01[k];
                d = f;
                B = vptr00[k];
                c = a + B;
                f = vptr01[k + 1];
                a = vptr00[k + 1];
                d += f;
                c += a;
                vptr00[k] = /* x0* */ c;
                d += 2 * c;
                F = 2 * a + f;
                vptr01[k + 1] = /* x0* */ F;
                vptr00[k + 1] = x1 * a;
                H = d + b;
                vptr01[k] = x_1 * H;

                d = f1;
                b = vptr11[k] + b + vptr21[k];
                c1 = a1 + vptr10[k] + B + vptr20[k];
                f1 = vptr11[k + 1] + f + vptr21[k + 1];
                a1 = vptr10[k + 1] + a + vptr20[k + 1];

                vptr10[k + 1] = /* x0* */ a1;
                d += f1;
                c1 += a1;
                d += 2 * c1;
                vptr11[k + 1] = x_1 * (2 * a1 + f1);
                vptr10[k] = x_1 * c1;
                vptr11[k] = x_2 * (b + d);
                k += 2;
            }
            if (k == xlength - 1) {

                b = vptr01[k];
                B = vptr00[k];
                c = 2 * a + B;
                vptr00[k] = /* x0* */ c;
                d = 2 * (f + c);
                F = 2 * a + f;
                H = d + b;
                vptr01[k] = x_1 * H;
                b = vptr11[k] + b + vptr21[k];
                c1 = vptr10[k] + vptr20[k] + B;
                c1 += 2 * a1;
                d = 2 * (f1 + c1);
                vptr10[k] = x_1 * c1;
                vptr11[k] = x_2 * (b + d);
            }

            j = 2; /*next lines will be line 0 and 1 */

        } else if (ylength == 1) {
            vptr00 = vector;

            vptr10 = vptr00 + FrameSize;
            ;

            vptr20 = vptr10 + FrameSize;
            ;

            if ((pairity == 1) || (xlength == 1)) {
                a = vptr00[0];
                vptr00[0] = x1 * a;
                a1 = vptr10[0] + a + vptr20[0];
                vptr10[0] = /* x0* */ a1;
                k = 1;
            } else {
                B = vptr00[0];
                a = vptr00[1];
                c = 2 * a + B;
                vptr00[0] = /* x0* */ c;
                vptr00[1] = x1 * a;
                a1 = vptr10[1] + a + vptr20[1];
                vptr10[1] = /* x0* */ a1;
                c1 = 2 * a1 + vptr10[0] + B + vptr20[0];
                vptr10[0] = x_1 * c1;
                k = 2;
            }

            while (k < xlength - 1) {

                B = vptr00[k];
                c = a + B;
                a = vptr00[k + 1];
                c += a;
                vptr00[k] = /* x0* */ c;
                vptr00[k + 1] = x1 * a;
                c1 = a1 + vptr10[k] + B + vptr20[k];
                a1 = vptr10[k + 1] + a + vptr20[k + 1];
                vptr10[k + 1] = /* x0* */ a1;
                c1 += a1;
                vptr10[k] = x_1 * c1;
                k += 2;
            }
            if (k == xlength - 1) {
                B = vptr00[k];
                c = 2 * a + B;
                vptr00[k] = /* x0* */ c;
                c1 = 2 * a1 + vptr10[k] + B + vptr20[k];
                vptr10[k] = x_1 * c1;
            }
            j = 1; /*next lines will be line 1 and 2 */
        }

        else /*if ((pairity == 1)*/

        {
            vptr00 = vector;

            vptr10 = vptr00 + FrameSize;
            ;

            vptr20 = vptr10 + FrameSize;
            ;

            if ((pairity == 1) || (xlength == 1)) {
                a = vptr00[0];
                vptr00[0] = a;
                a1 = vptr10[0] + a + vptr20[0];
                vptr10[0] = a1;
                k = 1;
            } else {
                B = vptr00[0];
                a = vptr00[1];
                c = 2 * a + B;
                vptr00[0] = c;
                vptr00[1] = a;
                a1 = vptr10[1] + a + vptr20[1];
                vptr10[1] = a1;
                c1 = 2 * a1 + vptr10[0] + B + vptr20[0];
                vptr10[0] = c1;
                k = 2;
            }

            while (k < xlength - 1) {

                B = vptr00[k];
                c = a + B;
                a = vptr00[k + 1];
                c += a;
                vptr00[k] = c;
                vptr00[k + 1] = a;
                c1 = a1 + vptr10[k] + B + vptr20[k];
                a1 = vptr10[k + 1] + a + vptr20[k + 1];
                vptr10[k + 1] = a1;
                c1 += a1;
                vptr10[k] = c1;
                k += 2;
            }
            if (k == xlength - 1) {
                B = vptr00[k];
                c = 2 * a + B;
                vptr00[k] = c;
                c1 = 2 * a1 + vptr10[k] + B + vptr20[k];
                vptr10[k] = c1;
            }
            j = 1; /*next lines will be line 1 and 2 */
        }

        while (j < ylength - 2) { /* starts main outer loop doing 
					   two lines each time  */

            vptr02 = vector + (j - 1) * xlength;
            vptr01 = vector + j * xlength;
            vptr00 = vector + (j + 1) * xlength;

            vptr12 = vptr02 + FrameSize;
            ;
            vptr11 = vptr01 + FrameSize;
            ;
            vptr10 = vptr00 + FrameSize;
            ;

            vptr21 = vptr11 + FrameSize;
            ;
            vptr20 = vptr10 + FrameSize;
            ;
            k = 0;
            if ((pairity == 0) && (xlength > 1)) {

                E = vptr02[0];
                b = vptr01[0];
                B = vptr00[0];
                f = vptr01[1];
                a = vptr00[1];
                d = E + 2 * f;
                vptr02[0] = /* x0* */ E;
                c = B + 2 * a;
                vptr00[0] = c;
                d += c;
                G = vptr02[1];
                F = a + f + G;
                vptr01[1] = /* x0* */ F;
                vptr02[1] = x1 * G;
                H = d + b;
                vptr01[0] = x_1 * H;
                e = vptr12[0];
                b = vptr11[0] + b + vptr21[0];
                f1 = vptr11[1] + f + vptr21[1];
                a1 = vptr10[1] + a + vptr20[1];
                c1 = 2 * a1 + vptr10[0] + B + vptr20[0];

                vptr10[1] = a1;
                d = e + 2 * f1;
                vptr12[0] = x_1 * e;
                d += c1;
                g = vptr12[1];
                vptr11[1] = x_1 * (a1 + f1 + g);
                vptr12[1] = /* x0* */ g;
                vptr10[0] = c1;
                vptr11[0] = x_2 * (b + d);
                k = 2;
            } else /*if ((pairity == 1)||(xlength==1))  */ {
                d = f;
                f = vptr01[0];
                a = vptr00[0];
                G = vptr02[0];
                F = a + f + G;
                vptr02[0] = x1 * G;
                vptr01[0] = /* x0* */ F;
                vptr00[0] = a;

                f1 = vptr11[0] + f + vptr21[0];
                a1 = vptr10[0] + a + vptr20[0];

                vptr10[0] = a1;
                g = vptr12[0];
                vptr11[0] = x_1 * (a1 + f1 + g);
                vptr12[0] = /* x0* */ g;
                k = 1;
            }

            /**********HERE IS THE MAIN INNER LOOP TO OPTIMIZE : :  ***********/
            /*  it is running  about  m N M /4   times   for one level 
		  biorthogonal filter  2m+1    on  N    M  images           */
            while (k < xlength - 1) {

                E = vptr02[k];
                b = vptr01[k];
                d = f;
                B = vptr00[k];
                c = a + B;
                f = vptr01[k + 1];
                a = vptr00[k + 1];
                d += E;
                vptr02[k] = /* x0* */ E;
                d += f;
                c += a;
                vptr00[k] = c;
                d += c;
                G = vptr02[k + 1];
                F = a + f + G;
                vptr02[k + 1] = x1 * G;
                vptr01[k + 1] = /* x0* */ F;
                H = d + b;
                vptr01[k] = x_1 * H;
                e = vptr12[k];
                d = f1;
                b = vptr11[k] + b + vptr21[k];
                c1 = a1 + vptr10[k] + B + vptr20[k];
                f1 = vptr11[k + 1] + f + vptr21[k + 1];
                a1 = vptr10[k + 1] + a + vptr20[k + 1];
                d += e;
                vptr12[k] = x_1 * e;
                vptr10[k + 1] = a1;
                d += f1;
                c1 += a1;
                d += c1;
                g = vptr12[k + 1];
                vptr11[k + 1] = x_1 * (a1 + f1 + g);
                vptr12[k + 1] = /* x0* */ g;
                vptr10[k] = c1;
                vptr11[k] = x_2 * (b + d);
                k += 2;
            }
            /**************END MAIN INNER LOOP***************************/
            if (k == xlength - 1) { /* rigth boundary  if one point at end */
                /* remains we reflect to get an extra point */

                E = vptr02[k];
                b = vptr01[k];
                d = 2 * f;
                B = vptr00[k];
                c = 2 * a + B;
                d += E;
                vptr02[k] = /* x0* */ E;
                vptr00[k] = c;
                d += c;
                H = d + b;
                vptr01[k] = x_1 * H;
                e = vptr12[k];
                d = 2 * f1;

                b = vptr11[k] + b + vptr21[k];
                c1 = 2 * a1 + vptr10[k] + B + vptr20[k];

                d += e;
                vptr12[k] = x_1 * e;
                d += c1;
                vptr10[k] = c1;
                vptr11[k] = x_2 * (b + d);
            }

            /************************/

            j += 2;
        }

        if (j == ylength - 2) { /* starts main outer loop doing 
					   two lines each time  */

            vptr02 = vector + (j - 1) * xlength;
            vptr01 = vector + j * xlength;
            vptr00 = vector + (j + 1) * xlength;

            vptr12 = vptr02 + FrameSize;
            ;
            vptr11 = vptr01 + FrameSize;
            ;
            vptr10 = vptr00 + FrameSize;
            ;

            vptr21 = vptr11 + FrameSize;
            ;
            vptr20 = vptr10 + FrameSize;
            ;
            k = 0;
            if ((pairity == 0) && (xlength > 1)) {
                E = vptr02[0];
                b = vptr01[0];
                B = vptr00[0];
                f = vptr01[1];
                a = vptr00[1];
                d = E + 2 * f;
                vptr02[0] = /* x0* */ E;
                c = B + 2 * a;
                vptr00[0] = /* x0* */ c;
                d += c;
                G = vptr02[1];
                F = a + f + G;
                vptr02[1] = x1 * G;
                vptr01[1] = /* x0* */ F;
                vptr00[1] = x1 * a;
                H = d + b;
                vptr01[0] = x_1 * H;
                e = vptr12[0];
                b = vptr11[0] + b + vptr21[0];
                f1 = vptr11[1] + f + vptr21[1];
                a1 = vptr10[1] + a + vptr20[1];
                c1 = 2 * a1 + vptr10[0] + B + vptr20[0];
                vptr10[1] = /* x0* */ a1;
                d = e + 2 * f1;
                vptr12[0] = x_1 * e;
                d += c1;
                g = vptr12[1];
                vptr11[1] = x_1 * (a1 + f1 + g);
                vptr12[1] = /* x0* */ g;
                vptr10[0] = x_1 * c1;
                vptr11[0] = x_2 * (b + d);
                k = 2;
            } else /*if ((pairity == 1)||(xlength==1))  */ {
                f = vptr01[0];
                a = vptr00[0];
                G = vptr02[0];
                F = a + f + G;
                vptr02[0] = x1 * G;
                vptr01[0] = /* x0* */ F;
                vptr00[0] = x1 * a;

                f1 = vptr11[0] + f + vptr21[0];
                a1 = vptr10[0] + a + vptr20[0];

                vptr10[0] = /* x0* */ a1;
                g = vptr12[0];
                vptr11[0] = x_1 * (a1 + f1 + g);
                vptr12[0] = /* x0* */ g;
                k = 1;
            }

            /**********HERE IS THE MAIN INNER LOOP TO OPTIMIZE : :  ***********/
            /*  it is running  about  m N M /4   times   for one level 
		  biorthogonal filter  2m+1    on  N    M  images           */
            while (k < xlength - 1) {

                E = vptr02[k];
                b = vptr01[k];
                d = f;
                B = vptr00[k];
                c = a + B;
                f = vptr01[k + 1];
                a = vptr00[k + 1];
                d += E;
                vptr02[k] = /* x0* */ E;
                d += f;
                c += a;
                vptr00[k] = /* x0* */ c;
                d += c;
                G = vptr02[k + 1];
                F = a + f + G;
                vptr02[k + 1] = x1 * G;
                vptr01[k + 1] = /* x0* */ F;
                vptr00[k + 1] = x1 * a;
                H = d + b;
                vptr01[k] = x_1 * H;
                e = vptr12[k];
                d = f1;
                b = vptr11[k] + b + vptr21[k];
                c1 = a1 + vptr10[k] + B + vptr20[k];
                f1 = vptr11[k + 1] + f + vptr21[k + 1];
                a1 = vptr10[k + 1] + a + vptr20[k + 1];
                d += e;
                vptr12[k] = x_1 * e;
                vptr10[k + 1] = /* x0* */ a1;
                d += f1;
                c1 += a1;
                d += c1;
                g = vptr12[k + 1];
                vptr11[k + 1] = x_1 * (a1 + f1 + +g);
                vptr10[k] = x_1 * c1;
                vptr11[k] = x_2 * (b + d);
                k += 2;
            }
            /**************END MAIN INNER LOOP***************************/
            if (k == xlength - 1) { /* rigth boundary  if one point at end */
                /* remains we reflect to get an extra point */

                E = vptr02[k];
                b = vptr01[k];
                d = 2 * f;
                B = vptr00[k];
                c = 2 * a + B;
                d += E;
                vptr02[k] = /* x0* */ E;
                vptr00[k] = /* x0* */ c;
                d += c;
                H = d + b;
                vptr01[k] = x_1 * H;
                e = vptr12[k];
                d = 2 * f1;
                b = vptr11[k] + b + vptr21[k];
                c1 = 2 * a1 + vptr10[k] + B + vptr20[k];

                d += e;
                vptr12[k] = x_1 * e;
                d += c1;
                vptr10[k] = x_1 * c1;
                vptr11[k] = x_2 * (b + d);
            }

            /************************/

            j += 2;
        }

        if (j == ylength - 1) { /* in case the last line remins to be done
				   it is done alone  */

            vptr02 = vector + (j - 1) * xlength;
            vptr01 = vector + j * xlength;

            vptr12 = vptr02 + FrameSize;
            ;
            vptr11 = vptr01 + FrameSize;
            ;

            vptr21 = vptr11 + FrameSize;
            ;
            k = 0;
            if ((xlength > 1) && (pairity == 0)) {
                E = vptr02[0];
                b = vptr01[0];
                f = vptr01[1];
                d = 2 * (E + f);
                vptr02[0] = /* x0* */ E;
                G = vptr02[1];
                F = f + 2 * G;
                vptr02[1] = x1 * G;
                vptr01[1] = /* x0* */ F;
                H = d + b;
                vptr01[0] = x_1 * H;
                e = vptr12[0];
                b = vptr11[0] + b + vptr21[0];
                f1 = vptr11[1] + f + vptr21[1];

                d = 2 * (e + f1);
                vptr12[0] = x_1 * e;
                g = vptr12[1];
                vptr11[1] = x_1 * (f1 + 2 * g);
                vptr12[1] = /* x0* */ g;
                vptr11[0] = x_2 * (b + d);
                k = 2;
            } else {
                f = vptr01[0];
                G = vptr02[0];
                F = f + 2 * G;
                vptr02[0] = x1 * G;
                vptr01[0] = /* x0* */ F;
                f1 = vptr11[0] + f + vptr21[0];
                g = vptr12[0];
                vptr11[0] = x_1 * (f1 + 2 * g);
                vptr12[0] = /* x0* */ g;
                k = 1;
            }
            while (k < xlength - 1)

            {

                E = vptr02[k];
                b = vptr01[k];
                d = f;
                f = vptr01[k + 1];
                d += 2 * E;
                vptr02[k] = /* x0* */ E;
                d += f;
                G = vptr02[k + 1];
                F = f + 2 * G;
                vptr02[k + 1] = x1 * G;
                vptr01[k + 1] = /* x0* */ F;
                H = d + b;
                vptr01[k] = x_1 * H;
                e = vptr12[k];
                d = f1;
                b = vptr11[k] + b + vptr21[k];
                f1 = vptr11[k + 1] + f + vptr21[k + 1];

                d += 2 * e;
                vptr12[k] = x_1 * e;
                d += f1;
                g = vptr12[k + 1];
                vptr11[k + 1] = x_1 * (f1 + 2 * g);
                vptr12[k + 1] = /* x0* */ g;
                vptr11[k] = x_2 * (b + d);
                k += 2;
            }

            if (k == xlength - 1) {

                E = vptr02[k];
                b = vptr01[k];
                d = 2 * (f + E);
                vptr02[k] = /* x0* */ E;
                H = d + b;
                vptr01[k] = x_1 * H;
                e = vptr12[k];
                b = vptr11[k] + b + vptr21[k];
                d = 2 * (f1 + e);
                vptr12[k] = x_1 * e;
                vptr11[k] = x_2 * (b + d);
            }
        }
        s += 2;
    } /*************end z-loop*************/

    /**********************************3333333333333333end ********/

    /****11111111111111111111start***********************/

    if ((s == zlength - 2) || ((zlength == 2) && (pairity == 0))) { /* starts main z- loop doing  two lines each time  */
        vector = Vector + s * FrameSize;
        framesize = FrameSize;
        if ((zlength == 2) && (pairity == 0)) framesize = -FrameSize;

        if ((ylength > 2) && (pairity == 0)) {
            j = 0;
            vptr01 = vector;
            vptr00 = vector + xlength;

            vptr11 = vptr01 + framesize;
            ;
            vptr10 = vptr00 + framesize;
            ;
            k = 0;
            if ((xlength > 1) && (pairity == 0)) {
                b = vptr01[0];
                B = vptr00[0];
                f = vptr01[1];
                a = vptr00[1];
                c = 2 * a + B;
                vptr00[0] = c;
                d = 2 * (c + f);
                F = 2 * a + f;
                vptr01[1] = /* x0* */ F;
                vptr00[1] = a;
                H = d + b;
                vptr01[0] = x_1 * H;
                b = vptr11[0] + 2 * b;
                f1 = vptr11[1] + 2 * f;
                a1 = vptr10[1] + 2 * a;
                c1 = 2 * a1 + vptr10[0] + 2 * B;

                vptr10[1] = a1;
                d = 2 * (c1 + f1);
                vptr11[1] = x_1 * (2 * a1 + f1);
                vptr10[0] = c1;
                vptr11[0] = x_2 * (b + d);
                k = 2;
            }

            else {
                f = vptr01[0];
                a = vptr00[0];
                F = 2 * a + f;
                vptr01[0] = /* x0* */ F;
                vptr00[0] = a;
                f1 = vptr11[0] + 2 * f;
                a1 = vptr10[0] + 2 * a;
                vptr10[0] = a1;
                vptr11[0] = x_1 * (2 * a1 + f1);
                k = 1;
            }

            while (k < xlength - 1) {

                b = vptr01[k];
                d = f;
                B = vptr00[k];
                c = a + B;
                f = vptr01[k + 1];
                a = vptr00[k + 1];
                d += f;
                c += a;
                vptr00[k] = c;
                d += 2 * c;
                F = 2 * a + f;
                vptr01[k + 1] = /* x0* */ F;
                vptr00[k + 1] = a;
                H = d + b;
                vptr01[k] = x_1 * H;
                d = f1;
                b = vptr11[k] + 2 * b;
                c1 = a1 + vptr10[k] + 2 * B;
                f1 = vptr11[k + 1] + 2 * f;
                a1 = vptr10[k + 1] + 2 * a;

                vptr10[k + 1] = /* x0* */ a1;
                d += f1;
                c1 += a1;
                d += 2 * c1;
                vptr11[k + 1] = x_1 * (2 * a1 + f1);
                vptr10[k] = c1;
                vptr11[k] = x_2 * (b + d);
                k += 2;
            }
            if (k == xlength - 1) {
                d = 2 * f;
                b = vptr01[k];
                B = vptr00[k];
                c = 2 * a + B;
                vptr00[k] = /* x0* */ c;
                d = 2 * (c + f);
                H = d + b;
                vptr01[k] = x_1 * H;
                b = vptr11[k] + 2 * b;
                c1 = 2 * a1 + vptr10[k] + 2 * B;
                d = 2 * (c1 + f1);
                vptr10[k] = c1;
                vptr11[k] = x_2 * (b + d);
            }

            j = 2; /*next lines will be line 0 and 1 */
        }

        else if (ylength == 2) {
            j = 0;
            k = 0;
            if (pairity == 0) {
                vptr01 = vector;
                vptr00 = vector + xlength;
            }
            if (pairity == 1) {
                vptr00 = vector;
                vptr01 = vector + xlength;
            }

            vptr11 = vptr01 + framesize;
            ;
            vptr10 = vptr00 + framesize;
            ;

            if ((xlength > 1) && (pairity == 0)) {
                b = vptr01[0];
                B = vptr00[0];
                f = vptr01[1];
                a = vptr00[1];
                c = 2 * a + B;
                vptr00[0] = /* x0* */ c;
                d = 2 * (c + f);
                F = 2 * a + f;
                vptr01[1] = /* x0* */ F;
                vptr00[1] = x1 * a;
                H = d + b;
                vptr01[0] = x_1 * H;

                b = vptr11[0] + 2 * b;
                f1 = vptr11[1] + 2 * f;
                a1 = vptr10[1] + 2 * a;
                c1 = 2 * a1 + vptr10[0] + 2 * B;

                vptr10[1] = /* x0* */ a1;

                d = 2 * (c1 + f1);
                vptr11[1] = x_1 * (2 * a1 + f1);
                c += 2 * c;
                vptr10[0] = x_1 * c1;
                vptr11[0] = x_2 * (b + d);
                k = 2;
            }

            else {
                f = vptr01[0];
                a = vptr00[0];
                F = 2 * a + f;
                vptr01[0] = /* x0* */ F;
                vptr00[0] = x1 * a;
                f1 = vptr11[0];
                a1 = vptr10[0];

                f1 = vptr11[0] + 2 * f;
                a1 = vptr10[0] + 2 * a;
                vptr10[0] = /* x0* */ a1;
                vptr11[0] = x_1 * (2 * a1 + f1);
                k = 1;
            }

            while (k < xlength - 1) {

                b = vptr01[k];
                d = f;
                B = vptr00[k];
                c = a + B;
                f = vptr01[k + 1];
                a = vptr00[k + 1];
                d += f;
                c += a;
                vptr00[k] = /* x0* */ c;
                d += 2 * c;
                F = 2 * a + f;
                vptr01[k + 1] = /* x0* */ F;
                vptr00[k + 1] = x1 * a;
                H = d + b;
                vptr01[k] = x_1 * H;
                d = f1;

                b = vptr11[k] + 2 * b;
                c1 = a1 + vptr10[k] + 2 * B;
                f1 = vptr11[k + 1] + 2 * f;
                a1 = vptr10[k + 1] + 2 * a;

                vptr10[k + 1] = /* x0* */ a1;
                d += f1;
                c1 += a1;
                d += 2 * c1;
                vptr11[k + 1] = x_1 * (2 * a1 + f1);
                vptr10[k] = x_1 * c1;
                vptr11[k] = x_2 * (b + d);
                k += 2;
            }
            if (k == xlength - 1) {
                d = 2 * f;
                b = vptr01[k];
                B = vptr00[k];
                c = 2 * a + B;
                vptr00[k] = /* x0* */ c;
                d += 2 * c;
                H = d + b;
                vptr01[k] = x_1 * H;
                b = vptr11[k] + 2 * b;
                d = 2 * f1;
                c1 = 2 * a1 + vptr10[k] + 2 * B;
                d += 2 * c1;
                vptr10[k] = x_1 * c1;
                vptr11[k] = x_2 * (b + d);
            }

            j = 2; /*next lines will be line 0 and 1 */

        }

        else if (ylength == 1) {

            vptr00 = vector;
            vptr10 = vptr00 + framesize;
            ;
            k = 0;
            if ((xlength > 1) && (pairity == 0)) {
                B = vptr00[0];
                a = vptr00[1];
                c = 2 * a + B;
                vptr00[0] = /* x0* */ c;
                vptr00[1] = x1 * a;

                a1 = vptr10[1] + 2 * a;
                c1 = 2 * a1 + vptr10[0] + 2 * B;
                vptr10[0] = x_1 * c1;
                vptr10[1] = /* x0* */ a1;
                k = 2;
            } else { /*xlength ==1 */
                a = vptr00[0];
                vptr00[0] = x1 * a;

                a1 = vptr10[0] + 2 * a;
                vptr10[0] = /* x0* */ a1;
                k = 1;
            }
            while (k < xlength - 1) {
                B = vptr00[k];
                c = a + B;
                a = vptr00[k + 1];
                c += a;
                vptr00[k] = /* x0* */ c;
                vptr00[k + 1] = x1 * a;
                c1 = a1 + vptr10[k];
                a1 = vptr10[k + 1] + 2 * a;

                c1 += a1 + 2 * B;

                vptr10[k + 1] = /* x0* */ a1;
                vptr10[k] = x_1 * c1;
                k += 2;
            }

            if (k == xlength - 1) {
                B = vptr00[k];
                c = 2 * a + B;
                vptr00[k] = /* x0* */ c;

                c1 = 2 * a1 + vptr10[k] + 2 * B;

                vptr10[k] = x_1 * c1;
            }
            j = 1; /*next lines will be line 1 and 2 */
        }

        else /*if pairity ==1 */

        {

            vptr00 = vector;
            vptr10 = vptr00 + framesize;
            ;
            k = 0;
            if ((xlength > 1) && (pairity == 0)) {
                B = vptr00[0];
                a = vptr00[1];
                c = 2 * a + B;
                vptr00[0] = c;
                vptr00[1] = a;

                a1 = vptr10[1] + 2 * a;
                c1 = 2 * a1 + vptr10[0] + 2 * B;
                vptr10[0] = c1;
                vptr10[1] = a1;
                k = 2;
            } else { /*xlength ==1 */
                a = vptr00[0];
                vptr00[0] = a;

                a1 = vptr10[0] + 2 * a;
                vptr10[0] = a1;
                k = 1;
            }
            while (k < xlength - 1) {
                B = vptr00[k];
                c = a + B;
                a = vptr00[k + 1];
                c += a;
                vptr00[k] = c;
                vptr00[k + 1] = a;
                c1 = a1 + vptr10[k];
                a1 = vptr10[k + 1] + 2 * a;

                c1 += a1 + 2 * B;

                vptr10[k + 1] = a1;
                vptr10[k] = c1;
                k += 2;
            }

            if (k == xlength - 1) {
                B = vptr00[k];
                c = 2 * a + B;
                vptr00[k] = c;

                c1 = 2 * a1 + vptr10[k] + 2 * B;

                vptr10[k] = c1;
            }
            j = 1; /*next lines will be line 1 and 2 */
        }
        while (j < ylength - 2) { /* starts main outer loop doing 
				   two lines each time  */

            vptr02 = vector + (j - 1) * xlength;
            vptr01 = vector + j * xlength;
            vptr00 = vector + (j + 1) * xlength;

            vptr12 = vptr02 + framesize;
            ;
            vptr11 = vptr01 + framesize;
            ;
            vptr10 = vptr00 + framesize;
            ;
            k = 0;
            if ((xlength > 1) && (pairity == 0)) {

                E = vptr02[0];
                b = vptr01[0];
                B = vptr00[0];
                f = vptr01[1];
                a = vptr00[1];
                d = E + 2 * f;
                vptr02[0] = /* x0* */ E;
                c = B + 2 * a;
                vptr00[0] = c;
                d += c;
                G = vptr02[1];
                F = a + f + G;
                vptr02[1] = x1 * G;
                vptr01[1] = /* x0* */ F;
                H = d + b;
                vptr01[0] = x_1 * H;

                e = vptr12[0];

                b = vptr11[0] + 2 * b;
                f1 = vptr11[1] + 2 * f;
                a1 = vptr10[1] + 2 * a;
                c1 = 2 * a1 + vptr10[0] + 2 * B;
                vptr10[1] = a1;
                d = e + 2 * f1;
                vptr12[0] = x_1 * e;
                d += c1;
                g = vptr12[1];
                vptr11[1] = x_1 * (a1 + f1 + g);
                vptr12[1] = /* x0* */ g;
                vptr10[0] = c1;
                vptr11[0] = x_2 * (b + d);
                k = 2;
            }

            else /* if (xlength==1)  pairity =1 loop */
            {
                d = f;
                f = vptr01[0];
                a = vptr00[0];
                G = vptr02[0];
                F = a + f + G;
                vptr02[0] = x1 * G;
                vptr01[0] = /* x0* */ F;
                vptr00[0] = a;
                f1 = vptr11[0] + 2 * f;
                a1 = vptr10[0] + 2 * a;

                vptr10[0] = a1;
                g = vptr12[0];
                vptr11[0] = x_1 * (a1 + f1 + g);
                vptr12[0] = /* x0* */ g;
                k = 1;
            }

            /**********HERE IS THE MAIN INNER LOOP TO OPTIMIZE : :  ***********/
            /*  it is running  about  m x N x M /4   times   for one level 
			biorthogonal filter  2m+1    on  N  x   M  images           */
            while (k < xlength - 1) {
                E = vptr02[k];
                b = vptr01[k];
                d = f;
                B = vptr00[k];
                c = a + B;
                f = vptr01[k + 1];
                a = vptr00[k + 1];
                d += E;
                vptr02[k] = /* x0* */ E;
                d += f;
                c += a;
                vptr00[k] = c;
                d += c;
                G = vptr02[k + 1];
                F = a + f + G;
                vptr02[k + 1] = x1 * G;
                vptr01[k + 1] = /* x0* */ F;
                vptr00[k + 1] = a;
                H = d + b;
                vptr01[k] = x_1 * H;
                e = vptr12[k];
                d = f1;
                b = vptr11[k] + 2 * b;
                c1 = a1 + vptr10[k] + 2 * B;
                f1 = vptr11[k + 1] + 2 * f;
                a1 = vptr10[k + 1] + 2 * a;

                d += e;
                vptr12[k] = x_1 * e;
                vptr10[k + 1] = a1;
                d += f1;
                c1 += a1;
                d += c1;
                g = vptr12[k + 1];
                vptr11[k + 1] = x_1 * (a1 + f1 + +g);
                vptr12[k + 1] = /* x0* */ g;
                vptr10[k] = c1;
                vptr11[k] = x_2 * (b + d);
                k += 2;
            }
            /**************END MAIN INNER LOOP***************************/
            if (k == xlength - 1) { /* rigth boundary  if one point at end */
                                    /* remains we reflect to get an extra point */

                E = vptr02[k];
                b = vptr01[k];
                d = 2 * f;
                B = vptr00[k];
                c = 2 * a + B;
                d += E;
                vptr02[k] = /* x0* */ E;
                vptr00[k] = c;
                d += c;
                H = d + b;
                vptr01[k] = x_1 * H;
                e = vptr12[k];
                d = 2 * f1;

                b = vptr11[k] + 2 * b;
                c1 = 2 * a1 + vptr10[k] + 2 * B;

                d += e;
                vptr12[k] = x_1 * e;
                d += c1;
                vptr10[k] = c1;
                vptr11[k] = x_2 * (b + d);
            }

            /************************/

            j += 2;
        }
        if (j == ylength - 2) { /* starts main outer loop doing 
				   two lines each time  */

            vptr02 = vector + (j - 1) * xlength;
            vptr01 = vector + j * xlength;
            vptr00 = vector + (j + 1) * xlength;

            vptr12 = vptr02 + framesize;
            ;
            vptr11 = vptr01 + framesize;
            ;
            vptr10 = vptr00 + framesize;
            ;
            k = 0;
            if ((xlength > 1) && (pairity == 0)) {

                E = vptr02[0];
                b = vptr01[0];
                B = vptr00[0];
                f = vptr01[1];
                a = vptr00[1];
                d = E + 2 * f;
                vptr02[0] = /* x0* */ E;
                c = B + 2 * a;
                vptr00[0] = /* x0* */ c;
                d += c;
                G = vptr02[1];
                F = a + f + G;
                vptr02[1] = x1 * G;
                vptr01[1] = /* x0* */ F;
                vptr00[1] = x1 * a;
                H = d + b;
                vptr01[0] = x_1 * H;
                e = vptr12[0];

                b = vptr11[0] + 2 * b;
                f1 = vptr11[1] + 2 * f;
                a1 = vptr10[1] + 2 * a;
                c1 = 2 * a1 + vptr10[0] + 2 * B;

                vptr10[1] = /* x0* */ a1;
                d = e + 2 * f1;
                vptr12[0] = x_1 * e;
                d += c1;
                g = vptr12[1];
                vptr11[1] = x_1 * (a1 + f1 + +g);
                vptr12[1] = /* x0* */ g;
                vptr10[0] = x_1 * c1;
                vptr11[0] = x_2 * (b + d);
                k = 2;
            }

            else /* if (xlength==1)  pairity =1 loop */
            {
                d = f;
                f = vptr01[0];
                a = vptr00[0];
                G = vptr02[0];
                F = a + f + G;
                vptr02[0] = x1 * G;
                vptr01[0] = /* x0* */ F;
                vptr00[0] = x1 * a;
                f1 = vptr11[0];
                a1 = vptr10[0];
                f1 = vptr11[0] + 2 * f;
                a1 = vptr10[0] + 2 * a;

                vptr10[0] = /* x0* */ a1;
                g = vptr12[0];
                vptr11[0] = x_1 * (a1 + f1 + g);
                vptr12[0] = /* x0* */ g;
                k = 1;
            }

            /**********HERE IS THE MAIN INNER LOOP TO OPTIMIZE : :  ***********/
            /*  it is running  about  m x N x M /4   times   for one level 
			biorthogonal filter  2m+1    on  N  x   M  images           */
            while (k < xlength - 1) {
                E = vptr02[k];
                b = vptr01[k];
                d = f;
                B = vptr00[k];
                c = a + B;
                f = vptr01[k + 1];
                a = vptr00[k + 1];
                d += E;
                vptr02[k] = /* x0* */ E;
                d += f;
                c += a;
                vptr00[k] = /* x0* */ c;
                d += c;
                G = vptr02[k + 1];
                F = a + f + G;
                vptr02[k + 1] = x1 * G;
                vptr01[k + 1] = /* x0* */ F;
                vptr00[k + 1] = x1 * a;
                H = d + b;
                vptr01[k] = x_1 * H;
                e = vptr12[k];
                d = f1;
                b = vptr11[k] + 2 * b;
                c1 = a1 + vptr10[k] + 2 * B;
                f1 = vptr11[k + 1] + 2 * f;
                a1 = vptr10[k + 1] + 2 * a;
                d += e;
                vptr12[k] = x_1 * e;
                vptr10[k + 1] = /* x0* */ a1;
                d += f1;
                c1 += a1;
                d += c1;
                B = 2 * F;
                g = vptr12[k + 1];
                vptr11[k + 1] = x_1 * (a1 + f1 + +g);
                vptr12[k + 1] = /* x0* */ g;
                vptr10[k] = x_1 * c1;
                vptr11[k] = x_2 * (b + d);
                k += 2;
            }
            /**************END MAIN INNER LOOP***************************/
            if (k == xlength - 1) { /* rigth boundary  if one point at end */
                                    /* remains we reflect to get an extra point */

                E = vptr02[k];
                b = vptr01[k];
                d = 2 * f;
                B = vptr00[k];
                c = 2 * a + B;
                d += E;
                vptr02[k] = /* x0* */ E;
                vptr00[k] = /* x0* */ c;
                d += c;
                H = d + b;
                vptr01[k] = x_1 * H;
                e = vptr12[k];
                d = 2 * f1;
                c1 = 2 * a1 + vptr10[k];
                b = vptr11[k] + 2 * b;
                c1 = 2 * a1 + vptr10[k] + 2 * B;
                d += e;
                vptr12[k] = x_1 * e;
                d += c1;
                vptr10[k] = x_1 * c1;
                vptr11[k] = x_2 * (b + d);
            }

            /************************/

            j += 2;
        }

        if (j == ylength - 1) { /* in case the last line remins to be done
				   it is done alone  */

            vptr02 = vector + (j - 1) * xlength;
            vptr01 = vector + j * xlength;

            vptr12 = vptr02 + framesize;
            ;
            vptr11 = vptr01 + framesize;
            ;
            k = 0;
            if ((xlength > 1) && (pairity == 0)) {

                E = vptr02[0];
                b = vptr01[0];
                f = vptr01[1];
                d = 2 * (E + f);
                vptr02[0] = /* x0* */ E;
                G = vptr02[1];
                F = f + 2 * G;
                vptr02[1] = x1 * G;
                vptr01[1] = /* x0* */ F;
                H = d + b;
                vptr01[0] = x_1 * H;
                e = vptr12[0];
                b = vptr11[0] + 2 * b;
                f1 = vptr11[1] + 2 * f;
                d = 2 * (e + f1);
                vptr12[0] = x_1 * e;
                g = vptr12[1];
                vptr11[1] = x_1 * (f1 + 2 * g);
                vptr12[1] = /* x0* */ g;
                vptr11[0] = x_2 * (b + d);
                k = 2;
            } else {
                f = vptr01[0];
                G = vptr02[0];
                F = f + 2 * G;
                vptr02[0] = x1 * G;
                vptr01[0] = /* x0* */ F;
                f1 = vptr11[0] + 2 * f;
                g = vptr12[0];
                vptr11[0] = x_1 * (f1 + 2 * g);
                vptr12[0] = /* x0* */ g;
                k = 1;
            }

            while (k < xlength - 1)

            {

                E = vptr02[k];
                b = vptr01[k];
                d = f;
                f = vptr01[k + 1];
                d += 2 * E;
                vptr02[k] = /* x0* */ E;
                d += f;
                G = vptr02[k + 1];
                F = f + 2 * G;
                vptr02[k + 1] = x1 * G;
                vptr01[k + 1] = /* x0* */ F;
                H = d + b;
                vptr01[k] = x_1 * H;

                e = vptr12[k];
                d = f1;
                b = vptr11[k] + 2 * b;
                f1 = vptr11[k + 1] + 2 * f;
                d += 2 * e;
                vptr12[k] = x_1 * e;
                d += f1;
                g = vptr12[k + 1];
                vptr11[k + 1] = x_1 * (f1 + 2 * g);
                vptr12[k + 1] = /* x0* */ g;
                vptr11[k] = x_2 * (b + d);
                k += 2;
            }

            if (k == xlength - 1) {
                e = vptr02[k];
                b = vptr01[k];
                d = 2 * (f + e);
                vptr02[k] = /* x0* */ e;
                H = d + b;
                vptr01[k] = x_1 * H;
                e = vptr12[k];
                b = vptr11[k] + 2 * b;
                vptr12[k] = x_1 * e;
                d = 2 * (f1 + e);
                vptr11[k] = x_2 * (b + d);
            }

        } /* end last line */

        s += 2;
    } /*************end z-loop*************/

    /******************11111111111111111111end ***/

    /*************2222222222222start************/
    if ((s == zlength - 1) || (zlength == 1)) { /* starts main z- loop doing  two lines each time  */
        if (zlength == 1)
            vector = Vector;
        else
            vector = Vector + s * FrameSize;
        if ((ylength > 2) && (pairity == 0))
        /*if(pairity == 0)   thus Zlength ==1 */
        {

            vptr01 = vector;
            vptr00 = vector + xlength;
            k = 0;

            if (xlength > 1) {
                b = vptr01[0];
                B = vptr00[0];
                f = vptr01[1];
                a = vptr00[1];
                c = 2 * a + B;
                vptr00[0] = c;
                d = 2 * (c + f);
                F = 2 * a + f;
                vptr01[1] = /* x0* */ F;
                vptr00[1] = a;
                H = d + b;
                vptr01[0] = x_1 * H;
                k = 2;
            } else /* xlength =1 */
            {
                b = vptr01[0];
                f = vptr01[0];
                a = vptr00[0];
                F = 2 * a + f;
                vptr01[0] = /* x0* */ F;
                vptr00[0] = a;
                k = 2;
            }

            while (k < xlength - 1) {

                b = vptr01[k];
                d = f;
                B = vptr00[k];
                c = a + B;
                f = vptr01[k + 1];
                a = vptr00[k + 1];
                d += f;
                c += a;
                vptr00[k] = c;
                d += 2 * c;
                F = 2 * a + f;
                vptr01[k + 1] = /* x0* */ F;
                vptr00[k + 1] = a;
                H = d + b;
                vptr01[k] = x_1 * H;
                k += 2;
            }
            if (k == xlength - 1) {
                b = vptr01[k];
                d = 2 * f;
                B = vptr00[k];
                c = 2 * a + B;
                vptr00[k] = c;
                d += 2 * c;
                F = 2 * a + f;
                H = d + b;
                vptr01[k] = x_1 * H;
            }
            j = 2; /*next lines will be line 0 and 1 */
        }

        else if (ylength == 2) {
            j = 0;

            k = 0;
            if (pairity == 0) {
                vptr01 = vector;
                vptr00 = vector + xlength;
            }
            if (pairity == 1) {
                vptr00 = vector;
                vptr01 = vector + xlength;
            }
            if ((xlength > 1) && (pairity == 0)) {
                b = vptr01[0];
                B = vptr00[0];
                f = vptr01[1];
                a = vptr00[1];
                c = 2 * a + B;
                vptr00[0] = /* x0* */ c;
                d = 2 * (c + f);
                F = 2 * a + f;
                vptr01[1] = /* x0* */ F;
                vptr00[1] = x1 * a;
                H = d + b;
                vptr01[0] = x_1 * H;
                k = 2;
            } else /* xlength =1 */
            {
                b = vptr01[0];
                f = vptr01[0];
                a = vptr00[0];
                F = 2 * a + f;
                vptr01[0] = /* x0* */ F;
                vptr00[0] = x1 * a;
                k = 1;
            }

            while (k < xlength - 1) {

                b = vptr01[k];
                d = f;
                B = vptr00[k];
                c = a + B;
                f = vptr01[k + 1];
                a = vptr00[k + 1];
                d += f;
                c += a;
                vptr00[k] = /* x0* */ c;
                d += 2 * c;
                F = 2 * a + f;
                vptr01[k + 1] = /* x0* */ F;
                vptr00[k + 1] = x1 * a;
                H = d + b;
                vptr01[k] = x_1 * H;
                k += 2;
            }
            if (k == xlength - 1) {
                b = vptr01[k];
                d = 2 * f;
                B = vptr00[k];
                c = 2 * a + B;
                vptr00[k] = /* x0* */ c;
                d += 2 * c;
                F = 2 * a + f;
                H = d + b;
                vptr01[k] = x_1 * H;
            }
            j = 2; /*next lines will be line 0 and 1 */
        }

        else if (ylength == 1) {

            {

                vptr00 = vector;
                k = 0;
                if ((pairity == 0) && (xlength > 1)) {
                    B = vptr00[0];
                    a = vptr00[1];
                    c = 2 * a + B;
                    vptr00[0] = /* x0* */ c;
                    vptr00[1] = x1 * a;
                    k = 2;
                } else {
                    a = vptr00[0];
                    vptr00[0] = x1 * a;
                    k = 1;
                }
                while (k < xlength - 1) {
                    B = vptr00[k];
                    c = a + B;
                    a = vptr00[k + 1];
                    c += a;
                    vptr00[k] = /* x0* */ c;
                    vptr00[k + 1] = x1 * a;
                    k += 2;
                }
                if (k == xlength - 1) {
                    B = vptr00[k];
                    c = 2 * a + B;
                    vptr00[k] = /* x0* */ c;
                }

                j = 1; /*next lines will be line 1 and 2 */
            }

        } else /*if (pairity == 1) */
        {

            vptr00 = vector;
            k = 1;
            if (xlength > 1) {
                a = vptr00[0];
            }

            while (k < xlength - 1) {
                B = vptr00[k];
                c = a + B;
                a = vptr00[k + 1];
                c += a;
                vptr00[k] = c;
                k += 2;
            }
            if (k == xlength - 1) {
                B = vptr00[k];
                c = 2 * a + B;
                vptr00[k] = c;
            }

            j = 1; /*next lines will be line 1 and 2 */
        }

        while (j < ylength - 2) {

            /* starts main outer j  loop doing 
				   two lines each time  */

            vptr02 = vector + (j - 1) * xlength;
            vptr01 = vector + j * xlength;
            vptr00 = vector + (j + 1) * xlength;
            k = 0;
            if ((pairity == 0) && (xlength > 1)) {

                E = vptr02[0];
                b = vptr01[0];
                B = vptr00[0];
                f = vptr01[1];
                a = vptr00[1];
                d = E + 2 * f;
                c = B + 2 * a;
                vptr00[0] = c;
                d += c;
                G = vptr02[1];
                F = a + f + G;
                vptr02[1] = x1 * G;
                vptr01[1] = /* x0* */ F;
                vptr00[1] = a;
                H = d + b;
                vptr01[0] = x_1 * H;
                k = 2;
            }

            else { /* left boundary: first single point at 0 */
                /* no reflection is needed  */
                d = f;
                f = vptr01[0];
                a = vptr00[0];
                G = vptr02[0];
                F = a + f + G;
                vptr02[0] = x1 * G;
                vptr01[0] = /* x0* */ F;
                vptr00[0] = a;
                k = 1;
            }

            /**********HERE IS THE MAIN INNER LOOP TO OPTIMIZE : :  ***********/
            /*  it is running  about  m x N x M /4   times   for one level 
		  biorthogonal filter  2m+1    on  N  x M  images           */
            while (k < xlength - 1) {

                E = vptr02[k];
                b = vptr01[k];
                d = f;
                B = vptr00[k];
                c = a + B;
                f = vptr01[k + 1];
                a = vptr00[k + 1];
                d += E;
                vptr02[k] = /* x0* */ E;
                d += f;
                c += a;
                vptr00[k] = /* x0* */ c;
                d += c;
                G = vptr02[k + 1];
                F = a + f + G;
                vptr02[k + 1] = x1 * G;
                vptr01[k + 1] = /* x0* */ F;
                vptr00[k + 1] = a;
                H = d + b;
                vptr01[k] = x_1 * H;
                k += 2;
            }
            /**************END MAIN INNER LOOP***************************/
            if (k == xlength - 1) { /* rigth boundary  if one point at end */
                /* remains we reflect to get an extra point */

                E = vptr02[k];
                b = vptr01[k];
                d = 2 * f;
                B = vptr00[k];
                c = 2 * a + B;
                d += E;
                vptr02[k] = /* x0* */ E;
                vptr00[k] = c;
                d += c;
                H = d + b;
                vptr01[k] = x_1 * H;
            }

            /************************/

            j += 2;
        }

        if (j == ylength - 2) {

            /* starts main outer j  loop doing 
				   two lines each time  */

            vptr02 = vector + (j - 1) * xlength;
            vptr01 = vector + j * xlength;
            vptr00 = vector + (j + 1) * xlength;
            k = 0;
            if ((pairity == 0) && (xlength > 1)) {

                E = vptr02[0];
                b = vptr01[0];
                B = vptr00[0];
                f = vptr01[1];
                a = vptr00[1];
                d = E + 2 * f;
                vptr02[0] = /* x0* */ E;
                c = B + 2 * a;
                vptr00[0] = /* x0* */ c;
                d += c;
                G = vptr02[1];
                F = a + f + G;
                vptr02[1] = x1 * G;
                vptr01[1] = /* x0* */ F;
                vptr00[1] = x1 * a;
                H = d + b;
                vptr01[0] = x_1 * H;
                k = 2;
            }

            else { /* left boundary: first single point at 0 */
                /* no reflection is needed  */
                f = vptr01[0];
                a = vptr00[0];
                G = vptr02[0];
                F = a + f + G;
                vptr02[0] = x1 * G;
                vptr01[0] = /* x0* */ F;
                vptr00[0] = x1 * a;
                k = 1;
            }

            /**********HERE IS THE MAIN INNER LOOP TO OPTIMIZE : :  ***********/
            /*  it is running  about  m x N x M /4   times   for one level 
		  biorthogonal filter  2m+1    on  N  x   M  images           */
            while (k < xlength - 1) {

                E = vptr02[k];
                b = vptr01[k];
                d = f;
                B = vptr00[k];
                c = a + B;
                f = vptr01[k + 1];
                a = vptr00[k + 1];
                d += E;
                vptr02[k] = /* x0* */ E;
                d += f;
                c += a;
                vptr00[k] = /* x0* */ c;
                d += c;
                G = vptr02[k + 1];
                F = a + f + G;
                vptr02[k + 1] = x1 * G;
                vptr01[k + 1] = /* x0* */ F;
                vptr00[k + 1] = x1 * a;
                H = d + b;
                vptr01[k] = x_1 * H;
                k += 2;
            }
            /**************END MAIN INNER LOOP***************************/
            if (k == xlength - 1) { /* rigth boundary  if one point at end */
                /* remains we reflect to get an extra point */

                E = vptr02[k];
                b = vptr01[k];
                d = 2 * f;
                B = vptr00[k];
                c = 2 * a + B;
                d += E;
                vptr02[k] = /* x0* */ E;
                vptr00[k] = /* x0* */ c;
                d += c;
                H = d + b;
                vptr01[k] = x_1 * H;
            }

            /************************/

            j += 2;
        }

        if (j == ylength - 1) {

            vptr02 = vector + (j - 1) * xlength;
            vptr01 = vector + j * xlength;
            k = 0;
            /* in case the last line remins to be done
		 it is done alone  */
            if ((xlength > 1) && (pairity == 0)) {
                E = vptr02[0];
                b = vptr01[0];
                f = vptr01[1];
                d = 2 * (E + f);
                G = vptr02[1];
                F = f + 2 * G;
                vptr01[1] = /* x0* */ F;
                vptr02[1] = x1 * G;
                H = d + b;
                vptr01[0] = x_1 * H;
                vptr02[0] = /* x0* */ E;
                k = 2;
            } else /* if (xlength==1)*/
            {
                f = vptr01[0];
                G = vptr02[0];
                F = f + 2 * G;
                vptr02[0] = x1 * G;
                vptr01[0] = /* x0* */ F;
                k = 1;
            }

            while (k < xlength - 1)

            {

                E = vptr02[k];
                b = vptr01[k];
                d = f;
                f = vptr01[k + 1];
                d += 2 * E;
                d += f;
                G = vptr02[k + 1];
                F = f + 2 * G;
                vptr01[k + 1] = /* x0* */ F;
                vptr02[k + 1] = x1 * G;
                H = d + b;
                vptr01[k] = x_1 * H;
                vptr02[k] = /* x0* */ E;
                k += 2;
            }
            if (k == xlength - 1) {
                E = vptr02[k];
                b = vptr01[k];
                d = 2 * f;
                d += 2 * E;
                H = d + b;
                vptr01[k] = x_1 * H;
                vptr02[k] = /* x0* */ E;
            }
        }
        s = 1;
    } /*************end z-loop*************/

    /**********************222222222end*******************/

    return;
}
