
/*file decompose3m.c */
/* file defining procedure for decomposing image into a bitstream
 using biorthogonal filters and bitcoding.


 
                                   Algorithm and code developped 
                                               by
                                      Jan-Olov Stromberg

                                KTH, Stockholm, Sweden
                     Fast Mathematical Algorithms&Hardware, Hamden, CT, USA


                             Preliminay version by Octobler 19, 1998

*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#define TRIG_UNSIGNEDCHAR
#include "bio_parameters.h"
/*#include "coding_Color.h"*/
#include "bio.h"
/*
#ifndef FLOAT
#define FLOAT float
#endif
*/

/*#define FULLIMAGE   0,0,0,0,0*/
#define FULLIMAGE 0, 0, 0, 0, 0
#define FLLSKIPIMAGE 0, 0
#define XPARTIALLENGTH (xlength)
#define YPARTIALLENGTH (ylength)
#define XFULLENGTH (xlength)
#define YFULLENGTH (ylength)

#define PRINT 0
#define NEW 1
/************************/

int wavelet_decompose3(FLOAT* inspacevector,
                       int Xlength,
                       int Ylength,
                       int Zlength,
                       char Filterlength,
                       char Levels,
                       char minZLevels,
                       char MaxZLevels,
                       char minXYLevels,
                       char MaxXYLevels,
                       char Skip,
                       FLOAT* covector,
                       int* colength_ptr,
                       char ifnotSilent)

{

    int j;
    int ifnotallskip = 1;
    FLOAT* vector = NULL;
    FLOAT* invp;
    FLOAT* vp;
    int Size;
    int mxlength, mylength, mzlength;
    int scale, xlength, ylength, zlength, maxscale;
    /*int finalxlength,finalylength,finalzlength;*/
    int LLLsubsize;
    int LLHsubsize;
    int LHLsubsize;
    int LHHsubsize;
    int HLHsubsize;
    int HHLsubsize;
    int HHHsubsize;
    int HLLsubsize;
    int LLsubsize;
    int LHsubsize;
    int HLsubsize;
    int HHsubsize;

    int LLHlength;
    int LHLlength;
    int LHHlength;
    int HLLlength;
    int zscale;
    int xyscale;
    int maxzscale;
    FLOAT* LLL = NULL, * LHH = NULL, * LLH = NULL, * LHL = NULL;
    FLOAT* HLL = NULL, * HHH = NULL, * HLH = NULL, * HHL = NULL;
    FLOAT* LLLv = NULL, * LHHv = NULL, * LLHv = NULL, * LHLv = NULL;
    FLOAT* HLLv = NULL, * HHHv = NULL, * HLHv = NULL, * HHLv = NULL;
    FLOAT* LL, *LH, *HL, *HH;
    FLOAT* H, *L;
    FLOAT* HH0;
    int skipsize = 0;
    int xylength;
    int xy0length;
    int z0length;
    int x0length;
    int y0length;

    void (*bioD_2d)();
    void (*bioD_skip_2d)();
    void (*bioD_2d_char)();
    void (*bioD_skip_2d_char)();

    void (*bioD_2d_y)();
    void (*bioD_3d)();
    void (*bioD_skip_3d)();
    void (*bioD_3d_char)();
    void (*bioD_skip_3d_char)();

    if (MaxXYLevels < Levels) {
        printf("Program has  yet implemented the parameter case  minXYLevels < Levels \n");
        return 0;
    }

    getfilter(&bioD_2d, &bioD_skip_2d,
              &bioD_2d_char, &bioD_skip_2d_char, Filterlength, 3);
    getfilter(&bioD_2d_y, &bioD_skip_3d,
              &bioD_3d_char, &bioD_skip_3d_char, Filterlength, 2);
    getfilter(&bioD_3d, &bioD_skip_3d,
              &bioD_3d_char, &bioD_skip_3d_char, Filterlength, 7);

    xlength = ((Xlength - 1) >> Skip) + 1;
    ylength = ((Ylength - 1) >> Skip) + 1;
    zlength = ((Zlength - 1) >> Skip) + 1;

    maxscale = Levels - 1;
    maxzscale = MaxZLevels - 1;
    scale = 0;

    Size = Xlength * Ylength * Zlength;

    LLL = inspacevector;

    while ((scale <= maxscale) && (scale < Skip)) {
        if ((scale == maxscale) && (scale >= minZLevels - 1)) LLL = covector;
        xlength = 1 + ((Xlength - 1) >> scale);
        ylength = 1 + ((Ylength - 1) >> scale);
        zlength = 1 + ((Zlength - 1) >> scale);
        if (scale > maxzscale) {
            zlength = 1 + ((Zlength - 1) >> (maxzscale + 1));
            z0length = zlength;
            invp = inspacevector;
            vp = vector;

            LL = LLL;
            for (j = 0; j < zlength; j++) {
                if (!scale)
                    bioD_skip_2d_char(invp, LHH, LHL, LLH, LL, xlength, ylength,
                                      FULLIMAGE,
                                      0);
                else
                    bioD_skip_2d(vp, LHH, LHL, LLH, LL, xlength, ylength,
                                 FULLIMAGE,
                                 0);

                invp += xlength * ylength;
                vp += xlength * ylength;
                LL += ((xlength + 1) / 2) * ((ylength + 1) / 2);
            }
            vector = LLL;
        } else {
            xlength = 1 + ((Xlength - 1) >> scale);
            ylength = 1 + ((Ylength - 1) >> scale);
            zlength = 1 + ((Zlength - 1) >> scale);

            if (!scale)
                bioD_skip_3d_char(inspacevector, HHH, HHL, HLH, HLL, LHH, LHL, LLH, LLL, xlength, ylength, zlength,
                                  0);
            else
                bioD_skip_3d(vector, HHH, HHL, HLH, HLL, LHH, LHL, LLH, LLL, xlength, ylength, zlength,
                             0);
            vector = LLL;
        }
        scale++;
    }

    xlength = 1 + ((Xlength - 1) >> scale);
    ylength = 1 + ((Ylength - 1) >> scale);
    zlength = 1 + ((Zlength - 1) >> scale);
    if (scale > maxzscale) zlength = 1 + ((Zlength - 1) >> (maxzscale + 1));

    skipsize = xlength * ylength * zlength;

    LLLv = covector;

    while (scale <= maxscale) {

        xlength = 1 + ((Xlength - 1) >> scale);
        ylength = 1 + ((Ylength - 1) >> scale);
        if (scale > maxzscale)
            zlength = 1 + ((Zlength - 1) >> (maxzscale + 1));
        else
            zlength = 1 + ((Zlength - 1) >> scale);

        mxlength = (xlength + 1) >> 1;
        mylength = (ylength + 1) >> 1;
        if (scale > maxzscale)
            mzlength = zlength;
        else
            mzlength = (zlength + 1) / 2;

        vector = LLL;
        LLLsubsize = mzlength * mylength * mxlength;
        LLHsubsize = mzlength * mylength * (xlength - mxlength);
        LHLsubsize = mzlength * (ylength - mylength) * mxlength;
        LHHsubsize = mzlength * (ylength - mylength) * (xlength - mxlength);
        HLLsubsize = (zlength - mzlength) * mylength * mxlength;
        HLHsubsize = (zlength - mzlength) * mylength * (xlength - mxlength);
        HHLsubsize = (zlength - mzlength) * (ylength - mylength) * mxlength;
        HHHsubsize = (zlength - mzlength) * (ylength - mylength) * (xlength - mxlength);

        HHHv = LLLv;
        HHLv = HHHv + HHHsubsize;
        HLHv = HHLv + HHLsubsize;
        if (minZLevels >= minXYLevels) {
            HLLv = HLHv + HLHsubsize;
            LHHv = HLLv + HLLsubsize;
            LHLv = LHHv + LHHsubsize;
            LLHv = LHLv + LHLsubsize;
            LLLv = LLHv + LLHsubsize;
        } else {
            LHHv = HLHv + HLHsubsize;
            LHLv = LHHv + LHHsubsize;
            LLHv = LHLv + LHLsubsize;
            HLLv = LLHv + LLHsubsize;
            LLLv = HLLv + HLLsubsize;
        }
        HLH = HLHv;
        HHL = HHLv;
        HHH = HHHv;
        if (scale < minXYLevels - 1) {
            HLL = HLLv + LLLsubsize;
        } else {
            HLL = HLLv;
        }
        if (scale < minZLevels - 1) {
            LLH = LLHv + LLLsubsize;
            LHL = LHLv + LLLsubsize;
            LHH = LHHv + LLLsubsize;
        } else {
            LLH = LLHv;
            LHL = LHLv;
            LHH = LHHv;
        }

        if ((scale == maxscale) && (scale >= minZLevels - 1) &&
            ((scale >= minXYLevels - 1) || (HLLsubsize == 0))) LLL = LLLv;

        if ((scale > maxzscale) && (scale < MaxXYLevels)) {
            z0length = zlength;
            invp = inspacevector;
            vp = vector;

            HH = LHH;
            HL = LHL;
            LH = LLH;
            LL = LLL;
            HHsubsize = (xlength >> 1) * (ylength >> 1);
            LHsubsize = ((xlength) >> 1) * ((ylength + 1) >> 1);
            HLsubsize = ((xlength + 1) >> 1) * ((ylength) >> 1);
            LLsubsize = ((xlength + 1) >> 1) * ((ylength + 1) >> 1);
            xylength = xlength * ylength;

            if ((!scale)) {
                for (j = 0; j < zlength; j++) {
                    bioD_2d_char(invp, HH, HL, LH, LL, xlength, ylength,
                                 FULLIMAGE,
                                 ifnotallskip);
                    invp += xylength;
                    vp += xylength;
                    HH += HHsubsize;
                    LH += LHsubsize;
                    HL += HLsubsize;
                    LL += LLsubsize;
                }
            } else {
                for (j = 0; j < zlength; j++) {
                    bioD_2d(vp, HH, HL, LH, LL, xlength, ylength,
                            FULLIMAGE,
                            ifnotallskip);
                    invp += xylength;
                    vp += xylength;
                    HH += HHsubsize;
                    LH += LHsubsize;
                    HL += HLsubsize;
                    LL += LLsubsize;
                }
            }

        } else if ((scale <= maxzscale) && (scale < MaxXYLevels)) {
            zlength = 1 + ((Zlength - 1) >> scale);
            z0length = ((zlength + 1) >> 1);
            if ((!scale)) {

                bioD_3d_char(inspacevector, HHH, HHL, HLH, HLL, LHH, LHL, LLH, LLL,
                             xlength, ylength, zlength, ifnotallskip);
            } else {
                bioD_3d(vector, HHH, HHL, HLH, HLL, LHH, LHL, LLH, LLL,
                        xlength, ylength, zlength, ifnotallskip);
            }
        } else if ((scale <= maxzscale) && (scale >= MaxXYLevels)) {
            xylength = xlength * ylength;
            if (zscale == minZLevels - 1) LLL = HLL + xylength * (zlength >> 1);
            bioD_2d_y(vector, HLL, LLL,
                      xylength, zlength, 1);
        }

        scale++;

        if (ifnotallskip) {

            if (minXYLevels <= minZLevels) {
                x0length = (xlength + 1) >> 1;
                y0length = (ylength + 1) >> 1;
                z0length = zlength >> 1;
                HLLlength = x0length * y0length * z0length;
                if (HLLlength > 0) {
                    xyscale = scale;

                    HH0 = HLLv;

                    while (xyscale < minXYLevels) {

                        HHsubsize = (x0length >> 1) * (y0length >> 1);
                        HLsubsize = ((x0length + 1) >> 1) * ((y0length) >> 1);
                        LHsubsize = ((x0length) >> 1) * ((y0length + 1) >> 1);
                        LLsubsize = ((x0length + 1) >> 1) * ((y0length + 1) >> 1);
                        xy0length = x0length * y0length;

                        HH = HH0;
                        HL = HH + HHsubsize * z0length;
                        LH = HL + HLsubsize * z0length;
                        HH0 = LH + LHsubsize * z0length;
                        LL = HLL;
                        vp = HLL;

                        if (xyscale == minXYLevels - 1) LL = HH0;
#if (PRINT)
                        printf("x0length=%d,y0length=%d,z0length=%d\n", x0length, y0length, z0length);
#endif
                        for (j = 0; j < z0length; j++) {
                            bioD_2d(vp, HH, HL, LH, LL, x0length, y0length,
                                    FULLIMAGE,
                                    ifnotallskip);
                            vp += xy0length;
                            HH += HHsubsize;
                            LH += LHsubsize;
                            HL += HLsubsize;
                            LL += LLsubsize;
                        }
                        xyscale++;
                        x0length = (x0length + 1) >> 1;
                        y0length = (y0length + 1) >> 1;
                    }
                }
            }

            z0length = ((zlength + 1) >> 1);
            xylength = ((ylength) >> 1) * ((xlength) >> 1);
            LHHlength = z0length * xylength;
            if (LHHlength > 0) {
                /*******************************************/
                zscale = scale;
                H = LHHv;
                L = LHH;
                vector = LHH;
                while (zscale < minZLevels) {

                    if (zscale == minZLevels - 1) L = H + xylength * (z0length >> 1);
                    bioD_2d_y(vector, H, L,
                              xylength, z0length, 1);
                    H += xylength * (z0length >> 1);
                    z0length = (1 + z0length) >> 1;
                    vector = L;
                    zscale++;
                }
                /***********************************************/
            }

            z0length = ((zlength + 1) >> 1);
            xylength = ((ylength) >> 1) * ((xlength + 1) >> 1);
            LHLlength = z0length * xylength;
            if (LHLlength > 0) {
                /*******************************************/
                zscale = scale;
                H = LHLv;
                L = LHL;
                vector = LHL;
                while (zscale < minZLevels) {

                    if (zscale == minZLevels - 1) L = H + xylength * (z0length >> 1);
                    bioD_2d_y(vector, H, L,
                              xylength, z0length, 1);
                    H += xylength * (z0length >> 1);
                    z0length = (1 + z0length) >> 1;
                    vector = L;
                    zscale++;
                }
                /***********************************************/
            }

            z0length = ((zlength + 1) >> 1);
            xylength = ((ylength + 1) >> 1) * (xlength >> 1);
            LLHlength = z0length * xylength;
            if (LLHlength > 0) {
                /*******************************************/
                zscale = scale;
                H = LLHv;
                L = LLH;
                vector = LLH;
                while (zscale < minZLevels) {
                    if (zscale == minZLevels - 1) L = H + xylength * (z0length >> 1);
                    bioD_2d_y(vector, H, L,
                              xylength, z0length, 1);
                    H += xylength * (z0length >> 1);
                    z0length = (1 + z0length) >> 1;
                    vector = L;

                    zscale++;
                }

                /***********************************************/
            }

            if (minXYLevels > minZLevels) {
                x0length = (xlength + 1) >> 1;
                y0length = (ylength + 1) >> 1;
                z0length = zlength >> 1;
                HLLlength = x0length * y0length * z0length;
                if (HLLlength > 0) {
                    xyscale = scale;

                    HH0 = HLLv;

                    while (xyscale < minXYLevels) {
                        HHsubsize = (x0length >> 1) * (y0length >> 1);
                        LHsubsize = ((x0length) >> 1) * ((y0length + 1) >> 1);
                        HLsubsize = ((x0length + 1) >> 1) * ((y0length) >> 1);
                        LLsubsize = ((x0length + 1) >> 1) * ((y0length + 1) >> 1);
                        xy0length = x0length * y0length;

                        HH = HH0;
                        HL = HH + HHsubsize * z0length;
                        LH = HL + HLsubsize * z0length;
                        HH0 = LH + LHsubsize * z0length;
                        LL = HLL;
                        vp = HLL;

                        if (xyscale == minXYLevels - 1) LL = HH0;
                        for (j = 0; j < z0length; j++) {
                            bioD_2d(vp, HH, HL, LH, LL, x0length, y0length,
                                    FULLIMAGE,
                                    ifnotallskip);
                            vp += xy0length;
                            HH += HHsubsize;
                            LH += LHsubsize;
                            HL += HLsubsize;
                            LL += LLsubsize;
                        }
                        xyscale++;
                        x0length = (x0length + 1) >> 1;
                        y0length = (y0length + 1) >> 1;
                    }
                }
            }
        }

        vector = LLL;
    }

    zscale = scale;

    if (zscale < minZLevels) {

        if (maxscale == -1) vector = inspacevector;
        /*******************************************/
        xlength = 1 + ((Xlength - 1) >> scale);
        ylength = 1 + ((Ylength - 1) >> scale);
        z0length = 1 + ((Zlength - 1) >> scale);

        xylength = xlength * ylength;
        H = LLLv;
        L = LLL;
        if (maxscale == -1) {
            vector = inspacevector;
        }

        while (zscale < minZLevels) {

            if (zscale == minZLevels - 1) L = H + xylength * (z0length >> 1);
            bioD_2d_y(vector, H, L,
                      xylength, z0length, 1);

            H += xylength * (z0length >> 1);
            z0length = (1 + z0length) >> 1;
            vector = L;

            LLL = LLLv;
            zscale++;
        }
    }
    if (LLL != LLLv) memcpy(LLLv, LLL, LLLsubsize * sizeof(FLOAT));

    /***********************************************/
    *colength_ptr = skipsize;

    return 0;
}

/************************/
