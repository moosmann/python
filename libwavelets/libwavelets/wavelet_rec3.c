#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define TRIG_UNSIGNEDCHAR
#include "bio_parameters.h"
#include "bio.h"
#ifndef FLOAT
#define FLOAT float
#endif

#ifndef max
#define max(s, t) (s > t ? s : t)
#endif
#ifndef min
#define min(s, t) (s < t ? s : t)
#endif
#define FULLIMAGE 0, 0, 0, 0, 0

int wavelet_reconstruct3(FLOAT* reccovector,
                         int colength,
                         FLOAT* outvector,
                         int Xlength,
                         int Ylength,
                         int Zlength,
                         char Filterlength,
                         char Levels,
                         char minZLevels,
                         char maxZLevels,
                         char minXYLevels,
                         char maxXYLevels,
                         char Skip,
                         char ifnotSilent)

{
    long time0 = 0;
    long time1 = 1;
    FLOAT* vector = NULL;
    FLOAT* outvp;
    int j;
    int scale, xlength, ylength, zlength, maxscale;
    int zscale, xylength, z0length, maxzscale;
    int mxlength, mylength, mzlength;
    int x0length, y0length;
    int xy0length;
    int xyscale;
    FLOAT* LLL = NULL, * LHH = NULL, * LLH = NULL, * LHL = NULL;
    FLOAT* HLL = NULL, * HHH = NULL, * HLH = NULL, * HHL = NULL;
    FLOAT* LLLv = NULL, * LHHv = NULL, * LLHv = NULL, * LHLv = NULL;
    FLOAT* HLLv = NULL;
    FLOAT* LLLv0 = NULL;
    FLOAT* HLL0;
    FLOAT* LL, *LH, *HL, *HH;
    FLOAT* vp;
    FLOAT* L, *H;
    int skipsize;
    int Size;
    int finalxlength, finalylength, finalzlength;
    int finalLLLsize;

    int LLLsubsize;
    int LLHsubsize;
    int LHLsubsize;
    int LHHsubsize;
    int HLHsubsize;
    int HHLsubsize;
    int HHHsubsize;
    int HLLsubsize;
    int LHsubsize;
    int HLsubsize;
    int HHsubsize;
    int LLsubsize;
    int ifnotallskip = 1;

    void (*bioR_2d)();
    void (*bioR_skip_2d)();
    void (*bioR_2d_char)();
    void (*bioR_skip_2d_char)();
    void (*bioR_2d_y)();
    void (*bioR_3d)();
    void (*bioR_skip_3d)();
    void (*bioR_3d_char)();
    void (*bioR_skip_3d_char)();

    getfilter(&bioR_2d, &bioR_skip_2d,
              &bioR_2d_char, &bioR_skip_2d_char, -Filterlength, 3);
    getfilter(&bioR_2d_y, &bioR_skip_3d,
              &bioR_3d_char, &bioR_skip_3d_char, -Filterlength, 2);
    getfilter(&bioR_3d, &bioR_skip_3d,
              &bioR_3d_char, &bioR_skip_3d_char, -Filterlength, 7);

    Size = Xlength * Ylength * Zlength;
    xlength = 1 + ((Xlength - 1) >> Skip);
    ylength = 1 + ((Ylength - 1) >> Skip);
    if (Skip < maxZLevels)
        zlength = 1 + ((Zlength - 1) >> Skip);
    else
        zlength = 1 + ((Zlength - 1) >> maxZLevels);
    skipsize = xlength * ylength * zlength;

    maxscale = Levels - 1;
    maxzscale = maxZLevels - 1;

    if (Levels == 0) skipsize = Xlength * Ylength * Zlength;

    vector = outvector;
    if (((maxscale < 0) && (minZLevels == 0)) || ((maxZLevels == 0) && (maxXYLevels == 0))) return 0;
    scale = maxscale;

    finalxlength = 1 + ((Xlength - 1) >> (maxscale + 1));
    finalylength = 1 + ((Ylength - 1) >> (maxscale + 1));
    if (maxscale > maxzscale)
        finalzlength = 1 + ((Zlength - 1) >> (maxzscale + 1));
    else
        finalzlength = 1 + ((Zlength - 1) >> (maxscale + 1));
    finalLLLsize = finalxlength * finalylength * finalzlength;

    LLLv = reccovector + skipsize;
    LLL = LLLv - finalLLLsize;
    LLLv0 = outvector + finalLLLsize;

    xlength = Xlength;
    ylength = Ylength;
    zlength = Zlength;

    /********************************************/

    if (minZLevels > Levels) {
        LLL = outvector;
        zscale = minZLevels - 1;
        vector = outvector;
        xlength = 1 + ((Xlength - 1) >> Levels);
        ylength = 1 + ((Ylength - 1) >> Levels);
        zlength = 1 + ((Zlength - 1) >> zscale);

        z0length = ((zlength + 1) >> 1);
        xylength = ylength * xlength;

        L = LLLv - xylength * z0length;
        H = L;

        while (zscale > maxscale) {
            zlength = 1 + ((Zlength - 1) >> zscale);
            H -= xylength * (zlength >> 1);
            bioR_2d_y(H, L, vector, xylength, zlength, 1);
            L = vector;
            zscale--;
        }
    }

    /************************************************/

    while (scale >= 0 * Skip) { /* over scales */

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

        vector = outvector;
        LLLsubsize = mzlength * mylength * mxlength;
        LLHsubsize = mzlength * mylength * (xlength - mxlength);
        LHLsubsize = mzlength * (ylength - mylength) * mxlength;
        LHHsubsize = mzlength * (ylength - mylength) * (xlength - mxlength);
        HLLsubsize = (zlength - mzlength) * mylength * mxlength;
        HLHsubsize = (zlength - mzlength) * mylength * (xlength - mxlength);
        HHLsubsize = (zlength - mzlength) * (ylength - mylength) * mxlength;
        HHHsubsize = (zlength - mzlength) * (ylength - mylength) * (xlength - mxlength);

        if (minXYLevels <= minZLevels) {
            LLH = LLLv - LLLsubsize - LLHsubsize;
            LHL = LLH - LHLsubsize;
            LHH = LHL - LHHsubsize;
            HLL = LHH - HLLsubsize;
            HLH = HLL - HLHsubsize;
        } else { /* minXYLevels > minZLevels */
            HLL = LLLv - LLLsubsize - HLLsubsize;
            LLH = HLL - LLHsubsize;
            LHL = LLH - LHLsubsize;
            LHH = LHL - LHHsubsize;
            HLH = LHH - HLHsubsize;
        }

        HHL = HLH - HHLsubsize;
        HHH = HHL - HHHsubsize;

        HLLv = HLL;
        LLHv = LLH;
        LHLv = LHL;
        LHHv = LHH;

        if (minXYLevels - 1 > scale)
            HLLv = HLL + LLLsubsize;
        if (minZLevels - 1 > scale) {
            LLHv = LLH + LLLsubsize;
            LHLv = LHL + LLLsubsize;
            LHHv = LHH + LLLsubsize;
        }
        LLLv0 = outvector + LLLsubsize;

        ifnotallskip = (scale < Skip ? 0 : 1);

        xylength = mylength * (xlength - mxlength);

        if (scale < maxscale) LLL = outvector;

        if ((minXYLevels > minZLevels) && (zlength - mzlength > 0) && (minXYLevels - 1 > scale)) {

            HLL0 = HLL + HLLsubsize;
            HLL = HLLv;

            xyscale = minXYLevels - 1;
            x0length = 1 + ((Xlength - 1) >> xyscale);
            y0length = 1 + ((Ylength - 1) >> xyscale);
            zlength = 1 + ((Zlength - 1) >> scale);
            z0length = zlength >> 1;
            LLsubsize = ((x0length + 1) >> 1) * ((y0length + 1) >> 1);
            LL = HLL0 - z0length * LLsubsize;
            HLL0 = LL;
            while (scale < xyscale) {
                x0length = 1 + ((Xlength - 1) >> xyscale);
                y0length = 1 + ((Ylength - 1) >> xyscale);
                vector = HLLv /* CHANGE MADE HERE */;
                xy0length = x0length * y0length;
                if (LLL != outvector) {
                    memcpy(outvector, LLL, LLLsubsize * sizeof(FLOAT));
                    LLL = outvector;
                }

                HHsubsize = (x0length >> 1) * (y0length >> 1);
                HLsubsize = ((x0length + 1) >> 1) * ((y0length) >> 1);
                LHsubsize = ((x0length) >> 1) * ((y0length + 1) >> 1);
                LLsubsize = ((x0length + 1) >> 1) * ((y0length + 1) >> 1);
                vp = vector + (z0length - 1) * xy0length;
                ;

                LL = LL + (z0length - 1) * LLsubsize;
                LH = HLL0 - LHsubsize;
                HL = LH - (z0length - 1) * LHsubsize - HLsubsize;
                HH = HL - (z0length - 1) * HLsubsize - HHsubsize;
                HLL0 = HH - (z0length - 1) * HHsubsize;

                for (j = 0; j < z0length; j++) {
                    bioR_2d(HH, HL, LH, LL,
                            vp, x0length, y0length,
                            FULLIMAGE,
                            ifnotallskip);
                    vp -= xy0length;
                    HH -= HHsubsize;
                    HL -= HLsubsize;
                    LH -= LHsubsize;
                    LL -= LLsubsize;
                }

                LL = vector;
                xyscale--;
            }
        } /**** end if(minXYLevels>minZLevels .. */

        xylength = ((ylength + 1) >> 1) * ((xlength) >> 1);
        if ((minZLevels - 1 > scale) && (xylength > 0)) {

            LLH = LLH + LLHsubsize;

            zscale = minZLevels - 1;
            zlength = 1 + ((Zlength - 1) >> zscale);

            z0length = (zlength + 1) >> 1;
            vector = LLHv;
            L = LLH - xylength * z0length;
            H = L;
            LLH = LLHv;
            while (zscale > scale) {
                zlength = 1 + ((Zlength - 1) >> zscale);
                H -= xylength * (zlength >> 1);
                if (zscale == 0) vector = outvector;
                bioR_2d_y(H, L, vector, xylength, zlength, 1);
                L = vector;
                zscale--;
            }
        }

        /************************************************/

        /********************************************/
        xylength = ((ylength) >> 1) * ((xlength + 1) >> 1);

        if ((minZLevels - 1 > scale) && (xylength > 0)) {
            zscale = minZLevels - 1;
            zlength = 1 + ((Zlength - 1) >> zscale);

            z0length = ((zlength + 1) >> 1);
            LHL = LHL + LHLsubsize;

            vector = LHLv;
            L = LHL - xylength * z0length;
            H = L;
            LHL = LHLv;
            while (zscale > scale) {
                zlength = 1 + ((Zlength - 1) >> zscale);
                H -= xylength * (zlength >> 1);
                if (zscale == 0) vector = outvector;
                bioR_2d_y(H, L, vector, xylength, zlength, 1);
                L = vector;
                zscale--;
            }
        }

        /************************************************/

        /********************************************/
        xylength = ((ylength) >> 1) * (xlength >> 1);
        if ((minZLevels - 1 > scale) && (xylength > 0)) {

            zscale = minZLevels - 1;
            zlength = 1 + ((Zlength - 1) >> zscale);

            z0length = ((zlength + 1) >> 1);
            LHH = LHH + LHHsubsize;

            vector = LHHv;
            L = LHH - xylength * z0length;
            H = L;
            LHH = LHHv;

            while (zscale > scale) {
                zlength = 1 + ((Zlength - 1) >> zscale);
                H -= xylength * (zlength >> 1);
                if (zscale == 0) vector = outvector;
                bioR_2d_y(H, L, vector, xylength, zlength, 1);
                L = vector;
                zscale--;
            }
        }

        zlength = 1 + ((Zlength - 1) >> scale);
        if ((minXYLevels <= minZLevels) && (zlength - mzlength > 0) && (minXYLevels - 1 > scale)) {
            HLL0 = HLL + HLLsubsize;

            xyscale = minXYLevels - 1;
            x0length = 1 + ((Xlength - 1) >> xyscale);
            y0length = 1 + ((Ylength - 1) >> xyscale);

            z0length = (zlength >> 1);
            LLsubsize = ((x0length + 1) >> 1) * ((y0length + 1) >> 1);
            if ((xyscale > scale) && (LLsubsize > 0)) {

                LL = HLL0 - z0length * LLsubsize;
                HLL0 = LL;
                if (LLL != outvector) {
                    memcpy(outvector, LLL, LLLsubsize * sizeof(FLOAT));
                    LLL = outvector;
                }
                HLL = HLL + LLLsubsize;
                vector = HLL;
            }

            while (xyscale > scale) {
                x0length = 1 + ((Xlength - 1) >> xyscale);
                y0length = 1 + ((Ylength - 1) >> xyscale);
                xy0length = x0length * y0length;

                HHsubsize = (x0length >> 1) * (y0length >> 1);
                LHsubsize = ((x0length) >> 1) * ((y0length + 1) >> 1);
                HLsubsize = ((x0length + 1) >> 1) * ((y0length) >> 1);
                LLsubsize = ((x0length + 1) >> 1) * ((y0length + 1) >> 1);
                vp = vector + (z0length - 1) * xy0length;
                ;

                LL = LL + (z0length - 1) * LLsubsize;
                LH = HLL0 - LHsubsize;
                HL = LH - (z0length - 1) * LHsubsize - HLsubsize;
                HH = HL - (z0length - 1) * HLsubsize - HHsubsize;
                HLL0 = HH - (z0length - 1) * HHsubsize;

                for (j = 0; j < z0length; j++) {
                    bioR_2d(HH, HL, LH, LL,
                            vp, x0length, y0length,
                            FULLIMAGE,
                            ifnotallskip);
                    vp -= xy0length;
                    HH -= HHsubsize;
                    HL -= HLsubsize;
                    LH -= LHsubsize;
                    LL -= LLsubsize;
                }

                LL = vector;

                xyscale--;
            }
        }

        /************************************************/

        vector = outvector;

        if ((scale > maxzscale) && (scale < maxXYLevels)) {
            zlength = 1 + ((Zlength - 1) >> (maxzscale + 1));
            z0length = zlength;
            xylength = xlength * ylength;
            LHHsubsize = (xlength >> 1) * (ylength >> 1);
            LLHsubsize = ((xlength) >> 1) * ((ylength + 1) >> 1);
            LHLsubsize = ((xlength + 1) >> 1) * ((ylength) >> 1);
            LLLsubsize = ((xlength + 1) >> 1) * ((ylength + 1) >> 1);

            outvp = outvector + (zlength - 1) * xylength;
            vp = vector + (zlength - 1) * xylength;
            ;

            HH = LHHv + (zlength - 1) * LHHsubsize;
            HL = LHLv + (zlength - 1) * LHLsubsize;
            LH = LLHv + (zlength - 1) * LLHsubsize;
            if ((scale == maxscale) && (scale >= minZLevels - 1))
                LL = LLLv - LLLsubsize;

            else
                LL = LLLv0 - LLLsubsize;

            if (scale == 0) {
                for (j = 0; j < zlength; j++) {

                    bioR_2d_char(HH, HL, LH, LL,
                                 outvp, xlength, ylength,
                                 FULLIMAGE,
                                 ifnotallskip);
                    outvp -= xylength;
                    vp -= xylength;
                    HH -= LHHsubsize;
                    HL -= LHLsubsize;
                    LH -= LLHsubsize;
                    LL -= LLLsubsize;
                }

            } else {
                for (j = 0; j < zlength; j++) {

                    bioR_2d(HH, HL, LH, LL,
                            vp, xlength, ylength,
                            FULLIMAGE,
                            ifnotallskip);
                    outvp -= xylength;
                    vp -= xylength;
                    HH -= LHHsubsize;
                    HL -= LHLsubsize;
                    LH -= LLHsubsize;
                    LL -= LLLsubsize;
                }
            }
        } else if ((scale <= maxzscale) && (scale < maxXYLevels)) {
            zlength = 1 + ((Zlength - 1) >> scale);
            z0length = ((zlength + 1) >> 1);

            if (scale == 0) {
                bioR_3d_char(HHH, HHL, HLH, HLL, LHH, LHL, LLH, LLL,
                             outvector, xlength, ylength, zlength,
                             ifnotallskip);
            } else {
                bioR_3d(HHH, HHL, HLH, HLL, LHH, LHL, LLH, LLL,
                        vector, xlength, ylength, zlength,
                        ifnotallskip);
            }

        } else if ((scale <= maxzscale) && (scale >= maxXYLevels)) {

            if (zscale == 0) vector = outvector;
            bioR_2d_y(HLL, LLL, vector, xlength * ylength, zlength, 1);
        }

        LLL = vector;
        scale--;
    }
    if ((scale == maxscale) && (scale >= minZLevels - 1))
        LLL = reccovector;
    else
        LLL = outvector;

    while (scale >= 0) {
        ifnotallskip = 0;
        xlength = 1 + ((Xlength - 1) >> scale);
        ylength = 1 + ((Ylength - 1) >> scale);

        if (scale > maxzscale)
            zlength = 1 + ((Zlength - 1) >> (maxzscale + 1));
        else
            zlength = 1 + ((Zlength - 1) >> scale);
        /************************************************/

        vector = outvector;
        LLH = NULL;
        LHL = NULL;
        LHH = NULL;
        HLL = NULL;
        HLH = NULL;
        HHL = NULL;
        HHH = NULL;
        if (scale > maxzscale) {
            zlength = 1 + ((Zlength - 1) >> (maxzscale + 1));
            z0length = zlength;
            xylength = xlength * ylength;
            mxlength = (xlength + 1) >> 1;
            mylength = (ylength + 1) >> 1;
            outvp = outvector + (zlength - 1) * xylength;
            vp = vector + (zlength - 1) * xylength;
            ;
            LLLsubsize = mylength * mxlength;
            LL = LLL + (zlength - 1) * LLLsubsize;
            HH = NULL;
            LH = NULL;
            HH = NULL;
            if (scale == 0) {
                for (j = 0; j < zlength; j++) {

                    bioR_2d_char(HH, HL, LH, LL,
                                 outvp, xlength, ylength,
                                 FULLIMAGE,
                                 ifnotallskip);
                    outvp -= xylength;
                    vp -= xylength;
                    LL -= LLLsubsize;
                }

            } else {
                for (j = 0; j < zlength; j++) {

                    bioR_2d(HH, HL, LH, LL,
                            vp, xlength, ylength,
                            FULLIMAGE,
                            ifnotallskip);
                    outvp -= xylength;
                    vp -= xylength;
                    LL -= LLLsubsize;
                }
            }

        } else {
            zlength = 1 + ((Zlength - 1) >> scale);
            if (scale == 0) {
                bioR_3d_char(HHH, HHL, HLH, HLL, LHH, LHL, LLH, LLL,
                             outvector, xlength, ylength, zlength,
                             ifnotallskip);
            } else {
                bioR_3d(HHH, HHL, HLH, HLL, LHH, LHL, LLH, LLL,
                        vector, xlength, ylength, zlength,
                        ifnotallskip);
            }
        }

        LLL = vector;
        scale--;
    }

    return 0;
}

/*********************/
