
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define TRIG_UNSIGNEDCHAR
#include "bio_parameters.h"
#define COLORS 3
#define LEVELS 6
#define SKIP 1
#ifndef FLOAT
#define FLOAT float
#endif
#ifndef max
#define max(s, t) (s > t ? s : t)
#endif
#ifndef min
#define min(s, t) (s < t ? s : t)
#endif
#ifndef HIGH_PRECISION
#define ONEMASK 0x7fffffff
#else
#define ONEMASK 0x7fffffffffffffff
#endif

int compimages(INTYPE* vector1,
               OUTTYPE* vector2,
               DIFFTYPE* diffvector,
               int Size,
               double* ptrmaxvalue,
               double* ptrposerror,
               double* ptrnegerror,
               double* ptrerror2,
               double* ptrnorm2,
               double* ptrerror1,
               double* ptrnorm1,
               double* ptrerrorlog10,
               int* ptrnonexactnumber,
               int* ptrsize,
               double* ptrsum) {
    INTYPE* ptr1;
    OUTTYPE* ptr2;

    int k, j;
    FLOAT maxvalue;
    FLOAT minvalue;
    double negerror, poserror;
    double error2, norm2;
    FLOAT a, b;
#ifndef HIGH_PRECISION
    unsigned int uiabsa, *uiaptr, *minptr, *maxptr;
    int inta, intb;
    const unsigned int OneMask = ONEMASK;
#else
    unsigned long long uiabsa, *uiaptr, *minptr, *maxptr;
	const unsigned long long OneMask = ONEMASK;
#endif
    double error1, norm1;
    double errorlog2, normlog2;
    /*int errorbits;*/
    int number;
    double sum = 0;
    int flag = 1;

    negerror = 0;
    poserror = 0;
    norm2 = 0;
    error2 = 0;
    error1 = 0;
    norm1 = 0;

    errorlog2 = 0;
    normlog2 = 0;
    number = 0;

    ptr1 = vector1;
    ptr2 = vector2;

    for (j = 0; j < 1; j++) {
        k = Size;

        maxvalue = 0.0;
        minvalue = 1e35;

#ifndef HIGH_PRECISION
        minptr = (unsigned int*)&minvalue;
        maxptr = (unsigned int*)&maxvalue;
        uiaptr = (unsigned int*)&a;
#else
		minptr = (unsigned long long*)&minvalue;
		maxptr = (unsigned long long*)&maxvalue;
		uiaptr = (unsigned long long*)&a;
#endif

        while (k-- > 0) {
#ifdef UNSIGNEDCHAR
            inta = (int)(0.5 + *(ptr1++));
            intb = (int)(0.5 + *(ptr2++));
            a = inta;
            b = intb;

#else

            a = *(ptr1++);
            b = *(ptr2++);

#endif
            uiabsa = (*uiaptr) & OneMask;
            if (uiabsa) {

                /*  minvalue =(absa<minvalue?absa:minvalue);*/
                *minptr = (uiabsa < *minptr ? uiabsa : *minptr);

                /*  maxvalue=(absa>maxvalue?absa:maxvalue); */
                *maxptr = (uiabsa > *maxptr ? uiabsa : *maxptr);
            }
            error2 += (a - b) * (a - b);

            norm2 += a * a;
            sum += a;

            poserror =
                (b - a > poserror ? b - a : poserror);

            negerror =
                (b - a < negerror ? b - a : negerror);
        }
    }

    *ptrerror2 += error2;

    if (maxvalue > *ptrmaxvalue) *ptrmaxvalue = maxvalue;
    if (poserror > *ptrposerror) *ptrposerror = poserror;
    if (negerror < *ptrnegerror) *ptrnegerror = negerror;

    *ptrnorm2 += norm2;
    *ptrsize += Size;
    if (ptrsum != NULL) *ptrsum += sum;
    return 0;
}

int show_compresults(maxvalue, poserror, negerror, error2, norm2, error1, norm1, errorlog10, nonexactnumber, size, SNR_ptr, typesize, sum) double maxvalue;
double poserror;
double negerror;
double error2;
double norm2;
double error1;
double norm1;
double errorlog10;
int nonexactnumber;
int size;
FLOAT* SNR_ptr;
int typesize;
double sum;
{
    double varians;
    double mean;

    /*
FLOAT fraction;

FLOAT  errorlog2;  
*/
    printf("\nSize of data set:%d pixels on %d bytes\n", size, size * typesize);
    /*error1 /= size;
norm1  /= size;

error1 +=0.00000001;
norm1 += 0.0001;
*/

    error2 /= size;
    norm2 /= size;
    mean = sum / size;
    varians = norm2 - mean * mean;

    /*
errorlog2  = errorlog10/ log10(2);
errorlog2 /= size;
*/

    error2 += 1.0e-45;
    norm2 += 1.0e-45;
    varians += 1.0e-45;
    /*
fraction = ((FLOAT)nonexactnumber)/size;
*/
    *SNR_ptr = (-10 * log10((double)error2 / norm2));

    *SNR_ptr = (-10 * log10((double)error2 / norm2));

#ifndef ONLY_SNR_PRINT
    printf("Max value: %e \n", maxvalue);
    printf("Mean value: %e\n", mean);
    printf("Norm2^2/Size =%e\n", norm2);
    printf("Varians = %e\n", varians);
    printf("Max positiv error: %e\nMax negativ error: %e\n", poserror, negerror);
    printf("Error2 ^2 /SIZE = %e\n", error2);
    printf("(Error2/Norm2) ^2 = %e\n", (double)error2 / norm2);
    printf("(Error2^2/Varianse)  = %e\n", (double)error2 / varians);
    printf("(Error2/(Max))^2/SIZE = %e\n", error2 / (maxvalue * maxvalue));
#endif
    printf("SignaltoNoise rate %f decibel\n", (-10.0 * log10((double)error2 / norm2)));
#ifndef ONLY_SNR_PRINT
    printf("     PSNR   = %f decibel\n", (-10.0 * log10(error2 / (maxvalue * maxvalue))));
    printf("VariansetoNoise rate %f decibel\n", (-10.0 * log10((double)error2 / varians)));
#endif

    /*
printf("Error1 /SIZE = %e\n",error1 );
printf("(Error1/Norm1)  = %e\n",error1/norm1);
printf("(Error1/(Max))/SIZE = %e\n",error1/maxvalue);
*/
    /*printf("Errorlog2 /SIZE = %f\n",errorlog2);


printf("Fraction of nonexact bytes =%f\n",fraction);*/
    /*
printf("We May correct the approximative file with"); 
printf(" bits = %d,\nwhich is in mean %f per byte\n",errorbits,((FLOAT)errorbits/size));
*/
    return 0;
}
