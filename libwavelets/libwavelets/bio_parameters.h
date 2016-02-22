#define TEST (fabs(b) > 1e150)
/*For storing and reading some array in reversed order: */
#define REVERSEVALUELOAD
/*For float data set use next line*/
#define FLOATDATA
/* internal float: */

/*   I/O  float : */
#ifndef HIGH_PRECISION
#define FLOAT float
#else
#define FLOAT double
#endif
/*suggestion */

#ifdef TRIG_UNSIGNEDCHAR
#ifdef FLOATDATA
#define INTYPE FLOAT
#define OUTTYPE FLOAT
#define DIFFTYPE FLOAT
#else
#define INTYPE unsigned char
#define OUTTYPE unsigned char
#define DIFFTYPE unsigned char
#define UNSIGNEDCHAR
#endif
#else
#define INTYPE FLOAT
#define OUTTYPE FLOAT
#define DIFFTYPE FLOAT
#endif
#define ORDERTYPE unsigned char

/*parameters used in the bio9xxxxx  filters  */
#define HEADERLENGTH 0
#define MAXXLENGTH 3000
#define MAXYLENGTH 3000
#define XLENGTH 512
#define YLENGTH 512
#define ZLENGTH 1
#define ALPHA (1.3)

/* parameters used by the program bitcode and bitdecode and peano   */

#define LEVELS 6
#define MAXLEVELS 10
#define MAXBYTECOUNT 150
#define SKIP 1
#define HHSKIP 2
#define MAXBITLEVEL 10000
#define FLOATBITLEVEL 24
#define NULLFLOAT 1e-18
#ifdef FLOATDATA
#define THRESHLDO 0.01788
#else
#define THRESHOLD 15.0
#endif
#define MAXPEANOLEVELS 0
#define BETA (1.0)
#define ROUND (0.5)
#define VBITLENGTH 0
#define VBITINCREASE 1
#define FLIPOFF 1
#ifdef FLOATDATA
#define FIRSTFLIPOFF 0
#else
#define FIRSTFLIPOFF 1
#endif
#define PEANOLEVEL -1
#define MANTSIZE 23
#define BITPRECISION 23 /*maximum 23 but not larger than MANTSIZE  ( 23+8=31) */
#define CODE_EMPTYLEVEL 1
#define MAKELOWENDIAN

/***** parameters below should not be changed  ********/

#define N0 (threshold)

/***parameters in analysis of threshold *****/

#define MAXITERATE_SNR_DEFAULT 10.0
#define TOLERANSLEVEL_SNR_DEFAULT 0.01
#define MAXITERATE_R_DEFAULT 12
#define TOLERANSLEVEL_R_DEFAULT 0.0001

#ifndef M_LN2
#define M_LN2 0.69314718055994530942 /* log e2 */
#endif

#define LCT 0
