#pragma once

#include "bio_parameters.h"

#if HIGH_PRECISION
#define MEX_INPUT_CLASS_TYPE mxDOUBLE_CLASS
#define MEX_INPUT_CLASS_NAME "double"
#else
#define MEX_INPUT_CLASS_TYPE mxSINGLE_CLASS
#define MEX_INPUT_CLASS_NAME "single"
#endif
