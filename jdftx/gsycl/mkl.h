/* gsycl/mkl.h - Wrapper to prevent mkl_cblas.h conflicts with gsl_cblas.h
 * 
 * In C++ mode, GSL's gsl_cblas.h defines CBLAS enums as 'enum X {...}'
 * while MKL's mkl_cblas.h defines them as 'typedef enum X X'.
 * This causes redefinition errors when both are included.
 * 
 * Solution: Skip mkl_cblas.h if GSL has already defined the enums.
 */

#ifndef MKL_H_PROTECTED
#define MKL_H_PROTECTED

/* Skip the full mkl.h if GSL cblas headers are already present */
#ifdef GSL_CBLAS_H
/* GSL cblas headers already define CBLAS enums - skip mkl_cblas.h */
/* Just include the types we need from MKL */
#include mkl_types.h
#else
#include <mkl.h>
#endif

#endif /* MKL_H_PROTECTED */
