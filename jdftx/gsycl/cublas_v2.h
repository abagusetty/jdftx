#pragma once
// cuBLAS -> oneMKL DPC++ BLAS shim -- DECLARATIONS ONLY.
//
// IMPORTANT (GSL/oneMKL header-conflict fix):
// This header intentionally does NOT include <oneapi/mkl/blas.hpp> or any
// other oneMKL C++ header. oneMKL's oneapi/mkl/types.hpp unconditionally
// pulls in mkl_cblas.h, which redefines CBLAS_ORDER/CBLAS_TRANSPOSE/etc.
// as 'typedef enum X {...} X'. JDFTx's core/BlasExtra.h first includes
// <gsl/gsl_cblas.h>, which defines the SAME enum names as plain 'enum X
// {...}' (no typedef) -- a C++ redefinition clash if both ever appear in
// the same translation unit.
//
// Original JDFTx CUDA build has no such clash because real cublas_v2.h
// never touches CBLAS enums at all. We restore that same separation here:
// this header only declares the cublas*-named entry points JDFTx calls
// (no MKL types leak into headers included by core/BlasExtra.h etc.);
// the actual oneMKL calls live in gsycl/cublas_impl.cpp, a standalone
// translation unit that never sees gsl_cblas.h. This lets oneMKL act as
// a pure GPU-device library (mirroring cuBLAS), while GSL keeps serving
// CPU-side BLAS/LAPACK untouched -- no original JDFTx source changed.

#include "sycl_device.hpp"   // defines double2, float2 -- no MKL/CBLAS here

// =====================================================================
// Types and status codes
// =====================================================================
typedef int cublasStatus_t;
inline constexpr cublasStatus_t CUBLAS_STATUS_SUCCESS = 0;

typedef int cublasHandle_t;
typedef int cublasOperation_t;
typedef int cublasSideMode_t;
typedef int cublasFillMode_t;
typedef int cublasDiagType_t;

// Operation enums
inline constexpr cublasOperation_t CUBLAS_OP_N = 0;
inline constexpr cublasOperation_t CUBLAS_OP_T = 1;
inline constexpr cublasOperation_t CUBLAS_OP_C = 2;

// Side mode
inline constexpr cublasSideMode_t CUBLAS_SIDE_LEFT = 0;
inline constexpr cublasSideMode_t CUBLAS_SIDE_RIGHT = 1;

// Fill mode
inline constexpr cublasFillMode_t CUBLAS_FILL_MODE_UPPER = 0;
inline constexpr cublasFillMode_t CUBLAS_FILL_MODE_LOWER = 1;

// Diag type
inline constexpr cublasDiagType_t CUBLAS_DIAG_NON_UNIT = 0;
inline constexpr cublasDiagType_t CUBLAS_DIAG_UNIT = 1;

// NOTE: double2/float2 come from sycl_device.hpp -- do NOT redefine here.
// JDFTx code casts complex<double>* to (const double2*)/(double2*) at call
// sites, so all complex entry points below take double2*.

// =====================================================================
// Declarations only -- implemented in gsycl/cublas_impl.cpp
// =====================================================================
cublasStatus_t cublasDgemm(cublasHandle_t handle,
                            cublasOperation_t transA, cublasOperation_t transB,
                            int m, int n, int k,
                            const double* alpha,
                            const double* A, int lda,
                            const double* B, int ldb,
                            const double* beta,
                            double* C, int ldc);

cublasStatus_t cublasZgemm(cublasHandle_t handle,
                            cublasOperation_t transA, cublasOperation_t transB,
                            int m, int n, int k,
                            const double2* alpha,
                            const double2* A, int lda,
                            const double2* B, int ldb,
                            const double2* beta,
                            double2* C, int ldc);

cublasStatus_t cublasDaxpy_v2(cublasHandle_t handle, int n,
                               const double* alpha,
                               const double* x, int incx,
                               double* y, int incy);

cublasStatus_t cublasZaxpy_v2(cublasHandle_t handle, int n,
                               const double2* alpha,
                               const double2* x, int incx,
                               double2* y, int incy);

cublasStatus_t cublasDdot_v2(cublasHandle_t handle, int n,
                              const double* x, int incx,
                              const double* y, int incy,
                              double* result);

cublasStatus_t cublasZdotc_v2(cublasHandle_t handle, int n,
                               const double2* x, int incx,
                               const double2* y, int incy,
                               double2* result);

cublasStatus_t cublasDscal_v2(cublasHandle_t handle, int n,
                               const double* alpha,
                               double* x, int incx);

cublasStatus_t cublasZscal_v2(cublasHandle_t handle, int n,
                               const double2* alpha,
                               double2* x, int incx);

cublasStatus_t cublasZdscal_v2(cublasHandle_t handle, int n,
                                const double* alpha,
                                double2* x, int incx);

cublasStatus_t cublasDnrm2_v2(cublasHandle_t handle, int n,
                               const double* x, int incx,
                               double* result);

cublasStatus_t cublasDznrm2_v2(cublasHandle_t handle, int n,
                                const double2* x, int incx,
                                double* result);

cublasStatus_t cublasDasum(cublasHandle_t handle, int n,
                            const double* x, int incx,
                            double* result);

cublasStatus_t cublasDtrsm(cublasHandle_t handle,
                            cublasSideMode_t side,
                            cublasFillMode_t uplo,
                            cublasOperation_t transA,
                            cublasDiagType_t diag,
                            int m, int n,
                            const double* alpha,
                            const double* A, int lda,
                            double* B, int ldb);

cublasStatus_t cublasZtrsm(cublasHandle_t handle,
                            cublasSideMode_t side,
                            cublasFillMode_t uplo,
                            cublasOperation_t transA,
                            cublasDiagType_t diag,
                            int m, int n,
                            const double2* alpha,
                            const double2* A, int lda,
                            double2* B, int ldb);

inline cublasStatus_t cublasCreate(cublasHandle_t* handle) {
    *handle = 0;
    return CUBLAS_STATUS_SUCCESS;
}

inline cublasStatus_t cublasDestroy(cublasHandle_t /*handle*/) {
    return CUBLAS_STATUS_SUCCESS;
}
