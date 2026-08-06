#pragma once
// cuBLAS → oneMKL BLAS shim
// Maps cublas_v2 API to Intel MKL BLAS calls on SYCL queues.
// Uses USM pointers directly (no explicit copy needed).

#include <mkl.h>
#include <sycl/sycl.hpp>
#include "sycl_device.hpp"

// Global handle (oneMKL uses workspace + queue internally)
extern sycl::queue* cublasQueue;
extern bool cublasInitialized;

inline void init_cublas() {
    if (!cublasInitialized) {
        cublasQueue = &jdftx_sycl::queue();
        cublasInitialized = true;
    }
}

// Convert CBLAS_TRANSPOSE to MKL layout
inline MKL_ORDER mkl_layout(CBLAS_TRANSPOSE T) {
    return (T == CblasNoTrans) ? RowMajor : ColMajor;
}

// cublasZgemm → mkl_zgemm
inline cublasStatus_t cublasZgemm(cublasHandle_t /*handle*/,
                                   cublasOperation_t transA, cublasOperation_t transB,
                                   int m, int n, int k,
                                   const cuDoubleComplex* alpha,
                                   const cuDoubleComplex* A, int lda,
                                   const cuDoubleComplex* B, int ldb,
                                   const cuDoubleComplex* beta,
                                   cuDoubleComplex* C, int ldc) {
    MKL_Complex16 _alpha = {*alpha->x, *alpha->y};
    MKL_Complex16 _beta = {*beta->x, *beta->y};
    mkl_zgemm(
        (transA == CUBLAS_OP_N) ? 'N' : 'T',
        (transB == CUBLAS_OP_N) ? 'N' : 'T',
        m, n, k,
        &_alpha,
        (const MKL_Complex16*)A, lda,
        (const MKL_Complex16*)B, ldb,
        &_beta,
        (MKL_Complex16*)C, ldc
    );
    return CUBLAS_STATUS_SUCCESS;
}

// cublasDgemm → mkl_dgemm
inline cublasStatus_t cublasDgemm(cublasHandle_t /*handle*/,
                                   cublasOperation_t transA, cublasOperation_t transB,
                                   int m, int n, int k,
                                   const double* alpha,
                                   const double* A, int lda,
                                   const double* B, int ldb,
                                   const double* beta,
                                   double* C, int ldc) {
    mkl_dgemm(
        (transA == CUBLAS_OP_N) ? 'N' : 'T',
        (transB == CUBLAS_OP_N) ? 'N' : 'T',
        m, n, k,
        alpha, A, lda,
        B, ldb,
        beta, C, ldc
    );
    return CUBLAS_STATUS_SUCCESS;
}

// cublasZaxpy → mkl_zaxpy
inline cublasStatus_t cublasZaxpy_v2(cublasHandle_t /*handle*/, int n,
                                      const cuDoubleComplex* alpha,
                                      const cuDoubleComplex* x, int incx,
                                      cuDoubleComplex* y, int incy) {
    MKL_Complex16 _alpha = {*alpha->x, *alpha->y};
    mkl_zaxpy(n, &_alpha,
              (const MKL_Complex16*)x, incx,
              (MKL_Complex16*)y, incy);
    return CUBLAS_STATUS_SUCCESS;
}

// cublasDaxpy → mkl_daxpy
inline cublasStatus_t cublasDaxpy_v2(cublasHandle_t /*handle*/, int n,
                                      const double* alpha,
                                      const double* x, int incx,
                                      double* y, int incy) {
    mkl_daxpy(n, alpha, x, incx, y, incy);
    return CUBLAS_STATUS_SUCCESS;
}

// cublasZdotc → mkl_zdotc
inline cublasStatus_t cublasZdotc_v2(cublasHandle_t /*handle*/, int n,
                                      const cuDoubleComplex* x, int incx,
                                      const cuDoubleComplex* y, int incy,
                                      cuDoubleComplex* result) {
    MKL_Complex16 result_mkl;
    mkl_zdotc(&result_mkl, n,
              (const MKL_Complex16*)x, incx,
              (const MKL_Complex16*)y, incy);
    result->x = result_mkl.real;
    result->y = result_mkl.imag;
    return CUBLAS_STATUS_SUCCESS;
}

// cublasDdot → mkl_ddot
inline cublasStatus_t cublasDdot_v2(cublasHandle_t /*handle*/, int n,
                                     const double* x, int incx,
                                     const double* y, int incy,
                                     double* result) {
    *result = mkl_ddot(n, x, incx, y, incy);
    return CUBLAS_STATUS_SUCCESS;
}

// cublasDscal → mkl_dscal
inline cublasStatus_t cublasDscal_v2(cublasHandle_t /*handle*/, int n,
                                      const double* alpha,
                                      double* x, int incx) {
    mkl_dscal(n, alpha, x, incx);
    return CUBLAS_STATUS_SUCCESS;
}

// cublasZscal → mkl_zscal
inline cublasStatus_t cublasZscal_v2(cublasHandle_t /*handle*/, int n,
                                      const cuDoubleComplex* alpha,
                                      cuDoubleComplex* x, int incx) {
    MKL_Complex16 _alpha = {*alpha->x, *alpha->y};
    mkl_zscal(n, &_alpha, (MKL_Complex16*)x, incx);
    return CUBLAS_STATUS_SUCCESS;
}

// cublasZdscal → mkl_zdscal (scale complex by double)
inline cublasStatus_t cublasZdscal_v2(cublasHandle_t /*handle*/, int n,
                                       const double* alpha,
                                       cuDoubleComplex* x, int incx) {
    mkl_zdscal(n, alpha, (MKL_Complex16*)x, incx);
    return CUBLAS_STATUS_SUCCESS;
}

// cublasDznrm2 → mkl_znrm2
inline cublasStatus_t cublasDznrm2_v2(cublasHandle_t /*handle*/, int n,
                                       const cuDoubleComplex* x, int incx,
                                       double* result) {
    *result = mkl_zznrm2(n, (const MKL_Complex16*)x, incx);
    return CUBLAS_STATUS_SUCCESS;
}

// cublasDnrm2 → mkl_dnrm2
inline cublasStatus_t cublasDnrm2_v2(cublasHandle_t /*handle*/, int n,
                                      const double* x, int incx,
                                      double* result) {
    *result = mkl_dnrm2(n, x, incx);
    return CUBLAS_STATUS_SUCCESS;
}

// Stub: cublasCreate
inline cublasStatus_t cublasCreate(cublasHandle_t* handle) {
    init_cublas();
    *handle = 0;  // dummy handle
    return CUBLAS_STATUS_SUCCESS;
}

// Stub: cublasDestroy
inline cublasStatus_t cublasDestroy(cublasHandle_t /*handle*/) {
    return CUBLAS_STATUS_SUCCESS;
}

// Stub: cublasStatus_t type
typedef int cublasStatus_t;
inline constexpr cublasStatus_t CUBLAS_STATUS_SUCCESS = 0;
