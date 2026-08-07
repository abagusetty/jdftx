#pragma once
// cuSOLVER -> oneMKL DPC++ LAPACK shim -- DECLARATIONS ONLY.
//
// Same GSL/oneMKL header-conflict rationale as cublas_v2.h: this header
// must NOT include any oneMKL C++ header (they transitively pull in
// mkl_cblas.h, which clashes with GSL's gsl_cblas.h CBLAS enums in the
// same translation unit as core/matrixLinalg.cpp). Declarations only here;
// implementation lives in gsycl/cusolver_impl.cpp.

#include "sycl_device.hpp"   // defines double2 -- no MKL/CBLAS here

// =====================================================================
// Types
// =====================================================================
typedef struct cusolverDnContext_ { int unused; } *cusolverDnHandle_t;

typedef int cusolverStatus_t;
inline constexpr cusolverStatus_t CUSOLVER_STATUS_SUCCESS = 0;
inline constexpr cusolverStatus_t CUSOLVER_STATUS_NOT_INITIALIZED = 1;
inline constexpr cusolverStatus_t CUSOLVER_STATUS_ALLOC_FAILED = 2;

// cublas-compatible enums shared with cublas_v2.h call sites
typedef int cublasFillMode_t;
inline constexpr cublasFillMode_t CUBLAS_FILL_MODE_UPPER = 0;
inline constexpr cublasFillMode_t CUBLAS_FILL_MODE_LOWER = 1;

typedef int cublasDiagType_t;
inline constexpr cublasDiagType_t CUBLAS_DIAG_NON_UNIT = 0;
inline constexpr cublasDiagType_t CUBLAS_DIAG_UNIT = 1;

typedef int cublasSideMode_t;
inline constexpr cublasSideMode_t CUBLAS_SIDE_LEFT = 0;
inline constexpr cublasSideMode_t CUBLAS_SIDE_RIGHT = 1;

typedef int cublasOperation_t;
inline constexpr cublasOperation_t CUBLAS_OP_N = 0;
inline constexpr cublasOperation_t CUBLAS_OP_T = 1;
inline constexpr cublasOperation_t CUBLAS_OP_C = 2;

typedef int cusolverEigMode_t;
inline constexpr cusolverEigMode_t CUSOLVER_EIG_MODE_NOVECTOR = 0;
inline constexpr cusolverEigMode_t CUSOLVER_EIG_MODE_VECTOR = 1;

typedef struct gesvdjInfo_st { int unused; } *gesvdjInfo_t;

// =====================================================================
// cusolverDnCreate / Destroy -- trivial, kept inline (no MKL involvement)
// =====================================================================
inline cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t* handle) {
    *handle = reinterpret_cast<cusolverDnHandle_t>(0x1); // non-null sentinel
    return CUSOLVER_STATUS_SUCCESS;
}
inline cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t) {
    return CUSOLVER_STATUS_SUCCESS;
}

inline cusolverStatus_t cusolverDnCreateGesvdjInfo(gesvdjInfo_t* info) {
    *info = reinterpret_cast<gesvdjInfo_t>(0x1);
    return CUSOLVER_STATUS_SUCCESS;
}
inline cusolverStatus_t cusolverDnDestroyGesvdjInfo(gesvdjInfo_t) {
    return CUSOLVER_STATUS_SUCCESS;
}

// =====================================================================
// Declarations only -- implemented in gsycl/cusolver_impl.cpp
// =====================================================================
cusolverStatus_t cusolverDnZpotrf_bufferSize(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              double2* A, int lda,
                                              int* lwork);

cusolverStatus_t cusolverDnZpotrf(cusolverDnHandle_t handle,
                                   cublasFillMode_t uplo, int n,
                                   double2* A, int lda,
                                   double2* work, int lwork,
                                   int* devInfo);

cusolverStatus_t cusolverDnZpotrs(cusolverDnHandle_t handle,
                                   cublasFillMode_t uplo,
                                   int n, int nrhs,
                                   const double2* A, int lda,
                                   double2* B, int ldb,
                                   int* devInfo);

cusolverStatus_t cusolverDnZgetrf_bufferSize(cusolverDnHandle_t handle,
                                              int m, int n,
                                              double2* A, int lda,
                                              int* lwork);

cusolverStatus_t cusolverDnZgetrf(cusolverDnHandle_t handle,
                                   int m, int n,
                                   double2* A, int lda,
                                   double2* work,
                                   int* devIpiv,
                                   int* devInfo);

cusolverStatus_t cusolverDnZgetrs(cusolverDnHandle_t handle,
                                   cublasOperation_t trans,
                                   int n, int nrhs,
                                   const double2* A, int lda,
                                   const int* devIpiv,
                                   double2* B, int ldb,
                                   int* devInfo);

cusolverStatus_t cusolverDnZheevd_bufferSize(cusolverDnHandle_t handle,
                                              cusolverEigMode_t jobz,
                                              cublasFillMode_t uplo,
                                              int n,
                                              const double2* A, int lda,
                                              const double* eigs,
                                              int* lwork);

cusolverStatus_t cusolverDnZheevd(cusolverDnHandle_t handle,
                                   cusolverEigMode_t jobz,
                                   cublasFillMode_t uplo,
                                   int n,
                                   double2* A, int lda,
                                   double* W,
                                   double2* work, int lwork,
                                   int* devInfo);

cusolverStatus_t cusolverDnZgesvdj_bufferSize(cusolverDnHandle_t handle,
                                               cusolverEigMode_t jobz,
                                               int econ,
                                               int m, int n,
                                               const double2* A, int lda,
                                               const double* S,
                                               const double2* U, int ldu,
                                               const double2* V, int ldvt,
                                               int* lwork,
                                               gesvdjInfo_t params);

cusolverStatus_t cusolverDnZgesvdj(cusolverDnHandle_t handle,
                                    cusolverEigMode_t jobz,
                                    int econ,
                                    int m, int n,
                                    double2* A, int lda,
                                    double* S,
                                    double2* U, int ldu,
                                    double2* V, int ldvt,
                                    double2* work, int lwork,
                                    int* devInfo,
                                    gesvdjInfo_t params);
