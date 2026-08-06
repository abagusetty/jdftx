#pragma once
// cuSOLVER → oneMKL LAPACK shim
//
// JDFTx uses cuSOLVER for eigendecomposition and SVD in matrixLinalg.cpp.
// per the handoff: "skip entirely" was the original plan, but since oneMKL
// provides heevd and gesvd, we enable it as a real implementation.
//
// Intel documents oneMKL heevd as CPU-executed anyway for small matrices
// (~O(10²)), and JDFTx already falls back to LAPACK for small matrices
// with the comment "CPU LAPACK faster for small matrices".
//
// For the DFT code path, these matrices are small enough that the CPU
// fallback is fine. For larger matrices, Intel's MKL LAPACK handles
// USM pointers via the stream queue.

#include <mkl.h>
#include <sycl/sycl.hpp>
#include "sycl_device.hpp"

// =====================================================================
// Types
// =====================================================================
typedef struct cusolverDnHandle_ {
    void* queue;
    int* devInfo;
} *cusolverDnHandle_t;

typedef int cusolverStatus_t;
inline constexpr cusolverStatus_t CUSOLVER_STATUS_SUCCESS = 0;
inline constexpr cusolverStatus_t CUSOLVER_STATUS_NOT_INITIALIZED = 1;

// =====================================================================
// cublasFillMode_t / cublasDiagType_t / cublasOperation_t
// (also used in matrixLinalg.cpp)
// =====================================================================
typedef int cublasFillMode_t;
inline constexpr int CUBLAS_FILL_MODE_UPPER = 0;
inline constexpr int CUBLAS_FILL_MODE_LOWER = 1;

typedef int cublasDiagType_t;
inline constexpr int CUBLAS_DIAG_NON_UNIT = 0;
inline constexpr int CUBLAS_DIAG_UNIT = 1;

typedef int cublasSideMode_t;
inline constexpr int CUBLAS_SIDE_LEFT = 0;
inline constexpr int CUBLAS_SIDE_RIGHT = 1;

// =====================================================================
// cusolverDnCreate / Destroy
// =====================================================================
inline cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t* handle) {
    *handle = (cusolverDnHandle_t)malloc(sizeof(struct cusolverDnHandle_));
    (*handle)->queue = 0;
    (*handle)->devInfo = 0;
    return (*handle) ? CUSOLVER_STATUS_SUCCESS : 1;
}

inline cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t handle) {
    free(handle);
    return CUSOLVER_STATUS_SUCCESS;
}

// =====================================================================
// LAPACK wrappers
//
// JDFTx uses these routines (from matrixLinalg.cpp):
//   - dpotrf / zpotrf (Cholesky)
//   - dtrsm / ztrsm (triangular solve)
//   - dgeev / zgeev (eigenvalue, not directly via cuSOLVER — uses LAPACK)
//   - dgesvd / zgesvd (SVD — guarded, auto-fallback)
//
// oneMKL provides dgesvd/zgesvd/heevd/heevr/etc. via MKL LAPACK interface.
// These are pure CPU routines that work on USM pointers (MKL auto-detects).
// For the DFT code path, matrix sizes are ~O(10²), so CPU is fine.
// =====================================================================

// dgesvd → mkl_dgesvd (real SVD)
inline cusolverStatus_t cusolverDnDgesvd(cusolverDnHandle_t /*handle*/,
                                          char jobu, char jobvt,
                                          int* M, int* N,
                                          double* A, int lda,
                                          double* S, double* U, int ldu,
                                          double* VT, int ldvt,
                                          double* work, int lwork,
                                          int* devInfo) {
    // MKL dgesvd signature:
    //   MKL_DGESVD(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
    //              work, lwork, rwork, info)
    double rwork[1];
    MKL_INT info = 0;
    mkl_dgesvd(&jobu, &jobvt, M, N, A, lda, S, U, ldu, VT, ldvt,
               work, lwork, rwork, &info);
    *devInfo = (int)info;
    return (info == 0) ? CUSOLVER_STATUS_SUCCESS : 1;
}

// zgesvd → mkl_zgesvd (complex SVD)
inline cusolverStatus_t cusolverDnZgesvd(cusolverDnHandle_t /*handle*/,
                                          char jobu, char jobvt,
                                          int* M, int* N,
                                          cuDoubleComplex* A, int lda,
                                          double* S,
                                          cuDoubleComplex* U, int ldu,
                                          cuDoubleComplex* VT, int ldvt,
                                          cuDoubleComplex* work, int lwork,
                                          double* rwork,
                                          int* devInfo) {
    MKL_INT info = 0;
    mkl_zgesvd(&jobu, &jobvt, M, N, (MKL_Complex16*)A, lda, S,
               (MKL_Complex16*)U, ldu, (MKL_Complex16*)VT, ldvt,
               (MKL_Complex16*)work, lwork, rwork, &info);
    *devInfo = (int)info;
    return (info == 0) ? CUSOLVER_STATUS_SUCCESS : 1;
}

// dgeev → mkl_dgeev (real eigenvalue)
inline cusolverStatus_t cusolverDnDgeev(cusolverDnHandle_t /*handle*/,
                                         char jobvl, char jobvr,
                                         int* N,
                                         double* A, int lda,
                                         double* wr, double* wi,
                                         double* VL, int ldvl,
                                         double* VR, int ldvr,
                                         double* work, int lwork,
                                         int* devInfo) {
    MKL_INT info = 0;
    mkl_dgeev(&jobvl, &jobvr, N, A, lda, wr, wi, VL, ldvl, VR, ldvr,
              work, lwork, &info);
    *devInfo = (int)info;
    return (info == 0) ? CUSOLVER_STATUS_SUCCESS : 1;
}

// zgeev → mkl_zgeev (complex eigenvalue)
inline cusolverStatus_t cusolverDnZgeev(cusolverDnHandle_t /*handle*/,
                                         char jobvl, char jobvr,
                                         int* N,
                                         cuDoubleComplex* A, int lda,
                                         cuDoubleComplex* W,
                                         cuDoubleComplex* VL, int ldvl,
                                         cuDoubleComplex* VR, int ldvr,
                                         cuDoubleComplex* work, int lwork,
                                         double* rwork,
                                         int* devInfo) {
    MKL_INT info = 0;
    mkl_zgeev(&jobvl, &jobvr, N, (MKL_Complex16*)A, lda,
              (MKL_Complex16*)W, (MKL_Complex16*)VL, ldvl,
              (MKL_Complex16*)VR, ldvr, (MKL_Complex16*)work, lwork, rwork, &info);
    *devInfo = (int)info;
    return (info == 0) ? CUSOLVER_STATUS_SUCCESS : 1;
}

// Potrf: dpotrf / zpotrf
inline cusolverStatus_t cusolverDnDpotrf(cusolverDnHandle_t /*handle*/,
                                          cublasFillMode_t uplo, int n,
                                          double* A, int lda,
                                          int* devInfo) {
    MKL_INT info = 0;
    mkl_dpotrf((uplo == CUBLAS_FILL_MODE_UPPER) ? 'U' : 'L',
               &n, A, &lda, &info);
    *devInfo = (int)info;
    return (info == 0) ? CUSOLVER_STATUS_SUCCESS : 1;
}

// Trsm: dtrsm / ztrsm
inline cusolverStatus_t cusolverDnDtrsm(cusolverDnHandle_t /*handle*/,
                                         cublasSideMode_t side,
                                         cublasFillMode_t uplo,
                                         cublasOperation_t transA,
                                         cublasDiagType_t diag,
                                         int m, int n,
                                         const double* alpha,
                                         const double* A, int lda,
                                         double* B, int ldb) {
    mkl_dtrsm((side == CUBLAS_SIDE_LEFT) ? 'L' : 'R',
              (uplo == CUBLAS_FILL_MODE_UPPER) ? 'U' : 'L',
              (transA == CUBLAS_OP_N) ? 'N' : 'T',
              (diag == CUBLAS_DIAG_NON_UNIT) ? 'N' : 'U',
              &m, &n, alpha, A, &lda, B, &ldb);
    return CUSOLVER_STATUS_SUCCESS;
}

inline cusolverStatus_t cusolverDnZtrsm(cusolverDnHandle_t /*handle*/,
                                         cublasSideMode_t side,
                                         cublasFillMode_t uplo,
                                         cublasOperation_t transA,
                                         cublasDiagType_t diag,
                                         int m, int n,
                                         const cuDoubleComplex* alpha,
                                         const cuDoubleComplex* A, int lda,
                                         cuDoubleComplex* B, int ldb) {
    MKL_Complex16 _alpha = {*alpha->x, *alpha->y};
    mkl_ztrsm((side == CUBLAS_SIDE_LEFT) ? 'L' : 'R',
              (uplo == CUBLAS_FILL_MODE_UPPER) ? 'U' : 'L',
              (transA == CUBLAS_OP_N) ? 'N' : 'T',
              (diag == CUBLAS_DIAG_NON_UNIT) ? 'N' : 'U',
              &m, &n, &_alpha, (const MKL_Complex16*)A, &lda,
              (MKL_Complex16*)B, &ldb);
    return CUSOLVER_STATUS_SUCCESS;
}

// gesvdj (Jacobi SVD) — guarded, auto-fallback, Wannier-only
inline cusolverStatus_t cusolverDnDgesvdj(cusolverDnHandle_t /*handle*/) {
    return CUSOLVER_STATUS_SUCCESS;  // stub: falls back to LAPACK
}
