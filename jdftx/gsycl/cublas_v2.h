#pragma once
// cuBLAS -> oneMKL DPC++ BLAS shim (oneMKL 2026.1 verified API)
// Maps cublas_v2 API actually used by JDFTx to oneapi::mkl::blas USM calls.
//
// Verified against: /opt/aurora/26.181.0/oneapi/mkl/2026.1/include/oneapi/mkl/blas/usm_decls.hpp
//   gemm(queue, transa, transb, m,n,k, alpha, a,lda, b,ldb, beta, c,ldc, mode, deps)
//   trsm(queue, side, uplo, trans, diag, m,n, alpha, a,lda, b,ldb, mode, deps)
//   axpy(queue, n, alpha, x,incx, y,incy, deps)
//   dot(queue, n, x,incx, y,incy, result, deps)
//   dotc(queue, n, x,incx, y,incy, result, deps)
//   scal(queue, n, alpha, x,incx, deps)          -- alpha type may differ from x type (zdscal)
//   nrm2(queue, n, x,incx, result, deps)         -- result type differs for complex (Tres)
//   asum(queue, n, x,incx, result, deps)         -- result type differs for complex (Tres)
// alpha/beta use value_or_pointer<T>, implicitly constructible from const T*.

#include <sycl/sycl.hpp>
#include <oneapi/mkl/blas.hpp>
#include <complex>
#include <cstdint>
#include "sycl_device.hpp"   // defines double2, float2, jdftx_sycl::queue()

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

// NOTE: double2/float2 come from sycl_device.hpp — do NOT redefine here.
// JDFTx code casts complex<double>* to (const double2*)/(double2*) at call
// sites, so all complex entry points below take double2*, matching call
// sites exactly (no extra reinterpret_cast needed inside the shim; we
// reinterpret double2* -> std::complex<double>* internally instead, since
// std::complex<double> is layout-compatible with two doubles per [complex.numbers]).

// =====================================================================
// Helper: convert cublas enums to oneMKL DPC++ enums
// =====================================================================
inline oneapi::mkl::transpose cublas_to_mkl_trans(cublasOperation_t trans) {
    switch(trans) {
        case CUBLAS_OP_N: return oneapi::mkl::transpose::nontrans;
        case CUBLAS_OP_T: return oneapi::mkl::transpose::trans;
        case CUBLAS_OP_C: return oneapi::mkl::transpose::conjtrans;
        default:          return oneapi::mkl::transpose::nontrans;
    }
}

inline oneapi::mkl::side cublas_to_mkl_side(cublasSideMode_t side) {
    return (side == CUBLAS_SIDE_LEFT) ? oneapi::mkl::side::left
                                       : oneapi::mkl::side::right;
}

inline oneapi::mkl::uplo cublas_to_mkl_uplo(cublasFillMode_t uplo) {
    return (uplo == CUBLAS_FILL_MODE_UPPER) ? oneapi::mkl::uplo::upper
                                             : oneapi::mkl::uplo::lower;
}

inline oneapi::mkl::diag cublas_to_mkl_diag(cublasDiagType_t diag) {
    return (diag == CUBLAS_DIAG_NON_UNIT) ? oneapi::mkl::diag::nonunit
                                           : oneapi::mkl::diag::unit;
}

// =====================================================================
// GEMM: cublasDgemm / cublasZgemm
// Call site: eblas_zgemm_gpu -> cublasZgemm(handle, tA, tB, M,N,K,
//              (const double2*)&alpha, (const double2*)A, lda, (const double2*)B, ldb,
//              (const double2*)&beta, (double2*)C, ldc)
// =====================================================================
inline cublasStatus_t cublasDgemm(cublasHandle_t /*handle*/,
                                   cublasOperation_t transA, cublasOperation_t transB,
                                   int m, int n, int k,
                                   const double* alpha,
                                   const double* A, int lda,
                                   const double* B, int ldb,
                                   const double* beta,
                                   double* C, int ldc) {
    auto evt = oneapi::mkl::blas::gemm(
        jdftx_sycl::queue(),
        cublas_to_mkl_trans(transA), cublas_to_mkl_trans(transB),
        static_cast<std::int64_t>(m), static_cast<std::int64_t>(n), static_cast<std::int64_t>(k),
        alpha, A, lda, B, ldb, beta, C, ldc,
        oneapi::mkl::blas::compute_mode::unset, {});
    evt.wait();
    return CUBLAS_STATUS_SUCCESS;
}

inline cublasStatus_t cublasZgemm(cublasHandle_t /*handle*/,
                                   cublasOperation_t transA, cublasOperation_t transB,
                                   int m, int n, int k,
                                   const double2* alpha,
                                   const double2* A, int lda,
                                   const double2* B, int ldb,
                                   const double2* beta,
                                   double2* C, int ldc) {
    using cplx = std::complex<double>;
    auto evt = oneapi::mkl::blas::gemm(
        jdftx_sycl::queue(),
        cublas_to_mkl_trans(transA), cublas_to_mkl_trans(transB),
        static_cast<std::int64_t>(m), static_cast<std::int64_t>(n), static_cast<std::int64_t>(k),
        reinterpret_cast<const cplx*>(alpha),
        reinterpret_cast<const cplx*>(A), lda,
        reinterpret_cast<const cplx*>(B), ldb,
        reinterpret_cast<const cplx*>(beta),
        reinterpret_cast<cplx*>(C), ldc,
        oneapi::mkl::blas::compute_mode::unset, {});
    evt.wait();
    return CUBLAS_STATUS_SUCCESS;
}

// =====================================================================
// AXPY: cublasDaxpy_v2 / cublasZaxpy_v2
// =====================================================================
inline cublasStatus_t cublasDaxpy_v2(cublasHandle_t /*handle*/, int n,
                                      const double* alpha,
                                      const double* x, int incx,
                                      double* y, int incy) {
    auto evt = oneapi::mkl::blas::axpy(
        jdftx_sycl::queue(), static_cast<std::int64_t>(n),
        alpha, x, incx, y, incy, {});
    evt.wait();
    return CUBLAS_STATUS_SUCCESS;
}

inline cublasStatus_t cublasZaxpy_v2(cublasHandle_t /*handle*/, int n,
                                      const double2* alpha,
                                      const double2* x, int incx,
                                      double2* y, int incy) {
    using cplx = std::complex<double>;
    auto evt = oneapi::mkl::blas::axpy(
        jdftx_sycl::queue(), static_cast<std::int64_t>(n),
        reinterpret_cast<const cplx*>(alpha),
        reinterpret_cast<const cplx*>(x), incx,
        reinterpret_cast<cplx*>(y), incy, {});
    evt.wait();
    return CUBLAS_STATUS_SUCCESS;
}

// =====================================================================
// DOT: cublasDdot_v2
// =====================================================================
inline cublasStatus_t cublasDdot_v2(cublasHandle_t /*handle*/, int n,
                                     const double* x, int incx,
                                     const double* y, int incy,
                                     double* result) {
    auto evt = oneapi::mkl::blas::dot(
        jdftx_sycl::queue(), static_cast<std::int64_t>(n),
        x, incx, y, incy, result, {});
    evt.wait();
    return CUBLAS_STATUS_SUCCESS;
}

// =====================================================================
// DOTC: cublasZdotc_v2
// =====================================================================
inline cublasStatus_t cublasZdotc_v2(cublasHandle_t /*handle*/, int n,
                                      const double2* x, int incx,
                                      const double2* y, int incy,
                                      double2* result) {
    using cplx = std::complex<double>;
    auto evt = oneapi::mkl::blas::dotc(
        jdftx_sycl::queue(), static_cast<std::int64_t>(n),
        reinterpret_cast<const cplx*>(x), incx,
        reinterpret_cast<const cplx*>(y), incy,
        reinterpret_cast<cplx*>(result), {});
    evt.wait();
    return CUBLAS_STATUS_SUCCESS;
}

// =====================================================================
// SCAL: cublasDscal_v2 / cublasZscal_v2 / cublasZdscal_v2
// Note: oneMKL scal(queue,n,alpha,x,incx) has alpha type Ts which may
// differ from x's type T (e.g. real alpha scaling complex x = zdscal).
// =====================================================================
inline cublasStatus_t cublasDscal_v2(cublasHandle_t /*handle*/, int n,
                                      const double* alpha,
                                      double* x, int incx) {
    auto evt = oneapi::mkl::blas::scal(
        jdftx_sycl::queue(), static_cast<std::int64_t>(n),
        alpha, x, incx, {});
    evt.wait();
    return CUBLAS_STATUS_SUCCESS;
}

inline cublasStatus_t cublasZscal_v2(cublasHandle_t /*handle*/, int n,
                                      const double2* alpha,
                                      double2* x, int incx) {
    using cplx = std::complex<double>;
    auto evt = oneapi::mkl::blas::scal(
        jdftx_sycl::queue(), static_cast<std::int64_t>(n),
        reinterpret_cast<const cplx*>(alpha),
        reinterpret_cast<cplx*>(x), incx, {});
    evt.wait();
    return CUBLAS_STATUS_SUCCESS;
}

// cublasZdscal_v2: real alpha (double) scaling complex x -- distinct oneMKL
// overload: scal(queue, n, double alpha, std::complex<double>* x, incx, deps)
inline cublasStatus_t cublasZdscal_v2(cublasHandle_t /*handle*/, int n,
                                       const double* alpha,
                                       double2* x, int incx) {
    using cplx = std::complex<double>;
    auto evt = oneapi::mkl::blas::scal(
        jdftx_sycl::queue(), static_cast<std::int64_t>(n),
        *alpha, reinterpret_cast<cplx*>(x), incx,
        std::vector<sycl::event>{});
    evt.wait();
    return CUBLAS_STATUS_SUCCESS;
}

// =====================================================================
// NRM2: cublasDnrm2_v2 / cublasDznrm2_v2
// oneMKL nrm2(queue,n,x,incx,result,deps): result type is always real (Tres)
// even for complex x -- matches cublasDznrm2_v2(..., double* result) exactly.
// =====================================================================
inline cublasStatus_t cublasDnrm2_v2(cublasHandle_t /*handle*/, int n,
                                      const double* x, int incx,
                                      double* result) {
    auto evt = oneapi::mkl::blas::nrm2(
        jdftx_sycl::queue(), static_cast<std::int64_t>(n),
        x, incx, result, {});
    evt.wait();
    return CUBLAS_STATUS_SUCCESS;
}

inline cublasStatus_t cublasDznrm2_v2(cublasHandle_t /*handle*/, int n,
                                       const double2* x, int incx,
                                       double* result) {
    using cplx = std::complex<double>;
    auto evt = oneapi::mkl::blas::nrm2(
        jdftx_sycl::queue(), static_cast<std::int64_t>(n),
        reinterpret_cast<const cplx*>(x), incx, result, {});
    evt.wait();
    return CUBLAS_STATUS_SUCCESS;
}

// =====================================================================
// ASUM: cublasDasum (real-only usage in JDFTx, matrixOperators.cu)
// =====================================================================
inline cublasStatus_t cublasDasum(cublasHandle_t /*handle*/, int n,
                                   const double* x, int incx,
                                   double* result) {
    auto evt = oneapi::mkl::blas::asum(
        jdftx_sycl::queue(), static_cast<std::int64_t>(n),
        x, incx, result, {});
    evt.wait();
    return CUBLAS_STATUS_SUCCESS;
}

// =====================================================================
// TRSM: cublasDtrsm / cublasZtrsm (in-place)
// Call site: cublasZtrsm(handle, side, uplo, transA, diag, N, N,
//              (const double2*)&alpha, (const double2*)T, N, (double2*)rhs, N)
// =====================================================================
inline cublasStatus_t cublasDtrsm(cublasHandle_t /*handle*/,
                                   cublasSideMode_t side,
                                   cublasFillMode_t uplo,
                                   cublasOperation_t transA,
                                   cublasDiagType_t diag,
                                   int m, int n,
                                   const double* alpha,
                                   const double* A, int lda,
                                   double* B, int ldb) {
    auto evt = oneapi::mkl::blas::trsm(
        jdftx_sycl::queue(),
        cublas_to_mkl_side(side), cublas_to_mkl_uplo(uplo),
        cublas_to_mkl_trans(transA), cublas_to_mkl_diag(diag),
        static_cast<std::int64_t>(m), static_cast<std::int64_t>(n),
        alpha, A, lda, B, ldb,
        oneapi::mkl::blas::compute_mode::unset, {});
    evt.wait();
    return CUBLAS_STATUS_SUCCESS;
}

inline cublasStatus_t cublasZtrsm(cublasHandle_t /*handle*/,
                                   cublasSideMode_t side,
                                   cublasFillMode_t uplo,
                                   cublasOperation_t transA,
                                   cublasDiagType_t diag,
                                   int m, int n,
                                   const double2* alpha,
                                   const double2* A, int lda,
                                   double2* B, int ldb) {
    using cplx = std::complex<double>;
    auto evt = oneapi::mkl::blas::trsm(
        jdftx_sycl::queue(),
        cublas_to_mkl_side(side), cublas_to_mkl_uplo(uplo),
        cublas_to_mkl_trans(transA), cublas_to_mkl_diag(diag),
        static_cast<std::int64_t>(m), static_cast<std::int64_t>(n),
        reinterpret_cast<const cplx*>(alpha),
        reinterpret_cast<const cplx*>(A), lda,
        reinterpret_cast<cplx*>(B), ldb,
        oneapi::mkl::blas::compute_mode::unset, {});
    evt.wait();
    return CUBLAS_STATUS_SUCCESS;
}
