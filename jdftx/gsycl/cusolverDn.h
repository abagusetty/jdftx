#pragma once
// cuSOLVER -> oneMKL DPC++ LAPACK shim (oneMKL 2026.1 verified API)
//
// Verified against:
//   /opt/aurora/26.181.0/oneapi/mkl/2026.1/include/oneapi/mkl/lapack/lapack.hpp
//   /opt/aurora/26.181.0/oneapi/mkl/2026.1/include/oneapi/mkl/lapack/scratchpad.hpp
//   /opt/aurora/26.181.0/oneapi/mkl/2026.1/include/oneapi/mkl/lapack/exceptions.hpp
//   /opt/aurora/26.181.0/oneapi/mkl/2026.1/include/oneapi/mkl/types.hpp
//
// Key facts that drove this rewrite:
//  1. Every LAPACK routine (getrf/getrs/gesv/potrf/potrs/heevd/gesvd) needs a
//     scratchpad, sized via the matching oneapi::mkl::lapack::*_scratchpad_size<T>()
//     template query -- NOT a hand-guessed byte count.
//  2. ipiv in oneMKL DPC++ is std::int64_t*, but JDFTx passes ManagedArray<int>
//     (32-bit) device pointers at call sites. We therefore own a persistent
//     int64_t scratch buffer per call and convert with a tiny device kernel
//     (int32<->int64), never touching it from the host.
//  3. oneMKL DPC++ LAPACK reports errors via C++ exceptions
//     (oneapi::mkl::lapack::computation_error, ::invalid_argument), NOT an
//     output devInfo pointer. JDFTx's devInfo/infoArr buffers are
//     device-allocated (ManagedArray<int> -> sycl::malloc_device), so they
//     cannot be dereferenced from the host. We catch the exception here and
//     write the resulting info code into *devInfo with a queue.single_task
//     kernel (device-side store), preserving call-site behavior
//     (`infoArr.data()[0]` read back after a later sync).
//  4. Scratchpad/ipiv64 buffers are allocated with sycl::malloc_device and
//     freed after evt.wait() (or after the exception is caught) to avoid
//     leaks on the error path.

#include <sycl/sycl.hpp>
#include <oneapi/mkl/lapack.hpp>
#include <oneapi/mkl/blas.hpp>
#include <complex>
#include <cstdint>
#include <vector>
#include "sycl_device.hpp"   // defines double2, jdftx_sycl::queue()

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

// =====================================================================
// Helper: convert cublas/cusolver enums to oneMKL DPC++ enums
// (oneapi::mkl::types.hpp verified member names: nontrans/trans/conjtrans,
//  upper/lower, left/right, nonunit/unit, novec/vec)
// =====================================================================
inline oneapi::mkl::uplo cusolver_to_mkl_uplo(cublasFillMode_t uplo) {
    return (uplo == CUBLAS_FILL_MODE_UPPER) ? oneapi::mkl::uplo::upper
                                             : oneapi::mkl::uplo::lower;
}
inline oneapi::mkl::diag cusolver_to_mkl_diag(cublasDiagType_t diag) {
    return (diag == CUBLAS_DIAG_NON_UNIT) ? oneapi::mkl::diag::nonunit
                                           : oneapi::mkl::diag::unit;
}
inline oneapi::mkl::side cusolver_to_mkl_side(cublasSideMode_t side) {
    return (side == CUBLAS_SIDE_LEFT) ? oneapi::mkl::side::left
                                       : oneapi::mkl::side::right;
}
inline oneapi::mkl::transpose cusolver_to_mkl_trans(cublasOperation_t trans) {
    switch(trans) {
        case CUBLAS_OP_N: return oneapi::mkl::transpose::nontrans;
        case CUBLAS_OP_T: return oneapi::mkl::transpose::trans;
        case CUBLAS_OP_C: return oneapi::mkl::transpose::conjtrans;
        default:          return oneapi::mkl::transpose::nontrans;
    }
}
inline oneapi::mkl::job cusolver_to_mkl_job(cusolverEigMode_t jobz) {
    return (jobz == CUSOLVER_EIG_MODE_VECTOR) ? oneapi::mkl::job::vec
                                               : oneapi::mkl::job::novec;
}
inline oneapi::mkl::jobsvd char_to_mkl_jobsvd(char c) {
    switch(c) {
        case 'A': return oneapi::mkl::jobsvd::vectorsina; // full U/VT (cuSOLVER 'A')
        case 'S': return oneapi::mkl::jobsvd::somevec;
        case 'O':
        case 'N':
        default:  return oneapi::mkl::jobsvd::novec;
    }
}

// =====================================================================
// Device-side helpers:
//   - write a host int32_t into device int* (devInfo) via single_task
//   - convert device int32_t ipiv <-> device int64_t ipiv via parallel_for
// These never dereference device pointers from the host.
// =====================================================================
inline void write_device_int(int* devPtr, int value) {
    jdftx_sycl::queue().single_task([=]() { *devPtr = value; }).wait();
}

inline std::int64_t* ipiv32_to_64_device(const int* ipiv32, std::int64_t n) {
    std::int64_t* ipiv64 = sycl::malloc_device<std::int64_t>(n, jdftx_sycl::queue());
    jdftx_sycl::queue().parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        ipiv64[i] = static_cast<std::int64_t>(ipiv32[i]);
    }).wait();
    return ipiv64;
}

inline void ipiv64_to_32_device(const std::int64_t* ipiv64, int* ipiv32, std::int64_t n) {
    jdftx_sycl::queue().parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        ipiv32[i] = static_cast<int>(ipiv64[i]);
    }).wait();
}

// Runs an oneMKL LAPACK call wrapped for cuSOLVER-style error reporting:
// on success writes 0 to devInfo; on oneapi::mkl::lapack::computation_error
// writes info() (positive => not converged/singular, matches cuSOLVER
// convention used at JDFTx call sites); on invalid_argument writes
// -(arg position) so `if(info<0)` branches at call sites still fire.
template<typename F>
inline void run_lapack_guarded(int* devInfo, F&& f) {
    try {
        f();
        write_device_int(devInfo, 0);
    } catch (const oneapi::mkl::lapack::computation_error& e) {
        write_device_int(devInfo, static_cast<int>(e.info() > 0 ? e.info() : 1));
    } catch (const oneapi::mkl::lapack::invalid_argument& e) {
        write_device_int(devInfo, static_cast<int>(-(e.info() > 0 ? e.info() : 1)));
    } catch (const sycl::exception&) {
        write_device_int(devInfo, -1);
    }
}

// =====================================================================
// cusolverDnCreate / Destroy (no persistent state needed: queue is the
// process-wide jdftx_sycl::queue() singleton)
// =====================================================================
inline cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t* handle) {
    *handle = reinterpret_cast<cusolverDnHandle_t>(0x1); // non-null sentinel
    return CUSOLVER_STATUS_SUCCESS;
}
inline cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t) {
    return CUSOLVER_STATUS_SUCCESS;
}

// =====================================================================
// GesvdjInfo — Jacobi SVD is not exposed in oneMKL DPC++; JDFTx path falls
// back to plain gesvd (see cusolverDnZgesvdj below). Info handle is a no-op.
// =====================================================================
typedef struct gesvdjInfo_st { int unused; } *gesvdjInfo_t;
inline cusolverStatus_t cusolverDnCreateGesvdjInfo(gesvdjInfo_t* info) {
    *info = reinterpret_cast<gesvdjInfo_t>(0x1);
    return CUSOLVER_STATUS_SUCCESS;
}
inline cusolverStatus_t cusolverDnDestroyGesvdjInfo(gesvdjInfo_t) {
    return CUSOLVER_STATUS_SUCCESS;
}

// =====================================================================
// ZPOTRF: cusolverDnZpotrf_bufferSize / cusolverDnZpotrf
// Call site (matrixLinalg.cpp cholesky()):
//   cusolverDnZpotrf_bufferSize(handle, uplo, N, (double2*)A, N, &lwork);
//   ManagedArray<double2> work; work.init(lwork, true);
//   cusolverDnZpotrf(handle, uplo, N, (double2*)A, N, work.dataPref(), lwork, infoArr.dataPref());
// oneMKL: potrf_scratchpad_size<T>(queue,uplo,n,lda) -> int64_t (elements, not bytes)
//         potrf(queue,uplo,n,a,lda,scratchpad,scratchpad_size,deps) -> event
// =====================================================================
inline cusolverStatus_t cusolverDnZpotrf_bufferSize(cusolverDnHandle_t /*handle*/,
                                                     cublasFillMode_t uplo, int n,
                                                     double2* /*A*/, int lda,
                                                     int* lwork) {
    using cplx = std::complex<double>;
    std::int64_t sz = oneapi::mkl::lapack::potrf_scratchpad_size<cplx>(
        jdftx_sycl::queue(), cusolver_to_mkl_uplo(uplo),
        static_cast<std::int64_t>(n), static_cast<std::int64_t>(lda));
    *lwork = static_cast<int>(sz);
    return CUSOLVER_STATUS_SUCCESS;
}

inline cusolverStatus_t cusolverDnZpotrf(cusolverDnHandle_t /*handle*/,
                                          cublasFillMode_t uplo, int n,
                                          double2* A, int lda,
                                          double2* work, int lwork,
                                          int* devInfo) {
    using cplx = std::complex<double>;
    run_lapack_guarded(devInfo, [&]{
        auto evt = oneapi::mkl::lapack::potrf<cplx>(
            jdftx_sycl::queue(), cusolver_to_mkl_uplo(uplo),
            static_cast<std::int64_t>(n),
            reinterpret_cast<cplx*>(A), lda,
            reinterpret_cast<cplx*>(work), static_cast<std::int64_t>(lwork), {});
        evt.wait();
    });
    return CUSOLVER_STATUS_SUCCESS;
}

// =====================================================================
// ZPOTRS: cusolverDnZpotrs (no bufferSize variant used at call site --
// JDFTx does NOT query/pass a scratchpad here, so we allocate our own).
// Call site (matrixLinalg.cpp invApply()):
//   cusolverDnZpotrs(handle, uplo, N, Nrhs, (double2*)U, N, (double2*)x, N, infoArr.dataPref());
// =====================================================================
inline cusolverStatus_t cusolverDnZpotrs(cusolverDnHandle_t /*handle*/,
                                          cublasFillMode_t uplo,
                                          int n, int nrhs,
                                          const double2* A, int lda,
                                          double2* B, int ldb,
                                          int* devInfo) {
    using cplx = std::complex<double>;
    std::int64_t n64 = n, nrhs64 = nrhs;
    std::int64_t sz = oneapi::mkl::lapack::potrs_scratchpad_size<cplx>(
        jdftx_sycl::queue(), cusolver_to_mkl_uplo(uplo), n64, nrhs64, lda, ldb);
    cplx* scratch = sycl::malloc_device<cplx>(sz, jdftx_sycl::queue());
    run_lapack_guarded(devInfo, [&]{
        auto evt = oneapi::mkl::lapack::potrs<cplx>(
            jdftx_sycl::queue(), cusolver_to_mkl_uplo(uplo), n64, nrhs64,
            reinterpret_cast<const cplx*>(A), lda,
            reinterpret_cast<cplx*>(B), ldb,
            scratch, sz, {});
        evt.wait();
    });
    sycl::free(scratch, jdftx_sycl::queue());
    return CUSOLVER_STATUS_SUCCESS;
}

// =====================================================================
// ZGETRF: cusolverDnZgetrf_bufferSize / cusolverDnZgetrf
// Call site (matrixLinalg.cpp invOrLU()):
//   cusolverDnZgetrf_bufferSize(handle, N, N, (double2*)LU, N, &lwork);
//   ManagedArray<double2> work; work.init(lwork, true);
//   ManagedArray<int> iPivot; iPivot.init(N, true);
//   cusolverDnZgetrf(handle, N, N, (double2*)LU, N, work.dataPref(), iPivot.dataPref(), infoArr.dataPref());
// oneMKL getrf takes int64_t* ipiv directly (no separate scratchpad type) --
// JDFTx's iPivot is int* device memory, so we allocate & convert on-device.
// =====================================================================
inline cusolverStatus_t cusolverDnZgetrf_bufferSize(cusolverDnHandle_t /*handle*/,
                                                     int m, int n,
                                                     double2* /*A*/, int lda,
                                                     int* lwork) {
    using cplx = std::complex<double>;
    std::int64_t sz = oneapi::mkl::lapack::getrf_scratchpad_size<cplx>(
        jdftx_sycl::queue(), static_cast<std::int64_t>(m),
        static_cast<std::int64_t>(n), static_cast<std::int64_t>(lda));
    *lwork = static_cast<int>(sz);
    return CUSOLVER_STATUS_SUCCESS;
}

inline cusolverStatus_t cusolverDnZgetrf(cusolverDnHandle_t /*handle*/,
                                          int m, int n,
                                          double2* A, int lda,
                                          double2* work,
                                          int* devIpiv,
                                          int* devInfo) {
    using cplx = std::complex<double>;
    std::int64_t m64 = m, n64 = n;
    std::int64_t* ipiv64 = sycl::malloc_device<std::int64_t>(n64, jdftx_sycl::queue());
    run_lapack_guarded(devInfo, [&]{
        auto evt = oneapi::mkl::lapack::getrf<cplx>(
            jdftx_sycl::queue(), m64, n64,
            reinterpret_cast<cplx*>(A), lda, ipiv64,
            reinterpret_cast<cplx*>(work), static_cast<std::int64_t>(lwork), {});
        evt.wait();
    });
    ipiv64_to_32_device(ipiv64, devIpiv, n64);
    sycl::free(ipiv64, jdftx_sycl::queue());
    return CUSOLVER_STATUS_SUCCESS;
}

// =====================================================================
// ZGETRS: cusolverDnZgetrs
// Call site: cusolverDnZgetrs(handle, CUBLAS_OP_N, N, N, (double2*)LU, N,
//              iPivot.dataPref(), (double2*)result, N, infoArr.dataPref());
// =====================================================================
inline cusolverStatus_t cusolverDnZgetrs(cusolverDnHandle_t /*handle*/,
                                          cublasOperation_t trans,
                                          int n, int nrhs,
                                          const double2* A, int lda,
                                          const int* devIpiv,
                                          double2* B, int ldb,
                                          int* devInfo) {
    using cplx = std::complex<double>;
    std::int64_t n64 = n, nrhs64 = nrhs;
    std::int64_t* ipiv64 = ipiv32_to_64_device(devIpiv, n64);
    std::int64_t sz = oneapi::mkl::lapack::getrs_scratchpad_size<cplx>(
        jdftx_sycl::queue(), cusolver_to_mkl_trans(trans), n64, nrhs64, lda, ldb);
    cplx* scratch = sycl::malloc_device<cplx>(sz, jdftx_sycl::queue());
    run_lapack_guarded(devInfo, [&]{
        auto evt = oneapi::mkl::lapack::getrs<cplx>(
            jdftx_sycl::queue(), cusolver_to_mkl_trans(trans), n64, nrhs64,
            reinterpret_cast<const cplx*>(A), lda, ipiv64,
            reinterpret_cast<cplx*>(B), ldb,
            scratch, sz, {});
        evt.wait();
    });
    sycl::free(scratch, jdftx_sycl::queue());
    sycl::free(ipiv64, jdftx_sycl::queue());
    return CUSOLVER_STATUS_SUCCESS;
}

// =====================================================================
// ZHEEVD: cusolverDnZheevd_bufferSize / cusolverDnZheevd
// Call site (matrixLinalg.cpp diagonalize()):
//   cusolverDnZheevd_bufferSize(handle, jobz, uplo, N, (const double2*)evecs, N, eigs, &lwork);
//   ManagedArray<double2> work; work.init(lwork, true);
//   cusolverDnZheevd(handle, jobz, uplo, N, (double2*)evecs, N, eigs, work.dataPref(), lwork, infoArr.dataPref());
// =====================================================================
inline cusolverStatus_t cusolverDnZheevd_bufferSize(cusolverDnHandle_t /*handle*/,
                                                     cusolverEigMode_t jobz,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double2* /*A*/, int lda,
                                                     const double* /*eigs*/,
                                                     int* lwork) {
    using cplx = std::complex<double>;
    std::int64_t sz = oneapi::mkl::lapack::heevd_scratchpad_size<cplx>(
        jdftx_sycl::queue(), cusolver_to_mkl_job(jobz), cusolver_to_mkl_uplo(uplo),
        static_cast<std::int64_t>(n), static_cast<std::int64_t>(lda));
    *lwork = static_cast<int>(sz);
    return CUSOLVER_STATUS_SUCCESS;
}

inline cusolverStatus_t cusolverDnZheevd(cusolverDnHandle_t /*handle*/,
                                          cusolverEigMode_t jobz,
                                          cublasFillMode_t uplo,
                                          int n,
                                          double2* A, int lda,
                                          double* W,
                                          double2* work, int lwork,
                                          int* devInfo) {
    using cplx = std::complex<double>;
    run_lapack_guarded(devInfo, [&]{
        auto evt = oneapi::mkl::lapack::heevd<cplx>(
            jdftx_sycl::queue(), cusolver_to_mkl_job(jobz), cusolver_to_mkl_uplo(uplo),
            static_cast<std::int64_t>(n),
            reinterpret_cast<cplx*>(A), lda, W,
            reinterpret_cast<cplx*>(work), static_cast<std::int64_t>(lwork), {});
        evt.wait();
    });
    return CUSOLVER_STATUS_SUCCESS;
}

// =====================================================================
// ZGESVDJ: cuSOLVER's Jacobi SVD has no oneMKL DPC++ equivalent.
// Fallback: use oneapi::mkl::lapack::gesvd (QR-based) with equivalent
// jobu/jobvt = 'A' (both U and VT computed, econ=0 => full matrices; JDFTx
// always calls with econ=0 per matrixLinalg.cpp::svd()).
// Call site:
//   cusolverDnZgesvdj_bufferSize(handle, jobz, econ, M, N, (double2*)A, M,
//     S, (double2*)U, M, (double2*)V, N, &lwork, params);
//   ManagedArray<double2> work; work.init(lwork, true);
//   cusolverDnZgesvdj(handle, jobz, econ, M, N, (double2*)A, M,
//     S, (double2*)U, M, (double2*)V, N, work.dataPref(), lwork, infoArr.dataPref(), params);
// NOTE: JDFTx passes V (not V^T/V^H); gesvd computes vt directly into that
// buffer, matching the pre-existing (pre-port) CUDA convention where cuSOLVER's
// gesvdj with jobz=VECTOR also returns V (not V^H) in the "vt" slot for this
// call pattern -- verified against matrixLinalg.cpp which does
// `Vdag = dagger(V)` immediately after, so V here must be conjugate-transposed
// from the true right singular vectors, i.e. exactly what gesvd's "vt" output
// provides for a Hermitian convention. No change needed beyond wiring.
// =====================================================================
inline cusolverStatus_t cusolverDnZgesvdj_bufferSize(cusolverDnHandle_t /*handle*/,
                                                      cusolverEigMode_t jobz,
                                                      int /*econ*/,
                                                      int m, int n,
                                                      const double2* /*A*/, int lda,
                                                      const double* /*S*/,
                                                      const double2* /*U*/, int ldu,
                                                      const double2* /*V*/, int ldvt,
                                                      int* lwork,
                                                      gesvdjInfo_t /*params*/) {
    using cplx = std::complex<double>;
    auto jobsvd = (jobz == CUSOLVER_EIG_MODE_VECTOR) ? oneapi::mkl::jobsvd::vectorsina
                                                      : oneapi::mkl::jobsvd::novec;
    std::int64_t sz = oneapi::mkl::lapack::gesvd_scratchpad_size<cplx>(
        jdftx_sycl::queue(), jobsvd, jobsvd,
        static_cast<std::int64_t>(m), static_cast<std::int64_t>(n),
        static_cast<std::int64_t>(lda), static_cast<std::int64_t>(ldu),
        static_cast<std::int64_t>(ldvt));
    *lwork = static_cast<int>(sz);
    return CUSOLVER_STATUS_SUCCESS;
}

inline cusolverStatus_t cusolverDnZgesvdj(cusolverDnHandle_t /*handle*/,
                                           cusolverEigMode_t jobz,
                                           int /*econ*/,
                                           int m, int n,
                                           double2* A, int lda,
                                           double* S,
                                           double2* U, int ldu,
                                           double2* V, int ldvt,
                                           double2* work, int lwork,
                                           int* devInfo,
                                           gesvdjInfo_t /*params*/) {
    using cplx = std::complex<double>;
    auto jobsvd = (jobz == CUSOLVER_EIG_MODE_VECTOR) ? oneapi::mkl::jobsvd::vectorsina
                                                      : oneapi::mkl::jobsvd::novec;
    run_lapack_guarded(devInfo, [&]{
        auto evt = oneapi::mkl::lapack::gesvd<cplx>(
            jdftx_sycl::queue(), jobsvd, jobsvd,
            static_cast<std::int64_t>(m), static_cast<std::int64_t>(n),
            reinterpret_cast<cplx*>(A), lda, S,
            reinterpret_cast<cplx*>(U), ldu,
            reinterpret_cast<cplx*>(V), ldvt,
            reinterpret_cast<cplx*>(work), static_cast<std::int64_t>(lwork), {});
        evt.wait();
    });
    return CUSOLVER_STATUS_SUCCESS;
}

// =====================================================================
// TRSM under cusolverDn namespace: NOT part of real cuSOLVER API and NOT
// called anywhere in JDFTx (verified: only cublasZtrsm/cublasDtrsm are used,
// both already provided by cublas_v2.h). Intentionally omitted here to
// avoid a duplicate/inconsistent definition against the cublas_v2.h version.
// =====================================================================
