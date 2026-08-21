// gsycl/cublas_impl.cpp -- oneMKL DPC++ BLAS implementation of the cublas_v2.h
// shim declarations. This is the ONLY translation unit that includes
// <oneapi/mkl/blas.hpp> (and therefore mkl_cblas.h transitively). It never
// includes <gsl/gsl_cblas.h>, so there is no CBLAS enum redefinition clash.
// This mirrors the real cuBLAS architecture: cublas_v2.h is a pure
// declarations header (like NVIDIA's), and the actual implementation lives
// in a separately-compiled unit (like NVIDIA's prebuilt libcublas.so).

#include "cublas_v2.h"
#include <sycl/sycl.hpp>
#include <oneapi/mkl/blas.hpp>
#include <complex>
#include <cstdint>

namespace {

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

// cuBLAS's Level-1 scalar-result routines (dot, dotc, nrm2, asum) default to
// CUBLAS_POINTER_MODE_HOST: the caller passes a *host* pointer for `result`,
// and cuBLAS transparently copies the device-computed reduction back to it
// before returning (JDFTx never calls cublasSetPointerMode, so this default
// is what every call site here relies on -- see e.g. eblas_zdotc_gpu/
// eblas_ddot_gpu in core/BlasExtra.cu, which pass the address of a plain
// stack variable). oneMKL has no such abstraction: its dot/dotc/nrm2/asum
// require `result` to be a device-writable (USM) pointer, full stop. Handing
// a host stack address straight through, as a literal translation would,
// makes the GPU write to an unmapped host address -- a page fault whose
// async reporting (see gsycl/README.md) can surface much later, misattributed
// to whatever kernel happens to be in flight at the time. Route every such
// result through a process-lifetime device-only scratch scalar and copy it
// to the caller's host pointer ourselves via an explicit blocking memcpy,
// reproducing cuBLAS's own behavior. (A malloc_shared scratch was tried
// first so the host could read it directly with no copy, but a direct host
// dereference right after evt.wait() intermittently segfaulted -- shared-USM
// host-visibility apparently isn't guaranteed synchronous with the compute
// event's completion on this driver stack. malloc_device + explicit
// queue().memcpy(...).wait() sidesteps that ambiguity entirely.)
template<typename T> T* scratchResult()
{   static T* buf = static_cast<T*>(sycl::malloc_device(sizeof(T), jdftx_sycl::queue()));
    return buf;
}
template<typename T> void copyResultToHost(T* hostResult, const T* deviceScratch)
{   jdftx_sycl::queue().memcpy(hostResult, deviceScratch, sizeof(T)).wait();
}

} // namespace

cublasStatus_t cublasDgemm(cublasHandle_t /*handle*/,
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

cublasStatus_t cublasZgemm(cublasHandle_t /*handle*/,
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

cublasStatus_t cublasDaxpy_v2(cublasHandle_t /*handle*/, int n,
                               const double* alpha,
                               const double* x, int incx,
                               double* y, int incy) {
    auto evt = oneapi::mkl::blas::axpy(
        jdftx_sycl::queue(), static_cast<std::int64_t>(n),
        alpha, x, incx, y, incy, {});
    evt.wait();
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZaxpy_v2(cublasHandle_t /*handle*/, int n,
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

cublasStatus_t cublasDdot_v2(cublasHandle_t /*handle*/, int n,
                              const double* x, int incx,
                              const double* y, int incy,
                              double* result) {
    double* scratch = scratchResult<double>();
    auto evt = oneapi::mkl::blas::dot(
        jdftx_sycl::queue(), static_cast<std::int64_t>(n),
        x, incx, y, incy, scratch, {});
    evt.wait();
    copyResultToHost(result, scratch);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZdotc_v2(cublasHandle_t /*handle*/, int n,
                               const double2* x, int incx,
                               const double2* y, int incy,
                               double2* result) {
    using cplx = std::complex<double>;
    cplx* scratch = scratchResult<cplx>();
    auto evt = oneapi::mkl::blas::dotc(
        jdftx_sycl::queue(), static_cast<std::int64_t>(n),
        reinterpret_cast<const cplx*>(x), incx,
        reinterpret_cast<const cplx*>(y), incy,
        scratch, {});
    evt.wait();
    copyResultToHost(reinterpret_cast<cplx*>(result), scratch);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDscal_v2(cublasHandle_t /*handle*/, int n,
                               const double* alpha,
                               double* x, int incx) {
    auto evt = oneapi::mkl::blas::scal(
        jdftx_sycl::queue(), static_cast<std::int64_t>(n),
        alpha, x, incx, {});
    evt.wait();
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZscal_v2(cublasHandle_t /*handle*/, int n,
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

cublasStatus_t cublasZdscal_v2(cublasHandle_t /*handle*/, int n,
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

cublasStatus_t cublasDnrm2_v2(cublasHandle_t /*handle*/, int n,
                               const double* x, int incx,
                               double* result) {
    double* scratch = scratchResult<double>();
    auto evt = oneapi::mkl::blas::nrm2(
        jdftx_sycl::queue(), static_cast<std::int64_t>(n),
        x, incx, scratch, {});
    evt.wait();
    copyResultToHost(result, scratch);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDznrm2_v2(cublasHandle_t /*handle*/, int n,
                                const double2* x, int incx,
                                double* result) {
    using cplx = std::complex<double>;
    double* scratch = scratchResult<double>();
    auto evt = oneapi::mkl::blas::nrm2(
        jdftx_sycl::queue(), static_cast<std::int64_t>(n),
        reinterpret_cast<const cplx*>(x), incx, scratch, {});
    evt.wait();
    copyResultToHost(result, scratch);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDasum(cublasHandle_t /*handle*/, int n,
                            const double* x, int incx,
                            double* result) {
    double* scratch = scratchResult<double>();
    auto evt = oneapi::mkl::blas::asum(
        jdftx_sycl::queue(), static_cast<std::int64_t>(n),
        x, incx, scratch, {});
    evt.wait();
    copyResultToHost(result, scratch);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrsm(cublasHandle_t /*handle*/,
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

cublasStatus_t cublasZtrsm(cublasHandle_t /*handle*/,
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
