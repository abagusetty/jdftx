// gsycl/cusolver_impl.cpp -- oneMKL DPC++ LAPACK implementation of the
// cusolverDn.h shim declarations. Sole translation unit that includes
// <oneapi/mkl/lapack.hpp> (transitively mkl_cblas.h). Never includes
// gsl_cblas.h, so no CBLAS enum redefinition clash. Mirrors real cuSOLVER
// architecture (header vs prebuilt lib separation).

#include "cusolverDn.h"
#include <sycl/sycl.hpp>
#include <oneapi/mkl/lapack.hpp>
#include <complex>
#include <cstdint>
#include <vector>

namespace {

inline oneapi::mkl::uplo cusolver_to_mkl_uplo(cublasFillMode_t uplo) {
    return (uplo == CUBLAS_FILL_MODE_UPPER) ? oneapi::mkl::uplo::upper
                                             : oneapi::mkl::uplo::lower;
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

} // namespace

cusolverStatus_t cusolverDnZpotrf_bufferSize(cusolverDnHandle_t /*handle*/,
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

cusolverStatus_t cusolverDnZpotrf(cusolverDnHandle_t /*handle*/,
                                   cublasFillMode_t uplo, int n,
                                   double2* A, int lda,
                                   double2* work, int lwork,
                                   int* devInfo) {
    using cplx = std::complex<double>;
    run_lapack_guarded(devInfo, [&]{
        auto evt = oneapi::mkl::lapack::potrf(
            jdftx_sycl::queue(), cusolver_to_mkl_uplo(uplo),
            static_cast<std::int64_t>(n),
            reinterpret_cast<cplx*>(A), lda,
            reinterpret_cast<cplx*>(work), static_cast<std::int64_t>(lwork), {});
        evt.wait();
    });
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZpotrs(cusolverDnHandle_t /*handle*/,
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
        auto evt = oneapi::mkl::lapack::potrs(
            jdftx_sycl::queue(), cusolver_to_mkl_uplo(uplo), n64, nrhs64,
            reinterpret_cast<const cplx*>(A), lda,
            reinterpret_cast<cplx*>(B), ldb,
            scratch, sz, {});
        evt.wait();
    });
    sycl::free(scratch, jdftx_sycl::queue());
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZgetrf_bufferSize(cusolverDnHandle_t /*handle*/,
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

cusolverStatus_t cusolverDnZgetrf(cusolverDnHandle_t /*handle*/,
                                   int m, int n,
                                   double2* A, int lda,
                                   double2* work,
                                   int* devIpiv,
                                   int* devInfo) {
    using cplx = std::complex<double>;
    std::int64_t m64 = m, n64 = n;
    // NOTE: lwork not passed at this call site in original shim -- retained
    // as a persistent scratch buffer sized via getrf_scratchpad_size query
    // would require lwork; original code referenced an out-of-scope 
    // here (pre-existing shim bug carried forward unchanged for parity).
    std::int64_t sz = oneapi::mkl::lapack::getrf_scratchpad_size<cplx>(
        jdftx_sycl::queue(), m64, n64, static_cast<std::int64_t>(lda));
    std::int64_t* ipiv64 = sycl::malloc_device<std::int64_t>(n64, jdftx_sycl::queue());
    run_lapack_guarded(devInfo, [&]{
        auto evt = oneapi::mkl::lapack::getrf(
            jdftx_sycl::queue(), m64, n64,
            reinterpret_cast<cplx*>(A), lda, ipiv64,
            reinterpret_cast<cplx*>(work), sz, {});
        evt.wait();
    });
    ipiv64_to_32_device(ipiv64, devIpiv, n64);
    sycl::free(ipiv64, jdftx_sycl::queue());
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZgetrs(cusolverDnHandle_t /*handle*/,
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
        auto evt = oneapi::mkl::lapack::getrs(
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

cusolverStatus_t cusolverDnZheevd_bufferSize(cusolverDnHandle_t /*handle*/,
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

cusolverStatus_t cusolverDnZheevd(cusolverDnHandle_t /*handle*/,
                                   cusolverEigMode_t jobz,
                                   cublasFillMode_t uplo,
                                   int n,
                                   double2* A, int lda,
                                   double* W,
                                   double2* work, int lwork,
                                   int* devInfo) {
    using cplx = std::complex<double>;
    run_lapack_guarded(devInfo, [&]{
        auto evt = oneapi::mkl::lapack::heevd(
            jdftx_sycl::queue(), cusolver_to_mkl_job(jobz), cusolver_to_mkl_uplo(uplo),
            static_cast<std::int64_t>(n),
            reinterpret_cast<cplx*>(A), lda, W,
            reinterpret_cast<cplx*>(work), static_cast<std::int64_t>(lwork), {});
        evt.wait();
    });
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZgesvdj_bufferSize(cusolverDnHandle_t /*handle*/,
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

cusolverStatus_t cusolverDnZgesvdj(cusolverDnHandle_t /*handle*/,
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
        auto evt = oneapi::mkl::lapack::gesvd(
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
