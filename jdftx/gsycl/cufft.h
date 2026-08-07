#pragma once
// cuFFT -> oneMKL DFT shim -- DECLARATIONS ONLY.
//
// Same GSL/oneMKL header-conflict rationale as cublas_v2.h/cusolverDn.h:
// this header must not include <oneapi/mkl/dft.hpp> (transitively pulls in
// mkl_cblas.h, clashing with gsl_cblas.h in the same translation unit as
// core/GpuUtil.h/GridInfo.cpp). The oneMKL DFT descriptor is only ever
// touched inside gsycl/cufft_impl.cpp via an opaque pointer here.

#include "sycl_device.hpp"  // defines double2, cudaError_t, cudaSuccess

// =====================================================================
// Type aliases and stubs
// =====================================================================
inline constexpr int CUFFT_FORWARD = -1;
inline constexpr int CUFFT_INVERSE = +1;
inline constexpr int CUFFT_Z2Z = 0;
inline constexpr int CUFFT_D2Z = 1;
inline constexpr int CUFFT_Z2D = 2;

// JDFTx casts complex<double>* as double2* -- same layout
typedef double2 cuDoubleComplex;

// =====================================================================
// Handle: opaque pointer to oneMKL DFT descriptor (defined only in
// cufft_impl.cpp, which is the sole TU that knows the real type).
// =====================================================================
struct cufftHandle {
    void* desc;   // opaque dft_descriptor_c16*
    int type;     // CUFFT_Z2Z, CUFFT_D2Z, CUFFT_Z2D
    bool initialized;
};

// =====================================================================
// Declarations only -- implemented in gsycl/cufft_impl.cpp
// =====================================================================
cudaError_t cufftPlan3d(cufftHandle* plan, int nx, int ny, int nz, int type);

cudaError_t cufftExecZ2Z(cufftHandle plan, const cuDoubleComplex* in,
                          cuDoubleComplex* out, int direction);

cudaError_t cufftExecD2Z(cufftHandle plan, const double* in,
                          cuDoubleComplex* out);

cudaError_t cufftExecZ2D(cufftHandle plan, const cuDoubleComplex* in,
                          double* out);

cudaError_t cufftDestroy(cufftHandle plan);
