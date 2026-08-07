#pragma once
// cuFFT -> oneMKL DFT shim (USM SYCL API)
// Maps cufft API to oneapi::mkl::dft::* USM calls.

#include <sycl/sycl.hpp>
#include <oneapi/mkl/dft.hpp>
#include sycl_device.hpp  // defines double2, cudaSuccess, cuDoubleConcept

// =====================================================================
// Type aliases and stubs (NO cudaSuccess here - defined in sycl_device.hpp)
// =====================================================================
typedef int cudaError_t;
inline constexpr int CUFFT_FORWARD = -1;
inline constexpr int CUFFT_INVERSE = +1;
inline constexpr int CUFFT_Z2Z = 0;
inline constexpr int CUFFT_D2Z = 1;
inline constexpr int CUFFT_Z2D = 2;

// JDFTx casts complex<double>* as double2* — same layout
typedef double2 cuDoubleComplex;

// =====================================================================
// Handle: wraps oneMKL DFT descriptor
// =====================================================================
typedef oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
                                      oneapi::mkl::dft::domain::COMPLEX> dft_descriptor_c16;

struct cufftHandle {
    dft_descriptor_c16* desc;
    int type;    // CUFFT_Z2Z, CUFFT_D2Z, CUFFT_Z2D
    bool initialized;
};

// =====================================================================
// Helper: create descriptor for 3D plan
// =====================================================================
static dft_descriptor_c16* create_plan(int type, int nx, int ny, int nz) {
    std::vector<std::int64_t> n = { (std::int64_t)nz, (std::int64_t)ny, (std::int64_t)nx };

    if (type == CUFFT_Z2Z) {
        // Complex -> Complex
        auto desc = new dft_descriptor_c16(n);
        desc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                        oneapi::mkl::dft::config_value::NOT_INPLACE);
        desc->commit(jdftx_sycl::queue());
        return desc;
    }
    else if (type == CUFFT_D2Z) {
        // Real -> Complex
        auto desc = new dft_descriptor_c16(n);
        desc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                        oneapi::mkl::dft::config_value::NOT_INPLACE);
        std::vector<std::int64_t> strides(6, 0);
        desc->set_value(oneapi::mkl::dft::config_param::FWD_STRIDES, strides);
        desc->set_value(oneapi::mkl::dft::config_param::BWD_STRIDES, strides);
        desc->commit(jdftx_sycl::queue());
        return desc;
    }
    else if (type == CUFFT_Z2D) {
        // Complex -> Real
        auto desc = new dft_descriptor_c16(n);
        desc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                        oneapi::mkl::dft::config_value::NOT_INPLACE);
        std::vector<std::int64_t> strides(6, 0);
        desc->set_value(oneapi::mkl::dft::config_param::FWD_STRIDES, strides);
        desc->set_value(oneapi::mkl::dft::config_param::BWD_STRIDES, strides);
        desc->commit(jdftx_sycl::queue());
        return desc;
    }
    return nullptr;
}

// =====================================================================
// cufftPlan3d
// =====================================================================
inline cudaError_t cufftPlan3d(cufftHandle* plan, int nx, int ny, int nz, int type) {
    plan->desc = create_plan(type, nx, ny, nz);
    plan->type = type;
    plan->initialized = (plan->desc != nullptr);
    return cudaSuccess;
}

// =====================================================================
// cufftExecZ2Z: complex -> complex (forward/inverse)
// =====================================================================
inline cudaError_t cufftExecZ2Z(cufftHandle plan, const cuDoubleComplex* in,
                                 cuDoubleComplex* out, int direction) {
    if (!plan.initialized || !plan.desc) return 1;
    
    if (direction == CUFFT_FORWARD) {
        oneapi::mkl::dft::compute_forward(*plan.desc,
            reinterpret_cast<const std::complex<double>*>(in),
            reinterpret_cast<std::complex<double>*>(out),
            {});
    } else {
        oneapi::mkl::dft::compute_backward(*plan.desc,
            reinterpret_cast<const std::complex<double>*>(in),
            reinterpret_cast<std::complex<double>*>(out),
            {});
    }
    return cudaSuccess;
}

// =====================================================================
// cufftExecD2Z: real -> complex (forward only)
// =====================================================================
inline cudaError_t cufftExecD2Z(cufftHandle plan, const double* in,
                                 cuDoubleComplex* out) {
    if (!plan.initialized || !plan.desc) return 1;
    
    oneapi::mkl::dft::compute_forward(*plan.desc,
        in,
        reinterpret_cast<std::complex<double>*>(out),
        {});
    return cudaSuccess;
}

// =====================================================================
// cufftExecZ2D: complex -> real (backward only)
// =====================================================================
inline cudaError_t cufftExecZ2D(cufftHandle plan, const cuDoubleComplex* in,
                                 double* out) {
    if (!plan.initialized || !plan.desc) return 1;
    
    oneapi::mkl::dft::compute_backward(*plan.desc,
        reinterpret_cast<const std::complex<double>*>(in),
        out,
        {});
    return cudaSuccess;
}

// =====================================================================
// cufftDestroy
// =====================================================================
inline cudaError_t cufftDestroy(cufftHandle plan) {
    if (plan.initialized && plan.desc) {
        delete plan.desc;
    }
    plan.desc = nullptr;
    plan.initialized = false;
    return cudaSuccess;
}
