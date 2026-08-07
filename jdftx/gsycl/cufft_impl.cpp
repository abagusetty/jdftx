// gsycl/cufft_impl.cpp -- oneMKL DPC++ DFT implementation of the cufft.h
// shim declarations. Sole translation unit that includes
// <oneapi/mkl/dft.hpp> (transitively mkl_cblas.h). Never includes
// gsl_cblas.h, so no CBLAS enum redefinition clash.

#include "cufft.h"
#include <sycl/sycl.hpp>
#include <oneapi/mkl/dft.hpp>
#include <complex>
#include <cstdint>
#include <vector>

typedef oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
                                      oneapi::mkl::dft::domain::COMPLEX> dft_descriptor_c16;

namespace {

dft_descriptor_c16* create_plan(int type, int nx, int ny, int nz) {
    std::vector<std::int64_t> n = { (std::int64_t)nz, (std::int64_t)ny, (std::int64_t)nx };

    if (type == CUFFT_Z2Z) {
        auto desc = new dft_descriptor_c16(n);
        desc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                        oneapi::mkl::dft::config_value::NOT_INPLACE);
        desc->commit(jdftx_sycl::queue());
        return desc;
    }
    else if (type == CUFFT_D2Z) {
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

} // namespace

cudaError_t cufftPlan3d(cufftHandle* plan, int nx, int ny, int nz, int type) {
    plan->desc = create_plan(type, nx, ny, nz);
    plan->type = type;
    plan->initialized = (plan->desc != nullptr);
    return cudaSuccess;
}

cudaError_t cufftExecZ2Z(cufftHandle plan, const cuDoubleComplex* in,
                          cuDoubleComplex* out, int direction) {
    if (!plan.initialized || !plan.desc) return 1;
    auto* desc = static_cast<dft_descriptor_c16*>(plan.desc);
    if (direction == CUFFT_FORWARD) {
        oneapi::mkl::dft::compute_forward(*desc,
            reinterpret_cast<const std::complex<double>*>(in),
            reinterpret_cast<std::complex<double>*>(out),
            {});
    } else {
        oneapi::mkl::dft::compute_backward(*desc,
            reinterpret_cast<const std::complex<double>*>(in),
            reinterpret_cast<std::complex<double>*>(out),
            {});
    }
    return cudaSuccess;
}

cudaError_t cufftExecD2Z(cufftHandle plan, const double* in,
                          cuDoubleComplex* out) {
    if (!plan.initialized || !plan.desc) return 1;
    auto* desc = static_cast<dft_descriptor_c16*>(plan.desc);
    oneapi::mkl::dft::compute_forward(*desc,
        in,
        reinterpret_cast<std::complex<double>*>(out),
        {});
    return cudaSuccess;
}

cudaError_t cufftExecZ2D(cufftHandle plan, const cuDoubleComplex* in,
                          double* out) {
    if (!plan.initialized || !plan.desc) return 1;
    auto* desc = static_cast<dft_descriptor_c16*>(plan.desc);
    oneapi::mkl::dft::compute_backward(*desc,
        reinterpret_cast<const std::complex<double>*>(in),
        out,
        {});
    return cudaSuccess;
}

cudaError_t cufftDestroy(cufftHandle plan) {
    if (plan.initialized && plan.desc) {
        delete static_cast<dft_descriptor_c16*>(plan.desc);
    }
    plan.desc = nullptr;
    plan.initialized = false;
    return cudaSuccess;
}
